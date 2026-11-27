#!/usr/bin/env python3
"""Annotate every dataset image with LFM3-VL-2B and schema-constrained decoding."""
import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import html
import io
import json
import os
from pathlib import Path
import random
import sqlite3
import shutil
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import quote, urlencode, urlsplit
import zlib

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

from annotate_mirage import prepare_image, read_jsonl, sha, write_json
from lfm3_schema import load_taxonomies, json_schema, grammar, prompt, caption_prompt, validate
from split_sprite_sheets import propose_crops, save_crop

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LOCAL = ROOT.parent / 'ds/mother3-sync-latest-worktree'
WORK = ROOT / 'notebook/dataset/taxonomy/lfm3_work'
ATLAS_URL = 'ssh://git@atlas.methil.group:2222/sprite-pixelart-dataset.git'
ATLAS_RAW = 'https://atlas.methil.group/api/repos/sprite-pixelart-dataset/raw'
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tif', '.tiff', '.avif'}
SHEET_TYPES = {'sprite_sheet', 'animation_strip', 'tile_set'}
_source_downloads = threading.BoundedSemaphore(8)


def http_json(url, payload=None, timeout=300):
    request = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def inventory(args, token):
    path = args.output / 'inventory.json'
    if path.exists():
        document = json.loads(path.read_text())
        if document['repo'] != args.repo or (args.revision != 'main' and document['revision'] != args.revision):
            raise ValueError('Existing inventory uses another repository/revision; use a new --output')
        if not document.get('complete'):
            raise ValueError('Refusing an incomplete inventory')
        return document
    local_revision = None
    if args.local_root.is_dir():
        result = subprocess.run(['git', '-C', str(args.local_root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
        if result.returncode == 0:
            local_revision = result.stdout.strip()
    if getattr(args, 'atlas_checkout', None):
        revision = subprocess.check_output(
            ['git', '-C', str(args.atlas_checkout), 'rev-parse', '--verify', args.revision + '^{commit}'],
            text=True,
        ).strip() if args.revision != 'main' else local_revision
        if not revision:
            raise ValueError('Atlas checkout has no checked out commit')
        origin = 'atlas_git_tree'
    else:
        api = HfApi(token=token)
        revision = api.repo_info(args.repo, repo_type='dataset', revision=args.revision).sha
        origin = 'git_ls_tree_at_exact_hub_commit' if local_revision == revision else 'paginated_hub_tree'
    if origin != 'paginated_hub_tree':
        names = subprocess.check_output(
            ['git', '-C', str(args.local_root), 'ls-tree', '-rz', '--name-only', revision]
        ).decode().split('\0')
    else:
        # RepoInfo.siblings is truncated for large repos. Always paginate the full tree.
        names = []
        for entry in api.list_repo_tree(args.repo, repo_type='dataset', revision=revision, recursive=True):
            if hasattr(entry, 'size'):
                names.append(entry.path)
            if len(names) and len(names) % 25000 == 0:
                print(f'Inventaire HF : {len(names):,} fichiers...', flush=True)
        origin = 'paginated_hub_tree'
    images = sorted(p for p in names if p.startswith(('data/', 'source_sheets/')) and Path(p).suffix.lower() in EXTENSIONS)
    document = {'repo': args.repo, 'revision': revision, 'complete': True, 'origin': origin,
                'local_revision': local_revision, 'paths': images,
                'counts_by_root': dict(Counter(p.split('/')[0] for p in images))}
    write_json(path, document)
    print(f'Inventaire complet : {len(images):,} images · {document["counts_by_root"]}', flush=True)
    return document


def selected_images(paths, benchmark, limit, everything):
    if everything:
        return paths
    if not benchmark:
        return paths[:limit]
    # Deterministic sample spread over collections and including source assets.
    groups = defaultdict(list)
    for path in paths:
        parts = path.split('/')
        groups[(parts[0], parts[1])].append(path)
    randomizer = random.Random(42)
    keys = sorted(groups)
    randomizer.shuffle(keys)
    selected = []
    for root, quota in [('data', max(1, benchmark * 3 // 4)), ('source_sheets', max(1, benchmark // 4))]:
        candidates = [key for key in keys if key[0] == root]
        before = len(selected)
        for key in candidates[:quota]:
            selected.append(randomizer.choice(groups[key]))
        remaining = quota - (len(selected) - before)
        if remaining > 0:
            already = set(selected)
            extra = [p for key in candidates for p in groups[key] if p not in already]
            selected.extend(randomizer.sample(extra, min(remaining, len(extra))))
    return selected[:benchmark]


def open_database(path):
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS jobs (path TEXT PRIMARY KEY, local_path TEXT, parent_path TEXT, '
               'bbox TEXT, frame INTEGER DEFAULT 0, status TEXT DEFAULT "pending", '
               'attempts INTEGER DEFAULT 0, result TEXT, error TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT)')
    db.execute('CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status)')
    return db


def unpack_record(value):
    return zlib.decompress(value).decode('utf-8') if isinstance(value, bytes) else value


def temporary_source_path(args, name):
    folder = getattr(args, 'source_tmp_dir', args.output / 'source_tmp')
    return folder / (sha(name.encode()) + Path(name).suffix)


def cleanup_source(args, name):
    if getattr(args, 'ephemeral_downloads', False):
        path = temporary_source_path(args, name)
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + '.part').unlink(missing_ok=True)


def source_path(name, args, revision, token):
    local = args.local_root / name
    if local.is_file():
        with local.open('rb') as stream:
            pointer = stream.read(42).startswith(b'version https://git-lfs.github.com/spec/v1')
        if not pointer:
            # Only use actual files from the matching immutable dataset checkout.
            return local
    atlas = getattr(args, 'atlas_checkout', None)
    if atlas or getattr(args, 'ephemeral_downloads', False):
        class SafeRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, request, fp, code, msg, headers, url):
                redirected = super().redirect_request(request, fp, code, msg, headers, url)
                if redirected and urlsplit(url).netloc != urlsplit(request.full_url).netloc:
                    redirected.remove_header('Authorization')
                return redirected
        destination = temporary_source_path(args, name)
        destination.parent.mkdir(exist_ok=True)
        partial = destination.with_suffix(destination.suffix + '.part')
        if atlas:
            if not token:
                raise RuntimeError('Atlas file needs Git LFS hydration or an Atlas PAT in ATLAS_TOKEN/the Git credential helper')
            url = ATLAS_RAW + '?' + urlencode({'ref': revision, 'path': name})
        else:
            url = f'https://huggingface.co/datasets/{args.repo}/resolve/{revision}/{quote(name, safe="/")}'
        headers = {'Authorization': 'Bearer ' + token} if token else {}
        opener = urllib.request.build_opener(SafeRedirect())
        with _source_downloads:
            for attempt in range(3):
                try:
                    with opener.open(urllib.request.Request(url, headers=headers), timeout=120) as response, partial.open('wb') as stream:
                        shutil.copyfileobj(response, stream, length=1024 * 1024)
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                        raise
                    delay = exc.headers.get('Retry-After', '')
                    time.sleep(min(30, int(delay)) if delay.isdigit() else 2 ** (attempt + 1))
        if atlas:
            with partial.open('rb') as stream:
                if stream.read(42).startswith(b'version https://git-lfs.github.com/spec/v1'):
                    raise FileNotFoundError('Atlas has the LFS pointer but not the image bytes')
        partial.replace(destination)
        return destination
    return Path(hf_hub_download(args.repo, name, repo_type='dataset', revision=revision, token=token))


def model_image(path, max_side=768):
    image, measured = prepare_image(path, 'white', False)
    target = min(512, max_side) if max(image.size) < min(512, max_side) else max_side
    if max(image.size) != target:
        factor = target / max(image.size)
        image = image.resize((max(1, round(image.width * factor)), max(1, round(image.height * factor))), Image.Resampling.NEAREST)
    measured['model_input']['size'] = list(image.size)
    measured['model_input']['nearest_neighbor_scale'] = image.width / measured['width']
    measured['model_input']['rgb_sha256'] = sha(image.tobytes())
    measured['model_input']['max_side'] = max_side
    measured['palette_entry_count'] = len(measured['palette_hex'])
    measured['palette_truncated'] = len(measured['palette_hex']) > 64
    measured['palette_hex'] = measured['palette_hex'][:64]
    return image, measured


def inference(job, args, revision, token, dimensions, leaves, system_prompt, fingerprint):
    started = time.monotonic()
    name, generated_path, parent, bbox, frame = job
    path = Path(generated_path) if generated_path else source_path(name, args, revision, token)
    image, measured = model_image(path, args.max_side)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode()
    schema = json_schema(dimensions, leaves, measured['palette_size'], measured['has_transparency'])
    caption_schema = {'type': 'object', 'additionalProperties': False,
                      'properties': {'caption_candidate': schema['properties']['caption_candidate']},
                      'required': ['caption_candidate']}
    context = (f"Original size: {measured['width']}x{measured['height']}. "
               f"Source has transparent pixels: {measured['has_transparency']}. "
               f"Exact visible RGB count (alpha > 0): {measured['visible_rgb_color_count']}. "
               f"palette_size must be {measured['palette_size']}. Return compact JSON.")
    image_content = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + encoded}}
    caption_payload = {'model': 'LiquidAI/LFM3-VL-2B', 'messages': [
        {'role': 'system', 'content': caption_prompt()},
        {'role': 'user', 'content': [image_content, {'type': 'text', 'text': context}]}],
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'sprite_caption', 'strict': True, 'schema': caption_schema}},
        'temperature': 0.1, 'top_k': 50, 'top_p': 1.0, 'min_p': 0.0,
        'repeat_penalty': 1.1, 'max_tokens': 256, 'cache_prompt': True,
        'reasoning_effort': 'none', 'stream': False, 'seed': 42}
    caption = None
    for attempt in range(2):
        caption_response = http_json(args.server.rstrip('/') + '/v1/chat/completions', caption_payload)
        choice = caption_response['choices'][0]
        candidate = json.loads(choice['message']['content']).get('caption_candidate')
        if (choice['finish_reason'] == 'stop' and isinstance(candidate, str) and
                1 <= len(candidate.strip()) <= 640 and candidate.startswith(('A ', 'An ')) and
                candidate.strip().endswith(('.', '!', '?'))):
            caption = candidate.strip()
            break
        caption_payload['seed'] += 1
    if caption is None:
        raise ValueError('No complete caption after two attempts')
    # Repeat the already-generated caption as a forced prefix. This keeps the
    # small model anchored in its visual description before choosing categories.
    labels_schema = {'type': 'object', 'additionalProperties': False,
                     'properties': {'caption_candidate': {'type': 'string', 'enum': [caption]},
                                    'labels': schema['properties']['labels']},
                     'required': ['caption_candidate', 'labels']}
    payload = {'model': 'LiquidAI/LFM3-VL-2B', 'messages': [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': [{'type': 'text', 'text': context},
            image_content]}],
        # llama.cpp compiles this exact schema to GBNF and masks disallowed tokens.
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'sprite_annotation', 'strict': True, 'schema': labels_schema}},
        'temperature': 0.1, 'top_k': 50, 'top_p': 1.0, 'min_p': 0.0,
        'repeat_penalty': 1.1, 'max_tokens': 768, 'cache_prompt': True,
        'reasoning_effort': 'none', 'stream': False, 'seed': 42}
    failures = []
    for attempt in range(2):
        try:
            response = http_json(args.server.rstrip('/') + '/v1/chat/completions', payload)
            choice = response['choices'][0]
            if choice['finish_reason'] != 'stop':
                raise ValueError('Truncated model response')
            parsed = json.loads(choice['message']['content'])
            parsed['caption_candidate'] = caption
            parsed = validate(parsed, dimensions, leaves,
                              measured['palette_size'], measured['has_transparency'])
            break
        except (ValueError, KeyError, urllib.error.URLError) as exc:
            failures.append(type(exc).__name__)
            if attempt:
                raise
            payload['seed'] += 1
    labels = parsed['labels']
    result = {
        'asset_id': sha(name.encode())[:24], 'source': {'path': name, 'parent_path': parent,
            'bbox': json.loads(bbox) if bbox else None, 'frame': frame},
        'model': 'LiquidAI/LFM3-VL-2B', 'served_model': response.get('model'),
        'server_fingerprint': response.get('system_fingerprint'), 'run_config_id': fingerprint,
        'dataset_revision': revision, 'source_image_sha256': sha(path.read_bytes()),
        'labels': labels, 'taxonomy_paths': {
            key: [leaves[key][v]['path'] for v in value] if isinstance(value, list)
            else leaves[key][value]['path'] for key, value in labels.items()},
        'caption_candidate': parsed['caption_candidate'], 'caption_status': 'candidate_unreviewed',
        'review_status': 'auto_labeled', 'needs_taxonomy_review': True,
        'need_rework': None if labels['visual_quality'] == 'unknown' else labels['visual_quality'] == 'needs_rework',
        'rights_status': 'unknown', 'measurements': measured,
        'decoding': {'backend': 'llama.cpp_json_schema_to_gbnf', 'reasoning': False,
                     'schema_sha256': sha(json.dumps(labels_schema, sort_keys=True).encode()),
                     'caption_schema_sha256': sha(json.dumps(caption_schema, sort_keys=True).encode()),
                     'validation_retries': failures},
        'timings': response.get('timings'), 'usage': response.get('usage'),
        'caption_timings': caption_response.get('timings'), 'caption_usage': caption_response.get('usage'),
        'wall_seconds': time.monotonic() - started,
        'annotated_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    preview = args.output / 'previews' / (result['asset_id'] + '.png')
    # Keep bounded review previews; originals and derived crops remain available.
    if args.benchmark or args.keep_previews or name in args.preview_paths:
        image.save(preview)
    return result, path


def schedule_splits(db, result, path, args, overrides):
    name = result['source']['path']
    generated = result['source']['parent_path'] is not None
    existing = args.existing_splits.get(name)
    if existing and not generated:
        result['split'] = {'status': 'existing_extraction_linked',
                           'existing_sprite_paths': existing,
                           'provenance': 'dataset_metadata_and_reviewed_sheet_manifest'}
        return
    with Image.open(path) as image:
        frame_count = getattr(image, 'n_frames', 1)
        # Every animated frame becomes a separate annotation, even if visually identical.
        if not generated and frame_count > 1:
            folder = args.output / 'crops' / result['asset_id']
            folder.mkdir(parents=True, exist_ok=True)
            for frame in range(frame_count):
                image.seek(frame)
                destination = folder / f'frame_{frame:05d}.png'
                image.convert('RGBA').save(destination)
                child = f'derived/{result["asset_id"]}/frame_{frame:05d}.png'
                db.execute('INSERT OR IGNORE INTO jobs(path,local_path,parent_path,frame) VALUES(?,?,?,?)',
                           (child, str(destination.resolve()), name, frame))
            result['split'] = {'status': 'frames_extracted', 'children': frame_count}
            return
        is_sheet = result['labels']['asset_type'] in SHEET_TYPES
        source_probe = not generated and name.startswith('source_sheets/')
        if not is_sheet and name not in overrides and not source_probe:
            result['split'] = {'status': 'not_a_sheet' if result['labels']['asset_type'] != 'unknown' else 'pending_classification'}
            return
        # Crops are not recursively split into detached limbs or pixels.
        if generated:
            result['split'] = {'status': 'pending_crop_review', 'reason': 'Crop still classified as a sheet'}
            return
        boxes, split = propose_crops(image, overrides.get(name))
        if source_probe:
            split['trigger'] = 'source_image_geometry_checked_independently_of_model_label'
            split['model_asset_type'] = result['labels']['asset_type']
            if not boxes and not is_sheet and max(image.size) <= 128:
                split = {'status': 'not_a_sheet', 'source_geometry_checked': True}
            elif not boxes and result['labels']['asset_type'] == 'background':
                split = {'status': 'not_a_sheet', 'source_geometry_checked': True}
        result['split'] = split
        if boxes:
            folder = args.output / 'crops' / result['asset_id']
            folder.mkdir(parents=True, exist_ok=True)
            for i, bbox in enumerate(boxes):
                destination = folder / f'crop_{i:05d}.png'
                save_crop(image, bbox, destination)
                child = f'derived/{result["asset_id"]}/crop_{i:05d}.png'
                db.execute('INSERT OR IGNORE INTO jobs(path,local_path,parent_path,bbox) VALUES(?,?,?,?)',
                           (child, str(destination.resolve()), name, json.dumps(bbox)))


def export(db, output, filename=None, compress=False):
    filename = filename or ('annotations_lfm3.jsonl.gz' if compress else 'annotations_lfm3.jsonl')
    temporary = output / (filename + '.tmp')
    opener = gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=3) if compress else temporary.open('w', encoding='utf-8')
    with opener as stream:
        for (result,) in db.execute('SELECT result FROM jobs WHERE status="done" ORDER BY rowid'):
            stream.write(unpack_record(result) + '\n')
    temporary.replace(output / filename)


def existing_splits(local_root, inventory_paths, output):
    """Reuse reviewed source-to-crop links; never use old captions or model labels."""
    cache = output / 'existing_splits.json'
    if cache.exists():
        return json.loads(cache.read_text())
    sheet_metadata = local_root / 'sheet_metadata.jsonl'
    if not sheet_metadata.exists():
        return {}
    reviewed = set()
    with sheet_metadata.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip() or line.startswith('version https://git-lfs'):
                continue
            row = json.loads(line)
            status = str(row.get('status', ''))
            if ('extracted_and_reviewed' in status or status in {
                'individual_sprite_components_extracted', 'labeled_character_frames_imported',
                'frame_sections_mapped_and_duplicate_cells_linked'}) and 'partial' not in status:
                reviewed.add(row.get('file_name'))
    paths_set, links = set(inventory_paths), defaultdict(set)
    metadata_files = [local_root / 'metadata.jsonl', *sorted((local_root / 'metadata_updates').glob('*.jsonl'))]
    for path in metadata_files:
        if not path.is_file():
            continue
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip() or line.startswith('version https://git-lfs'):
                    continue
                row = json.loads(line)
                source = row.get('source_file') or row.get('source_sheet')
                target = row.get('file_name')
                if isinstance(source, str) and source in reviewed and target in paths_set:
                    links[source].add(target)
    result = {source: sorted(targets) for source, targets in links.items() if targets}
    write_json(cache, result)
    print(f'{len(result):,} planches déjà découpées avec liens vérifiés', flush=True)
    return result


def progress(db, args, total_sources, run_started, initial_done):
    statuses = dict(db.execute('SELECT status,COUNT(*) FROM jobs GROUP BY status'))
    done = statuses.get('done', 0)
    elapsed = time.monotonic() - run_started
    rate = (done - initial_done) / elapsed if elapsed else 0
    summary = {'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
               'original_source_images': total_sources, 'jobs_including_crops': sum(statuses.values()),
               'statuses': statuses, 'images_per_second_this_run': rate,
               'eta_current_pending_hours': (statuses.get('pending', 0) + statuses.get('running', 0) + statuses.get('cpu_running', 0)) / rate / 3600 if rate else None,
               'eta_scope': 'Current queue only; further sheet crops, retries and review can increase it',
               'workers': args.workers}
    summary['generated_output_tokens_this_run'] = getattr(args, 'output_tokens_this_run', 0)
    summary['output_tokens_per_second_this_run'] = summary['generated_output_tokens_this_run'] / elapsed if elapsed else 0
    write_json(args.output / 'progress.json', summary)
    return summary


def review(db, output, limit=24):
    cards = []
    rows = list(db.execute('SELECT result FROM jobs WHERE status="done" ORDER BY rowid LIMIT ?', (limit,)))
    for (serialized,) in rows:
        row = json.loads(unpack_record(serialized))
        preview = output / 'previews' / (row['asset_id'] + '.png')
        image = ('<img src="data:image/png;base64,' + base64.b64encode(preview.read_bytes()).decode() + '">') if preview.exists() else ''
        cards.append('<article><h2>' + html.escape(row['source']['path']) + '</h2>' + image +
                     '<p>' + html.escape(row['caption_candidate']) + '</p><pre>' +
                     html.escape(json.dumps({'labels': row['labels'], 'split': row['split']}, ensure_ascii=False, indent=2)) + '</pre></article>')
    (output / 'review.html').write_text('<!doctype html><html lang="fr"><meta charset="utf-8"><title>LFM3 — annotations nouvelles</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:auto;padding:32px;background:#f2f4f7}article{padding:24px;background:white;border-radius:16px;margin:24px 0;overflow:auto}'
        'img{float:right;max-width:55%;max-height:512px;image-rendering:pixelated;margin:16px}h2{font-size:16px;overflow-wrap:anywhere}pre{white-space:pre-wrap}</style>'
        '<h1>Nouvelles annotations LFM3-VL-2B</h1><p>Décodage contraint, annotations non vérifiées. '
        'La grammaire garantit la structure ; elle ne garantit pas la justesse de la description. '
        'Les découpes automatiques restent à vérifier.</p>' + ''.join(cards) + '</html>', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='ethanzxv/pixelart-dataset')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--local-root', type=Path, default=DEFAULT_LOCAL)
    parser.add_argument('--atlas-checkout', type=Path, help='Use a local Atlas Git checkout as the pinned dataset source')
    parser.add_argument('--path-prefix', help='Only queue source images under this repository-relative directory')
    parser.add_argument('--output', type=Path, default=WORK / 'full')
    parser.add_argument('--server', default='http://127.0.0.1:8091')
    parser.add_argument('--backend', choices=['llama.cpp', 'vllm'], default='llama.cpp')
    parser.add_argument('--model-revision', default='23387beda43c60310348e83e25f78f33c3140d3a')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--all', action='store_true')
    selection.add_argument('--benchmark', type=int, default=0)
    selection.add_argument('--limit', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-side', type=int, default=768)
    parser.add_argument('--keep-previews', action='store_true')
    parser.add_argument('--split-overrides', type=Path, help='JSON mapping source paths to exact grid dimensions')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--compress-output', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if min(args.limit, args.workers, args.max_side) < 1 or args.benchmark < 0:
        parser.error('limits, workers and max-side must be positive')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.compress_output = args.backend == 'vllm' if args.compress_output is None else args.compress_output
    args.ephemeral_downloads = args.backend == 'vllm' or bool(args.atlas_checkout)
    db = open_database(args.output / 'annotations.sqlite3')
    if args.export_only:
        export(db, args.output, 'annotations_lfm3.snapshot.jsonl.gz' if args.compress_output else 'annotations_lfm3.snapshot.jsonl', compress=args.compress_output)
        review(db, args.output)
        return
    lock = (args.output / '.runner.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('A runner is already using this output directory')
    if args.ephemeral_downloads and (args.output / 'source_tmp').exists():
        for temporary in (args.output / 'source_tmp').iterdir():
            if temporary.is_file():
                temporary.unlink()
    load_dotenv(ROOT / 'dataset-viewer/.env', override=False)
    if args.atlas_checkout:
        args.local_root = args.atlas_checkout.resolve()
        origin = subprocess.run(['git', '-C', str(args.local_root), 'remote', 'get-url', 'origin'],
                                check=True, capture_output=True, text=True).stdout.strip()
        if origin != ATLAS_URL:
            parser.error('The Atlas checkout must use the Atlas dataset SSH URL as origin')
        args.repo = ATLAS_URL
        token = os.environ.get('ATLAS_TOKEN')
        if not token:
            credential = subprocess.run(
                ['git', 'credential', 'fill'], input='protocol=https\nhost=atlas.methil.group\n\n',
                cwd=args.local_root, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GIT_ASKPASS': ''},
                text=True, capture_output=True,
            )
            token = next((line.removeprefix('password=') for line in credential.stdout.splitlines()
                          if line.startswith('password=')), None) if credential.returncode == 0 else None
    else:
        token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_ACCESS_TOKEN')
    doc = inventory(args, token)
    current_local = subprocess.run(['git', '-C', str(args.local_root), 'rev-parse', 'HEAD'],
                                   text=True, capture_output=True)
    if current_local.returncode or current_local.stdout.strip() != doc['revision']:
        # Never mix current local files with an older pinned Hub revision.
        args.local_root = args.output / 'no_matching_local_checkout'
    if args.backend == 'vllm':
        props = http_json(args.server.rstrip('/') + '/v1/models')
        served_model = props['data'][0]['id']
        from lfm3_vllm import inference as target_inference, REVIEW_INSTRUCTION
    else:
        props = http_json(args.server.rstrip('/') + '/props')
        served_model = props.get('model_path') or props.get('model_alias') or ''
        target_inference = inference
    if 'lfm3-vl-2b' not in served_model.lower():
        raise SystemExit('The server is not serving LFM3-VL-2B')
    dimensions, leaves, documents = load_taxonomies()
    overrides = json.loads(args.split_overrides.read_text()) if args.split_overrides else {}
    config = {'pipeline': 'lfm3_independent_v4_direct_caption', 'repo': args.repo, 'revision': doc['revision'],
              'taxonomies': {key: sha(json.dumps(value, sort_keys=True).encode()) for key, value in documents.items()},
              'server_model': served_model, 'max_side': args.max_side,
              'schema_decoder': 'llama.cpp_json_schema_to_gbnf', 'reasoning': False,
              'split_overrides': overrides, 'temperature': 0.1, 'seed': 42,
              'prompt_sha256': sha(prompt(dimensions, leaves).encode()),
              'caption_prompt_sha256': sha(caption_prompt().encode()),
              'selection': {'all': args.all, 'benchmark': args.benchmark, 'limit': args.limit}}
    if args.backend == 'vllm':
        config.update(pipeline='lfm3_bf16_vllm_self_review_v5_compact_storage', model_revision=args.model_revision,
                      weight_dtype='bfloat16', quantization=None, schema_decoder='vllm_xgrammar',
                      self_review=True, review_prompt_sha256=sha(REVIEW_INSTRUCTION.encode()))
    config.update(compressed_output=args.compress_output, ephemeral_source_downloads=args.ephemeral_downloads,
                  retained_palette_entries=64)
    relevant = doc['paths']
    if args.path_prefix:
        prefix = args.path_prefix.rstrip('/') + '/'
        relevant = [path for path in relevant if path.startswith(prefix)]
        if not relevant:
            parser.error('No source images match --path-prefix')
        config['selection']['path_prefix'] = prefix
    fingerprint = sha(json.dumps(config, sort_keys=True).encode())
    previous = db.execute('SELECT value FROM config WHERE key="fingerprint"').fetchone()
    if previous and previous[0] != fingerprint:
        raise SystemExit('Configuration changed; use a different --output')
    db.execute('INSERT OR REPLACE INTO config VALUES("fingerprint",?)', (fingerprint,))
    write_json(args.output / 'run_manifest.json', {**config, 'config_id': fingerprint,
               'started_at_utc': datetime.now(timezone.utc).isoformat()})
    write_json(args.output / 'annotation.schema.json', json_schema(dimensions, leaves))
    (args.output / 'annotation.gbnf').write_text(grammar(dimensions, leaves))
    paths = selected_images(relevant, args.benchmark, args.limit, args.all)
    args.preview_paths = set(paths[:24])
    args.existing_splits = existing_splits(args.local_root, doc['paths'], args.output) if any(
        path.startswith('source_sheets/') for path in relevant
    ) else {}
    db.executemany('INSERT OR IGNORE INTO jobs(path) VALUES(?)', ((p,) for p in paths))
    db.execute('UPDATE jobs SET status="pending",attempts=0 WHERE status="running"')
    db.execute('UPDATE jobs SET attempts=0 WHERE status="pending" AND attempts>=3')
    # Failed images remain explicit and are retried on the next invocation.
    db.execute('UPDATE jobs SET status="pending",attempts=0 WHERE status="failed"')
    db.commit()
    export(db, args.output, compress=args.compress_output)  # Reconcile the readable export after an interrupted append.
    (args.output / 'previews').mkdir(exist_ok=True)
    system_prompt = prompt(dimensions, leaves)
    started = time.monotonic()
    initial_done = db.execute('SELECT COUNT(*) FROM jobs WHERE status="done"').fetchone()[0]
    measured_times, completions, consecutive_failures = [], 0, 0
    args.output_tokens_this_run = 0
    last_progress = 0
    print(f'PID {os.getpid()} · {len(paths):,} sources sélectionnées · workers={args.workers}', flush=True)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {}
            while True:
                while len(pending) < args.workers:
                    if shutil.disk_usage(args.output).free < 256 * 1024 * 1024:
                        raise RuntimeError('Disk space below 256 MiB; queue preserved before filling the disk')
                    clause = ' AND parent_path IS NULL' if args.benchmark else ''
                    job = db.execute('SELECT path,local_path,parent_path,bbox,frame FROM jobs WHERE status="pending" AND attempts<3' + clause + ' ORDER BY rowid LIMIT 1').fetchone()
                    if not job:
                        break
                    db.execute('UPDATE jobs SET status="running",attempts=attempts+1 WHERE path=?', (job[0],))
                    db.commit()
                    future = pool.submit(target_inference, job, args, doc['revision'], token, dimensions, leaves, system_prompt, fingerprint)
                    pending[future] = job
                if not pending:
                    break
                done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                if not done:
                    progress(db, args, len(relevant), started, initial_done)
                for future in done:
                    job = pending.pop(future)
                    try:
                        result, path = future.result()
                        schedule_splits(db, result, path, args, overrides)
                        serialized = json.dumps(result, ensure_ascii=False)
                        stored = zlib.compress(serialized.encode(), level=3) if args.compress_output else serialized
                        db.execute('UPDATE jobs SET status="done",result=?,error=NULL WHERE path=?', (stored, job[0]))
                        db.commit()
                        stream = (gzip.open(args.output / 'annotations_lfm3.jsonl.gz', 'at', encoding='utf-8', compresslevel=3)
                                  if args.compress_output else (args.output / 'annotations_lfm3.jsonl').open('a', encoding='utf-8'))
                        with stream:
                            stream.write(serialized + '\n')
                        cleanup_source(args, job[0])
                        usage = result.get('self_review')
                        if usage:
                            args.output_tokens_this_run += sum(usage[key].get('completion_tokens', 0) for key in ['draft_usage', 'review_usage'])
                        else:
                            args.output_tokens_this_run += sum((result.get(key) or {}).get('completion_tokens', 0) for key in ['usage', 'caption_usage'])
                        measured_times.append(result['wall_seconds'])
                        completions += 1
                        consecutive_failures = 0
                        print(f'{initial_done + completions:,} annotées · {job[0]} · {result["wall_seconds"]:.2f}s · {result["split"]["status"]}', flush=True)
                    except Exception as exc:
                        db.rollback()
                        cleanup_source(args, job[0])
                        # Report HTTP status/type without request headers or signed URLs.
                        attempts = db.execute('SELECT attempts FROM jobs WHERE path=?', (job[0],)).fetchone()[0]
                        error = type(exc).__name__ + (f' HTTP {exc.code}' if isinstance(exc, urllib.error.HTTPError) else '')
                        if isinstance(exc, ValueError):
                            error += ': ' + str(exc)[:500]
                        db.execute('UPDATE jobs SET status=?,error=? WHERE path=?', ('failed' if attempts >= 3 else 'pending', error, job[0]))
                        db.commit()
                        print(f'Échec {job[0]} : {error} ({attempts}/3)', flush=True)
                        # Bad annotations belong to the per-image retry/review
                        # queue. Only repeated infrastructure failures stop bulk work.
                        consecutive_failures = 0 if isinstance(exc, ValueError) else consecutive_failures + 1
                        if consecutive_failures >= 5:
                            raise RuntimeError('Five consecutive failures; stopping for diagnosis, queue preserved') from exc
                    if time.monotonic() - last_progress >= 1:
                        progress(db, args, len(relevant), started, initial_done)
                        last_progress = time.monotonic()
                    if completions and completions % 25 == 0:
                        review(db, args.output)
    finally:
        summary = progress(db, args, len(relevant), started, initial_done)
        export(db, args.output, compress=args.compress_output)
        review(db, args.output)
        split_statuses = Counter()
        for (serialized,) in db.execute('SELECT result FROM jobs WHERE status="done"'):
            split_statuses[json.loads(unpack_record(serialized))['split']['status']] += 1
        summary['split_statuses'] = dict(split_statuses)
        summary['all_annotation_jobs_complete'] = all(status == 'done' for status in summary['statuses'])
        summary['all_sheet_splits_complete'] = not any(k.startswith('pending') for k in split_statuses)
        if measured_times:
            summary['median_job_wall_seconds'] = statistics.median(measured_times)
            summary['mean_job_wall_seconds'] = statistics.mean(measured_times)
            rate = summary['images_per_second_this_run']
            summary['projected_original_images_hours'] = len(doc['paths']) / rate / 3600 if rate else None
            summary['projection_excludes_additional_crops_and_review'] = True
        write_json(args.output / 'summary.json', summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
