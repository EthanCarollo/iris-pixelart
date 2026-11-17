#!/usr/bin/env python3
"""Publish a consistent, append-only LFM3 checkpoint to the private source dataset."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tarfile

from dotenv import load_dotenv
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

from annotate_lfm3 import ROOT, unpack_record
from lfm3_schema import load_taxonomies, validate

WORK = ROOT / 'notebook/dataset/taxonomy/lfm3_work'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def prepare_snapshot(source, staging):
    """SQLite backup includes both writers and excludes all later transactions."""
    staging.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source / 'run_manifest.json').read_text())
    database = staging / 'annotations.sqlite3'
    with sqlite3.connect(f'file:{source / "annotations.sqlite3"}?mode=ro', uri=True) as live:
        live.execute('BEGIN')
        live.execute('SELECT COUNT(*) FROM jobs').fetchone()
        with sqlite3.connect(database) as snapshot:
            live.backup(snapshot, pages=256, sleep=0.01)
    db = sqlite3.connect(database)
    db.execute('PRAGMA journal_mode=DELETE')
    if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
        raise ValueError('SQLite snapshot integrity check failed')
    statuses = dict(db.execute('SELECT status,COUNT(*) FROM jobs GROUP BY status'))
    dimensions, leaves, _ = load_taxonomies()
    devices, split_states = Counter(), Counter()
    assets = {}
    asset_rows = []
    for path, local_path, parent, bbox, frame, status in db.execute(
            'SELECT path,local_path,parent_path,bbox,frame,status FROM jobs WHERE parent_path IS NOT NULL'):
        local = Path(local_path).resolve()
        relative = local.relative_to(source.resolve())
        if not local.is_file() or relative.parts[0] != 'crops':
            raise ValueError(f'Missing or invalid derived image: {path}')
        checksum = digest(local)
        remote = f'data/lfm3_extracted/{checksum}.png'
        assets.setdefault(remote, local)
        asset_rows.append({'path': path, 'dataset_path': remote, 'checkpoint_path': relative.as_posix(),
                           'parent_path': parent, 'bbox': json.loads(bbox) if bbox else None,
                           'frame': frame, 'sha256': checksum, 'annotation_status': status})
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, target)
    by_path = {row['path']: row for row in asset_rows}
    with gzip.open(staging / 'derived_assets.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for row in asset_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    count = 0
    with gzip.open(staging / 'annotations_lfm3.jsonl.gz', 'wt', encoding='utf-8', compresslevel=6) as stream:
        for path, serialized in db.execute('SELECT path,result FROM jobs WHERE status="done" ORDER BY rowid'):
            row = json.loads(unpack_record(serialized))
            if row['run_config_id'] != manifest['config_id']:
                raise ValueError('Mixed run configuration in the checkpoint')
            measured = row['measurements']
            validate({key: row[key] for key in ['labels', 'caption_candidate']}, dimensions, leaves,
                     measured['palette_size'], measured['has_transparency'])
            if not row.get('self_review', {}).get('performed'):
                raise ValueError('A BF16 record has no self-review')
            row['source']['dataset_path'] = by_path[path]['dataset_path'] if path in by_path else path
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            devices[row.get('execution_device', 'cuda')] += 1
            split_states[row.get('split', {}).get('status', 'unknown')] += 1
            count += 1
    # The resume database uses portable crop paths and pending states. Never
    # mutate the live queue, including its GPU and CPU claims.
    for asset in asset_rows:
        db.execute('UPDATE jobs SET local_path=? WHERE path=?', (asset['checkpoint_path'], asset['path']))
    db.execute('UPDATE jobs SET status="pending",attempts=0 WHERE status IN ("running","cpu_running")')
    db.execute('UPDATE jobs SET attempts=0 WHERE status="pending"')
    db.commit()
    db.close()
    with database.open('rb') as stream, gzip.open(staging / 'annotations.sqlite3.gz', 'wb', compresslevel=6) as target:
        shutil.copyfileobj(stream, target)
    database.unlink()
    for name in ['run_manifest.json', 'annotation.schema.json', 'annotation.gbnf', 'existing_splits.json']:
        if (source / name).is_file():
            shutil.copy2(source / name, staging / name)
    with (source / 'inventory.json').open('rb') as stream, gzip.open(staging / 'inventory.json.gz', 'wb') as target:
        shutil.copyfileobj(stream, target)
    for name in ['taxonomies.json', 'taxonomies_couleurs.json']:
        shutil.copy2(ROOT / 'notebook/dataset/taxonomy' / name, staging / name)
    summary = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'dataset_repo': manifest['repo'], 'source_dataset_revision': manifest['revision'],
               'run_config_id': manifest['config_id'], 'annotations': count,
               'snapshot_job_states': statuses, 'annotations_by_device': dict(devices),
               'split_states': dict(split_states), 'derived_jobs': len(asset_rows),
               'unique_derived_images': len(assets), 'model': 'LiquidAI/LFM3-VL-2B',
               'weight_dtype': 'bfloat16', 'quantization': None,
               'model_revision': manifest['model_revision'], 'self_review': True,
               'human_verified': False, 'complete': all(state == 'done' for state in statuses),
               'resume': 'Decompress SQLite and inventory; rebase relative crop local_path values to the restored directory.',
               'export': 'annotations_lfm3.jsonl.gz merges CPU and GPU; source.dataset_path locates each image on the Hub.'}
    write_json(staging / 'summary.json', summary)
    return summary, assets


def bundle_experiments(staging):
    """Explicit artifact allowlist: no tokens, SSH files, environments or logs."""
    roots = []
    for name in ['full', 'benchmark', 'benchmark_v2', 'benchmark_v3', 'benchmark_v4',
                 'caption_improvement', 'bf16_smoke']:
        if (WORK / name).is_dir():
            roots.append(WORK / name)
    for name in ['perf', 'perf_fast', 'perf128', 'bench_c8', 'bench_c16', 'bench_c32', 'bench_c64']:
        base = WORK / 'remote' / name
        if base.is_dir():
            roots.extend([base] if name.startswith('bench') else [p for p in base.iterdir() if p.is_dir()])
    mirage = ROOT / 'notebook/dataset/taxonomy/mirage_work/test_image'
    if mirage.is_dir():
        roots.append(mirage)
    names = {'annotations_lfm3.jsonl', 'annotations_lfm3.jsonl.gz', 'annotations_mirage.jsonl',
             'summary.json', 'response.json', 'dedicated_caption_response.json',
             'dedicated_caption_complete_response.json', 'direct_caption_response.json',
             'result.json', 'result_checked.json', 'result_layout_grammar.json', 'input.png'}
    selected = set()
    for base in roots:
        selected.update(path for path in base.iterdir() if path.is_file() and path.name in names)
        if (base / 'previews').is_dir():
            selected.update((base / 'previews').glob('*.png'))
    first = ROOT / 'notebook/dataset/analysis/mirage_work'
    selected.update(first / name for name in ['first_source.png', 'first_input.png', 'first_mirage.json'] if (first / name).is_file())
    for name in ['benchmark_comparison.json', 'cpu_full_benchmark.json', 'verified_server_environment.json']:
        if (WORK / 'remote' / name).is_file():
            selected.add(WORK / 'remote' / name)
    with tarfile.open(staging / 'experiments.tar.gz', 'w:gz') as archive:
        for path in sorted(selected):
            archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    write_json(staging / 'experiments_manifest.json', {
        'purpose': 'Benchmark and quality diagnostics; excluded from the current annotation export.',
        'files': [path.relative_to(ROOT).as_posix() for path in sorted(selected)]})


def publish(staging, summary, assets, token, snapshot_id):
    api = HfApi(token=token)
    repo = summary['dataset_repo']
    info = api.repo_info(repo, repo_type='dataset')
    if not info.private:
        raise ValueError('Expected the existing private dataset')
    prefix = f'annotations/lfm3_vl_bf16/checkpoints/{snapshot_id}'
    if api.get_paths_info(repo, prefix, repo_type='dataset', revision=info.sha):
        raise ValueError('Checkpoint already exists; use a fresh snapshot ID')
    operations = []
    # Content-addressed image paths make repeated checkpoints share identical
    # crops, while derived_assets.jsonl retains every frame and parent alias.
    existing = {entry.path: entry for entry in api.get_paths_info(repo, list(assets),
                repo_type='dataset', revision=info.sha)} if assets else {}
    for remote, local in assets.items():
        if remote in existing:
            cached = Path(hf_hub_download(repo, remote, repo_type='dataset', revision=info.sha, token=token))
            if digest(cached) != digest(local):
                raise ValueError('Content-addressed image collision')
        else:
            operations.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)))
    files = [path for path in staging.rglob('*') if path.is_file()]
    # Crop bytes live in data/, with the exact resume layout supplied by the
    # derived-assets map. Avoid storing a second image copy in each checkpoint.
    for path in files:
        if path.relative_to(staging).parts[0] != 'crops':
            operations.append(CommitOperationAdd(path_in_repo=f'{prefix}/{path.relative_to(staging).as_posix()}',
                                                 path_or_fileobj=str(path)))
    commit = api.create_commit(repo_id=repo, repo_type='dataset', revision='main',
        parent_commit=info.sha, operations=operations, num_threads=4,
        commit_message=f'feat(annotations): snapshot {summary["annotations"]} reviewed LFM3 BF16 annotations',
        commit_description='Merge a consistent GPU/CPU snapshot, resume state, source taxonomies, derived images and experiment diagnostics. Automatic annotations remain unverified by a human.')
    # Verify the remote checkpoint receipt against its locally computed hashes.
    remote_manifest = Path(hf_hub_download(repo, f'{prefix}/manifest.json', repo_type='dataset', revision=commit.oid, token=token))
    if digest(remote_manifest) != digest(staging / 'manifest.json'):
        raise ValueError('Remote checkpoint verification failed')
    return {'repo': repo, 'commit': commit.oid, 'commit_url': commit.commit_url,
            'checkpoint': prefix, 'annotations': summary['annotations'],
            'unique_derived_images': len(assets)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=WORK / 'remote/full_bf16')
    parser.add_argument('--snapshot-id', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    parser.add_argument('--upload', action='store_true', help='Publish after preparing and validating the snapshot')
    args = parser.parse_args()
    load_dotenv(ROOT / 'dataset-viewer/.env', override=False)
    staging = WORK / 'sync' / args.snapshot_id
    summary, assets = prepare_snapshot(args.source.resolve(), staging)
    bundle_experiments(staging)
    code_revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    files = {path.relative_to(staging).as_posix(): {'sha256': digest(path), 'size': path.stat().st_size}
             for path in staging.rglob('*') if path.is_file() and path.relative_to(staging).parts[0] != 'crops'}
    write_json(staging / 'manifest.json', {**summary, 'code_revision': code_revision, 'files': files})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.upload:
        receipt = publish(staging, summary, assets, os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN'), args.snapshot_id)
        write_json(staging / 'upload_receipt.json', receipt)
        print(json.dumps(receipt, indent=2), flush=True)
    else:
        print(f'Prepared checkpoint: {staging}. Use --upload with a new snapshot ID to publish.')


if __name__ == '__main__':
    main()
