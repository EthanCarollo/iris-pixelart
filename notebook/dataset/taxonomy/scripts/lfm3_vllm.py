"""Batched vLLM BF16 requests: draft, visual self-review, validated final annotation."""
import base64
import copy
from datetime import datetime, timezone
import http.client
import io
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

from annotate_mirage import sha
from lfm3_schema import json_schema, prompt, validate

_connections = threading.local()
_preprocessing = threading.BoundedSemaphore(4)
REVIEW_INSTRUCTION = (
    'VISUAL SELF-REVIEW. Inspect the image again and audit the provisional annotation. '
    'The pixels and exact source measurements are the evidence; the draft can be wrong. '
    'Check every label and every descriptive claim, especially single sprite versus sheet, '
    'humanoid versus object, body parts, view, pose, hat versus hair, colors and their positions. '
    'Correct contradictions between caption and labels. Remove unsupported identity, emotion, '
    'material, game role, and claims of glowing or light emission unsupported by surrounding pixels. '
    'Use unknown when a category cannot be established. Keep supported details. '
    'FIRST inspect instance_layout: a single creature with many limbs or a symmetric body is '
    'one subject, NOT a sprite sheet. A sprite sheet/strip requires multiple separate '
    'sprites or repeated frames. A single isolated icon is also one subject. '
    'The final caption must start directly with A or An, describe the subject and the spatial '
    'arrangement of distinctive parts, and end as a complete sentence. '
    'Return ONLY visual_checks, corrected caption_candidate and labels. '
    'Do not add commentary about the review. Recheck the evidence rather than confirming '
    'the draft merely because it is fluent.')

LAYOUT_SCHEMA = {'type': 'object', 'additionalProperties': False,
                 'properties': {'instance_layout': {'type': 'string', 'enum': [
                     'single_subject', 'multiple_distinct_sprites_or_frames',
                     'repeated_tiles', 'scene', 'uncertain']}},
                 'required': ['instance_layout']}


def constrain_layout(schema, layout):
    """Cross-field grammar: one isolated subject cannot decode to a spritesheet."""
    result = copy.deepcopy(schema)
    allowed = {
        'single_subject': {'single_sprite', 'background', 'interface_element', 'font_glyph', 'unknown'},
        'multiple_distinct_sprites_or_frames': {'sprite_sheet', 'animation_strip', 'unknown'},
        'repeated_tiles': {'tile_set', 'sprite_sheet', 'unknown'},
    }.get(layout)
    if allowed:
        labels = result['properties']['labels']
        for branch in labels.get('oneOf', [labels]):
            spec = branch['properties']['asset_type']
            spec['enum'] = [v for v in spec['enum'] if v in allowed]
    return result


def normalize_color_set(result, stage, changes):
    """Repeated identical colors carry no new information in a multi-label set."""
    colors = result.get('labels', {}).get('dominant_color_families')
    if isinstance(colors, list) and all(isinstance(value, str) for value in colors):
        unique = list(dict.fromkeys(colors))
        if unique != colors:
            changes.append({'stage': stage, 'field': 'dominant_color_families',
                            'operation': 'remove_identical_duplicates', 'before': colors, 'after': unique})
            result['labels']['dominant_color_families'] = unique
    return result


def decoding_schema(schema):
    """xgrammar doesn't enforce uniqueItems; the Python validator still must."""
    result = copy.deepcopy(schema)
    def visit(node):
        if isinstance(node, dict):
            node.pop('uniqueItems', None)
            if 'pattern' in node:
                # The caption regex already encodes its length bounds. xgrammar
                # rejects an additional minLength/maxLength alongside pattern.
                node.pop('minLength', None)
                node.pop('maxLength', None)
                # xgrammar 0.2.8's FSM hashing fails on this large bounded
                # repetition. Keep the A/An + final punctuation grammar and
                # enforce the 640-character cap in the final Python validator.
                if node['pattern'] == r'^A(n)? [^\n\r]{1,636}[.!?]$':
                    node['pattern'] = r'^A(n)? [^\n\r]+[.!?]$'
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(result)
    return result


def chat(server, messages, schema, seed=42, max_tokens=768):
    target = urlsplit(server)
    key = (target.hostname, target.port or 80)
    connection = getattr(_connections, 'connection', None)
    if connection is None or getattr(_connections, 'key', None) != key:
        connection = http.client.HTTPConnection(*key, timeout=300)
        _connections.connection, _connections.key = connection, key
    payload = {
        'model': 'LiquidAI/LFM3-VL-2B', 'messages': messages,
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'sprite_annotation', 'strict': True, 'schema': decoding_schema(schema)}},
        'temperature': 0.1, 'top_k': 50, 'top_p': 1.0, 'min_p': 0.0,
        'repetition_penalty': 1.1, 'max_tokens': max_tokens, 'seed': seed,
        'chat_template_kwargs': {'enable_thinking': False, 'preserve_thinking': False},
        'stream': False,
    }
    for attempt in range(3):
        try:
            connection.request('POST', '/v1/chat/completions', json.dumps(payload),
                               {'Content-Type': 'application/json'})
            response = connection.getresponse()
            data = response.read()
            if response.status != 200:
                if response.status in {500, 502, 503, 504} and attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f'vLLM HTTP {response.status}: {data.decode()[:1200]}')
            result = json.loads(data)
            choice = result['choices'][0]
            if choice['finish_reason'] != 'stop':
                raise ValueError('Truncated vLLM response')
            return json.loads(choice['message']['content']), result
        except (ConnectionError, http.client.HTTPException, TimeoutError):
            connection.close()
            if attempt == 2:
                raise
            connection = http.client.HTTPConnection(*key, timeout=300)
            _connections.connection = connection


def reviewed_annotation(server, image, measured, dimensions, leaves, system_prompt):
    schema = json_schema(dimensions, leaves, measured['palette_size'], measured['has_transparency'])
    draft_schema = copy.deepcopy(schema)
    draft_schema['properties'] = {'visual_checks': LAYOUT_SCHEMA, **draft_schema['properties']}
    draft_schema['required'] = ['visual_checks', *draft_schema['required']]
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    visual = {'type': 'image_url', 'image_url': {
        'url': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()}}
    facts = (f'Original size: {measured["width"]}x{measured["height"]}. '
             f'Original alpha has transparency: {measured["has_transparency"]}. '
             f'Exact visible RGB count: {measured["visible_rgb_color_count"]}. '
             f'palette_size must be {measured["palette_size"]}. Return compact JSON.')
    messages = [{'role': 'system', 'content': system_prompt +
                 '\nFIRST write visual_checks.instance_layout from the pixels: a single '
                 'creature with many limbs is one subject, not a sheet. A sheet requires '
                 'multiple separate sprites or frames. Then describe and label the image.'},
                {'role': 'user', 'content': [visual, {'type': 'text', 'text': facts}]}]
    draft = None
    normalizations = []
    for attempt in range(2):
        proposal, draft_response = chat(server, messages, draft_schema, seed=42 + attempt)
        normalize_color_set(proposal, 'draft', normalizations)
        try:
            draft_layout = proposal['visual_checks']['instance_layout']
            draft = validate({key: proposal[key] for key in ('caption_candidate', 'labels')},
                             dimensions, leaves, measured['palette_size'], measured['has_transparency'])
            break
        except ValueError:
            if attempt:
                raise
    review_schema = constrain_layout(schema, draft_layout)
    review_schema['properties'] = {'visual_checks': LAYOUT_SCHEMA, **review_schema['properties']}
    review_schema['required'] = ['visual_checks', *review_schema['required']]
    unconstrained_review_schema = copy.deepcopy(review_schema)
    unconstrained_review_schema['properties']['labels'] = copy.deepcopy(schema['properties']['labels'])
    # The model receives the same image again. A fresh request avoids hidden
    # conversation state and preserves both versions for an auditable comparison.
    review_messages = [
        {'role': 'system', 'content': system_prompt + '\n' + REVIEW_INSTRUCTION},
        {'role': 'user', 'content': [visual, {'type': 'text', 'text': facts +
            '\nProvisional annotation to verify against the image:\n' +
            json.dumps({**draft, 'visual_checks': {'instance_layout': draft_layout}}, ensure_ascii=False)}]}]
    for attempt in range(2):
        revised, review_response = chat(server, review_messages, review_schema, seed=142 + attempt)
        normalize_color_set(revised, 'review', normalizations)
        try:
            final = validate({key: revised[key] for key in ('caption_candidate', 'labels')},
                             dimensions, leaves, measured['palette_size'], measured['has_transparency'])
            layout = revised['visual_checks']['instance_layout']
            asset = final['labels']['asset_type']
            if layout == 'single_subject' and asset in {'sprite_sheet', 'animation_strip', 'tile_set'}:
                raise ValueError('Single subject contradicts sheet/tile_set label')
            if layout == 'multiple_distinct_sprites_or_frames' and asset == 'single_sprite':
                raise ValueError('Multiple frames contradict single_sprite label')
            break
        except ValueError as exc:
            if attempt:
                raise
            review_schema = constrain_layout(unconstrained_review_schema,
                                             revised['visual_checks']['instance_layout'])
            review_messages[-1]['content'].append({'type': 'text', 'text':
                f'Your previous review failed this consistency check: {exc}. '
                'Inspect the image and return a coherent corrected annotation.'})
    audit = {
        'performed': True, 'image_seen_again': True, 'passes': 2,
        'draft': draft, 'changed': final != draft,
        'changed_fields': (['caption_candidate'] if final['caption_candidate'] != draft['caption_candidate'] else []) +
                          ['labels.' + key for key in final['labels'] if final['labels'][key] != draft['labels'][key]],
        'draft_visual_checks': {'instance_layout': draft_layout},
        'normalizations': normalizations,
        'visual_checks': revised['visual_checks'],
        'review_prompt_sha256': sha(REVIEW_INSTRUCTION.encode()),
        'draft_usage': draft_response.get('usage'), 'review_usage': review_response.get('usage'),
    }
    return final, audit, schema, review_response


def inference(job, args, revision, token, dimensions, leaves, system_prompt, fingerprint):
    from annotate_lfm3 import model_image, source_path
    started = time.monotonic()
    name, generated_path, parent, bbox, frame = job
    path = Path(generated_path) if generated_path else source_path(name, args, revision, token)
    with _preprocessing:
        image, measured = model_image(path, args.max_side)
    final, audit, schema, response = reviewed_annotation(
        args.server, image, measured, dimensions, leaves, system_prompt)
    labels = final['labels']
    result = {
        'asset_id': sha(name.encode())[:24],
        'source': {'path': name, 'parent_path': parent, 'bbox': json.loads(bbox) if bbox else None, 'frame': frame},
        'model': 'LiquidAI/LFM3-VL-2B', 'served_model': response.get('model'),
        'execution_device': response.get('device', 'cuda'),
        'model_revision': args.model_revision, 'weight_dtype': 'bfloat16', 'quantization': None,
        'server_fingerprint': response.get('system_fingerprint'), 'run_config_id': fingerprint,
        'dataset_revision': revision, 'source_image_sha256': sha(path.read_bytes()),
        'labels': labels, 'taxonomy_paths': {
            key: [leaves[key][v]['path'] for v in value] if isinstance(value, list)
            else leaves[key][value]['path'] for key, value in labels.items()},
        'caption_candidate': final['caption_candidate'], 'caption_status': 'self_reviewed_candidate',
        'review_status': 'model_self_reviewed', 'needs_taxonomy_review': True,
        'need_rework': None if labels['visual_quality'] == 'unknown' else labels['visual_quality'] == 'needs_rework',
        'rights_status': 'unknown', 'measurements': measured, 'self_review': audit,
        'decoding': {'backend': ('transformers_cpu_xgrammar_json_schema' if response.get('device') == 'cpu'
                                else 'vllm_xgrammar_json_schema'), 'reasoning': False,
                     'schema_sha256': sha(json.dumps(decoding_schema(schema), sort_keys=True).encode())},
        'usage': response.get('usage'), 'wall_seconds': time.monotonic() - started,
        'annotated_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    if args.benchmark or args.keep_previews or name in args.preview_paths:
        image.save(args.output / 'previews' / (result['asset_id'] + '.png'))
    return result, path
