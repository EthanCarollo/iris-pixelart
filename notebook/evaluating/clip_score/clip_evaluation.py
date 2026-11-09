"""Paired image–prompt CLIP comparison; evaluation never generates images."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def load_prompts(path):
    prompts = {}
    with Path(path).open(encoding='utf-8') as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key, prompt = row['prompt_id'], row['prompt']
            if not isinstance(key, str) or not key.strip() or key in prompts:
                raise ValueError(f'Identifiant de prompt vide ou dupliqué ligne {number}: {key!r}')
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f'Prompt vide ligne {number}.')
            prompts[key] = prompt
    if not prompts:
        raise ValueError('Aucun prompt.')
    return prompts


def write_manifest_template(path, prompts, models, seeds):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'{path} existe déjà : choisis un nouveau chemin pour ne pas écraser ton manifeste.')
    if len(set(models)) != len(models) or not models or any(not m or '/' in m or '\\' in m or m in {'.', '..'} for m in models):
        raise ValueError('Les noms de modèles doivent être uniques et utilisables comme noms de dossiers.')
    if not seeds or len(set(map(str, seeds))) != len(seeds):
        raise ValueError('Les graines doivent être uniques et non vides.')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model_id', 'prompt_id', 'seed', 'image_path'])
        writer.writeheader()
        for model in models:
            for prompt_id in prompts:
                for seed in seeds:
                    writer.writerow({'model_id': model, 'prompt_id': prompt_id, 'seed': seed,
                                     'image_path': f'images/{model}/{prompt_id}/seed_{seed}.png'})
    return path


def load_manifest(path, prompts, baseline):
    path = Path(path).resolve()
    rows, keys = [], set()
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'model_id', 'prompt_id', 'seed', 'image_path'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f'Colonnes requises : {sorted(required)}')
        for line, row in enumerate(reader, 2):
            model, prompt_id, seed, image_path = (row[k].strip() for k in ('model_id','prompt_id','seed','image_path'))
            if not all((model, prompt_id, seed, image_path)):
                raise ValueError(f'Champ vide ligne {line}.')
            if prompt_id not in prompts:
                raise ValueError(f'Prompt inconnu ligne {line} : {prompt_id}')
            key = (model, prompt_id, seed)
            if key in keys:
                raise ValueError(f'Combinaison dupliquée ligne {line} : {key}')
            keys.add(key)
            image = Path(image_path).expanduser()
            if not image.is_absolute():
                image = path.parent / image
            if not image.is_file():
                raise FileNotFoundError(f'Image manquante ligne {line} : {image}')
            rows.append({'model_id': model, 'prompt_id': prompt_id, 'seed': seed, 'image_path': str(image.resolve())})
    if not rows:
        raise ValueError('Le manifeste ne contient aucune image.')
    by_model = defaultdict(set)
    for row in rows:
        by_model[row['model_id']].add((row['prompt_id'], row['seed']))
    if baseline not in by_model:
        raise ValueError(f'Modèle de référence absent : {baseline}')
    if len(by_model) < 2:
        raise ValueError('Il faut au moins deux modèles pour comparer.')
    expected = by_model[baseline]
    for model, combinations in sorted(by_model.items()):
        if combinations != expected:
            raise ValueError(f'Comparaison incomplète pour {model} : {len(expected - combinations)} combinaisons manquantes, {len(combinations - expected)} en trop par rapport à {baseline}.')
    used_prompts = {prompt_id for prompt_id, _ in expected}
    if used_prompts != set(prompts):
        raise ValueError(f'Le manifeste ne couvre pas tous les prompts : {sorted(set(prompts) - used_prompts)}')
    return rows


def _on_white(path):
    with Image.open(path) as image:
        image.load()
        rgba = image.convert('RGBA')
        base = Image.new('RGBA', rgba.size, 'white')
        base.alpha_composite(rgba)
        return base.convert('RGB')


def score_images(rows, prompts, model_name='openai/clip-vit-base-patch32', batch_size=8, device='cpu'):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    if batch_size < 1:
        raise ValueError('batch_size doit être positif.')
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    # Reject overlong prompts rather than silently scoring a truncated prompt.
    for prompt_id, prompt in prompts.items():
        tokens = processor.tokenizer(prompt, truncation=False)['input_ids']
        if len(tokens) > model.config.text_config.max_position_embeddings:
            raise ValueError(f'Prompt {prompt_id} trop long pour CLIP ({len(tokens)} tokens).')
    result = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        images = [_on_white(row['image_path']) for row in batch]
        inputs = processor(text=[prompts[row['prompt_id']] for row in batch], images=images,
                           return_tensors='pt', padding=True, truncation=False).to(device)
        with torch.inference_mode():
            output = model(**inputs, return_dict=True)
            image_features = torch.nn.functional.normalize(output.image_embeds.float(), dim=-1)
            text_features = torch.nn.functional.normalize(output.text_embeds.float(), dim=-1)
            similarities = (image_features * text_features).sum(dim=-1).cpu().numpy()
        for row, similarity in zip(batch, similarities):
            item = dict(row)
            item['prompt'] = prompts[row['prompt_id']]
            item['clip_model_revision'] = getattr(model.config, '_commit_hash', None) or 'unknown'
            item['image_sha256'] = hashlib.sha256(Path(row['image_path']).read_bytes()).hexdigest()
            item['clip_cosine'] = float(similarity)
            item['clip_similarity_0_100'] = 100 * max(0.0, float(similarity))
            result.append(item)
        print(f'Images évaluées : {min(offset + len(batch), len(rows))}/{len(rows)}', flush=True)
    return result


def compare_by_prompt(rows, baseline, bootstrap_samples=2000, random_seed=42):
    if bootstrap_samples < 1:
        raise ValueError('bootstrap_samples doit être positif.')
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['model_id'], row['prompt_id'])].append(float(row['clip_similarity_0_100']))
    means = {key: float(np.mean(values)) for key, values in grouped.items()}
    prompts = sorted(prompt for model, prompt in means if model == baseline)
    if not prompts:
        raise ValueError('Aucun prompt pour la référence.')
    rng = np.random.default_rng(random_seed)
    out = []
    for model in sorted({model for model, _ in means}):
        if any((model, prompt) not in means for prompt in prompts):
            raise ValueError(f'Prompts manquants pour {model}.')
        scores = np.array([means[(model, p)] for p in prompts])
        delta = scores - np.array([means[(baseline, p)] for p in prompts])
        draws = rng.integers(0, len(prompts), size=(bootstrap_samples, len(prompts)))
        confidence = np.percentile(delta[draws].mean(axis=1), [2.5, 97.5])
        out.append({'model_id': model, 'baseline': baseline, 'mean_clip_similarity_0_100': float(scores.mean()),
                    'delta_vs_baseline': float(delta.mean()), 'delta_ci_95_low': float(confidence[0]),
                    'delta_ci_95_high': float(confidence[1]), 'n_prompts': len(prompts),
                    'n_images': sum(len(grouped[(model, p)]) for p in prompts)})
    return out


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('Aucune ligne à sauvegarder.')
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
