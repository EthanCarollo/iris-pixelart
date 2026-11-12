from pathlib import Path
from collections import Counter
from datetime import datetime
import base64
import html
import json
import os
import random
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from IPython.display import HTML, display
from huggingface_hub import HfApi, hf_hub_download

DATASET_REPO = 'ethanzxv/pixelart-dataset'
HF_TOKEN = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN') or True

# Racine locale du projet et JSONL produit par le run courant.
cwd = Path.cwd().resolve()
project_roots = [p for p in (cwd, *cwd.parents) if (p / 'notebook/dataset').is_dir()]
PROJECT_ROOT = project_roots[0] if project_roots else cwd
LOCAL_ANNOTATION_CANDIDATES = [
    PROJECT_ROOT / 'notebook/dataset/taxonomy/taxonomy_work/annotations_v3_4_exact_palette.jsonl',
    PROJECT_ROOT / 'notebook/dataset/taxonomy/taxonomy_work/annotations_v3_4_full_color.jsonl',
]
LOCAL_ANNOTATIONS = next((candidate for candidate in LOCAL_ANNOTATION_CANDIDATES if candidate.is_file()), LOCAL_ANNOTATION_CANDIDATES[0])

# Priorité : override explicite, fichier vivant local, puis snapshot Qwen3.8 le plus récent sur HF.
if ANNOTATION_FILE_OVERRIDE:
    annotation_path = Path(hf_hub_download(repo_id=DATASET_REPO, repo_type='dataset', filename=ANNOTATION_FILE_OVERRIDE, token=HF_TOKEN))
    annotation_source = ANNOTATION_FILE_OVERRIDE
elif LOCAL_ANNOTATIONS.is_file():
    annotation_path = LOCAL_ANNOTATIONS
    annotation_source = f'workspace · {LOCAL_ANNOTATIONS}'
else:
    api = HfApi(token=HF_TOKEN)
    files = api.list_repo_files(DATASET_REPO, repo_type='dataset')
    remote_candidates = [
        name for name in files
        if name.endswith('.jsonl') and (
            name.startswith('annotations/dataset_lab_sync/')
            or name.startswith('annotations/taxonomy_runs/')
        )
    ]
    remote_candidates.sort(reverse=True)
    annotation_path = None
    annotation_source = None
    for remote_name in remote_candidates:
        downloaded = Path(hf_hub_download(repo_id=DATASET_REPO, repo_type='dataset', filename=remote_name, token=HF_TOKEN))
        try:
            with downloaded.open(encoding='utf-8') as stream:
                first = next((json.loads(line) for line in stream if line.strip()), {})
            model_name = str(first.get('model', '')).lower()
            if 'qwen3.8' in model_name and 'gyro' in model_name:
                annotation_path, annotation_source = downloaded, remote_name
                break
        except (OSError, json.JSONDecodeError):
            continue
    if annotation_path is None:
        raise FileNotFoundError('Aucun JSONL Qwen3.8 Gyro-S trouvé localement ou dans les snapshots HF.')

# Snapshot cohérent du JSONL : une éventuelle dernière ligne incomplète est ignorée.
qwen_rows = []
with annotation_path.open('rb') as stream:
    while True:
        line = stream.readline()
        if not line or not line.endswith(b'\n'):
            break
        if line.strip():
            qwen_rows.append(json.loads(line))
qwen_rows = [
    row for row in qwen_rows
    if 'qwen3.8' in str(row.get('model', '')).lower()
    and 'gyro' in str(row.get('model', '')).lower()
]
if not qwen_rows:
    raise ValueError(f'Aucune annotation Qwen3.8 Gyro-S dans {annotation_source}')

print(f'{len(qwen_rows):,} lignes Qwen3.8 Gyro-S · {len({r.get("asset_id") for r in qwen_rows}):,} asset_id uniques')
print('Source :', annotation_source)
