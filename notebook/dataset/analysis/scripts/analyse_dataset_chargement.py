from pathlib import Path
from collections import Counter
import json
import random
import sys

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from IPython.display import HTML, display

cwd = Path.cwd().resolve()
PROJECT_ROOT = next(p for p in (cwd, *cwd.parents) if (p / "notebook/dataset/atlas_dataset.py").is_file())
sys.path.insert(0, str(PROJECT_ROOT / "notebook/dataset"))
from atlas_dataset import AtlasDataset, annotation_asset_path

atlas_dataset = AtlasDataset(PROJECT_ROOT)
annotation_path = atlas_dataset.qwen_annotations(ANNOTATION_FILE_OVERRIDE)
annotation_source = f"Atlas {atlas_dataset.revision} · {annotation_path.relative_to(atlas_dataset.root)}"

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
