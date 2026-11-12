from pathlib import Path
from collections import Counter
from itertools import combinations
import gzip, json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

cwd = Path.cwd().resolve()
PROJECT_ROOT = next((p for p in (cwd, *cwd.parents) if (p / 'notebook/dataset').is_dir()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError('Racine du projet introuvable.')
WORK = PROJECT_ROOT / 'notebook/dataset/taxonomy/taxonomy_work'
INDEX_COMPLET = WORK / 'dataset_color_palettes.jsonl.gz'
INDEX_QWEN = WORK / 'annotations_v3_4_exact_palette.jsonl'
source = Path(SOURCE_OVERRIDE) if SOURCE_OVERRIDE else (INDEX_COMPLET if INDEX_COMPLET.is_file() else INDEX_QWEN)
if not source.is_file():
    raise FileNotFoundError('Aucune palette mesurée. Lance enrich_qwen_exact_palettes.py depuis la racine du projet.')
print('Source :', source)
print('Paires :', TOP_COULEURS_PAR_IMAGE, 'couleurs dominantes maximum par image')