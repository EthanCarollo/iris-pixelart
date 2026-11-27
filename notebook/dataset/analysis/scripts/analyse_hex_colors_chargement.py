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
local_index = PROJECT_ROOT / 'notebook/dataset/taxonomy/taxonomy_work/dataset_color_palettes.jsonl.gz'
if SOURCE_OVERRIDE or local_index.is_file():
    source = Path(SOURCE_OVERRIDE) if SOURCE_OVERRIDE else local_index
    if not source.is_file():
        raise FileNotFoundError(source)
    print('Source :', source)
else:
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "notebook/dataset"))
    from atlas_dataset import AtlasDataset

    atlas_dataset = AtlasDataset(PROJECT_ROOT)
    source = atlas_dataset.qwen_annotations(exact_palette=True)
    print('Source :', atlas_dataset.revision, source.relative_to(atlas_dataset.root))
print('Paires :', TOP_COULEURS_PAR_IMAGE, 'couleurs dominantes maximum par image')
