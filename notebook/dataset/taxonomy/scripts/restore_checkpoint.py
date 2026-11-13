#!/usr/bin/env python3
"""Restore a published Qwen taxonomy checkpoint without overwriting local work."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[4]
TAXONOMY = ROOT / 'notebook/dataset/taxonomy'
WORK = TAXONOMY / 'taxonomy_work'
REPO = 'ethanzxv/pixelart-dataset'
CHECKPOINT = 'annotations/checkpoints/20261113T093216Z'
REVISION = 'b399daded3dce679e6d4c264c357dc14d132df90'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=CHECKPOINT, help='Dossier du checkpoint sur le Hub')
    parser.add_argument('--revision', default=REVISION, help='Commit HF contenant le checkpoint')
    args = parser.parse_args()
    load_dotenv(ROOT / 'dataset-viewer/.env', override=False)
    token = os.environ.get('HF_TOKEN') or os.environ.get('HF_ACCESS_TOKEN') or None
    def download(name):
        return Path(hf_hub_download(repo_id=REPO, repo_type='dataset', revision=args.revision,
                                    filename=f'{args.checkpoint}/{name}', token=token))
    manifest = json.loads(download('manifest.json').read_text(encoding='utf-8'))
    for name in ('taxonomies.json', 'taxonomies_couleurs.json'):
        expected = manifest['files'][name]['sha256']
        if not (TAXONOMY / name).is_file() or digest(TAXONOMY / name) != expected:
            raise SystemExit(f'Taxonomie locale différente : {name}. Utilise la révision Git associée au checkpoint.')
    names = ('annotations_v3_4_full_color.jsonl', 'annotations_v3_4_exact_palette.jsonl')
    # Check every destination before writing either file; preserve newer local annotations.
    for name in names:
        destination = WORK / name
        if destination.exists() and digest(destination) != manifest['files'][name]['sha256']:
            raise SystemExit(f'{destination} existe et diffère du checkpoint : sauvegarde-le avant restauration.')
    WORK.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = download(name)
        if digest(source) != manifest['files'][name]['sha256']:
            raise SystemExit(f'Empreinte incorrecte : {name}')
        with tempfile.NamedTemporaryFile(dir=WORK, delete=False) as tmp:
            temporary = Path(tmp.name)
            try:
                with source.open('rb') as src:
                    shutil.copyfileobj(src, tmp)
                tmp.flush(); os.fsync(tmp.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(WORK / name)
    print(f"Checkpoint restauré : {manifest['annotations']} annotations · {manifest['measured_palettes']} palettes.")
    print(f"Révision source du dataset : {manifest['dataset_source_revision']}")

if __name__ == '__main__':
    main()
