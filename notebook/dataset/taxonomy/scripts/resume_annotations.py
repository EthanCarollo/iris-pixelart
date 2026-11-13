#!/usr/bin/env python3
"""Resume the Qwen taxonomy notebook from its saved JSONL checkpoint."""
import argparse
import ast
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
NOTEBOOK = ROOT / 'notebook/dataset/taxonomy/taxonomie_dataset.ipynb'
WORK = ROOT / 'notebook/dataset/taxonomy/taxonomy_work'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=int, default=10000, help='Total number of unique annotated images')
    parser.add_argument('--strata-url', default='http://127.0.0.1:8080')
    args = parser.parse_args()
    if args.target < 1:
        parser.error('--target doit être positif')
    WORK.mkdir(parents=True, exist_ok=True)
    lock = WORK / 'run_full_color.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise SystemExit(f'Verrou présent : {lock}. Vérifie qu’aucun run ne tourne avant de le retirer.')
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try:
        os.chdir(ROOT)
        notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        def cell(index): return ''.join(notebook['cells'][index]['source'])
        env = {'__name__': '__main__'}
        exec(compile(cell(5), str(NOTEBOOK), 'exec'), env)
        env['STRATA_URL'] = args.strata_url.rstrip('/')
        exec(compile(cell(7), str(NOTEBOOK), 'exec'), env)
        code = cell(9)
        tree = ast.parse(code)
        reader = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'read_records')
        exec(compile(ast.Module(body=[reader], type_ignores=[]), str(NOTEBOOK), 'exec'), env)
        done = len(env['read_records'](env['OUTPUT_PATH']))
        target = min(args.target, len(env['images']))
        print(f'Reprise : {done} annotations uniques ; objectif : {target}.', flush=True)
        if done >= target:
            return
        env['MAX_IMAGES'] = target - done
        exec(compile(code, str(NOTEBOOK), 'exec'), env)
        final = len(env['read_records'](env['OUTPUT_PATH']))
        print(f'Annotations enregistrées : {final}/{target}.', flush=True)
        if final < target:
            raise SystemExit('Lot incomplet ; les annotations enregistrées sont conservées.')
    finally:
        lock.unlink(missing_ok=True)

if __name__ == '__main__':
    main()
