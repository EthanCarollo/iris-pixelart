#!/usr/bin/env python3
"""One auxiliary CPU worker on the same durable queue, isolated from GPU claims."""
import argparse
import fcntl
import gzip
import json
import os
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace
import zlib

from dotenv import load_dotenv

from annotate_lfm3 import (ROOT, cleanup_source, open_database, schedule_splits,
                          unpack_record, write_json)
from lfm3_schema import load_taxonomies, prompt
from lfm3_vllm import inference


def claim_tail(db, min_pending=16384):
    # The GPU takes the queue head; CPU takes the tail while enough work remains.
    # Distinct cpu_running state also survives GPU-runner restarts.
    db.execute('CREATE TABLE IF NOT EXISTS cpu_attempts(path TEXT PRIMARY KEY, attempted_at REAL)')
    db.execute('BEGIN IMMEDIATE')
    try:
        job = db.execute('SELECT path,local_path,parent_path,bbox,frame FROM jobs WHERE status="failed" '
                         'AND path NOT IN (SELECT path FROM cpu_attempts) ORDER BY rowid LIMIT 1').fetchone()
        if not job:
            count = db.execute('SELECT COUNT(*) FROM jobs WHERE status="pending"').fetchone()[0]
            if count < min_pending:
                db.commit()
                return None
            job = db.execute('SELECT path,local_path,parent_path,bbox,frame FROM jobs '
                             'WHERE status="pending" AND path NOT IN (SELECT path FROM cpu_attempts) '
                             'ORDER BY rowid DESC LIMIT 1').fetchone()
        if not job:
            db.commit()
            return None
        db.execute('UPDATE jobs SET status="cpu_running",attempts=attempts+1 WHERE path=? AND status IN ("pending","failed")', (job[0],))
        db.execute('INSERT OR REPLACE INTO cpu_attempts VALUES(?,?)', (job[0], time.time()))
        db.commit()
        return job
    except BaseException:
        db.rollback()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--server', default='http://127.0.0.1:8093')
    parser.add_argument('--max-jobs', type=int, default=0)
    parser.add_argument('--min-pending', type=int, default=16384)
    args = parser.parse_args()
    args.output = args.output.resolve()
    lock = (args.output / '.cpu-worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    db = open_database(args.output / 'annotations.sqlite3')
    db.execute('PRAGMA busy_timeout=30000')
    # Only this auxiliary process owns cpu_running; repair its abandoned claim.
    db.execute('UPDATE jobs SET status="pending" WHERE status="cpu_running"')
    db.commit()
    manifest = json.loads((args.output / 'run_manifest.json').read_text())
    inventory = json.loads((args.output / 'inventory.json').read_text())
    if not manifest.get('self_review') or manifest.get('weight_dtype') != 'bfloat16':
        raise SystemExit('This CPU worker requires the existing BF16 reviewed run')
    load_dotenv(ROOT / 'dataset-viewer/.env', override=False)
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
    settings = SimpleNamespace(
        output=args.output, server=args.server, repo=manifest['repo'],
        local_root=ROOT.parent / 'ds/mother3-sync-latest-worktree',
        ephemeral_downloads=True, source_tmp_dir=args.output / 'source_tmp_cpu',
        max_side=manifest['max_side'], model_revision=manifest['model_revision'],
        benchmark=False, keep_previews=False, preview_paths=set(),
        existing_splits=json.loads((args.output / 'existing_splits.json').read_text()))
    settings.source_tmp_dir.mkdir(exist_ok=True)
    dimensions, leaves, _ = load_taxonomies()
    system_prompt = prompt(dimensions, leaves)
    started, done, failures, tokens, current = time.monotonic(), 0, 0, 0, None
    system_failures = 0
    def progress(state):
        elapsed = time.monotonic() - started
        write_json(args.output / 'cpu_progress.json', {
            'pid': os.getpid(), 'state': state, 'done_this_run': done, 'failures': failures,
            'current_image': current[0] if current else None,
            'images_per_second': done / elapsed if elapsed else 0,
            'output_tokens_per_second': tokens / elapsed if elapsed else 0,
            'execution_device': 'cpu', 'weight_dtype': 'bfloat16',
            'elapsed_seconds': elapsed})
    try:
        while not args.max_jobs or done < args.max_jobs:
            current = claim_tail(db, args.min_pending)
            if current is None:
                break
            progress('running')
            print('CPU annotating', current[0], flush=True)
            try:
                result, path = inference(current, settings, manifest['revision'], token,
                                         dimensions, leaves, system_prompt, manifest['config_id'])
                if result.get('execution_device') != 'cpu':
                    raise ValueError('Auxiliary endpoint did not confirm CPU execution')
                schedule_splits(db, result, path, settings, manifest.get('split_overrides', {}))
                serialized = json.dumps(result, ensure_ascii=False)
                db.execute('UPDATE jobs SET status="done",result=?,error=NULL WHERE path=? AND status="cpu_running"',
                           (zlib.compress(serialized.encode(), level=3), current[0]))
                db.commit()
                # A separate stream avoids interleaving gzip members with the
                # GPU writer. The final GPU export merges the canonical DB.
                with gzip.open(args.output / 'annotations_lfm3.cpu.jsonl.gz', 'at', encoding='utf-8') as stream:
                    stream.write(serialized + '\n')
                done += 1
                system_failures = 0
                tokens += sum(result['self_review'][key]['completion_tokens'] for key in ['draft_usage', 'review_usage'])
                print(f'CPU completed {done} · {result["wall_seconds"]:.2f}s', flush=True)
            except Exception as exc:
                db.rollback()
                failures += 1
                status = 'failed' if isinstance(exc, ValueError) else 'pending'
                db.execute('UPDATE jobs SET status=?,attempts=CASE WHEN ?="pending" THEN 0 ELSE attempts END,error=? '
                           'WHERE path=? AND status="cpu_running"',
                           (status, status, 'CPU retry: ' + type(exc).__name__, current[0]))
                db.commit()
                print('CPU failed:', type(exc).__name__, flush=True)
                # Preserve GPU throughput if the auxiliary service cannot serve.
                system_failures = 0 if isinstance(exc, ValueError) else system_failures + 1
                if system_failures >= 2:
                    break
            except BaseException:
                db.rollback()
                db.execute('UPDATE jobs SET status="pending" WHERE path=? AND status="cpu_running"', (current[0],))
                db.commit()
                raise
            finally:
                cleanup_source(settings, current[0])
                current = None
                progress('running')
    finally:
        if current:
            db.execute('UPDATE jobs SET status="pending" WHERE path=? AND status="cpu_running"', (current[0],))
            db.commit()
        progress('stopped')
        db.close()


if __name__ == '__main__':
    main()
