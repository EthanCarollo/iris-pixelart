#!/usr/bin/env python3
"""Start/resume the independent LFM3 annotation pipeline locally on Apple Silicon."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'notebook/dataset/taxonomy/lfm3_work'
MODEL_REVISION = '1add221a5ea6c047fcf1eaaf2d247939d7cf8849'
RUNTIME_VERSION = 'b12359'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--all', action='store_true')
    selection.add_argument('--benchmark', type=int)
    selection.add_argument('--limit', type=int, default=1)
    parser.add_argument('--output', type=Path, default=WORK / 'full')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--port', type=int, default=8091)
    parser.add_argument('--local-root', type=Path, default=ROOT.parent / 'ds/mother3-sync-latest-worktree')
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        parser.error('This launcher targets Apple Silicon. Use annotate_lfm3.py with a compatible external llama.cpp server.')
    if not 1 <= args.workers <= 4:
        parser.error('--workers must be between 1 and 4')
    load_dotenv(ROOT / 'dataset-viewer/.env', override=False)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    endpoint = f'http://127.0.0.1:{args.port}'
    try:
        with urllib.request.urlopen(endpoint + '/props', timeout=3) as response:
            props = json.load(response)
        if 'lfm3-vl-2b' not in str(props.get('model_path') or props.get('model_alias')).lower():
            raise SystemExit('The port is already serving another model; choose --port')
        server_pid = None
    except (OSError, urllib.error.URLError):
        cache = Path.home() / '.cache/iris-pixelart'
        binary = cache / f'llama-{RUNTIME_VERSION}/llama-server'
        if not binary.is_file():
            cache.mkdir(parents=True, exist_ok=True)
            archive = cache / f'llama-{RUNTIME_VERSION}.tar.gz'
            url = f'https://github.com/ggml-org/llama.cpp/releases/download/{RUNTIME_VERSION}/llama-{RUNTIME_VERSION}-bin-macos-arm64.tar.gz'
            urllib.request.urlretrieve(url, archive)
            with tarfile.open(archive) as source:
                source.extractall(cache, filter='data')
        token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
        model = hf_hub_download('LiquidAI/LFM3-VL-2B-GGUF', 'LFM3-VL-2B-Q8_0.gguf', revision=MODEL_REVISION, token=token)
        projector = hf_hub_download('LiquidAI/LFM3-VL-2B-GGUF', 'mmproj-LFM3-VL-2B-F16.gguf', revision=MODEL_REVISION, token=token)
        command = [str(binary), '-m', model, '--mmproj', projector, '--host', '127.0.0.1',
                   '--port', str(args.port), '-ngl', '99', '-c', str(8192 * args.workers),
                   '-np', str(args.workers), '--reasoning-budget', '0', '--reasoning-format', 'none',
                   '--no-reasoning-preserve', '--temp', '0.1', '--top-p', '1.0', '--top-k', '50',
                   '--min-p', '0.0', '--repeat-penalty', '1.1', '-fa', 'on']
        with (args.output / 'server.log').open('a') as log:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True, cwd=ROOT)
        server_pid = server.pid
        for attempt in range(60):
            if server.poll() is not None:
                raise SystemExit('llama.cpp did not start; see server.log')
            try:
                with urllib.request.urlopen(endpoint + '/health', timeout=2) as response:
                    if json.load(response).get('status') == 'ok':
                        break
            except OSError:
                pass
            time.sleep(1)
        else:
            raise SystemExit('Server is still loading; see server.log and rerun the launcher when ready')
    command = [sys.executable, str(Path(__file__).with_name('annotate_lfm3.py')),
               '--server', endpoint, '--workers', str(args.workers), '--output', str(args.output),
               '--local-root', str(args.local_root)]
    command += ['--all'] if args.all else ['--benchmark', str(args.benchmark)] if args.benchmark else ['--limit', str(args.limit)]
    if shutil.which('caffeinate'):
        command = ['caffeinate', '-i', *command]
    with (args.output / 'run.log').open('a') as log:
        runner = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True, cwd=ROOT)
    time.sleep(1)
    if runner.poll() is not None and runner.returncode:
        raise SystemExit('Runner did not start; see run.log (a run may already hold the output lock)')
    session = {'launcher_pid': os.getpid(), 'runner_pid': runner.pid,
               'server_pid_if_started_here': server_pid, 'server': endpoint,
               'output': str(args.output), 'all_images': args.all,
               'command': command, 'keeps_mac_awake_while_running': bool(shutil.which('caffeinate'))}
    (args.output / 'processes.json').write_text(json.dumps(session, indent=2) + '\n')
    print(json.dumps(session, indent=2))


if __name__ == '__main__':
    main()
