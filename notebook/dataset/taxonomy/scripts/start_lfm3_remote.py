#!/usr/bin/env python3
"""Start/resume local annotation jobs against the private remote BF16 vLLM API."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'notebook/dataset/taxonomy/lfm3_work/remote'


def main():
    load_dotenv(ROOT / 'dataset-viewer/.env', override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default=os.environ.get('IRIS_SSH_HOST'))
    parser.add_argument('--ssh-port', type=int, default=20395)
    parser.add_argument('--user', default='eth')
    parser.add_argument('--password-file', type=Path, default=WORK / 'ssh-password')
    parser.add_argument('--port', type=int, default=8092)
    parser.add_argument('--workers', type=int, default=128)
    parser.add_argument('--output', type=Path, default=WORK / 'full_bf16')
    parser.add_argument('--atlas-checkout', type=Path)
    parser.add_argument('--path-prefix')
    parser.add_argument('--revision', default='main')
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--all', action='store_true')
    choice.add_argument('--benchmark', type=int)
    choice.add_argument('--limit', type=int, default=1)
    args = parser.parse_args()
    if not args.host:
        parser.error('Supply --host or set IRIS_SSH_HOST in the ignored local .env')
    if args.atlas_checkout and args.output == WORK / 'full_bf16':
        parser.error('Use a separate --output for Atlas jobs to protect the existing checkpoint')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(exist_ok=True)
    endpoint = f'http://127.0.0.1:{args.port}'
    def ready():
        try:
            with urllib.request.urlopen(endpoint + '/v1/models', timeout=5) as response:
                data = json.load(response)
            return any('lfm3-vl-2b' in model['id'].lower() for model in data['data'])
        except OSError:
            return False
    tunnel_pid = None
    if not ready():
        if not args.password_file.is_file():
            parser.error('SSH password file is missing; supply --password-file (kept outside Git, mode 600)')
        command = [sys.executable, str(Path(__file__).with_name('lfm3_ssh_tunnel.py')),
                   '--host', args.host, '--ssh-port', str(args.ssh_port), '--user', args.user,
                   '--password-file', str(args.password_file.resolve()),
                   '--local-port', str(args.port), '--remote-port', str(args.port)]
        with (WORK / 'tunnel.log').open('a') as log:
            tunnel = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True, cwd=ROOT)
        tunnel_pid = tunnel.pid
        for _ in range(30):
            if ready():
                break
            if tunnel.poll() is not None:
                raise SystemExit('SSH tunnel failed; check tunnel.log and the remote vLLM server')
            time.sleep(1)
        else:
            raise SystemExit('Remote vLLM is not ready; check server.log on the SSH host')
    command = [sys.executable, str(Path(__file__).with_name('annotate_lfm3.py')),
               '--backend', 'vllm', '--server', endpoint, '--workers', str(args.workers),
               '--output', str(args.output)]
    command += ['--all'] if args.all else ['--benchmark', str(args.benchmark)] if args.benchmark else ['--limit', str(args.limit)]
    command += ['--revision', args.revision]
    if args.atlas_checkout:
        command += ['--atlas-checkout', str(args.atlas_checkout.resolve())]
    if args.path_prefix:
        command += ['--path-prefix', args.path_prefix]
    if shutil.which('caffeinate'):
        command = ['caffeinate', '-i', *command]
    with (args.output / 'run.log').open('a') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, cwd=ROOT)
    time.sleep(1)
    if process.poll() is not None and process.returncode:
        raise SystemExit('Runner failed to start; see run.log')
    data = {'runner_pid': process.pid, 'tunnel_pid_if_started_here': tunnel_pid,
            'server': endpoint, 'ssh_host': args.host, 'output': str(args.output),
            'workers': args.workers, 'backend': 'vllm', 'weight_dtype': 'bfloat16',
            'quantization': None, 'self_review': True, 'command': command}
    (args.output / 'processes.json').write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
