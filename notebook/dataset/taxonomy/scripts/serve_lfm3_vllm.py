#!/usr/bin/env python3
"""Serve unquantized LFM3-VL-2B BF16 on one NVIDIA GPU with continuous batching."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sysconfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='LiquidAI/LFM3-VL-2B')
    parser.add_argument('--revision', default='23387beda43c60310348e83e25f78f33c3140d3a')
    parser.add_argument('--port', type=int, default=8092)
    parser.add_argument('--max-seqs', type=int, default=128)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.88)
    parser.add_argument('--no-flashinfer-sampler', action='store_true')
    args = parser.parse_args()
    environment = os.environ.copy()
    environment.update(VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1', OMP_NUM_THREADS='1')
    cuda = Path(sysconfig.get_paths()['purelib']) / 'nvidia/cu13'
    if (cuda / 'bin/nvcc').is_file():
        # NVIDIA's pip wheels use lib/, while CUDA JIT builders expect lib64/
        # and the unversioned libcudart linker name.
        if (cuda / 'lib').is_dir() and not (cuda / 'lib64').exists():
            (cuda / 'lib64').symlink_to('lib', target_is_directory=True)
        if (cuda / 'lib/libcudart.so.13').is_file() and not (cuda / 'lib/libcudart.so').exists():
            (cuda / 'lib/libcudart.so').symlink_to('libcudart.so.13')
        environment['CUDA_HOME'] = str(cuda)
        environment['PATH'] = str(Path(sysconfig.get_paths()['scripts'])) + ':' + str(cuda / 'bin') + ':' + environment.get('PATH', '')
    if args.no_flashinfer_sampler:
        environment['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
    command = [str(Path(sysconfig.get_paths()['scripts']) / 'vllm'), 'serve', args.model,
        '--served-model-name', 'LiquidAI/LFM3-VL-2B', '--dtype', 'bfloat16', '--trust-remote-code',
        '--host', '127.0.0.1', '--port', str(args.port), '--max-model-len', '8192',
        '--max-num-seqs', str(args.max_seqs), '--max-num-batched-tokens', '8192',
        '--gpu-memory-utilization', str(args.gpu_memory_utilization), '--enable-prefix-caching',
        '--limit-mm-per-prompt', '{"image":1,"video":0}',
        '--mm-processor-kwargs', '{"size":{"shortest_edge":65536,"longest_edge":589824}}',
        '--mm-processor-cache-gb', '1', '--reasoning-parser', 'qwen3',
        '--structured-outputs-config', '{"backend":"xgrammar","disable_any_whitespace":true}',
        '--disable-log-requests']
    if not Path(args.model).is_dir():
        command += ['--revision', args.revision]
    print(json.dumps({'command': command, 'weight_dtype': 'bfloat16', 'quantization': None}), flush=True)
    subprocess.run(command, env=environment, check=True)


if __name__ == '__main__':
    main()
