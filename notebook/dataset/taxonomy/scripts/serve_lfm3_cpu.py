#!/usr/bin/env python3
"""Small CPU-only BF16 Transformers server with the same xgrammar constraints."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8093)
    parser.add_argument('--threads', type=int, default=6)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['MKL_NUM_THREADS'] = str(args.threads)
    os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
    os.environ['KMP_BLOCKTIME'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import xgrammar as xgr
    from xgrammar.contrib.hf import LogitsProcessor
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model), dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation='sdpa', local_files_only=True).eval()
    processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(
        processor.tokenizer, vocab_size=model.config.text_config.vocab_size), max_threads=1)
    model_lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def respond(self, code, value):
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            self.respond(200, {'data': [{'id': 'LiquidAI/LFM3-VL-2B', 'device': 'cpu',
                                        'dtype': 'bfloat16', 'quantization': None}]})
        def do_POST(self):
            if self.path != '/v1/chat/completions':
                self.respond(404, {'error': 'Unsupported endpoint'})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                messages = []
                for message in body['messages']:
                    content = message['content']
                    if isinstance(content, list):
                        converted = []
                        for part in content:
                            if part['type'] == 'image_url':
                                uri = part['image_url']['url']
                                if not uri.startswith('data:image/'):
                                    raise ValueError('Only inline images are accepted')
                                image = Image.open(io.BytesIO(base64.b64decode(uri.split(',', 1)[1]))).convert('RGB')
                                converted.append({'type': 'image', 'image': image})
                            else:
                                converted.append(part)
                        content = converted
                    messages.append({'role': message['role'], 'content': content})
                schema = body['response_format']['json_schema']['schema']
                compiled = compiler.compile_json_schema(json.dumps(schema), any_whitespace=False)
                with model_lock, torch.inference_mode():
                    inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
                        tokenize=True, return_dict=True, return_tensors='pt', enable_thinking=False)
                    inputs = {key: value.to(dtype=torch.bfloat16) if torch.is_floating_point(value) else value
                              for key, value in inputs.items()}
                    prompt_tokens = inputs['input_ids'].shape[1]
                    torch.manual_seed(body.get('seed', 42))
                    started = time.monotonic()
                    output = model.generate(**inputs, do_sample=True,
                        temperature=body.get('temperature', 0.1), top_k=body.get('top_k', 50),
                        top_p=body.get('top_p', 1.0), repetition_penalty=body.get('repetition_penalty', 1.1),
                        max_new_tokens=min(body.get('max_tokens', 768), 768),
                        logits_processor=[LogitsProcessor(compiled)])
                    elapsed = time.monotonic() - started
                    ids = output[0, prompt_tokens:]
                    text = processor.tokenizer.decode(ids, skip_special_tokens=True)
                reason = 'length' if len(ids) >= min(body.get('max_tokens', 768), 768) else 'stop'
                self.respond(200, {'model': 'LiquidAI/LFM3-VL-2B',
                    'system_fingerprint': 'transformers-cpu-bf16-xgrammar',
                    'choices': [{'finish_reason': reason, 'index': 0,
                                 'message': {'role': 'assistant', 'content': text}}],
                    'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': len(ids),
                              'total_tokens': prompt_tokens + len(ids)},
                    'timings': {'wall_seconds': elapsed, 'output_tokens_per_second': len(ids) / elapsed},
                    'device': 'cpu', 'weight_dtype': 'bfloat16'})
            except Exception as exc:
                self.respond(500, {'error': type(exc).__name__ + ': ' + str(exc)[:500]})
        def log_message(self, format, *args):
            pass
    print(json.dumps({'ready': True, 'device': 'cpu', 'dtype': str(next(model.parameters()).dtype),
                      'threads': args.threads, 'port': args.port}), flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
