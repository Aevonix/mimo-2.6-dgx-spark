#!/usr/bin/env python3
"""Run frozen synthetic requests. No agent access or tool execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen
import uuid

ROOT = Path(__file__).resolve().parents[1]


def parsed_json(text):
    text = text.strip()
    fenced = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    return json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def exact(expected, actual):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and expected.keys() == actual.keys() and all(exact(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(exact(a, b) for a, b in zip(expected, actual))
    return type(expected) is type(actual) and expected == actual


def run_case(base_url, body, timeout):
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('MIMO_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['MIMO_API_KEY']
    request = Request(base_url.rstrip('/') + '/chat/completions', data=json.dumps(body).encode(), headers=headers)
    start = time.perf_counter()
    first = visible = None
    content = ''
    usage = {}
    finish = None
    with urlopen(request, timeout=timeout) as response:
        if not body.get('stream'):
            result = json.load(response)
            return {'response': result, 'elapsed_s': time.perf_counter() - start}
        for line in response:
            if not line.startswith(b'data: '):
                continue
            payload = line[6:].strip()
            if payload == b'[DONE]':
                break
            event = json.loads(payload)
            if event.get('error'):
                raise RuntimeError(event['error'])
            usage = event.get('usage') or usage
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                now = time.perf_counter() - start
                text = delta.get('content') or ''
                reasoning = delta.get('reasoning_content') or delta.get('reasoning') or ''
                if (text or reasoning) and first is None:
                    first = now
                if text and visible is None:
                    visible = now
                content += text
                finish = choice.get('finish_reason') or finish
    elapsed = time.perf_counter() - start
    tokens = usage.get('completion_tokens')
    return {'content': content, 'content_sha256': hashlib.sha256(content.encode()).hexdigest(),
            'usage': usage, 'finish_reason': finish, 'elapsed_s': elapsed,
            'first_model_output_s': first, 'first_visible_output_s': visible,
            'e2e_output_tokens_per_s': tokens / elapsed if isinstance(tokens, int) else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True, help='OpenAI-compatible URL including /v1')
    parser.add_argument('--model', default='mimo-v2.6-pro-rl')
    parser.add_argument('--suite', choices=['frozen', 'context', 'api'], default='frozen')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output already exists; use a new path to preserve results')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.suite == 'api':
        for path in sorted((ROOT / 'fixtures/api').glob('*-request.json')):
            body = json.loads(path.read_text())
            body['model'] = args.model
            jobs.append((path.name.removesuffix('-request.json'), body, None))
    else:
        files = ['4k.jsonl', 'decode.jsonl'] if args.suite == 'frozen' else ['context-256k-input.jsonl']
        oracles = json.loads((ROOT / 'fixtures' / ('oracles.json' if args.suite == 'frozen' else 'context-256k-oracles.json')).read_text())
        for name in files:
            for line in (ROOT / 'fixtures' / name).read_text().splitlines():
                row = json.loads(line)
                prompt = row['conversations'][0]['value']
                # Original fixture manifest hashes json.dumps(prompt), including its quotes.
                expected = oracles[hashlib.sha256(json.dumps(prompt).encode()).hexdigest()]['expected']
                body = {'model': args.model, 'messages': [{'role': 'user', 'content': prompt}],
                        'temperature': 0, 'seed': 20260922, 'max_tokens': 4096,
                        'chat_template_kwargs': {'enable_thinking': args.suite == 'context'},
                        'stream': True, 'stream_options': {'include_usage': True},
                        'cache_salt': 'mimo-public-' + uuid.uuid4().hex}
                jobs.append((row['id'], body, expected))
    if not jobs:
        raise SystemExit('No cases found')
    failed = 0
    with args.output.open('x') as output:
        for case_id, body, expected in jobs:
            row = {'id': case_id, 'suite': args.suite}
            try:
                row.update(run_case(args.base_url, body, args.timeout))
                if args.suite == 'api':
                    choice = row['response']['choices'][0]
                    message = choice['message']
                    if case_id.startswith('schema'):
                        row['pass'] = exact({'status': 'ready', 'count': 7}, parsed_json(message.get('content') or ''))
                    else:
                        calls = message.get('tool_calls') or []
                        row['pass'] = (len(calls) == 1 and calls[0]['function']['name'] == 'record_receipt'
                                       and exact({'code': 'spark-check-17', 'count': 7}, json.loads(calls[0]['function']['arguments'])))
                    row['pass'] = row['pass'] and choice.get('finish_reason') in ['stop', 'tool_calls']
                else:
                    row['pass'] = row['finish_reason'] == 'stop' and exact(expected, parsed_json(row['content']))
            except Exception as error:
                row['pass'] = False
                row['error'] = f'{type(error).__name__}: {error}'
            failed += not row['pass']
            output.write(json.dumps(row, allow_nan=False) + '\n')
            output.flush()
            print(json.dumps({key: row.get(key) for key in ['id', 'pass', 'elapsed_s', 'e2e_output_tokens_per_s', 'error']}), flush=True)
    print(json.dumps({'cases': len(jobs), 'passed': len(jobs) - failed, 'failed': failed}))
    return bool(failed)


if __name__ == '__main__':
    raise SystemExit(main())
