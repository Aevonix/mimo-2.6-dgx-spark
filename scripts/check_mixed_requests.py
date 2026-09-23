#!/usr/bin/env python3
"""Synthetic normal-MiMo API qualification packet; never manages the server.

Default order: plain, JSON grammar, 596-token/max_tokens=1, then mixed C2.
Stops on first failed stage. Payloads contain generated public test text only.
The mixed scheduler shape is a target, not asserted from client concurrency.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import socket
import threading
import time
import urllib.request
import uuid

MODEL = 'mimo-v2.6-pro-rl'


def headers():
    out = {'Content-Type': 'application/json'}
    if os.environ.get('MIMO_API_KEY'):
        out['Authorization'] = 'Bearer ' + os.environ['MIMO_API_KEY']
    return out


def json_call(base, path, payload=None, timeout=30):
    req = urllib.request.Request(base.rstrip('/') + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers=headers())
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def body(text, maximum=32, temperature=0):
    return {'model': MODEL, 'messages': [{'role': 'user', 'content': text}],
            'temperature': temperature, 'max_tokens': maximum,
            'chat_template_kwargs': {'enable_thinking': False},
            'stream': True, 'stream_options': {'include_usage': True},
            'return_token_ids': True, 'cache_salt': 'synthetic-mimo-' + uuid.uuid4().hex}


def schema_body(text, items=3, maximum=64, temperature=0):
    out = body(text, maximum, temperature)
    out['response_format'] = {'type': 'json_schema', 'json_schema': {
        'name': 'synthetic_integer_list', 'strict': True,
        'schema': {'type': 'object', 'properties': {'items': {
            'type': 'array', 'items': {'type': 'integer', 'enum': list(range(10))},
            'minItems': items, 'maxItems': items}},
            'required': ['items'], 'additionalProperties': False}}}
    return out


def exact_prompt(base, template, target):
    original = template['messages'][0]['content']
    pad = 0
    for _ in range(12):
        text = original + '\nSynthetic padding:' + ' x' * pad
        payload = {'model': MODEL, 'messages': [{'role': 'user', 'content': text}],
                   'chat_template_kwargs': {'enable_thinking': False},
                   'add_generation_prompt': True}
        counted = json_call(base, '/tokenize', payload)['count']
        if counted == target:
            template['messages'] = payload['messages']
            return template
        pad += target - counted
        if pad < 0:
            raise ValueError('Requested prompt length is below the synthetic instruction length')
    raise ValueError('Could not construct exact prompt token length with the server tokenizer')


def validate_response(payload, result):
    """Validate complete synthetic responses, including the intended sequence."""
    if not result.get('done') or result.get('finish_reason') is None:
        raise ValueError('Response did not complete normally')
    finish = result['finish_reason']
    if 'response_format' in payload:
        if finish != 'stop':
            raise ValueError('Structured response did not stop naturally')
        parsed = json.loads(result['content'])
        spec = payload['response_format']['json_schema']['schema']['properties']['items']
        expected = [i % 10 for i in range(spec['minItems'])]
        if (type(parsed) is not dict or set(parsed) != {'items'}
                or type(parsed['items']) is not list
                or any(type(x) is not int for x in parsed['items'])
                or parsed['items'] != expected):
            raise ValueError('Synthetic schema semantic validation failed')
    elif payload['max_tokens'] == 1:
        completion_tokens = (result.get('usage') or {}).get('completion_tokens')
        if (finish not in ('stop', 'length') or type(completion_tokens) is not int
                or completion_tokens != 1):
            raise ValueError('Short response must complete with exactly one token')
    elif finish != 'stop' or result['content'].strip() != 'ready':
        raise ValueError('Plain response must stop naturally with ready')


@contextmanager
def response_deadline(response, deadline):
    """Interrupt an HTTP(S) body read even when no complete SSE line arrives."""
    # urllib's socket timeout alone measures inactivity, not elapsed time.
    connection = response.fp.raw._sock
    expired = threading.Event()

    def interrupt():
        expired.set()
        try:
            connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    timer = threading.Timer(max(0, deadline - time.monotonic()), interrupt)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
        if expired.is_set():
            raise TimeoutError('Synthetic request elapsed deadline')


def request(base, payload, output, timeout, gate=None, trigger_tokens=482):
    output.mkdir(mode=0o700)
    (output/'request.json').write_text(json.dumps(payload, indent=2)+'\n')
    started = time.monotonic()
    result = {'ok': False, 'started_monotonic': started, 'first_token_s': None,
              'output_token_ids_count': 0, 'visible_chunks': 0, 'content': '',
              'finish_reason': None, 'usage': None, 'done': False}
    req = urllib.request.Request(base.rstrip('/')+'/v1/chat/completions',
        data=json.dumps(payload).encode(), headers=headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response, response_deadline(response, started + timeout):
            for raw in response:
                elapsed = time.monotonic() - started
                if elapsed > timeout:
                    raise TimeoutError('Synthetic request elapsed deadline')
                if not raw.startswith(b'data:'):
                    continue
                event = raw[5:].strip()
                if event == b'[DONE]':
                    result['done'] = True
                    break
                data = json.loads(event)
                if data.get('error'):
                    raise RuntimeError('Server returned a streamed error')
                if data.get('usage'):
                    result['usage'] = data['usage']
                for choice in data.get('choices', []):
                    if choice.get('finish_reason'):
                        result['finish_reason'] = choice['finish_reason']
                    delta = choice.get('delta', {})
                    text = delta.get('content') or ''
                    if text:
                        result['content'] += text
                        result['visible_chunks'] += 1
                    tokens = choice.get('token_ids') or []
                    result['output_token_ids_count'] += len(tokens)
                    if tokens or text:
                        if result['first_token_s'] is None:
                            result['first_token_s'] = elapsed
                    if gate is not None and not gate.is_set() and result['output_token_ids_count'] >= trigger_tokens:
                        result['gate_at_output_tokens'] = result['output_token_ids_count']
                        result['gate_monotonic'] = time.monotonic()
                        gate.set()
        validate_response(payload, result)
        result['ok'] = True
    except Exception as exc:
        result['error_type'] = type(exc).__name__
    result['ended_monotonic'] = time.monotonic()
    result['elapsed_s'] = result['ended_monotonic'] - started
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def run(args):
    os.umask(0o077)
    output = args.output.expanduser().resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    base = args.url.rstrip('/').removesuffix('/v1')
    packet = {'model': MODEL, 'max_concurrent': 2,
              'incident_target': {'query_rows': [8, 596], 'total_query_rows': 604,
                'cached_computed_tokens': 9105, 'cached_output_tokens': 482,
                'draft_tokens': 7, 'cached_structured_output': True,
                'new_temperature': .6, 'new_max_tokens': 1},
              'limits': ['Original cached schema and sampling parameters are unavailable in these logs.',
                'Long prompt length is adjustable; 8624 approximates the observed cached context.',
                'Client overlap and token counts do not prove an exact scheduler row shape.',
                'Watermarking is omitted to retain the API default; no watermark kernel is inferred.',
                'No retries, restarts, profile changes or automatic recovery are performed.'],
              'selected_stages': args.stages.split(','), 'results': []}
    (output/'packet.json').write_text(json.dumps(packet, indent=2)+'\n')
    for stage in packet['selected_stages']:
        json_call(base, '/v1/models')
        if stage == 'plain':
            payload = body('Reply with the single word ready.')
        elif stage == 'json':
            payload = schema_body('Return an object with items containing the integers 0, 1, 2.')
        elif stage == 'short':
            payload = exact_prompt(base, body('Reply with one word.', 1, .6), args.short_prompt_tokens)
        elif stage == 'mixed':
            long = exact_prompt(base, schema_body(
                f'Return a JSON object with items containing exactly {args.items} integers. '
                'Repeat the sequence 0 through 9 as needed. Ignore the synthetic padding.',
                args.items, args.long_max_tokens, args.long_temperature), args.long_prompt_tokens)
            short = exact_prompt(base, body('Reply with one word.', 1, .6), args.short_prompt_tokens)
            gate = threading.Event()
            with ThreadPoolExecutor(max_workers=2) as pool:
                future = pool.submit(request, base, long, output/'mixed-long', args.timeout,
                                     gate, args.trigger_tokens)
                deadline = time.monotonic() + args.timeout
                while not gate.wait(.05) and not future.done() and time.monotonic() < deadline:
                    pass
                probe = None
                if gate.is_set() and not future.done():
                    if args.stagger_seconds:
                        time.sleep(args.stagger_seconds)
                    if not future.done():
                        probe = pool.submit(request, base, short, output/'mixed-short', args.timeout)
                long_result = future.result()
                short_result = probe.result() if probe else None
            overlap = bool(short_result and short_result['started_monotonic'] < long_result['ended_monotonic'])
            result = {'stage': stage, 'ok': bool(long_result['ok'] and short_result and short_result['ok'] and overlap),
                'overlap': overlap, 'gate_at_output_tokens': long_result.get('gate_at_output_tokens'),
                'long_ok': long_result['ok'], 'short_ok': short_result['ok'] if short_result else None,
                'scheduler_shape_verified': False}
        else:
            raise ValueError('Unknown stage')
        if stage != 'mixed':
            full = request(base, payload, output/stage, args.timeout)
            result = {'stage': stage, **{k: full.get(k) for k in ('ok', 'elapsed_s', 'usage', 'finish_reason', 'error_type')}}
        packet['results'].append(result)
        (output/'summary.json').write_text(json.dumps(packet, indent=2)+'\n')
        print(json.dumps(result), flush=True)
        if not result['ok']:
            return 1
    return 0


def self_test():
    from tempfile import TemporaryDirectory
    from unittest.mock import MagicMock, patch
    simple, structured = body('test', 1, .6), schema_body('test')
    assert simple['max_tokens'] == 1 and simple['temperature'] == .6
    assert 'watermarking' not in simple and 'response_format' not in simple
    assert structured['response_format']['type'] == 'json_schema'
    with patch(__name__+'.json_call', side_effect=[{'count': 20}, {'count': 596}]):
        padded = exact_prompt('http://unused', simple, 596)
    assert padded['messages'][0]['content'].endswith(' x' * 576)
    complete = {'done': True, 'finish_reason': 'stop', 'content': '{"items": [0, 1, 2]}'}
    validate_response(structured, complete)
    validate_response(body('test'), {**complete, 'content': 'ready'})
    validate_response(simple, {**complete, 'finish_reason': 'length',
                              'usage': {'completion_tokens': 1}})
    invalid = [
        {**complete, 'finish_reason': 'length'},
        {**complete, 'content': '{"items": [0, 1]}'},
        {**complete, 'content': '{"items": [0, 2, 1]}'},
        {**complete, 'content': '{"items": [0, true, 2]}'},
        {**complete, 'content': '{"items": [0, 1.0, 2]}'},
        {**complete, 'content': '{"items": [0, "1", 2]}'},
        {**complete, 'content': '{"items": {"0": 0, "1": 1, "2": 2}}'},
        {**complete, 'content': '[0, 1, 2]'},
    ]
    for result in invalid:
        try:
            validate_response(structured, result)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid structured response was accepted')
    released = threading.Event()
    response = MagicMock()
    response.__enter__.return_value = response
    response.fp.raw._sock.shutdown.side_effect = lambda _: released.set()

    def stalled_lines():
        yield b'data: {"choices":[{"delta":{"content":"rea"},"token_ids":[1]}]}\n'
        assert released.wait(1), 'Deadline did not interrupt the stalled stream'

    response.__iter__.side_effect = stalled_lines
    with TemporaryDirectory() as temp, patch('urllib.request.urlopen', return_value=response):
        result = request('http://unused', body('test'), Path(temp)/'stalled', .02)
    assert not result['ok'] and result['error_type'] == 'TimeoutError'
    assert result['content'] == 'rea' and result['first_token_s'] is not None
    response.fp.raw._sock.shutdown.assert_called_once_with(socket.SHUT_RDWR)
    print(json.dumps({'self_test': 'passed', 'network_calls': 0, 'gpu_calls': 0}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', help='Server URL, optionally ending in /v1; required for a network run')
    p.add_argument('--output', type=Path)
    p.add_argument('--stages', default='plain,json,short,mixed')
    p.add_argument('--short-prompt-tokens', type=int, default=596)
    p.add_argument('--long-prompt-tokens', type=int, default=8624)
    p.add_argument('--trigger-tokens', type=int, default=482)
    p.add_argument('--stagger-seconds', type=float, default=0)
    p.add_argument('--long-temperature', type=float, default=0)
    p.add_argument('--long-max-tokens', type=int, default=2048)
    p.add_argument('--items', type=int, default=512)
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.url or not args.output:
        p.error('--url and --output are required for a network run')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        p.error('--timeout must be finite and positive')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
