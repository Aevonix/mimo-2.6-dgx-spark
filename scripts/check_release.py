#!/usr/bin/env python3
"""CPU-only checks for the release archive, installation and captured answers."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
from benchmark import exact, parsed_json
from install_overlays import install
from launch import command

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checked = 0
    for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = ROOT / name
        if not path.resolve().is_relative_to(ROOT) or not path.is_file() or digest(path) != expected:
            raise ValueError(f'Release hash mismatch: {name}')
        checked += 1
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for entry in manifest['overlays']:
        if digest(ROOT / entry['source']) != entry['sha256']:
            raise ValueError(f"Overlay hash mismatch: {entry['source']}")
    for path in ROOT.rglob('*.py'):
        if not any(p in ['.git', 'build', 'local-results'] for p in path.relative_to(ROOT).parts):
            ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
    with tempfile.TemporaryDirectory() as directory:
        site = Path(directory)
        originals = []
        for entry in manifest['overlays']:
            if 'original_sha256' in entry:
                source = ROOT / 'upstream/vllm' / Path(entry['target']).name
                if digest(source) != entry['original_sha256']:
                    raise ValueError('Pinned original hash mismatch')
                target = site / entry['target']
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
                originals.append((entry, target))
        # A stale original must be rejected before any other module is replaced.
        stale_entry, stale = originals[-1]
        previous = stale.read_bytes()
        stale.write_bytes(previous + b'\n# different upstream\n')
        before = {str(p.relative_to(site)): digest(p) for p in site.rglob('*') if p.is_file()}
        try:
            install(ROOT, site)
        except ValueError as error:
            if 'Unsupported upstream source' not in str(error):
                raise
        else:
            raise AssertionError('Stale upstream was accepted')
        after = {str(p.relative_to(site)): digest(p) for p in site.rglob('*') if p.is_file()}
        if before != after:
            raise AssertionError('Failed install changed files')
        stale.write_bytes(previous)
        # Check that published diffs actually reproduce every replacement.
        for entry, target in originals:
            patch = ROOT / 'patches' / (target.name + '.patch')
            subprocess.run(['patch', '--batch', '-p1', '-i', str(patch)], cwd=site,
                           check=True, stdout=subprocess.DEVNULL)
            if digest(target) != entry['sha256']:
                raise AssertionError('Published patch differs from selected overlay')
        with contextlib.redirect_stdout(io.StringIO()):
            install(ROOT, site)
            install(ROOT, site)
        for entry in manifest['overlays']:
            if digest(site / entry['target']) != entry['sha256']:
                raise AssertionError('Installed file mismatch')
    config = json.loads((ROOT / 'config/cluster.example.json').read_text())
    for rank in range(8):
        spec, native = command(config, rank), command(config, rank, native=True)
        if (rank > 0) != ('--headless' in spec):
            raise AssertionError('Headless rank mismatch')
        if '--speculative-config' not in spec or '--speculative-config' in native:
            raise AssertionError('Native/speculative comparison changes incorrectly')
        index = spec.index('--speculative-config')
        if spec[:index] + spec[index + 2:] != native:
            raise AssertionError('Native differs by more than speculative configuration')
    oracles = json.loads((ROOT / 'fixtures/oracles.json').read_text())
    expected = {v['id']: v['expected'] for v in oracles.values()}
    for name in ['4k.jsonl', 'decode.jsonl']:
        for line in (ROOT / 'fixtures' / name).read_text().splitlines():
            row = json.loads(line)
            key = hashlib.sha256(json.dumps(row['conversations'][0]['value']).encode()).hexdigest()
            if oracles[key]['id'] != row['id']:
                raise AssertionError('Fixture/oracle hash mismatch')
    answers = json.loads((ROOT / 'results/normal-answers.json').read_text())
    passed = 0
    for key, row in answers.items():
        if hashlib.sha256(row['content'].encode()).hexdigest() != row['content_sha256']:
            raise AssertionError('Captured answer changed')
        actual = row['finish_reason'] == 'stop' and exact(expected[key], parsed_json(row['content']))
        if actual != row['expected_pass']:
            raise AssertionError(f'Rescored outcome differs: {key}')
        passed += actual
    if len(answers) != 12 or passed != 10:
        raise AssertionError('Unexpected frozen score')
    if exact({'ok': True}, {'ok': 1}) or exact({'n': 7}, {'n': 7.0}):
        raise AssertionError('Scorer coerces JSON types')
    try:
        parsed_json('```json\n{}\n```\n```json\n{}\n```')
    except ValueError:
        pass
    else:
        raise AssertionError('Scorer accepts duplicated fenced answers')
    subprocess.run([sys.executable, str(ROOT / 'whole-k/check_scheduler.py')], check=True)
    print(json.dumps({'status': 'PASS', 'hashed_files': checked, 'overlays': len(manifest['overlays']),
                      'patches': len(originals), 'launcher_ranks': 8, 'captured_outcomes_reproduced': 12,
                      'known_correct_answers': passed, 'gpu_inference_run': False}))


if __name__ == '__main__':
    main()
