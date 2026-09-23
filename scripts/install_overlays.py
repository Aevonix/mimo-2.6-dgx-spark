#!/usr/bin/env python3
"""Install the exact source-checked overlay into the pinned container image."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def install(package, site_packages):
    manifest = json.loads((package / 'manifest.json').read_text())
    pending = []
    for entry in manifest['overlays']:
        source = package / entry['source']
        target = site_packages / entry['target']
        if hashlib.sha256(source.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f"Release source mismatch: {entry['source']}")
        if target.exists():
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual == entry['sha256']:
                continue
            if actual != entry.get('original_sha256'):
                raise ValueError(f"Unsupported upstream source: {entry['target']}")
        elif 'original_sha256' in entry:
            raise ValueError(f"Missing upstream source: {entry['target']}")
        pending.append((source, target))
    for source, target in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    print(json.dumps({'installed': len(pending), 'total': len(manifest['overlays'])}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--site-packages', type=Path, required=True)
    args = parser.parse_args()
    install(args.package, args.site_packages)
