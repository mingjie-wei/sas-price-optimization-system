"""Fetch the pinned public H2 snapshot, preserving an existing matching file."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'data/source.json').read_text())
    target = ROOT / manifest['local_path']
    if target.exists():
        content = target.read_bytes()
    else:
        with urllib.request.urlopen(manifest['download_url'], timeout=120) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != manifest['sha256']:
        raise ValueError('Snapshot checksum differs. Review the source change before updating data/source.json.')
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    print(f'Verified {target.name}: {len(content):,} bytes')


if __name__ == '__main__':
    main()
