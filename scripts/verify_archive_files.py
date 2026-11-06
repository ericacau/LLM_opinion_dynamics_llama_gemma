"""Verify the unmodified imported Zenodo files without third-party dependencies."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    entries = json.loads((root / 'provenance/archive-files.json').read_text())
    failures = []
    for entry in entries:
        path = root / entry['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            failures.append(entry['path'])
    if failures:
        raise SystemExit('Missing or modified archive files:\n' + '\n'.join(failures))
    print(f'OK: {len(entries)} imported files match the verified Zenodo archives.')


if __name__ == '__main__':
    main()
