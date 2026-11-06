"""Download the raw results and verify against the recorded Zenodo checksum."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def verify(path, entry):
    digest = hashlib.md5()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return path.stat().st_size == entry['size'] and 'md5:' + digest.hexdigest() == entry['checksum']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results.zip'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    record = json.loads((root / 'provenance/zenodo-22938433.json').read_text())
    entry = next(f for f in record['files'] if f['key'] == 'results.zip')
    target = args.output.expanduser()
    if target.exists():
        if verify(target, entry):
            print(f'Already verified: {target}')
            return
        raise SystemExit(f'{target} exists but does not match the deposit; choose a different output path.')
    partial = target.with_name(target.name + '.part')
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {entry['size']:,} bytes from Zenodo to {partial}", flush=True)
    try:
        with partial.open('xb') as output:
            with urllib.request.urlopen(entry['links']['self'], timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        if not verify(partial, entry):
            raise SystemExit(f'Checksum/size mismatch; incomplete download retained at {partial}.')
        # A hard link avoids overwriting a target created during the download.
        target.hardlink_to(partial)
        partial.unlink()
    except FileExistsError as exc:
        raise SystemExit(f'Output or partial file already exists: {exc.filename}; choose another --output.') from exc
    print(f'Verified size and MD5: {target}. Extract locally beneath results/.')


if __name__ == '__main__':
    main()
