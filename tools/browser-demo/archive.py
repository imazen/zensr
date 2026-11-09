"""Copy a frozen browser run to a new archive and verify every byte hash."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(a):
    root = a.root.resolve()
    destination = a.destination.resolve()
    if destination.is_relative_to(root):
        raise ValueError('Archive must be outside the source run')
    if destination.exists():
        raise ValueError('Use a new archive directory; existing data is preserved')
    excluded = [Path(name) for name in a.exclude]
    if any(p.is_absolute() or '..' in p.parts for p in excluded):
        raise ValueError('Exclusions must be relative run paths')
    paths = [p for p in sorted(root.rglob('*')) if p.is_file()
             and not any(p.relative_to(root).is_relative_to(e) for e in excluded)]
    destination.mkdir(parents=True)
    records = []
    for source in paths:
        before = source.stat()
        relative = source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        identity = hashlib.sha256()
        copied = 0
        with source.open('rb') as stream, target.open('xb') as output:
            while block := stream.read(1024 * 1024):
                output.write(block)
                identity.update(block)
                copied += len(block)
                if copied % (64 * 1024 * 1024) == 0:
                    output.flush()
                    os.fdatasync(output.fileno())
            output.flush()
            os.fsync(output.fileno())
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'Run changed during archival: {relative}')
        expected = identity.hexdigest()
        if digest(target) != expected:
            raise ValueError(f'Archive hash differs: {relative}')
        records.append(dict(path=str(relative), bytes=copied, sha256=expected))
    manifest = dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    files=records, excluded=a.exclude, count=len(records),
                    total_bytes=sum(r['bytes'] for r in records))
    data = json.dumps(manifest, indent=2) + '\n'
    (destination / 'ARCHIVE_VERIFIED-v1.json').write_text(data)
    (root / 'ARCHIVE_VERIFIED-v1.json').write_text(data)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('destination', type=Path)
    p.add_argument('--exclude', action='append', default=[])
    main(p.parse_args())
