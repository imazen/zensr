#!/usr/bin/env python3
"""Acquire the pinned author-released MeanFlow inference package for evaluation."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = '33315831b6b342d3b8ddbdc70aa11cb84e55c66b'
CHECKPOINT = 'meanflow_jpeg_qf5_20.pth'
CHECKPOINT_BYTES = 261352510
CHECKPOINT_SHA256 = 'fbb573706c64ae90da85e47d1ff3377734264142460770c54ebadc50a4ba4caa'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / CHECKPOINT
    if not path.exists():
        temporary = path.with_suffix('.download-partial')
        url = f'https://huggingface.co/asandeistefan/jpeg-restore-meanflows/resolve/{REVISION}/{CHECKPOINT}?download=true'
        count = 0
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('xb') as out:
            while block := response.read(1024 * 1024):
                out.write(block)
                count += len(block)
                if count > CHECKPOINT_BYTES:
                    raise ValueError('checkpoint exceeds declared bytes')
        if count != CHECKPOINT_BYTES or sha(temporary) != CHECKPOINT_SHA256:
            raise ValueError('checkpoint identity differs from published LFS object')
        temporary.rename(path)
    if path.stat().st_size != CHECKPOINT_BYTES or sha(path) != CHECKPOINT_SHA256:
        raise ValueError('existing checkpoint differs from published LFS object')
    record = dict(checked=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  revision=REVISION, bytes=CHECKPOINT_BYTES, sha256=CHECKPOINT_SHA256,
                  source='https://huggingface.co/asandeistefan/jpeg-restore-meanflows')
    with (root / 'CHECKPOINT.json').open('x') as f:
        json.dump(record, f, indent=2)
        f.write('\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    main(p.parse_args().root)
