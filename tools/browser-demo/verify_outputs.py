"""Independent Pillow validation of the fixture and completed browser PNGs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def verify(root):
    Image.MAX_IMAGE_PIXELS = 201_600_000
    source = root / 'input-12mp.png'
    with Image.open(source) as image:
        pixels = np.asarray(image)
        x = np.arange(4000, dtype=np.uint32)
        for y in range(3000):
            expected = np.stack(((x + y * 3) % 256,
                                 (x * 3 + y) % 256, (x ^ y) % 256), axis=-1)
            if not np.array_equal(pixels[y], expected):
                raise ValueError(f'Fixture differs at row {y}')
    records = []
    for name in ['spanf-wasm', 'spanf-12mp', 'dscf-12mp',
                 'dscf-swiftshader-v3', 'spanf-webgpu-jpeg']:
        folder = root / name
        result = json.loads((folder / 'result.json').read_text())
        stats = result['stats']
        path = folder / 'output.png'
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            if image.size != (stats['outputWidth'], stats['outputHeight']) or image.mode != 'RGB':
                raise ValueError(f'Output shape or channels differ: {name}')
        if path.stat().st_size != stats['encodedBytes']:
            raise ValueError(f'Output length differs: {name}')
        if any(r['method'] != 'GET' for r in result['requests']):
            raise ValueError(f'Unexpected upload request: {name}')
        digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
        records.append(dict(name=name, bytes=path.stat().st_size, sha256=digest,
                            width=stats['outputWidth'], height=stats['outputHeight']))
    (root / 'png-verification.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    verify(parser.parse_args().root)
