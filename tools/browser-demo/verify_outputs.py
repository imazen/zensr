"""Independent Pillow validation of the fixture and completed browser PNGs."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def verify(root,cases,tiles=None,tile_case=None):
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
    for name in cases:
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
    if tiles:
        if not tile_case:raise ValueError('Native tile validation needs its output case')
        count=0
        with Image.open(root/tile_case/'output.png') as image:
            full=np.asarray(image)
            for path in tiles.iterdir():
                match=re.fullmatch(r'(\d+)-(\d+)\.png',path.name)
                if not match:continue
                x,y=map(int,match.groups())
                with Image.open(path) as tile:
                    actual=np.asarray(tile)
                    expected=full[y*512:y*512+tile.height,x*512:x*512+tile.width]
                    if not np.array_equal(actual,expected):raise ValueError(f'Native inspection tile differs: {path}')
                count+=1
        if count!=12:raise ValueError(f'Expected 12 inspection witnesses, got {count}')
        print(f'All {count} native inspection tiles match full-output RGB8 exactly')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--cases',nargs='+',default=['spanf-wasm','spanf-12mp','dscf-12mp','dscf-swiftshader-v3','spanf-webgpu-jpeg'])
    parser.add_argument('--tiles',type=Path)
    parser.add_argument('--tile-case')
    args=parser.parse_args()
    verify(args.root,args.cases,args.tiles,args.tile_case)
