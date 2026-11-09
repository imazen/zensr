"""Read back browser PNGs independently and score differences from author PyTorch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(a):
    reference=json.loads((a.root/'reference.json').read_text())
    cases=json.loads((a.root/'cases.json').read_text())
    for case in cases:
        with Image.open(a.root/(case['id']+'.png')) as image:
            assert image.mode=='RGB' and image.size==(case['width'],case['height'])
            assert image.tobytes()==Path(case['rgb']).read_bytes(), 'Fixture encoder changed samples'
    scores=[]
    for row in reference['records']:
        if row['precision']=='browser-mixed' and row['model']!='spanf':raise ValueError('Invalid control')
        folder=a.root/f"{row['case']}-{row['model']}-{a.backend}-x{row['scale']}"
        png=folder/'output.png'
        with Image.open(png) as image:
            assert image.mode=='RGB' and image.size==(row['width'],row['height'])
            candidate=folder/'output.rgb8'
            data=image.tobytes()
            if candidate.exists():assert candidate.read_bytes()==data
            else:candidate.write_bytes(data)
        assert sha(Path(row['rgb']))==row['sha256']
        score=json.loads(subprocess.check_output([str(a.scorer),row['rgb'],str(candidate),str(row['width']),str(row['height'])],text=True))
        score.update(case=row['case'],model=row['model'],precision=row['precision'],scale=row['scale'],backend=a.backend,
                     reference_sha256=row['sha256'],candidate_sha256=sha(candidate),png_sha256=sha(png))
        scores.append(score)
        print(json.dumps(score),flush=True)
    (a.root/('scores-'+a.backend+'.json')).write_text(json.dumps(dict(scorer_sha256=sha(a.scorer),png_oracle='Pillow '+Image.__version__,records=scores),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('scorer',type=Path)
    p.add_argument('--backend',choices=['wasm','webgpu'],default='wasm')
    main(p.parse_args())
