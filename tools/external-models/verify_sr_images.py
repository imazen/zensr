"""Independently decode retained SR PNGs and compare regenerated v1 evidence."""
import argparse
import hashlib
import json
from pathlib import Path
from sr_artifacts import validate_png


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(a):
    from PIL import Image
    previous = {}
    for path in a.previous:
        identity = json.loads((path/'IDENTITY.json').read_text())
        previous[identity['name']] = path
    decoded = {}
    records = []
    used_previous = set()
    for run in a.runs:
        identity = json.loads((run/'IDENTITY.json').read_text())
        complete = json.loads((run/'COMPLETE.json').read_text())
        if complete['identity_sha256'] != sha(run/'IDENTITY.json'):
            raise ValueError('completion identity differs')
        files = sorted((run/'cells').glob('*.json'))
        if len(files) != complete['inputs']:
            raise ValueError('cell coverage differs')
        recovered = 0
        for path in files:
            cell = json.loads(path.read_text())
            row = cell['row']
            old = None
            if identity['name'] in previous:
                old = json.loads((previous[identity['name']]/'cells'/path.name).read_text())
                if old['row'] != row or old['seed'] != cell['seed']:
                    raise ValueError('regenerated input or seed differs')
                used_previous.add(identity['name'])
            for role in ['baseline', 'native_span', 'candidate']:
                score = cell[role]
                validate_png(score, row['width'], row['height'], Path(identity['artifact_root']))
                image_path = Path(score['png']['path'])
                digest = score['png']['sha256']
                if digest not in decoded:
                    with Image.open(image_path) as image:
                        if image.format != 'PNG':
                            raise ValueError('not a PNG image')
                        pixels = image.convert('RGB')
                        decoded[digest] = (pixels.size, hashlib.sha256(pixels.tobytes()).hexdigest())
                if decoded[digest] != ((row['width'], row['height']), score['rgb8_sha256']):
                    raise ValueError('decoded PNG pixels differ from scored pixels')
                if old:
                    for key in ['ssim2', 'butter_max', 'butter_pnorm3', 'rgb8_sha256']:
                        if score[key] != old[role][key]:
                            raise ValueError(f'regenerated {role} {key} differs: {path.name}')
            recovered += int(old is not None)
        records.append(dict(run=str(run), identity_sha256=sha(run/'IDENTITY.json'),
            inputs=len(files), retained_image_records=3*len(files), recovered_inputs=recovered))
    if used_previous != set(previous):
        raise ValueError('requested previous run was not compared')
    result = dict(decoder='Pillow independent PNG decoder', unique_pngs=len(decoded),
        verifier_sha256=sha(Path(__file__)), runs=records)
    with a.output.open('x') as f:
        f.write(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--previous', nargs='*', type=Path, default=[])
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
