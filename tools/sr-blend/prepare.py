"""Retain a score-independent subset of existing SR outputs and source identities."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main(a):
    identity = a.cascade/'run-v1/IDENTITY.json'
    complete = json.loads((a.cascade/'run-v1/COMPLETE.json').read_text())
    if sha(identity) != complete['identity_sha256']:
        raise ValueError('Cascade identity does not match its completion record')
    a.output.mkdir(parents=True, exist_ok=False)
    retained = a.output/'retained'
    retained.mkdir()
    copied = {}

    def retain(path, digest):
        source = Path(path)
        if sha(source) != digest:
            raise ValueError(f'Artifact hash differs: {source}')
        target = retained/(digest+source.suffix)
        if digest not in copied:
            shutil.copyfile(source, target)
            if sha(target) != digest:
                raise ValueError('Retained copy differs')
            copied[digest] = dict(path=str(target), bytes=target.stat().st_size, source=str(source))
        return dict(path=str(target), sha256=digest)

    cases = []
    for path in sorted((a.cascade/'run-v1/cells').glob('*.json')):
        cell = json.loads(path.read_text())
        row = cell['row']
        if row['q'] not in (0, 5, 20, 50, 90) or row['ss'] not in ('clean', '420'):
            continue
        reference = retain(row['reference'], row['reference_sha256'])
        low = retain(row['input'], row['input_sha256'])
        models = []
        for model in ('spanf', 'dscf'):
            original = cell[model+'_direct']
            png = original['png']
            models.append(dict(model=model, png=retain(png['path'], png['sha256']),
                               rgb8_sha256=original['rgb8_sha256']))
        cases.append(dict(id=row['id'], source_id=row['source_id'], origin=row['origin'],
            category=row['content'], q=row['q'], sampling=row['ss'], width=row['width'], height=row['height'],
            low_width=row['low_width'], low_height=row['low_height'], original=reference, low=low, sr=models))
    sources = {c['source_id'] for c in cases}
    if len(sources) != 17 or len(cases) != 85:
        raise ValueError(f'Frozen subset coverage differs: {len(sources)} sources, {len(cases)} cases')
    for source in sources:
        if {c['q'] for c in cases if c['source_id']==source} != {0, 5, 20, 50, 90}:
            raise ValueError('Quality coverage differs')
    manifest = dict(selection='source identities and q0/5/20/50/90; clean or JPEG 420; direct SR, repair off; no score selection',
        source_cascade_identity_sha256=sha(identity), scales=[2,4],
        filters=['triangle','mitchell','catmull_rom','lanczos2'], intensity_quarters=[0,1,2,3,4],
        color='filter resize in encoded sRGB; convex blend of stored RGB8 samples before final RGB8 rounding',
        x2='Box resize stored x4 SR to x2; HR reference reduced with linear-light CatmullRom',
        cases=cases)
    (a.output/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (a.output/'RETENTION.json').write_text(json.dumps(dict(files=copied,unique_files=len(copied),
        total_bytes=sum(r['bytes'] for r in copied.values()),manifest_sha256=sha(a.output/'inputs.json')),indent=2)+'\n')
    print(json.dumps(dict(sources=len(sources),cases=len(cases),sr_pairs=len(cases)*2,files=len(copied),
        bytes=sum(r['bytes'] for r in copied.values()))))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cascade',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
