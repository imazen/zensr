"""Freeze a two-sample 2048-to-8192 CPU extension of the retained SR plans."""
import argparse
import copy
import json
from pathlib import Path

from cpu_plan import preserve, sha


def main(a):
    a.output.mkdir(exist_ok=True)
    sr = json.loads((a.sr_root / 'full-plan-v1.json').read_text())
    dscf = json.loads((a.dscf_root / 'dscf-plan-v1.json').read_text())
    # Repeat the decoded 128-pixel fixture without interpolation. This periodic
    # texture is only a latency fixture, never quality/generalization evidence.
    small = a.sr_root / 'input-128.f32'
    data = small.read_bytes()
    if len(data) != 3 * 128 * 128 * 4:
        raise ValueError('source fixture geometry differs')
    planes = []
    for c in range(3):
        rows = [data[(c * 128 + y) * 512:(c * 128 + y + 1) * 512] * 16
                for y in range(128)]
        planes.append(b''.join(rows) * 16)
    image = a.output / 'input-2048.f32'
    preserve(image, b''.join(planes))
    native = next(m for m in sr['candidates'] if m['id'] == 'native-spanf')
    models = copy.deepcopy([native, *dscf['candidates']])
    assets = json.loads((a.dscf_root / 'assets-dscf-plan-v1.json').read_text())
    assets.update({p: sha(Path(p)) for p in
                   [native['args'][1], native['args'][3], str(small), __file__]})
    for p, digest in assets.items():
        if sha(Path(p)) != digest:
            raise ValueError(f'asset changed: {p}')
    for model in models:
        model['args'][-1] = '2'
        model['timeout_seconds'] = 1800
        if model['id'] == 'DSCF-tract':
            # Rebind input facts in cpu_sr and infer the fully convolutional
            # graph at 2048; output geometry is checked by the existing worker.
            model['args'][3] = str(a.dscf_root / 'models/DSCF_x4_256x256.onnx')
    plan = dict(schema=1, baseline=models[0], candidates=models[1:],
        cases=[dict(id='2048-t4', fields=dict(n='2048', threads='4'),
                    inputs=dict(input=str(image), **{f'asset_{i}': p
                                for i, p in enumerate(sorted(assets))}))],
        metric=dict(extract=dict(kind='prefix', prefix='latency_ns='),
                    unit='ns', expected=2, divide_by=1),
        check=dict(kind='json', prefix='IDENTITY ', pointers=['/input_sha256', '/shape']),
        warmup=dscf['warmup'], blocks=1, aa=False, deadline_seconds=5400,
        settle_ms=0, seed=20261109)
    preserve(a.output / 'plan.json', (json.dumps(plan, indent=2) + '\n').encode())
    identity = dict(source_sha256=sha(small), input_sha256=sha(image),
                    recipe='repeat planar 128x128 fixture 16x16, no interpolation',
                    assets=assets, timed_calls_per_model=2, models=[m['id'] for m in models])
    preserve(a.output / 'IDENTITY.json', (json.dumps(identity, indent=2) + '\n').encode())
    print(json.dumps(dict(plan=str(a.output / 'plan.json'), models=identity['models'],
                         input_sha256=identity['input_sha256'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['sr-root', 'dscf-root', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    main(p.parse_args())
