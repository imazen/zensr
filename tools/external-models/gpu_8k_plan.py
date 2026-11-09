"""Freeze the two-call FP32 CUDA extension of the 8K CPU experiment."""
import argparse
import json
from pathlib import Path

from cpu_plan import preserve, sha


def main(a):
    source = Path(__file__).resolve().parent
    image = a.cpu_root / 'input-2048.f32'
    identity = json.loads((a.cpu_root / 'IDENTITY.json').read_text())
    if sha(image) != identity['input_sha256']:
        raise ValueError('CPU timing fixture changed')
    a.root.mkdir(exist_ok=True)
    assets = [image, a.python, a.python.parent.parent / 'pyvenv.cfg',
              source / 'gpu_8k_worker.py', source / 'gpu_8k_plan.py',
              source / 'sr_models.py', source / 'cpu_plan.py',
              a.root / 'spanf/team24_SPANF.py', a.root / 'spanf/team24_spanf.pth',
              a.sr_root / 'models/spanf_weights.raw', a.sr_root / 'models/spanf_weights_f16.raw',
              a.external / 'dscf/team23_DSCF.pth',
              a.external / 'dscf/source/models/team23_DSCF.py',
              a.external / 'dscf/ACQUISITION.json']
    env = dict(CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               NVIDIA_TF32_OVERRIDE='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1',
               TMPDIR=str(Path.home() / 'tmp/zensr-strata'))
    models = []
    for name in ['spanf', 'dscf']:
        args = ['-v', str(a.python), str(source / 'gpu_8k_worker.py'), '--name', name,
                '--spanf-root', str(a.root / 'spanf'),
                '--spanf-f32', str(a.sr_root / 'models/spanf_weights.raw'),
                '--spanf-mixed', str(a.sr_root / 'models/spanf_weights_f16.raw'),
                '--dscf-root', str(a.external / 'dscf'), '--input', '${input}',
                '--output', '${output}', '--samples', '2']
        models.append(dict(id=name + '-cuda-fp32', program='/usr/bin/time', args=args,
                           cwd=str(source.parents[1]), env=env,
                           revision='worker-sha256:' + sha(source / 'gpu_8k_worker.py'),
                           timeout_seconds=1200))
    plan = dict(schema=1, baseline=models[0], candidates=models[1:],
                cases=[dict(id='2048-to-8192', inputs=dict(input=str(image),
                            **{f'asset_{i}': str(p) for i, p in enumerate(assets)}))],
                metric=dict(extract=dict(kind='prefix', prefix='latency_ns='),
                            unit='ns', expected=2, divide_by=1),
                check=dict(kind='json', prefix='IDENTITY ', pointers=['/input_sha256', '/shape']),
                warmup='One full-size control before two resident inference calls. CUDA events synchronized per call. Pinned H2D and D2H measured separately; allocation, input reads, weight loading, checksums, codecs and IQA excluded. Inference output/scratch allocation included.',
                blocks=1, aa=False, deadline_seconds=2400, settle_ms=0, seed=20261109)
    preserve(a.root / 'plan.json', (json.dumps(plan, indent=2) + '\n').encode())
    preserve(a.root / 'IDENTITY.json', (json.dumps(dict(input_sha256=sha(image),
             assets={str(p): sha(p) for p in assets}, spanf_source_commit='b69fb668c0362deb696eecbfdebfaa5c2fcdfcb4'), indent=2) + '\n').encode())
    print(str(a.root / 'plan.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'cpu-root', 'sr-root', 'external', 'python']:
        p.add_argument('--' + name, type=Path, required=True)
    main(p.parse_args())
