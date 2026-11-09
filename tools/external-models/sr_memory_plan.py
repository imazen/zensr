"""Freeze matched output-storage and DSCF lifetime comparisons at 8K."""
import argparse
import json
from pathlib import Path
import shutil

from cpu_plan import preserve, sha


def main(a):
    source = Path(__file__).resolve().parent
    a.root.mkdir(exist_ok=True)
    binary = a.root / 'bin'
    binary.mkdir(exist_ok=True)
    for path in [a.worker, source / 'sr_memory_worker.py', source / 'sr_models.py', source / 'cpu_plan.py']:
        target = binary / path.name
        preserve(target, path.read_bytes())
        if path == a.worker:
            target.chmod(0o755)
    # Retain the private Rust helper and plan source for source inspection too.
    for path in [source / 'sr_memory_plan.py', source.parents[1] / 'tools/quality-int8/runtime/src/sr_memory.rs']:
        preserve(binary / path.name, path.read_bytes())
    cpu = json.loads((a.cpu_root / 'RESULTS.json').read_text())
    gpu = json.loads((a.gpu_root / 'RESULTS.json').read_text())
    expected = dict(cpu=next(r['output_sha256'] for r in cpu if r['model'] == 'dscf'),
                    cuda=next(r['output_sha256'] for r in gpu if r['model'] == 'dscf'))
    image = a.cpu_root / 'input-2048.f32'
    weights = a.sr_root / 'models/spanf_weights_f16.raw'
    assets = [*sorted(binary.iterdir()), image, weights, a.python,
              a.python.parent.parent / 'pyvenv.cfg',
              a.external / 'dscf/team23_DSCF.pth',
              a.external / 'dscf/source/models/team23_DSCF.py',
              a.cpu_root / 'RESULTS.json', a.gpu_root / 'RESULTS.json']
    preserve(a.root / 'IDENTITY.json', (json.dumps(dict(input_sha256=sha(image),
        assets={str(p): sha(p) for p in assets}, expected_float=expected), indent=2) + '\n').encode())
    for device in ['native', 'cpu', 'cuda']:
        env = dict(CUDA_VISIBLE_DEVICES='0' if device == 'cuda' else '',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', NVIDIA_TF32_OVERRIDE='0',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                   RAYON_NUM_THREADS='4', ZENSR_WINOGRAD='0', PYTHONDONTWRITEBYTECODE='1',
                   TMPDIR=str(Path.home() / 'tmp/zensr-strata'))
        models = []
        for lifetime in ['original', 'compact']:
            if device == 'native':
                kind = 'spanf-rgb${bits}' + ('-reference' if lifetime == 'original' else '')
                args = ['-v', str(binary / 'cpu_sr'), kind, str(weights), '${input}', '2048', '2048', '4', '2']
            else:
                args = ['-v', str(a.python), str(binary / 'sr_memory_worker.py'),
                        '--root', str(a.external / 'dscf'), '--input', '${input}',
                        '--device', device, '--lifetime', lifetime, '--bits', '${bits}',
                        '--expected-float', expected[device], '--samples', '2']
            models.append(dict(id=lifetime, program='/usr/bin/time', args=args,
                               cwd=str(source.parents[1]), env=env,
                               revision='frozen-worker:' + sha(binary / ('cpu_sr' if device == 'native' else 'sr_memory_worker.py')),
                               timeout_seconds=1800))
        cases = [dict(id=f'{device}-bits{bits}', fields=dict(bits=str(bits)),
                      inputs=dict(input=str(image), **{f'asset_{i}': str(p) for i, p in enumerate(assets)}))
                 for bits in ([8, 16] if device == 'native' else [0, 8])]
        plan = dict(schema=1, baseline=models[0], candidates=models[1:], cases=cases,
                    metric=dict(extract=dict(kind='prefix', prefix='latency_ns='), unit='ns', expected=2, divide_by=1),
                    check=dict(kind='json', prefix='IDENTITY ', pointers=['/input_sha256', '/shape', '/bits', '/output_sha256']),
                    warmup='One complete digest-only control. Two timed calls include final conversion/allocation; no codecs or IQA. FP32 inference unchanged. CUDA events for GPU and separate D2H timing. Native original converts after full f32 image; compact converts each existing-kernel tile. DSCF compact releases unused tensors earlier. Float DSCF hashes must match frozen prior backend outputs.',
                    blocks=1, aa=False, deadline_seconds=7200, settle_ms=0, seed=20261109)
        preserve(a.root / f'{device}-plan.json', (json.dumps(plan, indent=2) + '\n').encode())
    print(json.dumps(dict(root=str(a.root), invocations=12, samples_per_invocation=2)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'worker', 'python', 'external', 'sr-root', 'cpu-root', 'gpu-root']:
        p.add_argument('--' + name, type=Path, required=True)
    main(p.parse_args())
