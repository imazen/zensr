"""Verify two-call CUDA evidence and retain timing/VRAM/transfer summaries."""
import argparse
import csv
import gzip
import io
import json
import math
from pathlib import Path
import re

from cpu_plan import preserve, sha


def main(a):
    packet = json.loads(gzip.decompress((a.root / 'run-v1/packet.json.gz').read_bytes()))
    plan = json.loads((a.root / 'plan.json').read_text())
    normalized = json.loads(json.dumps(plan))
    normalized.update(record_env=[], env={}, clean_env=False)
    for kind in ['metric', 'check']:
        normalized[kind].setdefault('stream', 'stdout')
    for case in normalized['cases']:
        case.setdefault('fields', {})
        case['inputs'] = {k: str(Path(v).resolve()) for k, v in case['inputs'].items()}
    if not packet['complete'] or packet['plan'] != normalized:
        raise ValueError('incomplete packet or changed plan')
    if plan['blocks'] != 1 or plan['aa'] or plan['metric']['expected'] != 2:
        raise ValueError('two-call protocol differs')
    identity = json.loads((a.root / 'IDENTITY.json').read_text())
    for p, digest in identity['assets'].items():
        if sha(Path(p)) != digest:
            raise ValueError(f'asset changed: {p}')
    invocations = packet['invocations']
    if len(invocations) != 2 or sorted(i['arm'] for i in invocations) != [0, 1]:
        raise ValueError('model coverage differs')
    rows = []
    for invocation in sorted(invocations, key=lambda i: i['arm']):
        if invocation['status'] != 'completed' or invocation['exit_code'] != 0:
            raise ValueError('worker failed')
        output = Path(invocation['output'])
        for filename in ['stdout.txt', 'stderr.txt']:
            path = output / filename
            if sha(path) != invocation['artifacts'][str(path)]:
                raise ValueError('worker log changed')
        lines = (output / 'stdout.txt').read_text().splitlines()
        identities = [json.loads(x.removeprefix('IDENTITY ')) for x in lines if x.startswith('IDENTITY ')]
        samples = [json.loads(x.removeprefix('SAMPLE ')) for x in lines if x.startswith('SAMPLE ')]
        result = json.loads((output / 'RESULT.json').read_text())
        if len(identities) != 1 or result != dict(identity=identities[0], samples=samples):
            raise ValueError('result file differs from bound stdout')
        observed = identities[0]
        if (invocation['correctness'] != [identity['input_sha256'], [1, 3, 8192, 8192]]
                or observed['input_sha256'] != identity['input_sha256']
                or observed['shape'] != [1, 3, 8192, 8192]
                or observed['precision'] != 'FP32' or observed['tf32']):
            raise ValueError('identity or precision contract differs')
        if len(samples) != 2 or len(invocation['values_ns']) != 2:
            raise ValueError('sample count differs')
        for i, sample in enumerate(samples):
            if sample['iteration'] != i + 1:
                raise ValueError('iteration order differs')
            values = [v for k, v in sample.items() if k != 'iteration']
            if any(not math.isfinite(x) or x <= 0 for x in values):
                raise ValueError('invalid sample')
            if abs(invocation['values_ns'][i] - sample['inference_cuda_ms'] * 1e6) > 1:
                raise ValueError('packet timing differs from CUDA event')
        rss = re.findall(r'Maximum resident set size \(kbytes\): (\d+)',
                         (output / 'stderr.txt').read_text())
        if len(rss) != 1:
            raise ValueError('missing measured host RSS')
        mean = lambda key: sum(s[key] for s in samples) / 2
        rows.append(dict(model=observed['name'], iteration_1_ms=samples[0]['inference_cuda_ms'],
                         iteration_2_ms=samples[1]['inference_cuda_ms'], mean_ms=mean('inference_cuda_ms'),
                         mean_synchronized_host_ms=mean('inference_synchronized_host_ms'),
                         mean_h2d_ms=mean('pinned_h2d_cuda_ms'), mean_d2h_ms=mean('pinned_d2h_cuda_ms'),
                         peak_allocated_gib=max(s['peak_allocated_bytes'] for s in samples) / 2**30,
                         peak_reserved_gib=max(s['peak_reserved_bytes'] for s in samples) / 2**30,
                         host_peak_rss_kib=int(rss[0]), output_sha256=observed['output_sha256']))
    preserve(a.root / 'RESULTS.json', (json.dumps(rows, indent=2) + '\n').encode())
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    preserve(a.root / 'results.csv', stream.getvalue().encode())
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    main(p.parse_args())
