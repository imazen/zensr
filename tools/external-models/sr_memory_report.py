"""Verify the complete matched native/CPU/CUDA storage experiment."""
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
    identity = json.loads((a.root / 'IDENTITY.json').read_text())
    for path, digest in identity['assets'].items():
        if sha(Path(path)) != digest:
            raise ValueError(f'asset changed: {path}')
    rows = []
    for device in ['native', 'cpu', 'cuda']:
        packet = json.loads(gzip.decompress((a.root / f'{device}-v1/packet.json.gz').read_bytes()))
        plan = json.loads((a.root / f'{device}-plan.json').read_text())
        normalized = json.loads(json.dumps(plan))
        normalized.update(record_env=[], env={}, clean_env=False)
        for key in ['metric', 'check']:
            normalized[key].setdefault('stream', 'stdout')
        for case in normalized['cases']:
            case['inputs'] = {k: str(Path(v).resolve()) for k, v in case['inputs'].items()}
        if not packet['complete'] or packet['plan'] != normalized:
            raise ValueError(f'incomplete or changed {device} experiment')
        if plan['blocks'] != 1 or plan['aa'] or plan['metric']['expected'] != 2:
            raise ValueError('protocol differs')
        placement = packet['placement']
        classes = placement['topology']['classes']
        selected = [c for c in classes if c['name'] == placement['core_class']]
        if (placement['status'] != 'enforced_uniform' or placement['enforcement'] != 'cpuset'
                or len(selected) != 1 or not placement['effective_cpus']
                or not set(placement['effective_cpus']) <= set(selected[0]['cpus'])
                or selected[0]['performance_rank'] != max(c['performance_rank'] for c in classes)):
            raise ValueError('performance-core boundary differs')
        invocations = packet['invocations']
        if len(invocations) != 4 or sorted((i['case'], i['arm']) for i in invocations) != [(0,0), (0,1), (1,0), (1,1)]:
            raise ValueError('missing or duplicate configuration')
        paired = {}
        for invocation in sorted(invocations, key=lambda i: (i['case'], i['arm'])):
            values = invocation['values_ns']
            if (invocation['status'] != 'completed' or invocation['exit_code'] != 0
                    or len(values) != 2 or any(not math.isfinite(x) or x <= 0 for x in values)):
                raise ValueError('failed invocation or invalid samples')
            output = Path(invocation['output'])
            for filename in ['stdout.txt', 'stderr.txt']:
                path = output / filename
                if sha(path) != invocation['artifacts'][str(path)]:
                    raise ValueError('worker log changed')
            lines = (output / 'stdout.txt').read_text().splitlines()
            records = [json.loads(x.removeprefix('IDENTITY ')) for x in lines if x.startswith('IDENTITY ')]
            if len(records) != 1:
                raise ValueError('identity count differs')
            record = records[0]
            bits = int(plan['cases'][invocation['case']]['fields']['bits'])
            check = [identity['input_sha256'], [1,3,8192,8192], bits, record['output_sha256']]
            if (invocation['correctness'] != check or record['input_sha256'] != check[0]
                    or record['shape'] != check[1] or record['bits'] != bits):
                raise ValueError('input/shape/output mismatch')
            if device != 'native' and record['float_output_sha256'] != identity['expected_float'][device]:
                raise ValueError('changed float computation')
            if invocation['case'] in paired and paired[invocation['case']] != record['output_sha256']:
                raise ValueError('compact output differs from original conversion')
            paired[invocation['case']] = record['output_sha256']
            rss = re.findall(r'Maximum resident set size \(kbytes\): (\d+)',
                             (output / 'stderr.txt').read_text())
            if len(rss) != 1:
                raise ValueError('missing measured RSS')
            samples = [json.loads(x.removeprefix('SAMPLE ')) for x in lines if x.startswith('SAMPLE ')]
            if device != 'native' and (len(samples) != 2 or [s['latency_ns'] for s in samples] != values):
                raise ValueError('sample detail mismatch')
            rows.append(dict(model='SPANF' if device == 'native' else 'DSCF', device=device,
                bits=bits, variant='original' if invocation['arm'] == 0 else 'compact',
                iteration_1_ms=values[0]/1e6, iteration_2_ms=values[1]/1e6,
                mean_ms=sum(values)/2e6, peak_rss_kib=int(rss[0]),
                peak_rss_mib=int(rss[0])/1024,
                peak_allocated_gib=max(s['allocated_bytes'] for s in samples)/2**30 if device == 'cuda' else None,
                peak_reserved_gib=max(s['reserved_bytes'] for s in samples)/2**30 if device == 'cuda' else None,
                mean_d2h_ms=sum(s['d2h_ms'] for s in samples)/2 if device == 'cuda' else None,
                output_sha256=record['output_sha256']))
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
