"""Verify and summarize the frozen two-call 2048-to-8192 CPU packet."""
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
    identity = json.loads((a.root / 'IDENTITY.json').read_text())
    normalized = json.loads(json.dumps(plan))
    normalized.update(record_env=[], env={}, clean_env=False)
    normalized['metric'].setdefault('stream', 'stdout')
    normalized['check'].setdefault('stream', 'stdout')
    for case in normalized['cases']:
        case['inputs'] = {k: str(Path(v).resolve()) for k, v in case['inputs'].items()}
    if not packet['complete'] or packet['plan'] != normalized:
        raise ValueError('incomplete packet or changed plan')
    if plan['blocks'] != 1 or plan['aa'] or plan['metric']['expected'] != 2:
        raise ValueError('two-call exploratory protocol differs')
    placement = packet['placement']
    classes = placement['topology']['classes']
    selected = [c for c in classes if c['name'] == placement['core_class']]
    if (placement['status'] != 'enforced_uniform' or placement['enforcement'] != 'cpuset'
            or len(selected) != 1 or not placement['effective_cpus']
            or not set(placement['effective_cpus']) <= set(selected[0]['cpus'])
            or selected[0]['performance_rank'] != max(c['performance_rank'] for c in classes)):
        raise ValueError('performance-core boundary not verified')
    for p, digest in identity['assets'].items():
        if sha(Path(p)) != digest:
            raise ValueError(f'asset changed: {p}')
    if sha(a.root / 'input-2048.f32') != identity['input_sha256']:
        raise ValueError('input changed')
    models = [plan['baseline'], *plan['candidates']]
    invocations = packet['invocations']
    if len(invocations) != 3 or sorted(i['arm'] for i in invocations) != [0, 1, 2]:
        raise ValueError('model coverage differs')
    rows = []
    for invocation in sorted(invocations, key=lambda i: i['arm']):
        samples = invocation['values_ns']
        if (invocation['status'] != 'completed' or invocation['exit_code'] != 0
                or len(samples) != 2 or not all(math.isfinite(x) and x > 0 for x in samples)):
            raise ValueError('incomplete or invalid timing samples')
        output = Path(invocation['output'])
        for filename in ['stdout.txt', 'stderr.txt']:
            path = output / filename
            if sha(path) != invocation['artifacts'][str(path)]:
                raise ValueError('worker log changed')
        records = [json.loads(line.removeprefix('IDENTITY '))
                   for line in (output / 'stdout.txt').read_text().splitlines()
                   if line.startswith('IDENTITY ')]
        if len(records) != 1:
            raise ValueError('worker identity count differs')
        record = records[0]
        expected = [identity['input_sha256'], [1, 3, 8192, 8192]]
        if [record['input_sha256'], record['shape']] != expected or invocation['correctness'] != expected:
            raise ValueError('input/output contract differs')
        rss = re.findall(r'Maximum resident set size \(kbytes\): (\d+)',
                         (output / 'stderr.txt').read_text())
        if len(rss) != 1 or int(rss[0]) <= 0:
            raise ValueError('missing measured RSS')
        rows.append(dict(model=models[invocation['arm']]['id'],
                         iteration_1_seconds=samples[0] / 1e9,
                         iteration_2_seconds=samples[1] / 1e9,
                         mean_seconds=sum(samples) / 2e9,
                         peak_rss_kib=int(rss[0]), peak_rss_gib=int(rss[0]) / 1048576,
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
