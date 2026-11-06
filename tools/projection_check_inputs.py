#!/usr/bin/env python3
"""Prepare a bounded, explicitly worst-case diagnostic from retained artifacts.

No encoding, fitting, or broad artifact audit. References remain exact RGB8.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess


def prepare(report, root, output, count):
    if count < 1:
        raise ValueError('count must be positive')
    rows = json.loads(report.read_text())['decoded/budget-0.005'][:count]
    if not rows:
        raise ValueError('no diagnostic rows')
    output.mkdir(exist_ok=False)
    records = []
    with (output/'inputs.tsv').open('w') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(['id', 'jpeg', 'reference_rgb8', 'width', 'height'])
        for i, row in enumerate(rows):
            jpeg = root/row['jpeg_artifact']
            reference = root/row['reference_artifact']
            encoded = reference.read_bytes()
            raw = (subprocess.run(['zstd', '-q', '-d', '-c'], input=encoded,
                                  check=True, capture_output=True).stdout
                   if encoded[:4] == b'\x28\xb5\x2f\xfd' else encoded)
            if len(raw) != int(row['width'])*int(row['height'])*3:
                raise ValueError('reference length mismatch')
            path = output/f'reference-{i}.rgb8'
            path.write_bytes(raw)
            writer.writerow([f'worst-{i}', jpeg, path.resolve(), row['width'], row['height']])
            records.append(dict(row=row, jpeg_sha256=hashlib.sha256(jpeg.read_bytes()).hexdigest(),
                                reference_rgb8_sha256=hashlib.sha256(raw).hexdigest()))
    (output/'provenance.json').write_text(json.dumps(dict(
        selection='first worst cases from already-inspected decoded/budget-0.005 report; diagnostic only',
        source_report=str(report), source_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        records=records), indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('report', 'root', 'output'):
        p.add_argument(name, type=Path)
    p.add_argument('--count', type=int, default=4)
    a = p.parse_args()
    prepare(a.report, a.root, a.output, a.count)
