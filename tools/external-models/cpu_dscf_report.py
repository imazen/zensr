"""Report the complete six-case DSCF CPU matrix without relaxing other reports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from cpu_report import read_packet


def main(a):
    rows, packet = read_packet(a.packet, 'dscf')
    if ({r['model'] for r in rows} != {'native-span', 'DSCF-tract', 'dscf', '__aa'}
            or {(r['input_side'], r['requested_threads']) for r in rows}
            != {(n, t) for n in [64, 128, 256] for t in [1, 4]} or len(rows) != 24):
        raise ValueError('DSCF matrix membership differs')
    a.output.mkdir(exist_ok=True)
    with (a.output/'results.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    with (a.output/'IDENTITY.json').open('x') as f:
        f.write(json.dumps(dict(packet_sha256=hashlib.sha256(a.packet.read_bytes()).hexdigest(),
            fingerprint=packet['fingerprint'], placement=packet['placement'],
            invocations=len(packet['invocations'])), indent=2, sort_keys=True)+'\n')
    for row in rows:
        if row['model'] != '__aa':print(json.dumps(row))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
