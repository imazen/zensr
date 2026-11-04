#!/usr/bin/env python3
"""Choose a previously inspected large reference using geometry only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def prepare(source,output,edge):
    if output.exists():raise ValueError('refusing to replace a resource pilot manifest')
    with source.open() as f:
        reader=csv.DictReader(f,delimiter='\t');fields=reader.fieldnames;rows=list(reader)
    eligible=[r for r in rows if r['role']=='development_previous' and max(int(r['width']),int(r['height']))>=edge]
    if not eligible:raise ValueError('no previously inspected reference reaches requested edge without upscaling')
    selected=min(eligible,key=lambda r:(-min(int(r['width']),int(r['height']))/max(int(r['width']),int(r['height'])),r['file']))
    with output.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,delimiter='\t');writer.writeheader();writer.writerow(selected)
    record=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        selector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),edge=edge,
        rule='Previously inspected origin; eligible without upscaling; closest aspect ratio to square, filename tie break.',
        selected=selected)
    output.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--edge',type=int,required=True)
    args=parser.parse_args();prepare(args.source,args.output,args.edge)
