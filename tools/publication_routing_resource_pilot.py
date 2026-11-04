#!/usr/bin/env python3
"""Choose non-reserved large references for resource measurement using geometry only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def prepare(source,output,edge,count=1,role="development_previous"):
    if output.exists():raise ValueError('refusing to replace a resource pilot manifest')
    if count<1:raise ValueError('resource pilot reference count must be positive')
    if role not in ('development_previous','router_development'):raise ValueError('resource pilots cannot use reserved references')
    with source.open() as f:
        reader=csv.DictReader(f,delimiter='\t');fields=reader.fieldnames;rows=list(reader)
    eligible=[r for r in rows if r['role']==role and max(int(r['width']),int(r['height']))>=edge]
    if len(eligible)<count:raise ValueError('insufficient non-reserved references reach requested edge without upscaling')
    selected=sorted(eligible,key=lambda r:(-min(int(r['width']),int(r['height']))/max(int(r['width']),int(r['height'])),r['file']))[:count]
    with output.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,delimiter='\t');writer.writeheader();writer.writerows(selected)
    record=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        selector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),edge=edge,count=count,role=role,
        rule='Explicit non-reserved role; eligible without upscaling; closest aspect ratios to square, filename tie break. Resource measurement only; not a statistical origin sample.',
        selected=selected)
    output.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--edge',type=int,required=True)
    parser.add_argument('--references',type=int,default=1)
    parser.add_argument('--role',choices=('development_previous','router_development'),default='development_previous')
    args=parser.parse_args();prepare(args.source,args.output,args.edge,args.references,args.role)
