#!/usr/bin/env python3
"""Quarantine thumbnail-near reference groups before any policy fitting.

Reuses the established 16x16-luma/mean-absolute-distance audit. This is a
conservative suspected-duplicate screen, not proof of semantic identity.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from picker_leakage_audit import thumb, NEAR


def main():
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('output',type=Path)
    a=ap.parse_args()
    if a.output.exists():raise ValueError('refusing to overwrite frozen membership')
    with a.source.open() as f:
        reader=csv.DictReader(f,delimiter='\t');fields=reader.fieldnames;rows=list(reader)
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        parent[find(j)]=find(i)
    origins={}
    for i,r in enumerate(rows):
        if r['origin'] in origins:union(i,origins[r['origin']])
        origins[r['origin']]=i
    features=[]
    for r in rows:
        feature=thumb(r['path'])
        if feature is None:raise ValueError(f'failed fingerprint: {r["path"]}')
        features.append(feature)
    x=np.asarray(features,dtype=np.int16)
    pairs=[]
    for begin in range(0,len(rows),32):
        distances=np.abs(x[begin:begin+32,None,:]-x[None,:,:]).mean(axis=2)
        for local,j in zip(*np.where(distances < NEAR)):
            i=begin+int(local);j=int(j)
            if j<=i or rows[i]['origin']==rows[j]['origin']:continue
            pairs.append(dict(left=rows[i]['file'],right=rows[j]['file'],distance=float(distances[local,j])))
            union(i,j)
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):groups[find(i)].append(r)
    priority=['exclude_model_training_origin','development_previous','router_development','locked_test']
    changed=0
    for group in groups.values():
        old_origins=sorted({r['origin'] for r in group})
        identifier='audit:'+hashlib.sha256('\n'.join(old_origins).encode()).hexdigest()
        role=min((r['role'] for r in group),key=priority.index)
        for r in group:
            changed+=r['role']!=role;r['role']=role;r['origin']=identifier
    with a.output.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,delimiter='\t');writer.writeheader();writer.writerows(rows)
    result=dict(source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
                output_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
                threshold=NEAR,changed_roles=changed,cross_origin_near_pairs=pairs,
                counts={role:dict(files=sum(r['role']==role for r in rows),
                                  origins=len({r['origin'] for r in rows if r['role']==role})) for role in priority})
    a.output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cross_origin_near_pairs'},indent=2))
    print('cross-origin pairs',len(pairs))


if __name__=='__main__':main()
