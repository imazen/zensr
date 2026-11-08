#!/usr/bin/env python3
"""Combine the frozen QAT comparison with its matched GPU-PTQ control.

Requires complete runs, identical input bytes, identical baseline identity and
exact repeated baseline scores/control fields. Keeps original artifact paths.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ARTIFACTS=('forced_artifact','routed_artifact','identity_artifact','forced_butter_map','identity_butter_map')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(root):
    if not (root/'COMPLETE').is_file() or (root/'COMPLETE').read_text()!=sha(root/'results.tsv'):
        raise ValueError('incomplete or changed input run')
    with (root/'results.tsv').open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
    identity={r.split('\t',1)[0]:r.split('\t',1)[1] for r in (root/'IDENTITY.tsv').read_text().splitlines()}
    return rows,identity


def combine(first,second):
    base={r['id']:r for r in first if r['arm']=='baseline'}
    repeated={r['id']:r for r in second if r['arm']=='baseline'}
    if base.keys()!=repeated.keys():raise ValueError('baseline coverage differs')
    for key,r in base.items():
        for column,value in r.items():
            if column not in ARTIFACTS and repeated[key][column]!=value:
                raise ValueError(f'repeated baseline differs: {key} {column}')
    if {r['arm'] for r in second}!={'baseline','ptq8-gpu'}:raise ValueError('unexpected supplementary arm')
    extra=[r for r in second if r['arm']=='ptq8-gpu']
    if len(extra)!=len(base) or {r['id'] for r in extra}!=base.keys():raise ValueError('supplementary coverage differs')
    for r in extra:
        for column in ('origin','source_id','identity_ssim2','default_restore'):
            if r[column]!=base[r['id']][column]:raise ValueError('supplementary control differs')
    return extra


def main(a):
    first,one=load(a.primary);second,two=load(a.control)
    for key in ('inputs_sha256','training_plan_sha256','baseline'):
        if one[key]!=two[key]:raise ValueError('runtime input or baseline identity differs')
    if (a.primary/'inputs.tsv').read_bytes()!=(a.control/'inputs.tsv').read_bytes():raise ValueError('input manifests differ')
    extra=combine(first,second)
    a.output.mkdir()
    with (a.output/'results.tsv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(first[0]),delimiter='\t');writer.writeheader()
        for root,rows in [(a.primary,first),(a.control,extra)]:
            for row in rows:
                row=row.copy()
                for key in ARTIFACTS:row[key]=os.path.relpath(root/row[key],a.output)
                writer.writerow(row)
    (a.output/'inputs.tsv').write_bytes((a.primary/'inputs.tsv').read_bytes())
    (a.output/'IDENTITY.tsv').write_text('primary_identity_sha256\t'+sha(a.primary/'IDENTITY.tsv')+'\ncontrol_identity_sha256\t'+sha(a.control/'IDENTITY.tsv')+'\ninputs_sha256\t'+one['inputs_sha256']+'\ntraining_plan_sha256\t'+one['training_plan_sha256']+'\n')
    (a.output/'MERGE.json').write_text(json.dumps(dict(primary_results_sha256=sha(a.primary/'results.tsv'),
        control_results_sha256=sha(a.control/'results.tsv'),repeated_baseline_rows=len(extra),exact_baseline_comparison=True,
        tool_sha256=sha(Path(__file__)),interpretation='Additional untrained GPU PTQ numerical control; no learned model or policy re-selection.'),indent=2)+'\n')
    (a.output/'COMPLETE').write_text(sha(a.output/'results.tsv'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['primary','control','output']:p.add_argument(name,type=Path)
    main(p.parse_args())
