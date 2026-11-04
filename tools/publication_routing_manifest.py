#!/usr/bin/env python3
"""Freeze router-study origin membership before measuring candidate outputs.

Reuses the corpus-owned split and zensr grouping. Stronger document grouping
excludes related model-training pages instead of claiming they are independent.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
from corpus_split import group_key, split_map
from imazen26_canonical import manifest_rows


def origin(row):
    key = group_key(row, ia_group=True)
    folder = row['folder']
    if folder.startswith('6000-') and key:
        key = key[:2]  # All pages/rescans of one patent, not just the same page.
    if folder.startswith('5200-'):
        key = (folder, 'report')
    return json.dumps(key or (folder, row['path']), separators=(',', ':'))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--previous-scores', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a=ap.parse_args()
    if a.out.exists():
        raise SystemExit('refusing to overwrite a frozen manifest')
    canon=manifest_rows(); effective=split_map()
    recorded={}
    for line in Path('eval_split/imazen26_effective_split.tsv').read_text().splitlines():
        if line and not line.startswith('#'):
            path,bucket=line.split('\t');recorded[path]=bucket
    if effective != recorded:
        raise SystemExit('canonical split changed since the model/eval contract; resolve before fitting')
    by_src={(r['folder'],Path(r['path']).name):r for r in canon}
    groups=collections.defaultdict(list)
    for r in canon:
        groups[origin(r)].append(r)
    prior={r['file'] for r in csv.DictReader(a.previous_scores.open(), delimiter='\t')}
    pristine=list(csv.DictReader((line for line in (a.root/'PRISTINE_MANIFEST.tsv').open() if not line.startswith('#')), delimiter='\t'))
    previous_origins=set()
    for p in pristine:
        c=by_src[(p['subdir'],p['src_file'])]
        if p['dst_file'] in prior:
            previous_origins.add(origin(c))
    output=[]
    for p in pristine:
        c=by_src[(p['subdir'],p['src_file'])];o=origin(c)
        buckets={effective[r['path']] for r in groups[o]}
        if 'train' in buckets:
            role='exclude_model_training_origin'
        elif o in previous_origins:
            role='development_previous'
        elif 'validate' in buckets:
            role='router_development'
        else:
            role='locked_test'
        path=a.root/p['subdir']/p['dst_file']
        if not path.is_file():
            raise SystemExit(f'missing reference {path}')
        w,h=map(int,p['dst_wh'].split('x'))
        output.append(dict(path=str(path),file=p['dst_file'],content=c['folder'],origin=o,
                           role=role,network_split=effective[c['path']],ref_kind=p['ref_kind'],
                           width=w,height=h,reference_sha256=digest(path)))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]),delimiter='\t');writer.writeheader();writer.writerows(output)
    counts={}
    for role in sorted({r['role'] for r in output}):
        rows=[r for r in output if r['role']==role]
        counts[role]=dict(files=len(rows),origins=len({r['origin'] for r in rows}),
                          content=dict(collections.Counter(r['content'] for r in rows)),
                          reference_kind=dict(collections.Counter(r['ref_kind'] for r in rows)))
    summary=dict(manifest_sha256=digest(a.out),counts=counts,
                 source_manifest_sha256=digest(a.root/'PRISTINE_MANIFEST.tsv'),
                 effective_split_sha256=digest(Path('eval_split/imazen26_effective_split.tsv')))
    a.out.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
