#!/usr/bin/env python3
"""Freeze a source-separated QAT evaluation from retained JPEG/reference bytes.

Select using corpus metadata and completed-cell availability only. Never read
quality scores for selection. Stage just the selected inputs, not old outputs.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imazen26_canonical import manifest_rows
from publication_routing_manifest import origin


def sha(data): return hashlib.sha256(data).hexdigest()
def digest(path): return sha(path.read_bytes())
def table(path):
    with path.open() as stream:
        return list(csv.DictReader((l for l in stream if not l.startswith('#')),delimiter='\t'))


def main(a):
    a.output.mkdir()
    corpus={r['number']:r for r in manifest_rows()}
    splits={r['id']:r['split'] for r in table(a.split_map)}
    seen={r['imazen_id'] for r in table(a.original_pairs)}
    seen_origins={origin(corpus[i]) for i in seen}
    pristine={r['dst_file']:r for r in table(a.clean/'PRISTINE_MANIFEST.tsv')}
    qualities=list(range(5,101,5))
    families=['turbo','mozjpeg','jpegli','zenjpeg']
    source_inputs={str(p):digest(p) for p in [a.original_pairs,a.split_map,a.clean/'PRISTINE_MANIFEST.tsv',a.primary/'COVERAGE.tsv',a.secondary/'references.tsv']}
    candidates=collections.defaultdict(list)
    def eligible(filename,bucket):
        match=re.match(r'^(\d{4})_',filename)
        if not match:return False
        i=match[1]
        return splits[i]==bucket and i not in seen and origin(corpus[i]) not in seen_origins
    for filename in sorted({r['file'] for r in table(a.primary/'COVERAGE.tsv')}):
        if eligible(filename,'test'):
            p=pristine[filename]
            candidates[p['subdir']].append(p)
    selected=[];availability=[]
    def planned(root,p,sizes,encoders):
        path=a.clean/p['subdir']/p['dst_file']
        reference_sha=digest(path)
        cases=[]
        for size in sizes:
            for encoder in encoders:
                for ss in ('420','444'):
                    for quality in qualities:
                        key=sha(f'{reference_sha}\t{size}\t{encoder}\t{ss}\t{quality}'.encode())
                        cell=root/'cells'/f'{key}.tsv'
                        if not cell.is_file():return None
                        cases.append((cell,root,key,p))
        return cases
    for content,rows in sorted(candidates.items()):
        chosen=None
        for p in sorted(rows,key=lambda p:sha(('canon4-qat-eval-v1:'+p['dst_file'][:4]).encode())):
            cases=planned(a.primary,p,[512],families)
            if cases is not None:
                chosen=p;selected.extend(cases);break
        availability.append(dict(content=content,eligible_references=len(rows),selected_id=chosen['dst_file'][:4] if chosen else None))
    if not selected:raise ValueError('no complete primary reference grids')
    secondary=[]
    secondary_rows=table(a.secondary/'references.tsv')
    for r in sorted(secondary_rows,key=lambda r:sha(('canon4-qat-size-v1:'+r['file'][:4]).encode())):
        if not eligible(r['file'],'validate'):continue
        p=pristine[r['file']]
        cases=planned(a.secondary,p,[128,512,1024],['zenjpeg'])
        if cases is not None:
            secondary.append(r['file'][:4]);selected.extend(cases)
            if len(secondary)==2:break
    if len(secondary)!=2:raise ValueError('need two complete source-separated secondary size grids')
    refs=a.output/'references';refs.mkdir()
    jpegs=a.output/'jpegs';jpegs.mkdir()
    refs_done={};jpgs_done=set();manifest=[]
    def artifact_path(root, relative):
        path=root/relative
        if path.is_file():return path
        if root==a.secondary and a.secondary_archive is not None:
            archived=a.secondary_archive/relative
            if archived.is_file():return archived
        raise FileNotFoundError(path)

    for cell,root,key,p in selected:
        rows=table(cell)
        if len(rows)!=1:raise ValueError('incomplete retained cell')
        r=rows[0]
        if r['file']!=p['dst_file']:raise ValueError('wrong retained source')
        filename=r['jpeg_artifact'];source=artifact_path(root,filename)
        raw=source.read_bytes()
        if sha(raw)!=source.name.split('.')[0]:raise ValueError('JPEG bytes changed')
        jpg=jpegs/f'{sha(raw)}.jpg'
        if jpg not in jpgs_done:
            with jpg.open('xb') as stream:stream.write(raw)
            jpgs_done.add(jpg)
        ref_source=artifact_path(root,r['reference_artifact'])
        if str(ref_source) not in refs_done:
            compressed=ref_source.read_bytes()
            if sha(compressed)!=ref_source.name.split('.')[0]:raise ValueError('reference bytes changed')
            pixels=subprocess.run(['zstd','-q','-d','-c'],input=compressed,capture_output=True,check=True).stdout
            ref=refs/f'{sha(pixels)}.rgb8'
            if not ref.exists():
                with ref.open('xb') as stream:stream.write(pixels)
            refs_done[str(ref_source)]=(ref,len(pixels))
        ref,length=refs_done[str(ref_source)]
        if length!=int(r['width'])*int(r['height'])*3:raise ValueError('reference shape mismatch')
        manifest.append(dict(id=key,source_id=p['dst_file'][:4],origin=sha(origin(corpus[p['dst_file'][:4]]).encode()),
            content=p['subdir'],cohort='primary' if root==a.primary else 'secondary',
            size=r['size'],encoder=r['encoder'],ss=r['ss'],q=r['nominal_q'],
            jpeg=str(jpg.resolve()),reference_rgb8=str(ref.resolve()),width=r['width'],height=r['height'],
            jpeg_sha256=jpg.stem,reference_rgb8_sha256=ref.stem,retained_cell_sha256=digest(cell)))
    ids=[r['id'] for r in manifest]
    if len(set(ids))!=len(ids):raise ValueError('duplicate evaluation case')
    with (a.output/'inputs.tsv').open('x') as stream:
        w=csv.DictWriter(stream,fieldnames=list(manifest[0]),delimiter='\t');w.writeheader();w.writerows(manifest)
    record=dict(selector_sha256=digest(Path(__file__)),input_metadata_sha256=source_inputs,
        published_split_revision='187fbf338ce08e8e6654db7f04ddae58d5263da2',
        rule='One complete 512-pixel reference per available content stratum, hash-selected from published test IDs; exclude all original pair IDs and their publication origin groups. Two hash-selected published-validation origins with complete zenjpeg grids at 128/512/1024 form secondary size evidence. No score-based selection.',
        qualities=qualities,primary_encoders=families,subsampling=['420','444'],
        strata=availability,secondary_source_ids=secondary,cases=len(manifest),
        cohort_cases=dict(collections.Counter(r['cohort'] for r in manifest)),
        source_ids=sorted({r['source_id'] for r in manifest}),
        inputs_sha256=digest(a.output/'inputs.tsv'),
        limitations='Retained inputs previously used in routing studies; not a fresh corpus or population estimate. No QAT candidate test outputs informed selection. Historical pixel-artifact auditing remains cancelled; only selected JPEG/reference hashes and shapes were checked.')
    (a.output/'selection.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ['cases','cohort_cases','source_ids','secondary_source_ids']},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['primary','secondary','clean','original_pairs','split_map','output']:
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--secondary-archive',type=Path)
    main(p.parse_args())
