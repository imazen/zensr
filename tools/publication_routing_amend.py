#!/usr/bin/env python3
"""Finalize an explicitly approved reference exclusion and frozen subset recovery.

Preserves the original coverage and records both manifest identities. Does not
alter measurements, models, executable identity, or the frozen analysis code.
Artifact auditing remains a separate mandatory step.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil


def digest(data):
    return hashlib.sha256(data).hexdigest()


def finalize(root, recovery, manifest, recovery_manifest, excluded_sha, reason):
    if (root/'COMPLETE').exists():
        raise ValueError('original run already complete')
    if not (recovery/'COMPLETE').is_file():
        raise ValueError('recovery not complete')
    def identity(path):
        return dict(line.split('\t', 1) for line in (path/'IDENTITY.tsv').read_text().splitlines())
    original=identity(root); subset=identity(recovery)
    if original['manifest_sha256']!=digest(manifest.read_bytes()):
        raise ValueError('original manifest changed')
    if subset['manifest_sha256']!=digest(recovery_manifest.read_bytes()):
        raise ValueError('recovery manifest changed')
    if {k:v for k,v in original.items() if k!='manifest_sha256'}!={k:v for k,v in subset.items() if k!='manifest_sha256'}:
        raise ValueError('recovery changed scientific inputs')
    for name in ('FEATURES.tsv','ENCODERS.tsv'):
        if (root/name).read_bytes()!=(recovery/name).read_bytes():
            raise ValueError('recovery changed feature or encoder definitions')
    refs={r['reference_sha256']:r for r in csv.DictReader(io.StringIO(manifest.read_text()),delimiter='\t') if r['role']=='locked_test'}
    excluded=refs[excluded_sha]
    if int(excluded['width'])*int(excluded['height'])<=120_000_000:
        raise ValueError('exclusion is not an oversized reference')
    selected=list(csv.DictReader(io.StringIO(recovery_manifest.read_text()),delimiter='\t'))
    if any(r!=refs.get(r['reference_sha256']) or r['reference_sha256']==excluded_sha for r in selected):
        raise ValueError('recovery reference differs from original manifest')
    coverage=(root/'COVERAGE.tsv').read_bytes()
    rows=list(csv.DictReader(io.StringIO(coverage.decode()),delimiter='\t'))
    changed=0
    for r in rows:
        if r['file']==excluded['file'] and r['status']=='required':
            r['status']='excluded_by_caller_unsupported'; changed+=1
    if not changed:raise ValueError('no required cells excluded')
    planned={tuple(r[k] for k in ('file','size','encoder','ss','q')) for r in rows if r['status']=='required'}
    actual=set(); sources={}
    for directory in (root/'cells',recovery/'cells'):
        for path in directory.glob('*.tsv'):
            data=path.read_bytes(); cell=list(csv.DictReader(io.StringIO(data.decode()),delimiter='\t'))
            if len(cell)!=1:raise ValueError('incomplete cell')
            r=cell[0]
            key=tuple(r[k] for k in ('file','size','encoder','ss','nominal_q'))
            if key not in planned:raise ValueError('unexpected cell')
            if r['reference_sha256'] not in refs or r['reference_sha256']==excluded_sha:
                raise ValueError('unexpected reference')
            if r['file']!=refs[r['reference_sha256']]['file'] or r.get('role')!='locked_test':
                raise ValueError('reference metadata differs')
            if path.name in sources:
                if sources[path.name].read_bytes()!=data:raise ValueError('conflicting replay')
                continue
            if key in actual:raise ValueError('duplicate measurement')
            actual.add(key); sources[path.name]=path
    if actual!=planned:raise ValueError('amended grid incomplete')
    history=root/'reference-exclusion-history'
    history.mkdir(exist_ok=False)
    (history/'COVERAGE.tsv').write_bytes(coverage)
    shutil.copyfile(manifest,history/'original-manifest.tsv')
    shutil.copyfile(recovery_manifest,history/'recovery-manifest.tsv')
    for name,path in sources.items():
        target=root/'cells'/name
        if not target.exists():
            shutil.copyfile(path,target)
            if target.read_bytes()!=path.read_bytes():raise ValueError('copy differs')
    amended=io.StringIO(); writer=csv.DictWriter(amended,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    (root/'COVERAGE.tsv').write_text(amended.getvalue())
    record=dict(reason=reason,excluded_reference=excluded,excluded_cells=changed,
        completed_cells=len(actual),original_manifest_sha256=original['manifest_sha256'],
        recovery_manifest_sha256=subset['manifest_sha256'],original_coverage_sha256=digest(coverage),
        amended_coverage_sha256=digest(amended.getvalue().encode()),finalizer_sha256=digest(Path(__file__).read_bytes()),
        artifact_audit_required=True)
    (root/'REFERENCE_EXCLUSION.json').write_text(json.dumps(record,indent=2)+'\n')
    (root/'COMPLETE').write_text(f'{len(actual)}/{len(planned)} required cells completed\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','recovery','manifest','recovery_manifest'):p.add_argument(name,type=Path)
    p.add_argument('excluded_sha');p.add_argument('--reason',required=True)
    a=p.parse_args();finalize(a.root,a.recovery,a.manifest,a.recovery_manifest,a.excluded_sha,a.reason)
