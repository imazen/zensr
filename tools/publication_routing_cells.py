#!/usr/bin/env python3
"""Prepare reproducible pilot inputs and audit completed publication cells."""
import argparse
import csv
import hashlib
from pathlib import Path
import subprocess


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def pilot(source, output):
    if output.exists():raise ValueError('refusing to overwrite pilot manifest')
    with source.open() as f:
        reader=csv.DictReader(f,delimiter='\t');fields=reader.fieldnames;rows=list(reader)
    selected=[]
    for prefix in ('1000-', '8100-'):
        eligible=sorted((r for r in rows if r['role']=='development_previous' and r['content'].startswith(prefix)),key=lambda r:r['file'])
        if not eligible:raise ValueError(f'no pilot input for {prefix}')
        selected.append(eligible[0])
    with output.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,delimiter='\t');writer.writeheader();writer.writerows(selected)
    print(f'pilot: {len(selected)} references, sha256 {digest(output)}')


def required_coverage(root):
    with (root/'COVERAGE.tsv').open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
    seen=set();required=set()
    allowed={'required','excluded_no_upscale','excluded_metric_aspect_domain','excluded_by_caller_unsupported'}
    for r in rows:
        key=tuple(r[c] for c in ('file','size','encoder','ss','q'))
        if key in seen:raise ValueError('duplicate coverage entry')
        if r['status'] not in allowed:raise ValueError('unknown coverage status')
        seen.add(key)
        if r['status']=='required':required.add(key)
    if not required:raise ValueError('empty required coverage')
    return required


def inspect_artifact(artifact):
    from PIL import Image
    expected=artifact.name.split('.')[0]
    if digest(artifact)!=expected:raise ValueError(f'changed artifact {artifact}')
    if artifact.suffix=='.jpg':
        with Image.open(artifact) as image:
            image.load()
            if image.format!='JPEG':raise ValueError('non-JPEG encoded artifact')
            return dict(dimensions=image.size)
    if artifact.suffix!='.zst':raise ValueError('unknown artifact format')
    proc=subprocess.Popen(['zstd','-q','-d','-c',str(artifact)],stdout=subprocess.PIPE)
    size=0
    for chunk in iter(lambda:proc.stdout.read(1048576),b''):size+=len(chunk)
    if proc.wait()!=0:raise ValueError(f'corrupt compressed artifact {artifact}')
    return dict(bytes=size)


def audit(root):
    if not (root/'COMPLETE').is_file():raise ValueError('grid completion marker required')
    planned=required_coverage(root)
    if (root/'AUDITED.json').exists():
        import time
        history=root/'audit-history';history.mkdir(exist_ok=True)
        (root/'AUDITED.json').rename(history/f'{time.time_ns()}.json')
    seen=set();covered=set();count=0;verified={};cell_hash=hashlib.sha256()
    paths=sorted((root/'cells').glob('*.tsv'))
    if not paths:raise ValueError('no completed cells')
    required_artifacts=('jpeg_artifact','reference_artifact','decoded_artifact','restored_artifact',
                        'identity_butter_map','restored_butter_map')
    for path in paths:
        cell_hash.update(path.name.encode());cell_hash.update(path.read_bytes())
        with path.open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
        if len(rows)!=1:raise ValueError(f'incomplete cell {path}')
        row=rows[0]
        key=tuple(row[c] for c in ('reference_sha256','size','encoder','ss','nominal_q'))
        if key in seen:raise ValueError(f'duplicate {key}')
        seen.add(key)
        covered.add(tuple(row[c] for c in ('file','size','encoder','ss','nominal_q')))
        dims=(int(row['width']),int(row['height']))
        for name in required_artifacts:
            value=row[name];artifact=root/value
            expected_suffix='.jpg' if name=='jpeg_artifact' else ('.butter-f32le.zst' if name.endswith('_butter_map') else '.rgb8.zst')
            if not artifact.name.endswith(expected_suffix):raise ValueError('artifact kind mismatch')
            if value not in verified:verified[value]=inspect_artifact(artifact)
            info=verified[value]
            # Deduplicate expensive I/O, never the shape contract of each row.
            if name=='jpeg_artifact':
                if info['dimensions']!=dims:raise ValueError(f'independent decoder dimensions differ: {artifact}')
            else:
                bytes_per_pixel=4 if name.endswith('_butter_map') else 3
                if info['bytes']!=dims[0]*dims[1]*bytes_per_pixel:
                    raise ValueError(f'pixel/map size mismatch: {artifact}')
        count+=1
    if planned!=covered:raise ValueError(f'grid mismatch: missing {len(planned-covered)}, unexpected {len(covered-planned)}')
    import json,PIL
    from PIL import features
    (root/'AUDITED.json').write_text(json.dumps(dict(cells=count,artifacts=len(verified),cells_sha256=cell_hash.hexdigest(),audit_source_sha256=digest(Path(__file__)),independent_jpeg_decoder='Pillow/libjpeg',pillow_version=PIL.__version__,libjpeg_version=features.version('jpg')),indent=2)+'\n')
    print(f'{count} unique cells; {len(verified)} artifacts hashed, decompressed and shape-checked; JPEGs read independently')


def status(root):
    import datetime,json
    cells=sum(1 for _ in (root/'cells').glob('*.tsv')) if (root/'cells').exists() else 0
    required=None
    if (root/'COVERAGE.tsv').exists():
        with (root/'COVERAGE.tsv').open() as f:
            required=sum(r['status']=='required' for r in csv.DictReader(f,delimiter='\t'))
    pids=[]
    found=subprocess.run(['pgrep','-x','routing_study'],capture_output=True,text=True)
    for value in found.stdout.split():
        cmd=subprocess.run(['ps','-p',value,'-o','args='],capture_output=True,text=True).stdout.strip()
        if str(root) in cmd:pids.append(dict(pid=int(value),command=cmd))
    record=dict(checked=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(root),
        completed_cells=cells,required_cells=required,live_processes=pids,
        complete_marker=(root/'COMPLETE').exists(),artifact_audited=(root/'AUDITED.json').exists())
    if (root/'IDENTITY.tsv').exists():record['identity']=(root/'IDENTITY.tsv').read_text()
    (root/'PROGRESS.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('pilot');p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p=sub.add_parser('audit');p.add_argument('root',type=Path)
    p=sub.add_parser('status');p.add_argument('root',type=Path)
    a=ap.parse_args()
    if a.command=='pilot':pilot(a.source,a.output)
    elif a.command=='audit':audit(a.root)
    else:status(a.root)

if __name__=='__main__':main()
