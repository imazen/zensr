#!/usr/bin/env python3
"""Prepare reproducible pilot inputs and audit completed publication cells."""
import argparse
import csv
import hashlib
import datetime
import json
import os
import tempfile
import io
import functools
import shutil
from concurrent.futures import ThreadPoolExecutor
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



@functools.lru_cache(maxsize=1)
def zstd_program():
    path=shutil.which('zstd')
    if path is None:raise FileNotFoundError('zstd executable required for artifact audit')
    return path

def inspect_artifact(artifact):
    from PIL import Image
    from PIL.JpegImagePlugin import get_sampling
    expected=artifact.name.split('.')[0]
    if digest(artifact)!=expected:raise ValueError(f'changed artifact {artifact}')
    if artifact.suffix=='.jpg':
        with Image.open(artifact) as image:
            image.load()
            if image.format!='JPEG':raise ValueError('non-JPEG encoded artifact')
            sampling={0:'444',1:'422',2:'420'}.get(get_sampling(image),'unsupported')
            return dict(dimensions=image.size,ss=sampling)
    if artifact.suffix!='.zst':raise ValueError('unknown artifact format')
    proc=subprocess.Popen([zstd_program(),'-q','-d','-c',str(artifact)],stdout=subprocess.PIPE)
    size=0
    for chunk in iter(lambda:proc.stdout.read(1048576),b''):size+=len(chunk)
    if proc.wait()!=0:raise ValueError(f'corrupt compressed artifact {artifact}')
    return dict(bytes=size)


def write_record(path,record):
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'-',suffix='.partial',dir=path.parent)
    with os.fdopen(fd,'w') as f:
        json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(name,path)


def audit(root,workers=1):
    if workers<1:raise ValueError('audit workers must be positive')
    if not (root/'COMPLETE').is_file():raise ValueError('grid completion marker required')
    audit_source=digest(Path(__file__));coverage_hash=digest(root/'COVERAGE.tsv')
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
    def progress(stage,error=None):
        write_record(root/'AUDIT_PROGRESS.json',dict(stage=stage,checked=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            pid=os.getpid(),root=str(root),workers=workers,checked_cells=count,required_cells=len(planned),
            cell_files=len(paths),verified_artifacts=len(verified),audit_source_sha256=audit_source,
            coverage_sha256=coverage_hash,error=error))
    progress('verifying')
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Bound queued reads and decompressions independently of total grid size.
            for start in range(0,len(paths),128):
                batch=[];pending={}
                for path in paths[start:start+128]:
                    data=path.read_bytes();cell_hash.update(path.name.encode());cell_hash.update(data)
                    rows=list(csv.DictReader(io.StringIO(data.decode()),delimiter='\t'))
                    if len(rows)!=1:raise ValueError(f'incomplete cell {path}')
                    row=rows[0];batch.append(row)
                    key=tuple(row[c] for c in ('reference_sha256','size','encoder','ss','nominal_q'))
                    if key in seen:raise ValueError(f'duplicate {key}')
                    seen.add(key)
                    covered.add(tuple(row[c] for c in ('file','size','encoder','ss','nominal_q')))
                    for name in required_artifacts:
                        value=row[name];artifact=root/value
                        expected_suffix='.jpg' if name=='jpeg_artifact' else ('.butter-f32le.zst' if name.endswith('_butter_map') else '.rgb8.zst')
                        if not artifact.name.endswith(expected_suffix):raise ValueError('artifact kind mismatch')
                        if value not in verified and value not in pending:
                            pending[value]=pool.submit(inspect_artifact,artifact)
                for row in batch:
                    dims=(int(row['width']),int(row['height']))
                    for name in required_artifacts:
                        value=row[name];artifact=root/value
                        if value not in verified:verified[value]=pending.pop(value).result()
                        info=verified[value]
                        # Shared byte verification never bypasses any row's shape
                        # or chroma-sampling contract.
                        if name=='jpeg_artifact':
                            if info['dimensions']!=dims:raise ValueError(f'independent decoder dimensions differ: {artifact}')
                            if info['ss']!=row['ss']:raise ValueError(f'independent decoder subsampling differs: {artifact}')
                        else:
                            bytes_per_pixel=4 if name.endswith('_butter_map') else 3
                            if info['bytes']!=dims[0]*dims[1]*bytes_per_pixel:
                                raise ValueError(f'pixel/map size mismatch: {artifact}')
                    count+=1
                progress('verifying')
        if planned!=covered:raise ValueError(f'grid mismatch: missing {len(planned-covered)}, unexpected {len(covered-planned)}')
        if digest(Path(__file__))!=audit_source:raise ValueError('auditor source changed during run')
        if digest(root/'COVERAGE.tsv')!=coverage_hash:raise ValueError('coverage changed during audit')
        import PIL
        from PIL import features
        write_record(root/'AUDITED.json',dict(cells=count,artifacts=len(verified),cells_sha256=cell_hash.hexdigest(),
            audit_source_sha256=audit_source,coverage_sha256=coverage_hash,artifact_workers=workers,
            zstd_binary_sha256=digest(Path(zstd_program())),independent_jpeg_decoder='Pillow/libjpeg',
            pillow_version=PIL.__version__,libjpeg_version=features.version('jpg')))
        progress('complete')
    except BaseException as error:
        progress('failed',f'{type(error).__name__}: {error}')
        raise
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
    p=sub.add_parser('audit');p.add_argument('root',type=Path);p.add_argument('--workers',type=int,default=1)
    p=sub.add_parser('status');p.add_argument('root',type=Path)
    a=ap.parse_args()
    if a.command=='pilot':pilot(a.source,a.output)
    elif a.command=='audit':audit(a.root,a.workers)
    else:status(a.root)

if __name__=='__main__':main()
