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


def audit(root):
    from PIL import Image
    seen=set();covered=set();count=0;verified=set();cell_hash=hashlib.sha256()
    paths=sorted((root/'cells').glob('*.tsv'))
    if not paths:raise ValueError('no completed cells')
    for path in paths:
        cell_hash.update(path.name.encode());cell_hash.update(path.read_bytes())
        with path.open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
        if len(rows)!=1:raise ValueError(f'incomplete cell {path}')
        row=rows[0]
        key=(row['reference_sha256'],row['size'],row['encoder'],row['ss'],row['nominal_q'])
        if key in seen:raise ValueError(f'duplicate {key}')
        seen.add(key)
        covered.add((row['file'],row['size'],row['encoder'],row['ss'],row['nominal_q']))
        for name,value in row.items():
            if name.endswith('_artifact') or name.endswith('_butter_map'):
                artifact=root/value
                expected=artifact.name.split('.')[0]
                if value in verified:continue
                verified.add(value)
                if digest(artifact)!=expected:raise ValueError(f'changed artifact {artifact}')
                if artifact.suffix=='.jpg':
                    with Image.open(artifact) as image:
                        image.load()
                        if image.size!=(int(row['width']),int(row['height'])):
                            raise ValueError(f'independent decoder dimensions differ: {artifact}')
                if artifact.suffix=='.zst':
                    proc=subprocess.Popen(['zstd','-q','-d','-c',str(artifact)],stdout=subprocess.PIPE)
                    size=0
                    for chunk in iter(lambda:proc.stdout.read(1048576),b''):size+=len(chunk)
                    if proc.wait()!=0:raise ValueError(f'corrupt compressed artifact {artifact}')
                    channels=4 if '.butter-f32le.' in artifact.name else 3
                    if size!=int(row['width'])*int(row['height'])*channels:
                        raise ValueError(f'pixel/map size mismatch: {artifact}')
        count+=1
    coverage=root/'COVERAGE.tsv'
    if coverage.exists():
        with coverage.open() as f:
            planned={(r['file'],r['size'],r['encoder'],r['ss'],r['q']) for r in csv.DictReader(f,delimiter='\t') if r['status']=='required'}
        if planned!=covered:raise ValueError(f'grid mismatch: missing {len(planned-covered)}, unexpected {len(covered-planned)}')
    import json
    (root/'AUDITED.json').write_text(json.dumps(dict(cells=count,artifacts=len(verified),cells_sha256=cell_hash.hexdigest(),independent_jpeg_decoder='Pillow/libjpeg'),indent=2)+'\n')
    print(f'{count} unique cells; {len(verified)} artifacts hashed, decompressed and shape-checked; JPEGs read independently')


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('pilot');p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p=sub.add_parser('audit');p.add_argument('root',type=Path)
    a=ap.parse_args()
    if a.command=='pilot':pilot(a.source,a.output)
    else:audit(a.root)

if __name__=='__main__':main()
