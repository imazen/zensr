#!/usr/bin/env python3
"""Explicit immutable table snapshots for already-audited study inputs.

Bulk pixel artifacts stay at their canonical location. A snapshot is a separate
input root, not an implicit cache of a live cells directory.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from publication_routing_cells import digest, zstd_program

METADATA=('COMPLETE','COVERAGE.tsv','AUDITED.json','IDENTITY.tsv','FEATURES.tsv','ENCODERS.tsv')
FORMAT='zensr-audited-tables-v1'


def pack(root,output):
    if output.exists():raise ValueError('refusing to replace a table snapshot')
    metadata={name:(root/name).read_bytes() for name in METADATA}
    audit=json.loads(metadata['AUDITED.json'])
    if audit['coverage_sha256']!=hashlib.sha256(metadata['COVERAGE.tsv']).hexdigest():
        raise ValueError('coverage changed after audit')
    if audit['audit_source_sha256']!=digest(Path(__file__).with_name('publication_routing_cells.py')):
        raise ValueError('auditor source differs')
    paths=sorted((root/'cells').glob('*.tsv'))
    if len(paths)!=audit['cells']:raise ValueError('audited cell count differs')
    output.mkdir(parents=True)
    partial=output/'cells.tar.zst.partial';cell_hash=hashlib.sha256();total=0
    proc=subprocess.Popen([zstd_program(),'-q','-T2','-3','-o',str(partial)],stdin=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdin,mode='w|') as archive:
            for index,path in enumerate(paths,1):
                data=path.read_bytes();cell_hash.update(path.name.encode());cell_hash.update(data)
                info=tarfile.TarInfo(path.name);info.size=len(data);info.mode=0o444
                archive.addfile(info,io.BytesIO(data));total+=len(data)
                if index%4096==0:print(f'packed {index}/{len(paths)} cells',flush=True)
        proc.stdin.close()
        if proc.wait()!=0:raise ValueError('table compression failed')
    finally:
        if proc.poll() is None:proc.kill()
        if not proc.stdin.closed:proc.stdin.close()
        proc.wait()
    if cell_hash.hexdigest()!=audit['cells_sha256']:raise ValueError('cells changed after audit')
    for name,data in metadata.items():
        if (root/name).read_bytes()!=data:raise ValueError('metadata changed while packing')
        (output/name).write_bytes(data)
    archive=output/'cells.tar.zst';partial.rename(archive)
    record=dict(format=FORMAT,source_root=str(root),cells=len(paths),cell_bytes=total,
        cells_sha256=cell_hash.hexdigest(),archive_bytes=archive.stat().st_size,
        archive_sha256=digest(archive),metadata_sha256={n:hashlib.sha256(b).hexdigest() for n,b in metadata.items()},
        packer_sha256=digest(Path(__file__)))
    (output/'TABLE_SNAPSHOT.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


def cell_bytes(root):
    marker=root/'TABLE_SNAPSHOT.json'
    if not marker.exists():
        for path in sorted((root/'cells').glob('*.tsv')):yield path.name,path.read_bytes()
        return
    if (root/'cells').exists():raise ValueError('ambiguous live cells and immutable snapshot')
    record=json.loads(marker.read_text())
    if record['format']!=FORMAT:raise ValueError('unknown table snapshot format')
    for name in METADATA:
        if digest(root/name)!=record['metadata_sha256'][name]:raise ValueError('snapshot metadata changed')
    archive=root/'cells.tar.zst'
    if digest(archive)!=record['archive_sha256']:raise ValueError('table snapshot archive changed')
    proc=subprocess.Popen([zstd_program(),'-q','-d','-c',str(archive)],stdout=subprocess.PIPE)
    previous='';count=0;cell_hash=hashlib.sha256()
    try:
        with tarfile.open(fileobj=proc.stdout,mode='r|') as stream:
            for entry in stream:
                if not entry.isfile() or Path(entry.name).name!=entry.name or not entry.name.endswith('.tsv'):
                    raise ValueError('unexpected table snapshot member')
                if entry.name<=previous:raise ValueError('unordered or duplicate snapshot member')
                previous=entry.name
                with stream.extractfile(entry) as f:data=f.read()
                cell_hash.update(entry.name.encode());cell_hash.update(data);count+=1
                yield entry.name,data
        # Consume the stream's final padding before waiting for its producer.
        for _ in iter(lambda:proc.stdout.read(1048576),b''):pass
        if proc.wait()!=0:raise ValueError('corrupt table snapshot stream')
        if count!=record['cells'] or cell_hash.hexdigest()!=record['cells_sha256']:
            raise ValueError('table snapshot content differs')
    finally:
        if proc.poll() is None:proc.kill()
        proc.stdout.close();proc.wait()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();pack(args.root,args.output)
