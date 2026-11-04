#!/usr/bin/env python3
"""Retain the exact encoder dependency bytes identified by a study."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def capture(source,output):
    if output.exists():raise ValueError('refusing to replace provenance snapshot')
    with source.open() as f:entries=list(csv.reader(f,delimiter='\t'))
    if not entries or any(len(r)!=2 for r in entries):raise ValueError('invalid encoder identity table')
    output.mkdir(parents=True)
    records=[]
    for name,expected in entries:
        if len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):raise ValueError('invalid SHA256')
        original=Path(name);destination=output/expected
        fd,temporary=tempfile.mkstemp(prefix='copy-',suffix='.partial',dir=output)
        with os.fdopen(fd,'wb') as target,original.open('rb') as src:
            shutil.copyfileobj(src,target,1024*1024);target.flush();os.fsync(target.fileno())
        temporary=Path(temporary)
        if digest(temporary)!=expected:raise ValueError(f'encoder dependency changed: {name}; captured bytes retained at {temporary}')
        try:os.link(temporary,destination)
        except FileExistsError:
            if digest(destination)!=expected:raise ValueError('conflicting retained dependency')
        temporary.unlink()
        records.append(dict(source=name,sha256=expected,bytes=destination.stat().st_size,stored=expected,source_mode=original.stat().st_mode & 0o777))
    record=dict(identity_sha256=digest(source),capture_source_sha256=digest(Path(__file__)),dependencies=records,
                note='Captured dependency bytes, not a relocatable environment. Reproduction must resolve executable paths, loader search paths and permissions explicitly.')
    (output/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(dependencies=len(records),unique_artifacts=len({r['sha256'] for r in records})),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('identity',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();capture(args.identity,args.output)
