#!/usr/bin/env python3
"""Check stored artifact hashes before resuming an interrupted frozen writer.

This is not the complete artifact/coverage audit and never produces AUDITED.json.
"""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
from publication_routing_cells import retained_artifact_columns, validate_retention_row


def verify(root,quarantine=False):
    for process in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=process.read_bytes().split(b'\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if args and Path(args[0].decode(errors='replace')).name=='routing_study' and str(root).encode() in args:
            raise ValueError('stop the study before resume verification')
    columns=retained_artifact_columns(root)
    referenced=set();cells=0
    for path in sorted((root/'cells').glob('*.tsv')):
        with path.open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
        if len(rows)!=1:raise ValueError(f'incomplete committed cell: {path}')
        validate_retention_row(rows[0],columns)
        referenced.update(rows[0][column] for column in columns);cells+=1
        if cells%4096==0:print(f'read {cells} completed cells',flush=True)
    print(f'checking artifacts for {cells} completed cells',flush=True)
    bad=[];count=0;found=set()
    for path in sorted((root/'artifacts').glob('*/*')):
        with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        name=str(path.relative_to(root));found.add(name);expected=path.name.split('.')[0]
        if actual!=expected:bad.append(dict(path=name,expected=expected,actual=actual,referenced=name in referenced))
        count+=1
        if count%4096==0:print(f'checked {count} artifact hashes',flush=True)
    missing=sorted(referenced-found)
    if missing:raise ValueError(f'committed cells reference missing artifacts: {missing[:10]}')
    record=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checked=datetime.datetime.now(datetime.timezone.utc).isoformat(),cells=cells,artifacts=count,mismatches=bad,quarantined=[])
    # Never move a file referenced by a completed cell, or hide that corruption.
    if bad and quarantine and not any(item['referenced'] for item in bad):
        destination=root/'resume-history'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        destination.mkdir(parents=True)
        for item in bad:
            source=root/item['path'];target=destination/source.name;source.rename(target)
            record['quarantined'].append(str(target.relative_to(root)))
    target=root/'RESUME_CHECK.json';temporary=target.with_suffix('.partial')
    temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(target)
    if bad and not record['quarantined']:raise ValueError(f'artifact hash mismatch: {bad[:10]}')
    print(json.dumps(record),flush=True)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--quarantine-unreferenced',action='store_true')
    args=parser.parse_args();verify(args.root,args.quarantine_unreferenced)
