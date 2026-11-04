#!/usr/bin/env python3
"""Copy and verify a quiescent artifact stage, retaining its original directory."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import uuid

from publication_routing_remote_stage import frozen_request
from publication_routing_stage import digest,sync_directory


def migrate(config,source,destination,backup):
    if frozen_request(config)['excluded']:raise ValueError('open artifact descriptors remain')
    if source.is_symlink() or backup.exists():raise ValueError('unexpected source or existing backup')
    if any(destination.glob('*/*')):raise ValueError('destination is not empty')
    subprocess.run(['rsync','-rlt','--links','--checksum',str(source)+'/',str(destination)+'/'],check=True)
    records=[]
    for path in sorted(source.glob('*/*')):
        relative=path.relative_to(source);target=destination/relative
        if path.is_symlink():
            expected=str(Path(config['alias_root'])/relative)
            if os.readlink(path)!=expected or not target.is_symlink() or os.readlink(target)!=expected:
                raise ValueError('archive alias changed during migration')
        else:
            expected=path.name.split('.')[0]
            if digest(path)!=expected or target.is_symlink() or digest(target)!=expected:
                raise ValueError(f'artifact changed during migration: {relative}')
            with target.open('rb') as f:os.fsync(f.fileno())
        records.append(relative.as_posix())
        if len(records)%256==0:print(json.dumps(dict(verified=len(records))),flush=True)
    if {p.relative_to(destination).as_posix() for p in destination.glob('*/*')}!=set(records):
        raise ValueError('destination membership differs')
    # A paused thread may have completed mkdir on the old stage before its
    # next path lookup. Supply every content-addressed prefix in the new stage.
    for index in range(256):
        directory=destination/f'{index:02x}';directory.mkdir(exist_ok=True)
        sync_directory(directory)
    sync_directory(destination)
    if frozen_request(config)['excluded']:raise ValueError('artifact descriptors changed while frozen')
    link=source.parent/('.nvme-stage-link-'+uuid.uuid4().hex);link.symlink_to(destination.absolute())
    source.rename(backup)
    try:os.replace(link,source)
    except BaseException:
        backup.rename(source)
        raise
    sync_directory(source.parent)
    result=dict(source=str(source),destination=str(destination),retained_backup=str(backup),
                files=len(records),members=records)
    (source.parent/'nvme-migration.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='members'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('config','source','destination','backup'):parser.add_argument(name,type=Path)
    args=parser.parse_args()
    migrate(json.loads(args.config.read_text()),args.source,args.destination,args.backup)
