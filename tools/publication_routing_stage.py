#!/usr/bin/env python3
"""Move closed content-addressed artifacts from a frozen writer to durable storage.

The caller freezes the entire writer scope first. Open files stay in staging.
No artifact is released until its destination is hashed and synced. This is a
storage operation, not an artifact/coverage audit or permission to fit a study.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def transfer(source,destination):
    """Preserve conflicting/corrupt bytes; release source only after verification."""
    expected=source.name.split('.')[0]
    if not re.fullmatch('[0-9a-f]{64}',expected):raise ValueError('not a content-addressed artifact')
    if source.is_symlink() or destination.is_symlink():raise ValueError('symlink artifact')
    if digest(source)!=expected:raise ValueError(f'corrupt staged artifact: {source}')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        fd,name=tempfile.mkstemp(prefix='.transfer-',dir=destination.parent)
        temporary=Path(name)
        # Failed copies remain intact for diagnosis; never overwrite a target.
        with os.fdopen(fd,'wb') as out,source.open('rb') as inp:
            for data in iter(lambda:inp.read(1024*1024),b''):out.write(data)
            out.flush();os.fsync(out.fileno())
        if digest(temporary)!=expected:raise ValueError('transfer digest mismatch')
        try:os.link(temporary,destination)
        except FileExistsError:pass
        if digest(destination)!=expected:raise ValueError(f'conflicting destination: {destination}')
        temporary.unlink()
    elif digest(destination)!=expected:raise ValueError(f'conflicting destination: {destination}')
    # Sync pre-existing copies as well, then their directory, before release.
    with destination.open('rb') as f:os.fsync(f.fileno())
    sync_directory(destination.parent)
    size=source.stat().st_size
    source.unlink();sync_directory(source.parent)
    return size


def drain(source,destination,scope):
    source=source.resolve();destination=destination.resolve()
    if source==destination or source in destination.parents or destination in source.parents:
        raise ValueError('staging and destination must be disjoint')
    group=subprocess.check_output(['systemctl','--user','show',scope,'--value','-p','ControlGroup'],text=True).strip()
    if not group:raise ValueError('missing writer scope')
    cgroup=Path('/sys/fs/cgroup')/group.lstrip('/')
    if 'frozen 1' not in (cgroup/'cgroup.events').read_text().splitlines():
        raise ValueError('writer scope must be frozen')
    pids=set()
    for path in cgroup.rglob('cgroup.procs'):pids.update(path.read_text().split())
    if not pids:raise ValueError('empty writer scope')
    opened=set()
    for pid in pids:
        for fd in (Path('/proc')/pid/'fd').iterdir():
            try:
                stat=fd.stat();opened.add((stat.st_dev,stat.st_ino))
            except FileNotFoundError:pass
    count=total=skipped=0
    for path in sorted(source.glob('*/*')):
        if not path.is_file():raise ValueError(f'unexpected staged entry: {path}')
        stat=path.stat()
        if (stat.st_dev,stat.st_ino) in opened:skipped+=1;continue
        total+=transfer(path,destination/path.relative_to(source));count+=1
    return dict(transferred=count,bytes=total,open_files_retained=skipped)


def watch(source,destination,scope):
    """Operational free-space guard, not a filesystem quota or hard byte cap."""
    def free():
        stat=os.statvfs(source)
        return stat.f_bavail*stat.f_frsize
    minimum=10*1024**3
    threshold=max(minimum,free()-8*1024**3)
    if free()<=minimum:raise ValueError('insufficient staging headroom')
    print(json.dumps(dict(stage='watching',free_bytes=free(),drain_threshold_bytes=threshold)),flush=True)
    while True:
        state=subprocess.check_output(['systemctl','--user','show',scope,'--value','-p','ActiveState'],text=True).strip()
        if state!='active':
            print(json.dumps(dict(stage='writer-ended',scope_state=state,final_drain_required=True)),flush=True)
            return
        if free()<threshold:
            subprocess.run(['systemctl','--user','freeze',scope],check=True)
            print(json.dumps(dict(stage='draining',free_bytes=free())),flush=True)
            # Failures deliberately leave the writer frozen and artifacts intact.
            result=drain(source,destination,scope)
            if free()<=minimum:raise ValueError('space remains low; writer left frozen')
            print(json.dumps(dict(stage='drained',free_bytes=free(),**result)),flush=True)
            subprocess.run(['systemctl','--user','thaw',scope],check=True)
        time.sleep(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path);parser.add_argument('destination',type=Path)
    parser.add_argument('--frozen-scope',required=True)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    if args.watch:watch(args.source,args.destination,args.frozen_scope)
    else:print(json.dumps(drain(args.source,args.destination,args.frozen_scope)),flush=True)
