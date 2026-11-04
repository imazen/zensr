#!/usr/bin/env python3
"""Storage-local staging operations, coordinated by a frozen writer's owner."""
import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import uuid

from publication_routing_stage import transfer, digest


def identities():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),Path(__file__).with_name('publication_routing_stage.py'))}


def usage(source):
    allocated=payload=files=aliases=0
    for directory in source.iterdir():
        if not directory.is_dir() or directory.is_symlink():raise ValueError('unexpected staging directory')
        with os.scandir(directory) as entries:
            for entry in entries:
                stat=entry.stat(follow_symlinks=False)
                allocated+=stat.st_blocks*512
                if entry.is_symlink():aliases+=1
                elif entry.is_file(follow_symlinks=False):payload+=stat.st_size;files+=1
                else:raise ValueError('unexpected staged entry')
    space=os.statvfs(source)
    return dict(allocated_bytes=allocated,payload_bytes=payload,files=files,
                aliases=aliases,free_bytes=space.f_bavail*space.f_frsize,tools=identities())


def sync_filesystem(path):
    """Linux storage backend: commit file data and namespace in one barrier."""
    libc=ctypes.CDLL(None,use_errno=True)
    syncfs=libc.syncfs;syncfs.argtypes=[ctypes.c_int];syncfs.restype=ctypes.c_int
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:
        if syncfs(fd)!=0:
            error=ctypes.get_errno();raise OSError(error,os.strerror(error))
    finally:os.close(fd)


def prepare_copy(source,destination):
    expected=source.name.split('.')[0]
    if not re.fullmatch('[0-9a-f]{64}',expected):raise ValueError('not a content-addressed artifact')
    if source.is_symlink() or destination.is_symlink():raise ValueError('symlink artifact')
    if digest(source)!=expected:raise ValueError('corrupt staged artifact')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        fd,name=tempfile.mkstemp(prefix='.transfer-',dir=destination.parent)
        temporary=Path(name)
        with os.fdopen(fd,'wb') as out,source.open('rb') as inp:
            for data in iter(lambda:inp.read(1024*1024),b''):out.write(data)
        if digest(temporary)!=expected:raise ValueError('transfer digest mismatch')
        try:os.link(temporary,destination)
        except FileExistsError:pass
        if digest(destination)!=expected:raise ValueError('conflicting destination')
        temporary.unlink()
    elif digest(destination)!=expected:raise ValueError('conflicting destination')
    return source.stat().st_size


def transfer_group(paths,source,destination,alias_root):
    # No source pathname changes before every copy passes and syncfs succeeds.
    # Failed copies and partial destinations remain available for diagnosis.
    def prepare(path):return prepare_copy(path,destination/path.relative_to(source))
    with ThreadPoolExecutor(max_workers=4) as pool:sizes=list(pool.map(prepare,paths))
    sync_filesystem(destination)
    for path in paths:
        alias=path.parent/('.archive-alias-'+uuid.uuid4().hex)
        alias.symlink_to(alias_root/path.relative_to(source))
        os.replace(alias,path)
    sync_filesystem(source)
    return sum(sizes)


def drain(source,destination,request):
    # The coordinator owns systemctl on the compute host. The storage host
    # cannot independently observe that cgroup; bind its request and response.
    frozen=(request.get('version')=='zensr-frozen-native-stage-v1' and request.get('writer_frozen') is True)
    ended=(request.get('version')=='zensr-ended-native-stage-v1' and request.get('writer_ended') is True
           and request.get('excluded')==[])
    if not (frozen or ended):
        raise ValueError('frozen writer or verified ended writer request required')
    if not re.fullmatch('[0-9a-f]{32}',request.get('nonce','')):raise ValueError('invalid request nonce')
    if request.get('tools')!=identities():raise ValueError('native stage tool identity changed')
    source=source.resolve();destination=destination.resolve()
    if source==destination or source in destination.parents or destination in source.parents:
        raise ValueError('staging and archive must be disjoint')
    if not destination.is_dir():raise ValueError('archive directory required')
    alias_root=Path(request['alias_root'])
    if not alias_root.is_absolute():raise ValueError('absolute client archive path required')
    excluded=set(request['excluded'])
    if any(not re.fullmatch(r'[0-9a-f]{2}/[0-9a-f]{64}\.[a-z0-9.-]+',p) for p in excluded):
        raise ValueError('invalid open-artifact path')
    ready=[];skipped=0
    for path in sorted(source.glob('*/*')):
        relative=path.relative_to(source)
        if path.is_symlink():
            if os.readlink(path)!=str(alias_root/relative):raise ValueError('unexpected client archive alias')
            continue
        if not path.is_file():raise ValueError('unexpected staged entry')
        if relative.as_posix() in excluded:skipped+=1;continue
        ready.append(path)
    count=total=0
    def move(path):
        relative=path.relative_to(source)
        return transfer(path,destination/relative,keep_alias=True,alias_target=alias_root/relative)
    if request.get('batch_sync',False):
        for offset in range(0,len(ready),256):
            batch=ready[offset:offset+256]
            total+=transfer_group(batch,source,destination,alias_root);count+=len(batch)
            print(json.dumps(dict(phase='transfer-progress',nonce=request['nonce'],artifacts=count,bytes=total)),flush=True)
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for size in pool.map(move,ready):
                count+=1;total+=size
                if count%256==0:
                    print(json.dumps(dict(phase='transfer-progress',nonce=request['nonce'],artifacts=count,bytes=total)),flush=True)
    return dict(phase='transfer-complete',nonce=request['nonce'],transferred=count,
                bytes=total,open_files_retained=skipped,usage=usage(source),tools=identities())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('usage','drain'))
    parser.add_argument('source',type=Path);parser.add_argument('destination',type=Path,nargs='?')
    args=parser.parse_args()
    if args.operation=='usage':result=usage(args.source)
    else:
        if args.destination is None:parser.error('drain requires destination')
        result=drain(args.source,args.destination,json.load(sys.stdin))
    print(json.dumps(result),flush=True)
