#!/usr/bin/env python3
"""Storage-local staging operations, coordinated by a frozen writer's owner."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from publication_routing_stage import transfer


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


def drain(source,destination,request):
    # The coordinator owns systemctl on the compute host. The storage host
    # cannot independently observe that cgroup; bind its request and response.
    if request.get('version')!='zensr-frozen-native-stage-v1' or request.get('writer_frozen') is not True:
        raise ValueError('frozen writer request required')
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
