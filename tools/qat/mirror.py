#!/usr/bin/env python3
"""Archive immutable QAT artifacts in one sequential stream to project storage."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main(a):
    for name in a.members:
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or not (a.root/p).exists():raise ValueError('invalid archive member')
        if (name.endswith('evaluation') or name=='primary') and not (a.root/name/'COMPLETE').is_file():
            raise ValueError('archive only completed experiment stages')
    if a.destination.exists():raise ValueError('archive already exists')
    a.destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=a.destination.with_suffix(a.destination.suffix+'.partial')
    digest=hashlib.sha256();copied=0
    def record(stage,**extra):
        result=dict(pid=os.getpid(),stage=stage,copied_bytes=copied,
                    checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    destination=str(a.destination),members=a.members,**extra)
        new=a.progress.with_suffix('.new');new.write_text(json.dumps(result,indent=2)+'\n');new.replace(a.progress)
    record('copying')
    try:
        with temporary.open('xb') as output:
            process=subprocess.Popen(['tar','-C',str(a.root),'-cf','-',*a.members],stdout=subprocess.PIPE)
            assert process.stdout is not None
            with process.stdout as stream:
                while block:=stream.read(1024*1024):
                    output.write(block);digest.update(block);copied+=len(block)
                    if copied%(64*1024*1024)==0:record('copying')
                    if copied%(256*1024*1024)==0:
                        # Dirty NFS pages count against the cgroup even though
                        # the Python stream itself uses only a few MiB.
                        output.flush();os.fdatasync(output.fileno())
                        os.posix_fadvise(output.fileno(),0,copied,os.POSIX_FADV_DONTNEED)
            if process.wait()!=0:raise ValueError('archive command failed')
            record('synchronizing',sha256=digest.hexdigest());output.flush();os.fsync(output.fileno())
        if a.destination.exists():raise ValueError('archive destination appeared')
        temporary.rename(a.destination)
        record('complete',sha256=digest.hexdigest())
        print(f'{digest.hexdigest()}  {a.destination}',flush=True)
    except BaseException as error:
        record('failed',failure=repr(error));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('destination',type=Path);p.add_argument('progress',type=Path)
    p.add_argument('members',nargs='+');main(p.parse_args())
