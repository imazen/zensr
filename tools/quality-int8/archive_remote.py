#!/usr/bin/env python3
"""Stream immutable experiment files into a bounded Docker writer over SSH."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

RECEIVER = r'''
import hashlib,json,os,sys
from pathlib import Path
os.nice(19)
p=Path('/archive')/sys.argv[1]
p.parent.mkdir(parents=True,exist_ok=True)
temporary=p.with_suffix(p.suffix+'.partial')
if p.exists():raise ValueError('archive destination already exists')
h=hashlib.sha256();count=0
with temporary.open('xb') as out:
    while block:=sys.stdin.buffer.read(1024*1024):
        out.write(block);h.update(block);count+=len(block)
        if count%(64*1024*1024)==0:
            out.flush();os.fdatasync(out.fileno())
            os.posix_fadvise(out.fileno(),0,count,os.POSIX_FADV_DONTNEED)
    out.flush();os.fsync(out.fileno())
temporary.rename(p)
directory=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
try:os.fsync(directory)
finally:os.close(directory)
print(json.dumps(dict(bytes=count,sha256=h.hexdigest(),file_fsync=True,directory_fsync=True)),flush=True)
'''


def main(a):
    destination = Path(a.destination)
    if destination.is_absolute() or '..' in destination.parts:
        raise ValueError('destination must remain inside the server bind mount')
    members = json.loads(a.members.read_text())
    for name in members:
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or not (a.root / p).exists():
            raise ValueError('invalid archive member')
    docker = ['docker', 'run', '--rm', '-i', '--read-only', '--pull=never',
              '--memory=2g', '--memory-swap=2g', '--cpuset-cpus=0-3',
              '--cpu-shares=256', '--mount',
              f'type=bind,src={a.server_parent},dst=/archive',
              a.image, 'python', '-B', '-c', RECEIVER, a.destination]
    remote = ['ssh', '-o', 'BatchMode=yes', a.host, shlex.join(docker)]
    command = ['ssh', '-o', 'BatchMode=yes', a.relay, shlex.join(remote)]
    count = 0
    digest = hashlib.sha256()

    def record(stage, **extra):
        value = dict(stage=stage, sent_bytes=count, destination=a.destination,
                     checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     image=a.image, members=members, **extra)
        temporary = a.progress.with_suffix('.new')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(a.progress)

    record('starting')
    tar = subprocess.Popen(['tar', '-C', str(a.root), '-cf', '-', *members],
                           stdout=subprocess.PIPE)
    writer = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        record('streaming', tar_pid=tar.pid, ssh_pid=writer.pid)
        with tar.stdout as source:
            while block := source.read(1024 * 1024):
                writer.stdin.write(block)
                digest.update(block)
                count += len(block)
                if count % (64 * 1024 * 1024) == 0:
                    record('streaming', tar_pid=tar.pid, ssh_pid=writer.pid)
        writer.stdin.close()
        response = writer.stdout.read()
        if tar.wait() != 0 or writer.wait() != 0:
            raise ValueError('archive stream or remote writer failed')
        result = json.loads(response)
        if result['bytes'] != count or result['sha256'] != digest.hexdigest():
            raise ValueError('local/remote archive identity differs')
        if not result['file_fsync'] or not result['directory_fsync']:
            raise ValueError('server durability was not established')
        record('complete', **result)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        record('failed', failure=repr(error))
        for process in [tar, writer]:
            if process.poll() is None:
                process.terminate()
            process.wait()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'members', 'progress']:
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ['relay', 'host', 'image', 'server-parent', 'destination']:
        parser.add_argument('--' + name, required=True)
    main(parser.parse_args())
