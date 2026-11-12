"""Verify selected retained tar members over the existing bounded SSH relay."""
import argparse
import datetime
import json
from pathlib import Path
import shlex
import subprocess

READBACK = r'''
import hashlib,json,os,sys,tarfile
os.nice(19)
rows=json.loads(sys.argv[2])
with tarfile.open('/archive/'+sys.argv[1]) as archive:
    for row in rows:
        member=archive.getmember(row['member'])
        with archive.extractfile(member) as source:
            row['readback_sha256']=hashlib.file_digest(source,'sha256').hexdigest()
        if row['readback_sha256']!=row['sha256'] or member.size!=row['bytes']:
            raise ValueError('Retained archive member differs')
print(json.dumps(rows))
'''


def main(a):
    relative = Path(a.archive)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Archive path must remain in the server bind mount')
    checks = json.loads(a.members.read_text())
    if not checks or a.receipt.exists():
        raise ValueError('Readback needs explicit checks and a new receipt path')
    docker = ['docker', 'run', '--rm', '--read-only', '--pull=never', '--network=none',
              '--memory=2g', '--memory-swap=2g', '--cpuset-cpus=0-3', '--cpu-shares=256',
              '--mount', f'type=bind,src={a.server_parent},dst=/archive,readonly',
              a.image, 'python', '-B', '-c', READBACK, a.archive, json.dumps(checks)]
    remote = ['ssh', '-o', 'BatchMode=yes', a.host, shlex.join(docker)]
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', a.relay, shlex.join(remote)],
                            capture_output=True, text=True, check=True)
    rows = json.loads(result.stdout)
    if len(rows) != len(checks) or any(r['readback_sha256'] != c['sha256'] or
                                     r['member'] != c['member'] for r, c in zip(rows, checks)):
        raise ValueError('Readback response differs from requested identities')
    receipt = dict(archive=a.archive, checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   image=a.image, checks=rows)
    with a.receipt.open('x') as f:
        json.dump(receipt, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(verified_members=len(rows), archive=a.archive)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['members', 'receipt']:
        p.add_argument('--' + name, type=Path, required=True)
    for name in ['archive', 'relay', 'host', 'image', 'server-parent']:
        p.add_argument('--' + name, required=True)
    main(p.parse_args())
