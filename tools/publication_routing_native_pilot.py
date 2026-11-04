#!/usr/bin/env python3
"""Replay the frozen generator across a native archive transfer, preserving runs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from publication_routing_remote_stage import control,transfer_batch


def pilot(config,manifest,source,archive,reference,guard_scope,logs):
    if (source/'cells').exists():raise ValueError('pilot already contains cells; choose a new pilot directory')
    logs.mkdir(parents=True,exist_ok=True)
    control(config,'freeze')
    subprocess.run(['systemctl','--user','freeze',guard_scope],check=True)
    try:
        command=['./target/release/routing_study',str(manifest),str(source),'router_development','512','50','turbo','6','']
        def generate(name):
            with (logs/name).open('w') as output:
                subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,check=True)
            cells=list((source/'cells').glob('*.tsv'))
            if len(cells)!=12:raise ValueError('pilot cell count differs')
            if any(p.read_bytes()!=(reference/'cells'/p.name).read_bytes() for p in cells):
                raise ValueError('pilot differs from prior frozen replay')
        generate('generation.log')
        result=transfer_batch(config,logs)
        if result['transferred']!=66:raise ValueError('pilot artifact count differs')
        artifacts=list((source/'artifacts').glob('*/*'))
        if len(artifacts)!=66:raise ValueError('unexpected pilot artifacts')
        for path in artifacts:
            if not path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=path.name.split('.')[0]:
                raise ValueError('client archive alias is not readable/hash-correct')
        (source/'cells').rename(source/'cells.before-native-transfer')
        generate('regeneration-through-aliases.log')
        if transfer_batch(config,logs)['transferred']!=0:raise ValueError('replay unexpectedly created artifacts')
        for path in source.iterdir():
            if path.name in ('artifacts','scratch','cells.before-native-transfer'):continue
            target=archive/path.name
            if target.exists():raise ValueError(f'archive metadata already exists: {target}')
            if path.is_dir():shutil.copytree(path,target)
            else:shutil.copyfile(path,target)
        with (logs/'audit.log').open('w') as output:
            subprocess.run([sys.executable,'tools/publication_routing_cells.py','audit',str(archive)],
                           stdout=output,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(cells=12,artifacts=66,replayed_after_native_transfer=True,
                              archive_audited=True,source=str(source),archive=str(archive))),flush=True)
    finally:
        subprocess.run(['systemctl','--user','thaw',guard_scope],check=True)
        control(config,'thaw')


def transfer_pilot(config,source,archive,reference,logs):
    """Replay retained artifacts through a transfer; leave the writer frozen."""
    from publication_routing_remote_stage import frozen_request
    frozen_request(config)
    if source.exists() or archive.exists():raise ValueError('fresh pilot directories required')
    logs.mkdir(parents=True,exist_ok=True);source.mkdir(parents=True);archive.mkdir(parents=True)
    shutil.copytree(reference/'artifacts',source/'artifacts')
    (archive/'artifacts').mkdir()
    for name in ('COMPLETE','COVERAGE.tsv','ENCODERS.tsv','FEATURES.tsv','IDENTITY.tsv'):
        shutil.copyfile(reference/name,archive/name)
    shutil.copytree(reference/'cells',archive/'cells')
    paths=list((source/'artifacts').glob('*/*'))
    result=transfer_batch(config,logs)
    if result['transferred']!=len(paths):raise ValueError('transfer pilot artifact count differs')
    for path in paths:
        if not path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=path.name.split('.')[0]:
            raise ValueError('client archive alias is not readable/hash-correct')
    if transfer_batch(config,logs)['transferred']!=0:raise ValueError('repeated drain changed artifacts')
    with (logs/'audit.log').open('w') as output:
        subprocess.run([sys.executable,'tools/publication_routing_cells.py','audit',str(archive)],
                       stdout=output,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(dict(artifacts=len(paths),archive_audited=True,transfer_only=True,
                          source=str(source),archive=str(archive))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('config','manifest','source','archive','reference'):parser.add_argument(name,type=Path)
    parser.add_argument('guard_scope');parser.add_argument('logs',type=Path)
    args=parser.parse_args()
    pilot(json.loads(args.config.read_text()),args.manifest,args.source,args.archive,args.reference,args.guard_scope,args.logs)
