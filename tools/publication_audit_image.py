#!/usr/bin/env python3
"""Capture an offline-buildable Linux x86_64 artifact-audit image context.

Reuse the installed Pillow package and native utility bytes. No package manager
runs at image startup or build time; the base must be an explicit image digest.
"""
import argparse
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

import PIL
from publication_routing_provenance import capture, digest


def prepare(output, base):
    if platform.system()!='Linux' or platform.machine()!='x86_64':
        raise ValueError('this native utility bundle requires Linux x86_64')
    if sys.version_info[:2]!=(3,12):raise ValueError('the audit image requires Python 3.12 package ABI')
    if not re.fullmatch(r'[a-zA-Z0-9./_-]+@sha256:[0-9a-f]{64}',base):
        raise ValueError('an explicit base-image digest is required')
    if output.exists():raise ValueError('refusing to replace an audit image context')
    sources={}
    package=Path(PIL.__file__).parent
    for tree in (package,package.parent/'pillow.libs'):
        if not tree.is_dir():raise ValueError(f'missing Pillow wheel tree: {tree}')
        for path in sorted(tree.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                sources['python/'+str(path.relative_to(package.parent))]=path.resolve()
    for command in ('zstd','time'):
        executable=shutil.which(command)
        if executable is None:raise ValueError(f'missing utility: {command}')
        binary=Path(executable).resolve();sources['native/'+command]=binary
        result=subprocess.run(['ldd',str(binary)],check=True,text=True,capture_output=True).stdout
        if 'not found' in result:raise ValueError('unresolved native utility dependency')
        for name in re.findall(r'(?:=>\s+)?(/[^\s]+)',result):
            path=Path(name).resolve();destination='native/'+Path(name).name
            if destination in sources and digest(sources[destination])!=digest(path):
                raise ValueError('conflicting native library basename')
            sources[destination]=path
    if 'native/ld-linux-x86-64.so.2' not in sources:raise ValueError('missing native loader')
    sources['publication_routing_cells.py']=Path(__file__).with_name('publication_routing_cells.py')
    output.mkdir(parents=True)
    identity=output/'sources.tsv'
    identity.write_text(''.join(f'{p}\t{digest(p)}\n' for p in sorted(set(sources.values()))))
    capture(identity,output/'snapshot')
    records=[]
    context=output/'context';context.mkdir()
    for destination,source in sorted(sources.items()):
        sha=digest(source);retained=output/'snapshot'/sha;target=context/destination
        if not retained.is_file():raise ValueError('source changed after capture')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(retained,target)
        if digest(target)!=sha:raise ValueError('materialized context changed')
        target.chmod(source.stat().st_mode & 0o777)
        records.append(dict(path=destination,sha256=sha,bytes=target.stat().st_size))
    (context/'bin').mkdir()
    for command in ('zstd','time'):
        target=context/'bin'/command
        target.write_text('#!/bin/sh\nexec /opt/audit/native/ld-linux-x86-64.so.2 --argv0 '+command+' --library-path /opt/audit/native /opt/audit/native/'+command+' "$@"\n')
        target.chmod(0o755)
    (context/'Dockerfile').write_text(f'''FROM {base}
COPY . /opt/audit/
ENV PYTHONPATH=/opt/audit/python PATH=/opt/audit/bin:$PATH PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["nice", "-n", "19", "ionice", "-c", "3", "time", "-v", "python3", "/opt/audit/publication_routing_cells.py"]
''')
    for path in [context/'Dockerfile',*sorted((context/'bin').iterdir())]:
        records.append(dict(path=str(path.relative_to(context)),sha256=digest(path),bytes=path.stat().st_size))
    manifest=dict(base=base,python_abi='3.12',pillow=PIL.__version__,files=records,
        packager_sha256=digest(Path(__file__)),auditor_sha256=digest(sources['publication_routing_cells.py']),
        note='Build offline from the pinned preinstalled base; record the resulting image ID. Native tools use the captured loader/libraries without changing Python library lookup.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(files=len(records),context=str(context),auditor_sha256=manifest['auditor_sha256'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--base',required=True)
    args=parser.parse_args();prepare(args.output,args.base)
