#!/usr/bin/env python3
"""Package selected QAT/control models and their reproduction metadata, never pixels."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import subprocess
import tarfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pair_release import make_archive


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while b:=stream.read(1024*1024):h.update(b)
    return h.hexdigest()


def verify_archive(path, members):
    """Verify every regular nested member against the exact source inventory."""
    process = subprocess.Popen(['zstd', '-dc', str(path)], stdout=subprocess.PIPE)
    found = set()
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            for member in archive:
                name = Path(member.name)
                if (member.name not in members or member.name in found or
                        not member.isfile() or name.is_absolute() or '..' in name.parts):
                    raise ValueError('unexpected or unsafe archive member')
                found.add(member.name)
                digest = hashlib.sha256()
                with archive.extractfile(member) as source:
                    while block := source.read(1024 * 1024):
                        digest.update(block)
                if digest.hexdigest() != members[member.name]:
                    raise ValueError('archive member hash mismatch')
        if process.wait() != 0 or found != set(members):
            raise ValueError('incomplete archive')
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            process.wait()


def main(a):
    if not (a.root/'primary/COMPLETE').is_file() or not (a.root/'final-evaluation/COMPLETE').is_file():
        raise ValueError('complete training and all controls before packaging')
    if sha(a.baseline)!='c3db9373791270bbe5493d3ee98cd4f62296510fe6931073e2966f433e9a5cf4':
        raise ValueError('wrong baseline')
    a.output.mkdir();payload=a.output/'payload';payload.mkdir()
    catalog=[];included_checkpoints=set()
    originals=['baseline','ptq8','ptq8-gpu']+[f'{kind}-s{seed}-selected' for seed in [7,17,29] for kind in ['float','qat8']]
    sources=[(name,a.root/'primary'/name) for name in originals]
    sources += [(f'float-ptq-s{seed}',a.root/'float-ptq-controls'/f'float-ptq-s{seed}') for seed in [7,17,29]]
    for name,source in sources:
        meta=json.loads((source/'meta.json').read_text())
        if sha(source/'weights.raw')!=meta['weights_sha256']:raise ValueError('changed model export')
        target=payload/'models'/name;target.mkdir(parents=True)
        for filename in ['meta.json','weights.raw','gold_in_17x13.raw','gold_out_17x13.raw']:
            shutil.copyfile(source/filename,target/filename)
        packed=source/'weights_q8pc.bin'
        if packed.exists():
            if sha(packed)!=meta['packed_weights_sha256']:raise ValueError('changed packed model')
            shutil.copyfile(packed,target/packed.name)
        checkpoint=None
        if name.endswith('-selected'):
            checkpoint=a.root/'primary'/name.removesuffix('-selected')/f"checkpoint-{meta['selected_step']:06d}.pth"
            destination=payload/'selected-checkpoints'/f'{name}.pth';destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(checkpoint,destination);included_checkpoints.add(checkpoint)
        catalog.append(dict(name=name,metadata=meta,files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(target.iterdir())},
                            selected_shadow_checkpoint=str(destination.relative_to(payload)) if checkpoint else None))
    shutil.copyfile(a.baseline,payload/'baseline-shipped-f16.raw')
    for name,source in [('TRAINING-PLAN.json',a.root/'primary/PLAN.json'),('SELECTED.json',a.root/'primary/selected.json'),
                        ('STORAGE.json',a.root/'storage.json'),('QUALITY-SUMMARY.json',a.root/'final-summary.json'),
                        ('BACKEND-PARITY.json',a.root/'backend-check.json'),('FLOAT-POSTQUANT-PLAN.json',a.root/'float-ptq-controls/CONTROL_PLAN.json')]:
        shutil.copyfile(source,payload/name)
    for name in ['LICENSE-AGPL3','LICENSE-COMMERCIAL']:
        shutil.copyfile(Path(name),payload/name)
    not_uploaded=[]
    for path in sorted((a.root/'primary').glob('*/checkpoint-*.pth')):
        if path not in included_checkpoints:
            not_uploaded.append(dict(relative_path=str(path.relative_to(a.root)),bytes=path.stat().st_size,sha256=sha(path)))
    inventory=dict(models=catalog,not_uploaded=dict(unselected_optimizer_checkpoints=not_uploaded,
        metric_artifacts='Exact RGB8 outputs, JPEG/reference inputs and per-pixel Butteraugli maps stay in the experiment archive, outside git and this model-only package.'),
        quantized_format='Q8PC v1: per-output-channel signed int8 conv weights; f32 scales/biases/PReLU; f32 execution',
        status='research candidates; current f16 serving default is unchanged')
    (a.output/'CANON4-QAT-20261108.json').write_text(json.dumps(inventory,indent=2)+'\n')
    shutil.copyfile(a.output/'CANON4-QAT-20261108.json',payload/'MODEL-CATALOG.json')
    shutil.copyfile(a.readme,payload/'README.md');shutil.copyfile(a.readme,a.output/'CANON4-QAT-README.md')
    names=[str(p.relative_to(payload)) for p in sorted(payload.rglob('*')) if p.is_file()]
    artifact=make_archive(payload,a.output/'canon4-qat-w8a32-20261108.tar.zst',names)
    verify_archive(a.output/artifact['name'],artifact['members'])
    (a.output/'archive-verification.json').write_text(json.dumps(dict(archive=artifact['name'],bytes=artifact['bytes'],sha256=artifact['sha256'],verified_members=len(names)),indent=2)+'\n')
    files=['canon4-qat-w8a32-20261108.tar.zst','CANON4-QAT-20261108.json','CANON4-QAT-README.md']
    (a.output/'CANON4-QAT-SHA256SUMS').write_text(''.join(sha(a.output/name)+'  '+name+'\n' for name in files))
    print(json.dumps(dict(models=len(catalog),archive_bytes=artifact['bytes'],verified_members=len(names),unselected_checkpoints_not_uploaded=len(not_uploaded)),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','baseline','output','readme']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
