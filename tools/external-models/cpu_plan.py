#!/usr/bin/env python3
"""Freeze codec-free CPU SR inputs, assets and a zenbench external-process plan."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def preserve(path,data):
    if path.exists():
        if path.read_bytes()!=data:raise ValueError(f'existing frozen bytes differ: {path}')
    else:
        with path.open('xb') as f:f.write(data)


def main(a):
    a.root.mkdir(exist_ok=True)
    source=Path(__file__).resolve().parent
    reference=a.external/'sr-inputs-v1/7109-reference.rgb'
    low=a.external/'sr-inputs-v1/7109-0-clean.rgb'
    if reference.stat().st_size!=3*512*512 or low.stat().st_size!=3*128*128:
        raise ValueError('frozen source dimensions differ')
    for n in ([256] if a.matrix=='native-large' else [64,128]):
        pixels=reference.read_bytes() if n==256 else low.read_bytes()
        sw=512 if n==256 else 128
        start=0 if n==256 else (128-n)//2
        step=2 if n==256 else 1
        values=[pixels[((y*step+start)*sw+x*step+start)*3+c]/255 for c in range(3) for y in range(n) for x in range(n)]
        data=b''.join(struct.pack('<f',v) for v in values)
        preserve(a.root/f'input-{n}.f32',data)
    assets=[a.worker,a.python,a.python.parent.parent/'pyvenv.cfg',source/'cpu_worker.py',source/'sr_models.py',source/'cpu_plan.py',reference,low]
    assets.extend(sorted((a.root/'models').rglob('*.json')))
    assets.extend(sorted((a.root/'models').rglob('*.raw')))
    assets.extend(sorted((a.root/'models').glob('*.onnx')))
    if a.matrix=='full':
        for model in ['s3diff','resshift']:
            assets.extend(p for p in sorted((a.external/model/'source').rglob('*')) if p.is_file() and p.suffix in ('.py','.yaml','.yml'))
            assets.append(a.external/model/'ACQUISITION.json')
        for folder in ['s3diff/sd-turbo','s3diff/weights','resshift']:
            assets.extend(p for p in sorted((a.external/folder).glob('**/*')) if p.is_file() and p.suffix in ('.pth','.pkl','.safetensors'))
    assets=list(dict.fromkeys(assets))
    identity={str(p):sha(p) for p in assets}
    preserve(a.root/f'assets-{a.output.stem}.json',(json.dumps(identity,sort_keys=True,indent=2)+'\n').encode())
    env={'RAYON_NUM_THREADS':'${threads}','OMP_NUM_THREADS':'${threads}',
        'OPENBLAS_NUM_THREADS':'${threads}','MKL_NUM_THREADS':'${threads}',
        'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'TMPDIR':str(Path.home()/'tmp/zensr-strata'),'ZENSR_WINOGRAD':'0'}
    models=[]
    def native(name,kind,path):
        models.append(dict(id=name,program='/usr/bin/time',args=['-v',str(a.worker),kind,str(path),'${input}','${n}','${n}','${threads}','3'],
            cwd=str(source.parents[1]),env=env,revision='worker-sha256:'+sha(a.worker),timeout_seconds=1200))
    native('native-span','adopted',a.root/'models/adopted/nomosuni_span_4x')
    native('native-spanf','spanf',a.root/'models/spanf_weights_f16.raw')
    for name in ['general_x4v3','general_wdn_x4v3','animevideo_x4v3']:
        native(name,'adopted',a.root/'models/adopted'/name)
    for name in ['EFDN_baseline','NanoSR','TSR','SPANF']:
        native(name+'-tract','tract',a.root/('models/'+name+'_x4_${n}x${n}.onnx'))
    if a.matrix=='full':
        for name in ['s3diff','resshift-v3','resshift-bicubic']:
            # Keep the venv interpreter path as argv, since the external runner
            # canonicalizes program symlinks. time also retains per-child RSS.
            models.append(dict(id=name,program='/usr/bin/time',args=['-v',str(a.python),str(source/'cpu_worker.py'),
                '--name',name,'--root',str(a.external/('s3diff' if name=='s3diff' else 'resshift')),
                '--input','${input}','--height','${n}','--width','${n}','--threads','${threads}','--samples','3'],
                cwd=str(source.parents[1]),env=env,revision='adapter-sha256:'+sha(source/'sr_models.py'),timeout_seconds=1200))
    cases=[]
    for n in ([256] if a.matrix=='native-large' else [64,128]):
        for threads in [1,4]:
            cases.append(dict(id=f'{n}-t{threads}',fields=dict(n=str(n),threads=str(threads)),
                inputs=dict(input=str(a.root/f'input-{n}.f32'),
                    **{f'asset_{i}':str(p) for i,p in enumerate(assets)})))
    if a.pilot:cases=[c for c in cases if c['id']=='64-t4']
    if not cases:raise ValueError('empty case selection')
    plan=dict(schema=1,baseline=models[0],candidates=models[1:],cases=cases,
        metric=dict(extract=dict(kind='prefix',prefix='latency_ns='),unit='ns',expected=3,divide_by=1),
        check=dict(kind='json',prefix='IDENTITY ',pointers=['/input_sha256','/shape']),
        warmup='Two seeded full inference/geometry/determinism checks per process; native tiled thread-parity also checked. Timed SR calls include output/scratch allocation and model-specific resize/conditioning/color fix; input reads, model loading/packing/tract planning, RNG reseeding, output hashing/destruction, codecs and IQA excluded.',
        blocks=1 if a.pilot else 3,aa=True,deadline_seconds=7200,settle_ms=0,seed=20261109)
    preserve(a.output,(json.dumps(plan,sort_keys=True,indent=2)+'\n').encode())
    print(json.dumps(dict(plan=str(a.output),models=len(models),cases=len(cases),blocks=plan['blocks'],invocations=len(cases)*plan['blocks']*(len(models)+1))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','external','worker','python','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--matrix',choices=['full','native-large'],default='full')
    p.add_argument('--pilot',action='store_true')
    main(p.parse_args())
