#!/usr/bin/env python3
"""Fixed public ×4 SR comparison, with GPU inference and parallel scalar scorers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import threading
import sys

import numpy as np
import torch
from sr_models import load


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    part=path.with_suffix(f'.partial-{os.getpid()}-{threading.get_ident()}')
    with part.open('x') as f:
        json.dump(value,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    part.replace(path)


class Scorer:
    def __init__(self,a,index):
        self.log=(a.output/f'scorer-{index}.log').open('a')
        env=dict(os.environ,RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        self.process=subprocess.Popen([str(a.scorer),'score',str(a.native_weights)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,env=env)
        if self.receive()!=dict(ready=True,span_sha256=sha(a.native_weights)):raise ValueError('native model differs')

    def receive(self):
        line=self.process.stdout.readline()
        if not line:raise RuntimeError(f'scorer {self.process.pid} exited; inspect its log')
        return json.loads(line)

    def send(self,value,payload=None):
        self.process.stdin.write(json.dumps(value).encode()+b'\n')
        if payload is not None:self.process.stdin.write(payload)
        self.process.stdin.flush()
        return self.receive()

    def prepare(self,row):
        header=self.send(dict(action='prepare',row=row))
        n=header['input_bytes']
        if n!=12*row['low_width']*row['low_height'] or header['id']!=row['id']:raise ValueError('input dimensions differ')
        payload=bytearray(n);view=memoryview(payload)
        while view:
            count=self.process.stdout.readinto(view)
            if not count:raise RuntimeError('incomplete input pipe')
            view=view[count:]
        return header,np.frombuffer(payload,dtype='<f4').reshape(1,3,row['low_height'],row['low_width']).copy()

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write(b'{"action":"quit"}\n');self.process.stdin.flush();self.process.stdin.close()
        status=self.process.wait();self.log.close()
        if status:raise RuntimeError(f'scorer exit {status}')


def validate(cell,row):
    if cell['row']!=row:raise ValueError('resumed row differs')
    for name in ['baseline','native_span','candidate']:
        score=cell[name]
        for key in ['ssim2','butter_max','butter_pnorm3']:
            if not np.isfinite(float(score[key])):raise ValueError('nonfinite score')
        if len(score['rgb8_sha256'])!=64:raise ValueError('invalid output hash')


def main(a):
    if not 1<=a.jobs<=6:raise ValueError('invalid worker budget')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    rows=[json.loads(line) for line in a.inputs.read_text().splitlines()]
    if len(rows)!=697 or len({r['id'] for r in rows})!=697:raise ValueError('fixed SR coverage differs')
    if a.pilot:rows=[r for r in rows if r['source_id']=='1017' and r['q'] in (0,10,50,90)]
    if a.pilot and len(rows)!=7:raise ValueError('pilot controls missing')
    a.output.mkdir(exist_ok=True);(a.output/'cells').mkdir(exist_ok=True)
    identity=dict(schema='zensr-external-sr-v1',inputs_sha256=sha(a.inputs),scorer_sha256=sha(a.scorer),native_weights_sha256=sha(a.native_weights),
        acquisition_sha256=sha(a.model_root/'ACQUISITION.json'),script_sha256=sha(Path(__file__)),adapter_sha256=sha(Path(__file__).with_name('sr_models.py')),
        python=sys.version,packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),device=torch.cuda.get_device_name(0),torch_native_sha256=sha(Path(torch._C.__file__)),
        jobs=a.jobs,deterministic=True,tf32=False,cublas=os.environ['CUBLAS_WORKSPACE_CONFIG'],name=a.name,pilot=a.pilot,
        source_files={str(p.relative_to(a.model_root/'source')):sha(p) for p in sorted((a.model_root/'source').rglob('*')) if p.is_file() and p.suffix in ('.py','.yaml','.yml')},
        seeds='input id prefix xor 20261109 mod 2^63; reset each input',input_ids=[r['id'] for r in rows],
        precision='S3Diff f32; ResShift f32 weights with author fp16 autocast',retention='scalar scores, hashes and encoded inputs; no restored RGB or diagnostic maps')
    path=a.output/'IDENTITY.json'
    if path.exists() and json.loads(path.read_text())!=identity:raise ValueError('resumed identity differs')
    if not path.exists():save(path,identity)
    completed=0;pending=[]
    for row in rows:
        path=a.output/'cells'/(row['id']+'.json')
        if path.exists():validate(json.loads(path.read_text()),row);completed+=1
        else:pending.append(row)
    progress_lock=threading.Lock()
    scorers=[]
    def progress(stage,error=None):
        save(a.output/'PROGRESS.json',dict(stage=stage,completed=completed,required=len(rows),pid=os.getpid(),
            scorer_pids=[s.process.pid for s in scorers],checked=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=error))
    progress('loading')
    model=load(a.model_root,a.name)
    save(a.output/'MODEL.json',dict(parameters=model.parameters))
    gpu_lock=threading.Lock();halt=threading.Event();iterator=iter(pending)
    try:
        scorers=[Scorer(a,i) for i in range(a.jobs)]
        def worker(index):
            nonlocal completed
            try:
                while not halt.is_set():
                    with progress_lock:row=next(iterator,None)
                    if row is None:return
                    header,low=scorers[index].prepare(row)
                    seed=(int(row['id'][:16],16)^20261109)&((1<<63)-1)
                    with gpu_lock:
                        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                        prediction=model(torch.from_numpy(low).cuda()).cpu().float().contiguous()
                    if list(prediction.shape)!=[1,3,row['height'],row['width']]:raise ValueError('candidate output geometry differs')
                    payload=prediction.numpy().astype('<f4',copy=False).tobytes()
                    result=scorers[index].send(dict(action='score',id=row['id'],bytes=len(payload)),payload)
                    if result['id']!=row['id']:raise ValueError('candidate output id differs')
                    cell=dict(row=row,baseline=header['baseline'],native_span=header['native_span'],candidate=result['score'],seed=seed)
                    validate(cell,row);save(a.output/'cells'/(row['id']+'.json'),cell)
                    with progress_lock:
                        completed+=1
                        if completed%16==0 or completed==len(rows):progress('evaluating');print(f'completed {completed}/{len(rows)}',flush=True)
            except BaseException:
                halt.set();raise
        progress('evaluating')
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for future in as_completed([pool.submit(worker,i) for i in range(a.jobs)]):future.result()
        if completed!=len(rows):raise ValueError('incomplete SR comparison')
        save(a.output/'COMPLETE.json',dict(inputs=completed,identity_sha256=sha(a.output/'IDENTITY.json')))
        progress('complete')
    except BaseException as error:
        progress('failed',repr(error));raise
    finally:
        for scorer in scorers:scorer.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['inputs','scorer','native-weights','model-root','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--name',choices=['s3diff','resshift-v3','resshift-bicubic'],required=True)
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--pilot',action='store_true')
    main(p.parse_args())
