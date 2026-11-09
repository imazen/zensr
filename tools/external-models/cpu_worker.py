#!/usr/bin/env python3
"""Synchronous CPU-only SR timing worker for zenbench's external protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from sr_models import load


def main(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_available():
        raise ValueError('CPU benchmark requires all GPUs hidden')
    if not 1<=a.threads<=8 or not 1<=a.samples<=10:
        raise ValueError('invalid explicit work budget')
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=True
    model=load(a.root,a.name,device='cpu')
    modules=[model.model,model.de] if a.name=='s3diff' else [model.model,model.autoencoder]
    if any(p.device.type!='cpu' or p.dtype!=torch.float32 for m in modules for p in m.parameters()):
        raise ValueError('CPU FP32 model parameter contract differs')
    data=np.fromfile(a.input,dtype='<f4')
    if data.size!=3*a.height*a.width or not np.isfinite(data).all():
        raise ValueError('input geometry/values differ')
    image=torch.from_numpy(data.reshape(1,3,a.height,a.width))
    def check(output):
        if output.device.type!='cpu' or list(output.shape)!=[1,3,a.height*4,a.width*4] or not torch.isfinite(output).all():
            raise ValueError('invalid CPU output')
        return hashlib.sha256(output.detach().contiguous().numpy().tobytes()).hexdigest()
    torch.manual_seed(20261109)
    first=model(image);digest=check(first)
    torch.manual_seed(20261109)
    second=model(image)
    if not torch.equal(first,second):raise ValueError('repeated seeded CPU outputs differ')
    del first,second
    print('IDENTITY '+json.dumps(dict(name=a.name,device='cpu',precision='float32',threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),mkldnn=torch.backends.mkldnn.enabled,
        parameters=model.parameters,input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
        output_sha256=digest,shape=[1,3,a.height*4,a.width*4],torch=torch.__version__)),flush=True)
    if a.validate_only:return
    for _ in range(a.samples):
        torch.manual_seed(20261109)
        started=time.perf_counter_ns()
        output=model(image)
        elapsed=time.perf_counter_ns()-started
        if check(output)!=digest:raise ValueError('timed output differs from seeded CPU control')
        print(f'latency_ns={elapsed}',flush=True)
        del output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',choices=['s3diff','resshift-v3','resshift-bicubic'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--height',type=int,required=True);p.add_argument('--width',type=int,required=True)
    p.add_argument('--threads',type=int,required=True);p.add_argument('--samples',type=int,default=3)
    p.add_argument('--validate-only',action='store_true')
    main(p.parse_args())
