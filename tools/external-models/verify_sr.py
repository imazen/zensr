#!/usr/bin/env python3
"""Compare SR tensor adapters with unchanged author inference code."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from sr_models import load


def main(a):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    model=load(a.root,a.name)
    source=a.root/('source/src/inference_s3diff.py' if a.name=='s3diff' else 'source/sampler.py')
    tree=ast.parse(source.read_text())
    if a.name!='s3diff':
        author=next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='ResShiftSampler')
        method=next(m for m in author.body if isinstance(m,ast.FunctionDef) and m.name=='sample_func')
        scope=dict(torch=torch,F=F,math=math)
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),scope)
        context=SimpleNamespace(configs=model.config,model=model.model,base_diffusion=model.diffusion,
            autoencoder=model.autoencoder,padding_offset=64,sf=4)
    records=[]
    for h,w in [(128,72),(65,71)]:
        low=torch.linspace(0,1,3*h*w,device='cuda').reshape(1,3,h,w)
        torch.manual_seed(20261109)
        if a.name=='s3diff':
            from PIL import Image
            from torchvision import transforms
            from utils.wavelet_color import wavelet_color_fix, adain_color_fix
            main_node=next(c for c in tree.body if isinstance(c,ast.FunctionDef) and c.name=='main')
            loop=next(c for c in main_node.body if isinstance(c,ast.For))
            captured=[]
            scope=dict(torch=torch,F=F,math=math,os=os,transforms=transforms,config=SimpleNamespace(sf=4),
                dl_val=[{'lr_path':['input.png'],'lr':low.cpu()}],net_sr=model.model,net_de=model.de,
                accelerator=SimpleNamespace(unwrap_model=lambda m:m),
                wavelet_color_fix=wavelet_color_fix,adain_color_fix=adain_color_fix,
                args=SimpleNamespace(pos_prompt='A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.',
                    neg_prompt='oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth',
                    align_method='wavelet',output_dir='memory-only'))
            # Execute the unchanged author loop, capturing its final PIL save in RAM.
            with torch.inference_mode(),patch.object(Image.Image,'save',lambda image,*args,**kwargs:captured.append(image.copy())):
                exec(compile(ast.Module(body=[loop],type_ignores=[]),str(source),'exec'),scope)
            if len(captured)!=1:raise ValueError('author output count differs')
            expected=transforms.ToTensor()(captured[0]).unsqueeze(0)
        else:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                expected=(scope['sample_func'](context,low*2-1,mask=None)*.5+.5).float()
        torch.manual_seed(20261109)
        actual=model(low).cpu()
        expected=expected.cpu()
        if not torch.equal(actual,expected):raise ValueError(f'author sampling parity differs at {h}x{w}')
        records.append(dict(height=h,width=w,equal=True,output_sha256=hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest()))
    a.output.write_text(json.dumps(dict(name=a.name,author_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),records=records),sort_keys=True)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--name',choices=['s3diff','resshift-v3','resshift-bicubic'],required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
