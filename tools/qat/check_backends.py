#!/usr/bin/env python3
"""Explicit GPU diagnostic: CPU PTQ and GPU QAT initial packed operands."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
import canon4 as q


def main(a):
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise ValueError('CUDA required for this explicit diagnostic')
    torch.cuda.set_per_process_memory_fraction(.1)
    baseline=q.load_baseline(a.baseline)
    results=[]
    for name,layer in baseline.named_modules():
        if not isinstance(layer,nn.Conv2d):continue
        cpu,scpu=q.quantized(layer.weight)
        gpu,sgpu=q.quantized(layer.weight.cuda())
        gpu=gpu.cpu();sgpu=sgpu.cpu()
        codes_cpu=torch.round(cpu/scpu[:,None,None,None]).to(torch.int8)
        codes_gpu=torch.round(gpu/sgpu[:,None,None,None]).to(torch.int8)
        results.append(dict(layer=name,scale_bit_differences=int((scpu.view(torch.int32)!=sgpu.view(torch.int32)).sum()),
            integer_code_differences=int((codes_cpu!=codes_gpu).sum()),
            dequantized_bit_differences=int((cpu.view(torch.int32)!=gpu.view(torch.int32)).sum()),
            max_absolute_operand_difference=float((cpu-gpu).abs().max().detach())))
    cpu=q.pack(baseline);gpu=q.pack(baseline.cuda())
    result=dict(baseline_sha256=q.BASELINE_SHA,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        layers=results,packed_cpu_sha256=hashlib.sha256(cpu).hexdigest(),
        packed_gpu_sha256=hashlib.sha256(gpu).hexdigest(),packed_identical=cpu==gpu)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    if a.export is not None:
        q.export(baseline,a.export,True,dict(role='untrained_gpu_int8_control',baseline_sha256=q.BASELINE_SHA))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('baseline',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--export',type=Path)
    main(p.parse_args())
