#!/usr/bin/env python3
"""Quantize the already-selected float controls without training or re-selection."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import canon4 as q


def main(a):
    if not (a.training/'COMPLETE').is_file():raise ValueError('training must be complete')
    if not torch.cuda.is_available():raise ValueError('matching GPU conversion required')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.1)
    a.output.mkdir()
    # Keep the original training and baseline immutable while constructing controls.
    (a.output/'baseline').symlink_to((a.training/'baseline').resolve(),target_is_directory=True)
    shutil.copyfile(a.training/'PLAN.json',a.output/'PLAN.json')
    records=[]
    for seed in [7,17,29]:
        source=a.training/f'float-s{seed}-selected'
        meta=json.loads((source/'meta.json').read_text())
        if q.sha(source/'weights.raw')!=meta['weights_sha256']:raise ValueError('changed float control')
        model=q.load_flat(np.fromfile(source/'weights.raw',dtype='<f4')).cuda()
        record=q.export(model,a.output/f'float-ptq-s{seed}',True,dict(role='selected_float_then_gpu_ptq',seed=seed,
            selected_step=meta['selected_step'],source_weights_sha256=meta['weights_sha256'],
            checkpoint_selection='unchanged existing float-forward validation selection; no re-selection'))
        records.append(record)
    (a.output/'CONTROL_PLAN.json').write_text(json.dumps(dict(method='Deterministic INT8 conversion of each already-selected float control. No optimizer steps or checkpoint re-selection.',
        torch=torch.__version__,quantizer_device=torch.cuda.get_device_name(),models=records),indent=2)+'\n')
    (a.output/'COMPLETE').write_text(q.sha(a.output/'PLAN.json')+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('training',type=Path);p.add_argument('output',type=Path)
    main(p.parse_args())
