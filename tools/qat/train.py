#!/usr/bin/env python3
"""Paired canon4 float-control / W8A32 QAT fine-tuning with immutable inputs.

Run under run-heavy. Checkpoints, optimizer state, losses, selected exports and
PID-backed progress stay outside git. No test images participate in selection.
"""
import argparse
import copy
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

import canon4 as q


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.new')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def validation(model, lr, hr, ntrain, batch):
    total = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(ntrain, len(lr), batch):
            x = lr[start:start+batch].permute(0,3,1,2).float().div(255)
            y = hr[start:start+batch].permute(0,3,1,2).float().div(255)
            value = torch.sqrt((model(x)-y).square()+1e-6).mean()
            if not torch.isfinite(value): raise ValueError('non-finite validation loss')
            total += value.item() * len(x)
    model.train()
    return total / (len(lr)-ntrain)


def main(args):
    if args.steps < 1 or args.batch < 1 or args.lr <= 0:
        raise ValueError('invalid training configuration')
    if len(set(args.seeds)) != len(args.seeds): raise ValueError('duplicate seeds')
    if args.device == 'cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device == 'cuda': torch.cuda.set_per_process_memory_fraction(0.70)
    baseline = q.load_baseline(args.baseline)
    rows, ntrain = q.checked_pairs(args.dataset, args.split_map)
    args.output.mkdir()
    source_paths = [Path(__file__),Path(q.__file__),Path(q.original.__file__)]
    inputs = {name:q.sha(args.dataset/name) for name in ('meta.json','pairs.tsv','lr_u8.npy','hr_u8.npy')}
    corpus_revision = subprocess.check_output(['git','-C',str(args.split_map.parent.parent),'rev-parse','HEAD'],text=True).strip()
    if corpus_revision != '187fbf338ce08e8e6654db7f04ddae58d5263da2':
        raise ValueError('unrecognized published split revision')
    plan = dict(format='canon4-w8a32-qat-v1', build_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={p.name:q.sha(p) for p in source_paths}, dataset_sha256=inputs,
        baseline_sha256=q.BASELINE_SHA, published_split_sha256=q.sha(args.split_map),
        published_split_revision=corpus_revision, train_rows=ntrain, validation_rows=len(rows)-ntrain,
        seeds=args.seeds, arms=args.arms, steps=args.steps, batch=args.batch, lr=args.lr,
        warmup_steps=min(250,args.steps//10), qboost=3, loss='Charbonnier(GT)+0.25*Charbonnier(frozen-f16-canon4)',
        checkpoint_selection='minimum internal-validation GT Charbonnier, checked every 500 steps and at final step',
        quantization='conv weights symmetric int8 per output channel; biases and PReLU f32; activations f32',
        scale_policy='dynamic detached per-output-channel absmax/127 on every forward; export uses identical native rounding',
        inherited_baseline_split_exposure=True, no_new_imazen26_heldout_training=True,
        numeric=dict(torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),
            cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),tf32=False,amp=False,
            deterministic=True,threads=args.threads,device=args.device,
            device_name=torch.cuda.get_device_name() if args.device=='cuda' else platform.processor()))
    write_json(args.output/'PLAN.json',plan)
    plan_sha=q.sha(args.output/'PLAN.json')
    progress=dict(pid=os.getpid(),host=platform.node(),stage='loading_data',required_runs=len(args.seeds)*len(args.arms),
        completed_runs=0,required_steps=args.steps,step=0,run_directory=str(args.output),plan_sha256=plan_sha)
    def update(**state):
        progress.update(state);progress['checked_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(args.output/'progress.json',progress)
    update()
    # Keep uint8 pairs on-device; only the current batch is expanded to f32.
    lr=torch.from_numpy(np.load(args.dataset/'lr_u8.npy',mmap_mode='r').copy()).to(args.device)
    hr=torch.from_numpy(np.load(args.dataset/'hr_u8.npy',mmap_mode='r').copy()).to(args.device)
    pool=list(range(ntrain))
    for i,r in enumerate(rows[:ntrain]):
        if int(r['clean']) or int(r['q']) >= 85: pool.extend([i]*3)
    pool=torch.tensor(pool,dtype=torch.long,device=args.device)
    teacher=copy.deepcopy(baseline).to(args.device).eval().requires_grad_(False)
    q.export(baseline,args.output/'baseline',False,dict(plan_sha256=plan_sha,role='deployed_f16_expanded_to_f32'))
    q.export(baseline,args.output/'ptq8',True,dict(plan_sha256=plan_sha,role='untrained_int8_control'))
    results=[]
    started=time.monotonic()
    try:
        for seed in args.seeds:
            for arm in args.arms:
                torch.manual_seed(seed)
                rng=np.random.default_rng(seed+1000)
                m=copy.deepcopy(baseline).to(args.device)
                if arm=='qat8': q.enable_qat(m)
                opt=torch.optim.AdamW(m.parameters(),lr=args.lr,weight_decay=0)
                best=float('inf');best_step=0;best_sd=None
                run=args.output/f'{arm}-s{seed}';run.mkdir()
                with (run/'loss.tsv').open('x',buffering=1) as log:
                    log.write('step\tgt_charbonnier\tteacher_charbonnier\tvalidation_gt_charbonnier\tlr\n')
                    for step in range(1,args.steps+1):
                        warm=plan['warmup_steps']
                        fraction=(step-warm)/max(1,args.steps-warm)
                        rate=args.lr*(step/max(1,warm) if step<=warm else 0.02+0.98*0.5*(1+math.cos(math.pi*fraction)))
                        for group in opt.param_groups: group['lr']=rate
                        indices=pool[torch.from_numpy(rng.integers(0,len(pool),args.batch)).to(args.device)]
                        x=lr[indices].permute(0,3,1,2).float().div(255)
                        y=hr[indices].permute(0,3,1,2).float().div(255)
                        with torch.no_grad(): t=teacher(x)
                        prediction=m(x)
                        gt=torch.sqrt((prediction-y).square()+1e-6).mean()
                        kd=torch.sqrt((prediction-t).square()+1e-6).mean()
                        loss=gt+0.25*kd
                        if not torch.isfinite(loss): raise ValueError('non-finite training loss')
                        opt.zero_grad(set_to_none=True);loss.backward()
                        nn.utils.clip_grad_norm_(m.parameters(),1.0,error_if_nonfinite=True)
                        opt.step()
                        if step%100==0 or step==args.steps:
                            update(stage='training',arm=arm,seed=seed,step=step,elapsed_seconds=time.monotonic()-started)
                        if step%500==0 or step==args.steps:
                            val=validation(m,lr,hr,ntrain,args.batch)
                            log.write(f'{step}\t{gt.item():.12g}\t{kd.item():.12g}\t{val:.12g}\t{rate:.12g}\n')
                            if val<best:
                                best=val;best_step=step
                                best_sd=copy.deepcopy(m.state_dict())
                            checkpoint=run/f'checkpoint-{step:06d}.pth'
                            torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),numpy_rng=rng.bit_generator.state,
                                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if args.device=='cuda' else None,
                                step=step,best_step=best_step,best_validation=best,best_model=best_sd,plan_sha256=plan_sha),checkpoint)
                            update(checkpoint=str(checkpoint),best_step=best_step,best_validation=best)
                            print(f'{arm} seed={seed} step={step}/{args.steps} validation_charbonnier={val:.8f} best_step={best_step}',flush=True)
                    m.load_state_dict(best_sd)
                    metadata=q.export(m,args.output/f'{arm}-s{seed}-selected',arm=='qat8',
                        dict(plan_sha256=plan_sha,seed=seed,role=arm,selected_step=best_step,validation_charbonnier=best))
                    results.append(metadata)
                    write_json(args.output/'selected.json',results)
                    update(completed_runs=len(results))
                del m,opt,best_sd
                if args.device=='cuda': torch.cuda.empty_cache()
        update(stage='complete',step=args.steps)
        (args.output/'COMPLETE').write_text(plan_sha+'\n')
    except BaseException as error:
        update(stage='failed',failure=repr(error));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--split-map',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--seeds',type=int,nargs='+',default=[7,17,29])
    p.add_argument('--arms',nargs='+',choices=['float','qat8'],default=['float','qat8'])
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--threads',type=int,default=2)
    main(p.parse_args())
