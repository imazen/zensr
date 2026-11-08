#!/usr/bin/env python3
"""Summarize frozen paired canon4 QAT scores without selecting on test results."""
import argparse
import collections
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()


def limit(identity):
    return Decimal('.5')-Decimal('.04')*min(Decimal(10),max(Decimal(0),identity-Decimal(90)))


def balanced(values):
    groups=collections.defaultdict(list)
    for origin,value in values:groups[origin].append(value)
    return {k:statistics.fmean(v) for k,v in groups.items()}


def interval(means):
    a=np.array(list(means.values()),dtype=np.float64)
    rng=np.random.default_rng(20261108)
    bootstrap=np.mean(a[rng.integers(0,len(a),(10000,len(a)))],axis=1)
    return [float(v) for v in np.quantile(bootstrap,[.025,.975])]


def summarize(rows,mode):
    score=mode+'_ssim2'
    baseline={r['id']:r for r in rows if r['arm']=='baseline'}
    arms=collections.defaultdict(list)
    for r in rows:arms[r['arm']].append(r)
    expected=set(baseline)
    if not expected:raise ValueError('missing baseline')
    result={}
    for arm,values in sorted(arms.items()):
        if len(values)!=len(expected) or {r['id'] for r in values}!=expected:
            raise ValueError('incomplete/duplicate paired arm coverage')
        gains=[];deltas=[];harm=[];added_harm=[];worse01=[];worst=[]
        butter=[];coverage=[]
        for r in values:
            base=baseline[r['id']]
            for key in ['origin','identity_ssim2','default_restore']:
                if r[key]!=base[key]:raise ValueError('paired control differs')
            identity=Decimal(r['identity_ssim2']);actual=Decimal(r[score]);original=Decimal(base[score])
            if not all(v.is_finite() for v in [identity,actual,original]):raise ValueError('non-finite IQA')
            gain=actual-identity;delta=actual-original
            key=r['origin']
            gains.append((key,float(gain)));deltas.append((key,float(delta)))
            harmed=gain < -limit(identity);original_harmed=original-identity < -limit(identity)
            harm.append((key,float(harmed)));added_harm.append((key,float(harmed and not original_harmed)))
            worse01.append((key,float(delta < Decimal('-.1'))))
            coverage.append((key,float(r['default_restore'])))
            value=float(r['forced_butter_pnorm3']) if mode=='forced' or r['default_restore']=='1' else float(r['identity_butter_pnorm3'])
            old=float(base['forced_butter_pnorm3']) if mode=='forced' or base['default_restore']=='1' else float(base['identity_butter_pnorm3'])
            butter.append((key,value-old))
            worst.append((float(delta),r['id'],r['source_id'],r['encoder'],r['ss'],r['q']))
        delta_origins=balanced(deltas)
        result[arm]=dict(cells=len(values),origins=len(delta_origins),
            mean_gain_ssim2=statistics.fmean(balanced(gains).values()),
            mean_vs_baseline_ssim2=statistics.fmean(delta_origins.values()),
            paired_origin_bootstrap_95_vs_baseline=interval(delta_origins),
            harm_fraction=statistics.fmean(balanced(harm).values()),
            additional_harm_fraction=statistics.fmean(balanced(added_harm).values()),
            loss_gt_point1_vs_baseline_fraction=statistics.fmean(balanced(worse01).values()),
            worst_gain_ssim2=min(v for _,v in gains),
            worst_vs_baseline=sorted(worst)[:10],
            mean_butter_pnorm3_delta_vs_baseline=statistics.fmean(balanced(butter).values()),
            frozen_default_coverage=statistics.fmean(balanced(coverage).values()))
    # Report every seed and paired control, without choosing a best test arm.
    seed_means=[result[f'qat8-s{s}-selected']['mean_vs_baseline_ssim2'] for s in [7,17,29]]
    comparisons={}
    for seed in [7,17,29]:
        q={r['id']:r for r in arms[f'qat8-s{seed}-selected']}
        for control in ['ptq8',f'float-s{seed}-selected']:
            deltas=[(r['origin'],float(Decimal(q[r['id']][score])-Decimal(r[score]))) for r in arms[control]]
            means=balanced(deltas)
            comparisons[f'qat8-s{seed}-vs-{control}']=dict(mean=statistics.fmean(means.values()),paired_origin_bootstrap_95=interval(means))
    # Exploratory two-way uncertainty: resample training seeds and origins
    # together, preserving each seed's matched float-control pairing.
    joint={}
    origins=sorted({r['origin'] for r in rows})
    for kind in ['baseline','ptq8','matched_float']:
        matrix=[]
        for seed in [7,17,29]:
            q={r['id']:r for r in arms[f'qat8-s{seed}-selected']}
            control=f'float-s{seed}-selected' if kind=='matched_float' else kind
            values=balanced([(r['origin'],float(Decimal(q[r['id']][score])-Decimal(r[score]))) for r in arms[control]])
            matrix.append([values[o] for o in origins])
        matrix=np.array(matrix)
        rng=np.random.default_rng(20261108)
        si=rng.integers(0,3,(10000,3,1))
        oi=rng.integers(0,len(origins),(10000,1,len(origins)))
        sample=matrix[si,oi].mean(axis=(1,2))
        joint[kind]=dict(mean=float(matrix.mean()),exploratory_seed_origin_bootstrap_95=[float(v) for v in np.quantile(sample,[.025,.975])])
    return dict(arms=result,paired_comparisons=comparisons,qat_seed_mean_vs_baseline=statistics.fmean(seed_means),
                qat_seed_range_vs_baseline=[min(seed_means),max(seed_means)],qat_across_seeds=joint)


def main(a):
    if not (a.root/'COMPLETE').is_file():raise ValueError('incomplete evaluation')
    if (a.root/'COMPLETE').read_text()!=sha(a.root/'results.tsv'):raise ValueError('changed score table')
    with (a.root/'results.tsv').open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
    result=dict(report_sha256=sha(Path(__file__)),results_sha256=sha(a.root/'results.tsv'),
        identity_sha256=sha(a.root/'IDENTITY.tsv'),
        harm_definition='Loss >0.5 at identity SSIM2 <=90; linearly tapers to >0.1 at 100. Strict decimal comparison.',
        uncertainty='Paired bootstrap over origins, 10000 resamples, seed 20261108. Small designed cohort; pointwise intervals are not population guarantees. Three training seeds are reported separately; exploratory joint intervals also resample those three seeds. This does not establish uncertainty over arbitrary training runs.',
        limitations='Previously inspected routing-study inputs. No test-output checkpoint selection. Frozen canon4 routing, no model-specific recalibration. Weight-only compression with f32 execution; no integer speed claim.')
    for cohort in sorted({r['cohort'] for r in rows}):
        selected=[r for r in rows if r['cohort']==cohort]
        groups={'all':selected}
        if cohort=='secondary':
            groups.update({size:[r for r in selected if r['size']==size] for size in sorted({r['size'] for r in selected},key=int)})
        result[cohort]={name:{mode:summarize(group,mode) for mode in ['routed','forced']} for name,group in groups.items()}
        if cohort=='primary':
            subgroups={'encoder_'+encoder:[r for r in selected if r['encoder']==encoder] for encoder in sorted({r['encoder'] for r in selected})}
            subgroups['q5_to_60']=[r for r in selected if float(r['q'])<=60]
            subgroups['q65_to_100']=[r for r in selected if float(r['q'])>=65]
            result['primary_subgroups']={name:{mode:summarize(group,mode) for mode in ['routed','forced']} for name,group in subgroups.items()}

    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    for cohort in ['primary','secondary']:
        print(cohort)
        for arm,r in result[cohort]['all']['routed']['arms'].items():
            print(arm,'gain',r['mean_gain_ssim2'],'delta',r['mean_vs_baseline_ssim2'],'harm',r['harm_fraction'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    main(p.parse_args())
