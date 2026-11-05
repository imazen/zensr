#!/usr/bin/env python3
"""Scatter the exact frozen routing decisions against baseline quality."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import publication_routing_analysis as study
from publication_routing_scores_only import read_score_tables


def render(root,selection,report,output,policy):
    if output.exists():raise ValueError('refusing to replace scatter output')
    data=json.loads(report.read_text());freeze=selection/'selection.freeze.json'
    if study.sha(freeze)!=data['selection_sha256']:raise ValueError('selection changed')
    contract=json.loads((selection/'fit.contract.json').read_text())
    if study.environment_contract()!=contract['environment']:raise ValueError('numerical environment changed')
    rows,digest=read_score_tables(root,'locked_test')
    if digest!=data['test_cells_sha256']:raise ValueError('score tables differ from report')
    expected=data['policies'][policy];choice=expected['operating_point']['selection']
    frozen=json.loads(freeze.read_text());spec=frozen['models'][choice['model']]
    path=selection/(choice['model']+'.joblib')
    if study.sha(path)!=spec['sha256']:raise ValueError('model changed')
    model=joblib.load(path);x=study.inputs(rows,spec['features'])
    pg,pr=study.predict(model,x)
    restored=study.decisions(pg,pr,choice,study.in_domain(model,x))
    gain,harm,limits=study.labels(rows);groups=np.array([r['origin'] for r in rows])
    actual=study.summary(gain,harm,limits,restored,groups)
    for key in ('origin_mean','origin_harm','coverage','worst'):
        if not np.isclose(actual[key],expected[key],rtol=0,atol=1e-12):
            raise ValueError('decision replay differs from frozen report: '+key)
    baseline=np.array([float(r['identity_ssim2']) for r in rows]);applied=np.where(restored,gain,0.)
    harmed=harm & restored;good=restored & ~harm;skipped=~restored
    output.mkdir(parents=True)
    with gzip.open(output/'points.csv.gz','wt') as f:
        writer=csv.writer(f);writer.writerow(['reference_sha256','origin','encoder','ss','nominal_q','identity_ssim2','potential_model_gain','policy_gain','harm_threshold','restored','harmed'])
        for i,r in enumerate(rows):writer.writerow([r['reference_sha256'],r['origin'],r['encoder'],r['ss'],r['nominal_q'],baseline[i],gain[i],applied[i],limits[i],int(restored[i]),int(harmed[i])])
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    curve_x=np.linspace(min(0.,baseline.min()),max(100.,baseline.max()),1000)
    curve_y=-(.5-.04*np.clip(curve_x-90,0,10))
    for ax in axes:
        for mask,color,label,size,alpha in [(skipped,'#aeb5bd','Skipped (gain = 0)',4,.08),(good,'#1e70b8','Restored, below harm threshold',5,.15),(harmed,'#c82731','Restored, harmed',12,.65)]:
            ax.scatter(baseline[mask],applied[mask],s=size,c=color,alpha=alpha,linewidths=0,rasterized=True,label=f'{label}: {int(mask.sum()):,}')
        ax.plot(curve_x,curve_y,'k--',lw=1.1,label='Harm boundary')
        ax.axhline(0,color='0.35',lw=.6);ax.grid(alpha=.15)
        ax.set_xlabel('Baseline SSIM2 (restoration disabled)')
        ax.set_ylabel('Policy-applied SSIM2 change')
    axes[0].set_title('All decisions; full measured range')
    worst=int(np.argmin(applied));axes[0].annotate(f'Worst: {applied[worst]:.3f}\n{rows[worst]["encoder"]} {rows[worst]["ss"]} q{rows[worst]["nominal_q"]}',xy=(baseline[worst],applied[worst]),xytext=(-110,25),textcoords='offset points',fontsize=9,arrowprops=dict(arrowstyle='->'))
    legend=axes[0].legend(fontsize=8,loc='upper right',markerscale=2)
    for handle in legend.legend_handles:handle.set_alpha(.8)
    axes[1].set_xlim(0,100);axes[1].set_ylim(-2,6);axes[1].set_title('Zoom: baseline 0–100, change −2 to +6')
    outside=(baseline<0)|(baseline>100)|(applied < -2)|(applied>6)
    fig.suptitle(f'{policy}: {len(rows):,} reserved decisions, {len(set(groups))} origins\nOrigin-balanced gain +{actual["origin_mean"]:.3f}; harm {100*actual["origin_harm"]:.3f}%',fontsize=13)
    note=f'Every cell plotted; overlapping points are not counts. Zoom omits {int(outside.sum()):,} out-of-window decisions.\nHarm: loss >0.5 below SSIM2 90, tapering to 0.1 at 100. Pixel-artifact audit incomplete (owner cancelled).'
    fig.text(.01,.005,note,fontsize=8,va='bottom');fig.get_layout_engine().set(rect=(0,.10,1,.78))
    for suffix in ('png','svg','pdf'):fig.savefig(output/('scatter.'+suffix),dpi=180)
    plt.close(fig)
    meta=dict(policy=policy,cells=len(rows),restored=int(restored.sum()),harmed_cells=int(harmed.sum()),origin_mean=actual['origin_mean'],origin_harm=actual['origin_harm'],zoom_omitted=int(outside.sum()),report_sha256=study.sha(report),selection_sha256=study.sha(freeze),cells_sha256=digest,renderer_sha256=study.sha(__file__),artifact_verification=data['artifact_verification'])
    (output/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','selection','report','output'):p.add_argument(name,type=Path)
    p.add_argument('--policy',default='decoded/budget-0.005')
    a=p.parse_args();render(a.root,a.selection,a.report,a.output,a.policy)
