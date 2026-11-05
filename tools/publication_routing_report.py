#!/usr/bin/env python3
"""Render frozen reserved-test results; never select a deployment policy."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(source, output):
    if output.exists():raise ValueError('refusing to overwrite a report package')
    data=json.loads(source.read_text());policies=data['policies']
    output.mkdir(parents=True)
    fields=['policy','origins','n','origin_mean','origin_harm','coverage','worst',
            'harm_among_restored','origin_harm_among_restored','origin_coverage',
            'mean','harm','mean_loss','origin_mean_loss','harmed_origins','excess_loss','origin_excess_loss','content_macro_gain',
            'content_macro_harm','flat01_harm','flat05_harm','gain_low','gain_high',
            'harm_low','harm_high','gain_difference_vs_conservative',
            'gain_difference_vs_conservative_low','gain_difference_vs_conservative_high','router_artifact_bytes',
            'any_harm_origin_fraction','any_harm_origin_low','any_harm_origin_high']
    with (output/'policies.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for name,p in policies.items():
            row={k:p[k] for k in fields if k in p};row['policy']=name
            row.update(flat01_harm=p['harm_sensitivity']['0.1']['origin_harm'],
                       flat05_harm=p['harm_sensitivity']['0.5']['origin_harm'],
                       gain_low=p['origin_gain_95'][0],gain_high=p['origin_gain_95'][1],
                       harm_low=p['origin_harm_95'][0],harm_high=p['origin_harm_95'][1],
                       gain_difference_vs_conservative=p['paired_vs_conservative']['mean_gain_difference'],
                       gain_difference_vs_conservative_low=p['paired_vs_conservative']['gain_difference_95'][0],
                       gain_difference_vs_conservative_high=p['paired_vs_conservative']['gain_difference_95'][1])
            row['router_artifact_bytes']=(p.get('operating_point') or {}).get('router_artifact_bytes')
            incidence=p['harmed_origin_incidence']
            row.update(any_harm_origin_fraction=incidence['fraction'],any_harm_origin_low=incidence['interval_95'][0],any_harm_origin_high=incidence['interval_95'][1])
            writer.writerow(row)
    (output/'operating-points.json').write_text(json.dumps({name:p.get('operating_point') for name,p in policies.items()},indent=2)+'\n')
    worst={name:p['worst_origin_cases'] for name,p in policies.items()}
    (output/'worst-cases.json').write_text(json.dumps(worst,indent=2)+'\n')
    families=sorted({name.split('/')[0] for name in policies if '/' in name})
    baseline_names=['unchanged','current_fidelity','current_conservative','ungated']
    fig,axes=plt.subplots(math.ceil(len(families)/3),3,figsize=(13,3.8*math.ceil(len(families)/3)),squeeze=False,sharex=True,sharey=True,layout='constrained')
    for ax,family in zip(axes.flat,families):
        for baseline,marker in zip(baseline_names,('x','+','s','D')):
            p=policies[baseline]
            ax.scatter(100*p['origin_harm'],p['origin_mean'],marker=marker,s=30,color='0.5',label=baseline)
        points=[(name,p) for name,p in policies.items() if name.startswith(family+'/')]
        labels=collections.defaultdict(list)
        for name,p in points:
            x=100*p['origin_harm'];y=p['origin_mean']
            # Percentile bootstrap intervals need not bracket the point estimate.
            # Draw their endpoints directly rather than clipping negative errors.
            ax.plot([100*v for v in p['origin_harm_95']],[y,y],color='C0',alpha=.5)
            ax.plot([x,x],p['origin_gain_95'],color='C0',alpha=.5)
            ax.scatter(x,y,s=22,color='C0')
            budget=float(name.split('budget-')[1])
            labels[(x,y)].append(f'{100*budget:g}%')
        for point,budgets in labels.items():
            ax.annotate(', '.join(budgets),point,xytext=(3,3),textcoords='offset points',fontsize=8)
        ax.set_title(family.replace('_',' '));ax.set_xlabel('Reserved-set origin-balanced harm (%)')
        ax.set_ylabel('Reserved-set mean SSIM2 gain');ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(families):]:ax.set_visible(False)
    axes.flat[0].legend(fontsize=7,loc='best')
    scope=data.get('evaluation_scope','unspecified').replace('_',' ')
    development=data.get('development_sources',{})
    training_scope=development.get('scope','unspecified')
    scope+='; training '+training_scope.replace('_',' ')
    verification=data.get('artifact_verification',{})
    audit_note='Pixel-artifact audit incomplete (cancelled by owner).' if verification.get('status')=='incomplete_owner_cancelled' else ''
    if audit_note:scope+='; pixel audit incomplete'
    fig.suptitle(f'Frozen routing operating points ({scope}) — labels are development harm budgets\nBars: pointwise 95% origin-bootstrap intervals; gray markers: fixed baselines',fontsize=12)
    footer=.8/fig.get_figheight()
    fig.get_layout_engine().set(rect=(0,footer,1,1-footer))
    fig.text(.02,.1/fig.get_figheight(),
        'Harm curve: loss >0.5 at/below SSIM2 90, tapering linearly to 0.1 at 100; the taper is a proposal.\n'
        'Origins are separated against documented model-source policy; historical per-pair training provenance is incomplete.\n'
        'Legacy-chooser provenance remains unresolved. Intervals are pointwise; deployment selection remains open.',
        fontsize=8,va='bottom')
    fig.savefig(output/'frontier.svg');fig.savefig(output/'frontier.pdf')
    fig.savefig(output/'frontier.png',dpi=160);plt.close(fig)
    text=['# Frozen routing operating points','',
          audit_note,
          'Evaluation scope: '+data.get('evaluation_scope','unspecified')+'.',
          'Development scope: '+training_scope+'. '+development.get('limitation',''),
          'No deployment policy has been selected. Budgets below are development targets; held-out harm may exceed them.',
          '', '| Policy | Mean gain | Harm % | Restore % | Worst SSIM2 change |',
          '|---|---:|---:|---:|---:|']
    for name,p in policies.items():
        text.append(f'| {name} | {p["origin_mean"]:.4f} | {100*p["origin_harm"]:.3f} | {100*p["coverage"]:.2f} | {p["worst"]:.4f} |')
    text += ['', 'Gain and harm are origin-balanced. Restore coverage is cell-weighted.',
             'The CSV retains both origin-balanced and cell-weighted gain, harm, coverage, harm among restored cases, and loss severity.',
             'The CSV also reports any-harm origin incidence with a pointwise exact binomial interval. This distinct estimand counts an origin once if any tested variant is harmed; its independent, identically sampled origin assumption does not establish deployment-population coverage.',
             'The oracle uses reference outcomes and is not deployable.',
             'Post-inference policies pay restoration cost even when rejecting output.',
             data.get('uncertainty',''),data.get('legacy_chooser',''),data.get('training_provenance',''),data.get('router_size_scope',''),
             'Size generalization, encoder mode coverage and runtime cost require their own evidence; this table alone does not establish publication readiness.',
             '', '[Frozen configurations, feature lists and router artifact sizes](operating-points.json).',
             '[Detailed policies and intervals](policies.csv). [Frontier figure](frontier.svg).',
             '[Worst losses by origin, with retained artifact paths](worst-cases.json). Cases are diagnostics, not inputs to policy selection.', '']
    (output/'report.md').write_text('\n'.join(text))
    provenance=dict(report_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    selection_sha256=data['selection_sha256'],test_cells_sha256=data['test_cells_sha256'],
                    matplotlib=matplotlib.__version__,renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if verification:provenance['artifact_verification']=verification
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();render(args.source,args.output)
