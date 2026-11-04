#!/usr/bin/env python3
"""Fit canon4 curves from paired production scores and verified routing metadata.

Only image-hash calibration files choose curves and the zero-AC threshold.
Validation files score the frozen choice. No encoder's nominal dial substitutes
for the probe value consumed by the runtime. Outputs JSON for review/codegen.
"""
import argparse
from decimal import Decimal
import collections
import csv
import json
import statistics as st
import subprocess
import re
from gate_recalibrate import hash_stem, interp


def score_delta(restored, identity):
    """Preserve TSV decimal boundaries before converting to binary float."""
    return float(Decimal(restored) - Decimal(identity))


def key(r):
    return r['file'], r['encoder'], r['ss'], float(r['q'])


def load(scores, metadata):
    arms = collections.defaultdict(dict)
    for path in scores:
        for r in csv.DictReader(open(path), delimiter='\t'):
            k = key(r)
            if r['arm'] in arms[k]:
                raise ValueError(f'duplicate {k} {r["arm"]}')
            arms[k][r['arm']] = r
    meta = {}
    for path in metadata:
        for r in csv.DictReader(open(path), delimiter='\t'):
            k = key(r)
            if k in meta:
                raise ValueError(f'duplicate metadata {k}')
            meta[k] = r
    if arms.keys() != meta.keys():
        raise ValueError(f'grid mismatch: scores-only {len(arms.keys()-meta.keys())}, metadata-only {len(meta.keys()-arms.keys())}')
    out = []
    for k, a in arms.items():
        m = meta[k]
        base = a['identity_off']
        if float(base['ssim2']) != float(m['identity_ssim2']):
            raise ValueError(f'identity changed {k}: {base["ssim2"]} vs {m["identity_ssim2"]}')
        if (base['probe_family'], base['probe_q']) != (m['probe_family'], m['probe_q']):
            raise ValueError(f'original probe changed {k}')
        d = dict(m)
        d.update(x=round(float(m['routing_q']),6), z=float(m['zero_ac_frac']),
                 cal=hash_stem(k[0]) % 2 == 0,
                 raw=score_delta(a['model_off']['ssim2'], base['ssim2']),
                 gain=score_delta(a['model_proj']['ssim2'], base['ssim2']))
        d['axis'] = 'distance' if m['routing_scale']=='ButteraugliDistance' else 'quality'
        out.append(d)
    return out


def isotonic(points, increasing):
    # Weighted least-squares isotonic projection of per-cell medians, PAVA.
    sign = 1 if increasing else -1
    blocks = []
    for x, y, n in points:
        blocks.append(([x], sign*y*n, n))
        while len(blocks)>1 and blocks[-2][1]/blocks[-2][2] > blocks[-1][1]/blocks[-1][2]:
            b = blocks.pop(); a = blocks.pop()
            blocks.append((a[0]+b[0], a[1]+b[1], a[2]+b[2]))
    return [(x, round(sign*s/n, 6)) for xs,s,n in blocks for x in xs]


def fit(data, threshold):
    curves = {}
    # Four independent encoder legs; variants do not multiply zenjpeg's weight.
    core = [r for r in data if r['encoder'] in ('turbo','mozjpeg','jpegli','zenjpeg') and r['cal']]
    for axis in ('quality','distance'):
        for ss in ('420','444'):
            rows = [r for r in core if r['axis']==axis and r['ss']==ss]
            for content in ('pooled','graphic','photo'):
                cells = collections.defaultdict(list)
                for r in rows:
                    if content=='pooled' or ((r['z']>=threshold)==(content=='graphic')):
                        cells[r['x']].append(r['gain'])
                if not cells:
                    raise ValueError(f'empty calibration stratum {axis} {ss} {content}')
                pts = [(x,st.median(v),len(v)) for x,v in sorted(cells.items())]
                curves[f'{axis}_{ss}_{content}'] = isotonic(pts,axis=='distance')
    return curves


def gate_group(r):
    if r['axis']=='distance': return 'distance_'+r['ss']
    return ('mozjpeg_' if r['routing_family'].startswith('Mozjpeg') else 'ijg_')+r['ss']


def fit_gates(data):
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in data:
        if r['cal']: groups[gate_group(r)][r['x']].append(r['gain'])
    gates={}
    for group,cells in groups.items():
        distance=group.startswith('distance')
        xs=sorted(cells,reverse=not distance)
        selected=[]
        for x in xs:
            ds=cells[x]
            if sum(d < -.1 for d in ds)/len(ds) < .5: break
            selected.append(x)
        gates[group]=(max(selected) if distance else min(selected)) if selected else (-1.0 if distance else 101.0)
    return gates


def gated(r,gates):
    t=gates[gate_group(r)]
    return r['x']<=t if r['axis']=='distance' else r['x']>=t


def aq_deltas(data):
    paired=collections.defaultdict(dict)
    for r in data:
        if r['cal'] and r['encoder'] in ('zenjpeg','zenjpeg:noaq'):
            paired[(r['file'],r['ss'],r['x'])][r['encoder']]=r['gain']
    cells=collections.defaultdict(list)
    for (_,ss,x),v in paired.items():
        if len(v)!=2: raise ValueError('missing AQ pair')
        cells[ss,x].append(v['zenjpeg:noaq']-v['zenjpeg'])
    return {ss:[(x,round(st.median(v),6)) for (s,x),v in sorted(cells.items()) if s==ss] for ss in ('420','444')}


def evaluate(rows, curves, threshold, gates, aq, margin=0.0, min_gain=0.25):
    values=[]; restored=0; harmed=0
    for r in rows:
        content='graphic' if r['z']>=threshold else 'photo'
        pred=interp(curves[f'{r["axis"]}_{r["ss"]}_{content}'],r['x'])
        if r['encoder']=='zenjpeg:noaq': pred+=interp(aq[r['ss']],r['x'])
        shifted={k:v+(margin if k.startswith('distance') else -margin) for k,v in gates.items()}
        yes=pred>=min_gain and not gated(r,shifted)
        values.append(r['gain'] if yes else 0)
        restored+=yes; harmed+=yes and r['gain'] < -0.1
    return dict(n=len(rows),mean=st.fmean(values),restored=restored/len(rows),harmed=harmed/len(rows))


def legacy_evaluate(rows):
    revision='077cad7bbc04faa12b561ccb70fbeed0f84c5c70'
    source=subprocess.check_output(['git','show',revision+':crates/zensr-zenjpeg/src/api.rs'],text=True)
    source+=subprocess.check_output(['git','show',revision+':crates/zensr-zenjpeg/src/record.rs'],text=True)
    def table(name):
        block=re.search(r'const '+name+r':.*?= \[(.*?)\];',source,re.S).group(1)
        return [(float(x),float(y)) for x,y in re.findall(r'\(([\d.]+),\s*(-?[\d.]+)\)',block)]
    total=[]; restored=0; harmed=0
    for r in rows:
        dist=r['axis']=='distance'; ss=r['ss']; x=r['x']
        content='GRAPHIC' if r['z']>=.25 else 'PHOTO'
        name=content+('_DIST' if dist else '')+ss
        pred=interp(table(name),x)
        if dist and r['encoder']=='zenjpeg:noaq': pred+=interp(table('AQ_OFF_DELTA_'+ss),x)
        gate=x <= (1.3 if ss=='444' else .6) if dist else x >= (88 if ss=='444' else 94.5)
        yes=pred>=.25 and not gate
        total.append(r['gain'] if yes else 0); restored+=yes; harmed+=yes and r['gain']<-.1
    return dict(n=len(rows),mean=st.fmean(total),restored=restored/len(rows),harmed=harmed/len(rows))


def write_rust(result,path):
    lines=['//! Canon4 calibration, 2026-11-04. Generated by tools/fit_canon4.py.',
           '//! 33 calibration images; the 30 validation images do not choose these values.',
           '//! Evidence/method: benchmarks/ship_rt24i_canon4_2026-11-04.md.',
           '// Measured decimal data, not mathematical constants. Preserve codegen precision.',
           '#![allow(clippy::excessive_precision, clippy::approx_constant)]', '']
    def array(name,points,comment):
        lines.append('/// '+comment)
        lines.append(f'pub(crate) const {name}: &[(f32, f32)] = &[')
        lines.extend(f'    ({x:.6f}, {y:.6f}),' for x,y in points)
        lines.append('];\n')
    for key,points in result['curves'].items():
        axis,ss,content=key.split('_')
        prefix={'pooled':'','graphic':'GRAPHIC','photo':'PHOTO'}[content]
        name=(prefix+'_DIST' if prefix else 'DIST')+ss if axis=='distance' else (prefix or 'G')+ss
        array(name,points,f'{axis} {ss} {content}: isotonic fit of paired projected-gain medians.')
    for ss,points in result['aq_deltas'].items():
        array('AQ_OFF_DELTA_'+ss,points,'Paired AQ-off minus AQ-on projected gain, calibration images only.')
    lines.append(f"pub(crate) const GRAPHIC_ZERO_AC_THRESHOLD: f32 = {result['threshold']:.6f};")
    for name,value in result['gates'].items():
        lines.append('/// Tail gate: at least half of calibration cells lose more than 0.1 SSIM2.')
        lines.append(f'pub(crate) const GATE_{name.upper()}: f32 = {value:.6f};')
    with open(path,'w') as f:f.write('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scores',nargs='+',required=True);ap.add_argument('--metadata',nargs='+',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--rust-output')
    a=ap.parse_args();data=load(a.scores,a.metadata)
    # Select using calibration only. Validation is never used to choose t.
    gates=fit_gates(data); aq=aq_deltas(data)
    candidates=[]
    for t in [i/100 for i in range(5,56,5)]:
        curves=fit(data,t)
        score=evaluate([r for r in data if r['cal']],curves,t,gates,aq)
        candidates.append((score['mean'],t,curves,score))
    _,threshold,curves,score=max(candidates,key=lambda v:(v[0],-abs(v[1]-.25)))
    result=dict(threshold=threshold,curves=curves,gates=gates,aq_deltas=aq,calibrate=score,
                validate=evaluate([r for r in data if not r['cal']],curves,threshold,gates,aq),
                threshold_search=[dict(threshold=t,**s) for _,t,_,s in candidates])
    result['conservative_validate']=evaluate([r for r in data if not r['cal']],curves,threshold,gates,aq,margin=6.0)
    result['legacy_validate']=legacy_evaluate([r for r in data if not r['cal']])
    result['core_validate']=evaluate([r for r in data if not r['cal'] and r['encoder'] in ('turbo','mozjpeg','jpegli','zenjpeg')],curves,threshold,gates,aq)
    result['cells']=[]
    cells=collections.defaultdict(list)
    for r in data:
        cells[(r['encoder'],r['ss'],float(r['q']),r['cal'])].append(r)
    for (enc,ss,q,cal),rs in sorted(cells.items()):
        result['cells'].append(dict(encoder=enc,ss=ss,q=q,calibrate=cal,n=len(rs),raw=st.median(r['raw'] for r in rs),gain=st.median(r['gain'] for r in rs),harmed=sum(r['gain'] < -.1 for r in rs),routing_x=sorted(set(r['x'] for r in rs))))
    if a.rust_output: write_rust(result,a.rust_output)
    with open(a.out,'w') as f: json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ('threshold','gates','calibrate','validate','legacy_validate','core_validate')},indent=2))

if __name__=='__main__': main()
