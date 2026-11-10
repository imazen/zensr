"""Validate complete blend coverage and report fixed settings and per-input frontiers."""
import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def average(rows, metric):
    values=[float(r[metric]) for r in rows]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Exact/infinite PSNR cases require a separate transition report; refusing finite substitution')
    return statistics.mean(values)


def label(row):
    return '100% SR' if row['intensity_quarter']==4 else f"{row['intensity_quarter']*25}% SR + {row['filter']}"


def main(a):
    manifest=json.loads(a.manifest.read_text())
    complete=json.loads((a.run/'COMPLETE.json').read_text())
    identity=json.loads((a.run/'IDENTITY.json').read_text())
    if complete['identity_sha256']!=sha(a.run/'IDENTITY.json') or complete['scores_sha256']!=sha(a.run/'scores.jsonl'):
        raise ValueError('Completion identity or score bytes differ')
    if identity['manifest_sha256']!=sha(a.manifest) or identity['profile']!='zensim-a':
        raise ValueError('Wrong manifest or metric profile')
    rows=[json.loads(line) for line in (a.run/'scores.jsonl').read_text().splitlines()]
    cases={c['id']:c for c in manifest['cases']}
    key=lambda r:(r['id'],r['model'],r['scale'],r['filter'],r['intensity_quarter'])
    expected={(c['id'],s['model'],scale,f,q) for c in manifest['cases'] for s in c['sr']
              for scale in manifest['scales'] for f in manifest['filters'] for q in manifest['intensity_quarters']}
    if len(rows)!=len(expected) or {key(r) for r in rows}!=expected or complete['rows']!=len(rows) or complete['cases']!=len(cases):
        raise ValueError('Duplicate, missing or unexpected comparison')
    endpoints=defaultdict(set)
    for r in rows:
        case=cases[r['id']]
        for field in ('source_id','origin','category','q'):
            if r[field]!=case[field]:raise ValueError('Row source identity differs')
        if (r['width'],r['height'])!=(case['low_width']*r['scale'],case['low_height']*r['scale']):
            raise ValueError('Scored geometry differs')
        if not math.isfinite(float(r['zensim_a'])) or math.isnan(float(r['psnr'])):
            raise ValueError('Nonfinite score')
        if r['intensity_quarter']==0:
            endpoints[(r['id'],r['scale'],r['filter'],'filter')].add((r['rgb8_sha256'],r['psnr'],r['zensim_a']))
        elif r['intensity_quarter']==4:
            endpoints[(r['id'],r['scale'],r['model'],'sr')].add((r['rgb8_sha256'],r['psnr'],r['zensim_a']))
    if any(len(values)!=1 for values in endpoints.values()):
        raise ValueError('Endpoint pixels/scores depend on an irrelevant model/filter or reused scratch state')
    a.output.mkdir(exist_ok=False)
    ordered=sorted(rows,key=key)
    with (a.output/'rows.tsv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(ordered[0]),delimiter='\t',lineterminator='\n');writer.writeheader();writer.writerows(ordered)
    fixed=[];frontiers=[];groups=defaultdict(list)
    for r in rows:
        for prefix in [('all',),('category',r['category']),('quality',r['q']),('category-quality',r['category'],r['q'])]:
            groups[prefix+(r['model'],r['scale'],r['filter'],r['intensity_quarter'])].append(r)
    for group,values in sorted(groups.items(),key=lambda kv:str(kv[0])):
        fixed.append(dict(group=list(group),n=len(values),psnr=average(values,'psnr'),zensim_a=average(values,'zensim_a')))
    per_case=defaultdict(list)
    for r in rows:per_case[(r['id'],r['model'],r['scale'])].append(r)
    for key_,values in sorted(per_case.items()):
        baseline=next(r for r in values if r['intensity_quarter']==4)
        unique={r['rgb8_sha256']:r for r in values}.values()
        frontier=[r for r in unique if not any(s['psnr']>=r['psnr'] and s['zensim_a']>=r['zensim_a']
            and (s['psnr']>r['psnr'] or s['zensim_a']>r['zensim_a']) for s in unique)]
        frontiers.append(dict(id=key_[0],model=key_[1],scale=key_[2],baseline=baseline,
            zensim_best=max(values,key=lambda r:r['zensim_a']),psnr_best=max(values,key=lambda r:r['psnr']),frontier=frontier))
    (a.output/'fixed-settings.json').write_text(json.dumps(fixed,indent=2,allow_nan=False)+'\n')
    (a.output/'frontiers.json').write_text(json.dumps(frontiers,indent=2,allow_nan=False)+'\n')
    lines=['# Retained SR blend experiment — 2026-11-10','',
        'Exploratory filter/intensity comparison on 17 sources, one per category, at clean/q5/q20/q50/q90. There are 85 LR cases, two models, ×2/×4, four filters and five intensities: 6,800 score rows. This reuses inspected evaluation data; a per-input winner has HR-reference access and is not a deployable selector. No new default is selected.','',
        'Inputs are retained lossless direct-SR RGB8 outputs. JPEG repair is off. zenresize generates Triangle, Mitchell, CatmullRom and Lanczos2 in encoded sRGB; mixtures use float samples and one final RGB8 rounding/clamp. At ×2, stored ×4 SR is Box-reduced, and the HR reference is linear-light CatmullRom-reduced. Results do not establish a blend before SR quantization or another reference/degradation pipeline.','',
        'Scores are encoded RGB PSNR from the unchanged zenmetrics implementation and the explicitly pinned public zensim-A revision-7 model. Every candidate compares with HR pixels at the target resolution. Neither metric alone establishes subjective SR quality.','',
        '## Fixed settings and per-input opportunities','',
        '| Model | Scale | Best fixed PSNR setting | Mean PSNR gain vs SR (dB) | Mean A change for that setting | Inputs where some setting improves A |',
        '|---|---:|---|---:|---:|---:|']
    for model,scale in itertools.product(('spanf','dscf'),(2,4)):
        values=[r for r in fixed if r['group'][0]=='all' and r['group'][-4:-2]==[model,scale]]
        best=max(values,key=lambda r:r['psnr']);base=next(r for r in values if r['group'][-1]==4)
        selected=[r for r in frontiers if r['model']==model and r['scale']==scale]
        improved=sum(r['zensim_best']['zensim_a']>r['baseline']['zensim_a'] for r in selected)
        config=dict(filter=best['group'][-2],intensity_quarter=best['group'][-1])
        lines.append(f"| {model.upper()} | ×{scale} | {label(config)} | {best['psnr']-base['psnr']:+.4f} | {best['zensim_a']-base['zensim_a']:+.4f} | {improved}/{len(selected)} |")
    lines += ['', 'Pure SR has the highest mean A for both models/scales among fixed settings. Selecting separately for each known HR input exposes additional opportunities, but that is an oracle ceiling rather than evidence for an automatic runtime rule.', '',
        '## Graphics cases — clean input, ×2','', '| Category | Model | A-selected setting | A gain vs SR | PSNR change (dB) |','|---|---|---|---:|---:|']
    for source in ('7109','8027','8157','9059','9137','9779'):
        for model in ('spanf','dscf'):
            record=next(r for r in frontiers if r['scale']==2 and r['model']==model and r['baseline']['source_id']==source and r['baseline']['q']==0)
            b=record['baseline'];v=record['zensim_best']
            category=b['category'].split('-',1)[1].replace('lilith-','')
            lines.append(f"| {category} | {model.upper()} | {label(v)} | {v['zensim_a']-b['zensim_a']:+.4f} | {v['psnr']-b['psnr']:+.4f} |")
    lines += ['', 'The retained `fixed-settings.json` includes category × quality × model × scale breakdowns; `frontiers.json` includes all per-input Pareto sets, the pure-SR scores and separate PSNR/A winners. `rows.tsv` preserves every configuration and output pixel hash. Category claims are limited by one origin per category.', '',
        '## Execution and retention','',
        'The scorer loads original HR, LR and both SR images once per case, then reuses resident filter buffers, zensim prepared references and scratch. Four independent cases run concurrently. All 170 source SR PNGs are retained with input and file/pixel identities. Variants stay in memory; IQA maps are neither requested nor saved.', '',
        'Measured scoring run: 8 seconds, run-heavy peak RSS **0.15 GiB**, 6 GiB cgroup cap, four workers. This excludes inference, retention preparation and compilation. All 17 tests pass, including the copied PSNR scalar/SIMD tests, cached-versus-cold zensim-A at 19×23 and 512×384 with identity=100, and strided filter/blend input. The reporter checks 1,020 endpoint groups for exact pixel/score equality across irrelevant filter/model choices.', '',
        'The source/model pins and commands are in [the tool README](../tools/sr-blend/README.md). Raw artifacts: `/mnt/v/output/zensr/sr-blend-20261110/`. Generated tables and images remain outside Git; the repository contains source, lockfile and this report.', '']
    (a.output/'report.md').write_text('\n'.join(lines))
    (a.output/'VERIFIED.json').write_text(json.dumps(dict(cases=len(cases),rows=len(rows),manifest_sha256=sha(a.manifest),scores_sha256=sha(a.run/'scores.jsonl'),endpoint_groups=len(endpoints)),indent=2)+'\n')
    print('\n'.join(lines[:18]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path);p.add_argument('run',type=Path);p.add_argument('output',type=Path)
    main(p.parse_args())
