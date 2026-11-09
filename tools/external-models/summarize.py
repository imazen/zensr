#!/usr/bin/env python3
"""Report paired external-model scores by quality and category, with exact harm boundaries."""
import argparse
from collections import defaultdict
import csv
from decimal import Decimal as D
import hashlib
import json
from pathlib import Path


def band(q):
    q=int(q)
    return 'clean' if q==0 else ('05-20' if q<=20 else '25-60' if q<=60 else '65-100')


def summary(rows):
    gain=[D(r['score'])-D(r['baseline']) for r in rows]
    harmed=[g < -(D('.5') if D(r['baseline'])<=90 else max(D('.1'), D('.5')-D('.04')*(D(r['baseline'])-90))) for g,r in zip(gain,rows)]
    worst=min(range(len(rows)),key=lambda i:gain[i])
    return dict(n=len(rows),mean_gain=str(sum(gain)/len(rows)),harm=sum(harmed),harm_percent=str(D(sum(harmed))*100/len(rows)),
        worse=sum(g<0 for g in gain),worst_gain=str(gain[worst]),worst_id=rows[worst]['id'],
        mean_butter_pnorm3_delta=str(sum(D(r['butter'])-D(r['baseline_butter']) for r in rows)/len(rows)))


def main(a):
    root=a.root
    output=[]
    native={r['id']:r for r in csv.DictReader(a.native.open(),delimiter='\t') if r['arm']=='native'} if a.native else {}
    inputs={r['id']:r for r in csv.DictReader(a.inputs.open(),delimiter='\t')} if a.inputs else {}
    runs=['jpeg-evaluation-v1'] if a.task=='jpeg' else ['sr-s3diff-v1','sr-resshift-v3-v1','sr-resshift-bicubic-v1']
    for run in runs:
        path=root/run
        complete=json.loads((path/'COMPLETE.json').read_text())
        files=sorted((path/'cells').glob('*.json'))
        if len(files)!=complete['inputs']:raise ValueError('coverage differs')
        for file in files:
            cell=json.loads(file.read_text())
            if a.task=='jpeg':
                row=cell['input'];base=cell['identity']['identity_ssim2'];base_butter=cell['identity']['identity_butter_pnorm3']
                if row!=inputs[row['id']]:raise ValueError('input membership differs')
                arms=[(f'meanflow-{step}-{stage}',score[stage]) for step,score in cell['schedules'].items() for stage in ['raw','projected']]
                n=native[row['id']]
                arms += [(f'dejpeg7-{stage}',dict(ssim2=n[stage+'_ssim2'],butter_pnorm3=n[stage+'_butter_pnorm3'])) for stage in ['raw','projected']]
            else:
                row=cell['row'];base=cell['baseline']['ssim2'];base_butter=cell['baseline']['butter_pnorm3']
                arms=[(run,cell['candidate'])]
                if run==runs[0]:arms += [('native-span',cell['native_span'])]
            for arm,score in arms:
                output.append(dict(id=row['id'],arm=arm,content=row['content'],source_id=row['source_id'],q=row['q'],ss=row['ss'],
                    cohort=row.get('cohort','primary'),size=row.get('size','512'),encoder=row.get('encoder','zenjpeg' if row['q'] else 'none'),
                    baseline=base,baseline_butter=base_butter,score=score['ssim2'],butter=score['butter_pnorm3']))
    groups=defaultdict(list)
    for r in output:
        keys=[('overall',r['cohort'],r['arm']),('q-band',r['cohort'],r['arm'],band(r['q'])),
            ('category-band',r['cohort'],r['arm'],r['content'],band(r['q'])),
            ('category-quality',r['cohort'],r['arm'],r['content'],str(r['q'])),
            ('encoder-band',r['cohort'],r['arm'],r['encoder'],str(r['ss']),band(r['q'])),
            ('size-band',r['cohort'],r['arm'],str(r['size']),band(r['q']))]
        for key in keys:groups[key].append(r)
    report=[dict(group=list(key),**summary(rows)) for key,rows in sorted(groups.items())]
    a.output.mkdir(exist_ok=True)
    with (a.output/'rows.tsv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]),delimiter='\t');writer.writeheader();writer.writerows(output)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    digest=hashlib.sha256((a.output/'rows.tsv').read_bytes()).hexdigest()
    (a.output/'IDENTITY.json').write_text(json.dumps(dict(task=a.task,rows=len(output),rows_sha256=digest,runs=runs),sort_keys=True)+'\n')
    for row in report:
        if row['group'][0]=='q-band' and row['group'][1]=='primary':print(json.dumps(row))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--task',choices=['jpeg','sr'],required=True)
    p.add_argument('--native',type=Path);p.add_argument('--inputs',type=Path);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
