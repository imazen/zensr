"""Verify final pixels and report the frozen SR cascade factorial comparison."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from cascade_contract import ARMS, validate
from cascade_stats import paired
from summarize import band


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(a):
    from PIL import Image
    identity=json.loads((a.run/'IDENTITY.json').read_text())
    complete=json.loads((a.run/'COMPLETE.json').read_text())
    if complete['identity_sha256']!=sha(a.run/'IDENTITY.json'):raise ValueError('completion identity differs')
    cells=[json.loads(p.read_text()) for p in sorted((a.run/'cells').glob('*.json'))]
    if len(cells)!=complete['inputs'] or {c['row']['id'] for c in cells}!=set(identity['input_ids']):
        raise ValueError('cascade report coverage differs')
    if identity['pilot']:
        if not a.pilot or len(cells)!=7:raise ValueError('pilot requires explicit report opt-in')
    elif a.pilot or len(cells)!=697:raise ValueError('full cascade coverage required')
    decoded={};old_count=0
    for c in cells:
        r=c['row'];validate(c,r,Path(identity['artifact_root']))
        for score,w,h in [(c[name],r['width'],r['height']) for name in ('baseline',)+ARMS]+[(c['routing']['preview'],r['low_width'],r['low_height'])]:
            png=score['png'];digest=png['sha256']
            if digest not in decoded:
                with Image.open(png['path']) as image:
                    if image.format!='PNG':raise ValueError('retained artifact is not PNG')
                    pixels=image.convert('RGB')
                    decoded[digest]=(pixels.size,hashlib.sha256(pixels.tobytes()).hexdigest(),png['bytes'])
            if decoded[digest][:2]!=((w,h),score['rgb8_sha256']):raise ValueError('PNG pixels differ from scored pixels')
        if a.dscf_control:
            previous=json.loads((a.dscf_control/'cells'/(r['id']+'.json')).read_text())
            if previous['row']!=r:raise ValueError('DSCF control input differs')
            for key in ['ssim2','butter_max','butter_pnorm3','rgb8_sha256']:
                if previous['candidate'][key]!=c['dscf_direct'][key]:raise ValueError('direct DSCF control differs')
                if previous['baseline'][key]!=c['baseline'][key]:raise ValueError('CatmullRom control differs')
            old_count+=1
    groups=defaultdict(list)
    for c in cells:
        r=c['row']
        for key in [('overall',),('q-band',band(r['q'])),('category-band',r['content'],band(r['q'])),
            ('category-quality',r['content'],str(r['q'])),('chroma-band',str(r['ss']),band(r['q']))]:groups[key].append(c)
    reports=[]
    for key,group in sorted(groups.items()):
        restored=[c for c in group if c['routing']['applied']]
        stats=dict(group=list(key),n=len(group),restored=len(restored),
            quality={arm:paired(group,arm,'baseline') for arm in ARMS},
            repair_effect={m:dict(all=paired(group,m+'_repair',m+'_direct'),
                restored=paired(restored,m+'_repair',m+'_direct')) for m in ['spanf','dscf']},
            model_effect={stage:paired(group,'dscf_'+stage,'spanf_'+stage) for stage in ['direct','repair']})
        reports.append(stats)
    a.output.mkdir(exist_ok=True)
    with (a.output/'summary.json').open('x') as f:f.write(json.dumps(reports,indent=2)+'\n')
    rows=[]
    for c in cells:
        r=c['row']
        for arm in ARMS:
            p=c[arm];b=c['baseline']
            rows.append(dict(id=r['id'],source_id=r['source_id'],origin=r['origin'],content=r['content'],q=r['q'],ss=r['ss'],
                restored=int(c['routing']['applied']),arm=arm,ssim2=p['ssim2'],butter_pnorm3=p['butter_pnorm3'],butter_max=p['butter_max'],
                baseline_ssim2=b['ssim2'],baseline_butter_pnorm3=b['butter_pnorm3'],rgb8_sha256=p['rgb8_sha256'],png_sha256=p['png']['sha256'],png=p['png']['path']))
    with (a.output/'rows.tsv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    record=dict(run_identity_sha256=sha(a.run/'IDENTITY.json'),reporter_sha256=sha(Path(__file__)),
        statistics_sha256=sha(Path(__file__).with_name('cascade_stats.py')),input_count=len(cells),
        rows=len(rows),rows_sha256=sha(a.output/'rows.tsv'),unique_pngs=len(decoded),png_bytes=sum(v[2] for v in decoded.values()),
        dscf_control_inputs=old_count,decoder='Pillow independent PNG decode',pilot=identity['pilot'])
    with (a.output/'VERIFIED.json').open('x') as f:f.write(json.dumps(record,indent=2,sort_keys=True)+'\n')
    print(json.dumps(record))
    for r in reports:
        if r['group'][0] in ('overall','q-band'):print(json.dumps(r))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--dscf-control',type=Path,required=True)
    p.add_argument('--pilot',action='store_true')
    main(p.parse_args())
