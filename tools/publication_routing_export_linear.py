#!/usr/bin/env python3
"""Export the owner-selected frozen linear router and independent replay vectors."""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import publication_routing_analysis as study
from publication_routing_scores_only import read_score_tables


def export(root,selection,report,source,golden,replay,jpeg_replay=None):
    data=json.loads(report.read_text());freeze=selection/'selection.freeze.json'
    if study.sha(freeze)!=data['selection_sha256']:raise ValueError('selection changed')
    frozen=json.loads(freeze.read_text());point=data['policies']['decoded/budget-0.005']['operating_point']['selection']
    if point['model']!='decoded-linear-1' or point['domain_guard'] or point['skip']:
        raise ValueError('not the selected linear policy')
    spec=frozen['models'][point['model']];path=selection/(point['model']+'.joblib')
    if study.sha(path)!=spec['sha256']:raise ValueError('model changed')
    transform,gain_model,risk_model,_=model=joblib.load(path)
    scaler=transform.named_transformers_['numeric'];categorical=transform.named_transformers_['categorical']
    assert list(risk_model.classes_)==[0,1]
    definitions=dict(line.split('\t',1) for line in (root/'FEATURES.tsv').read_text().splitlines())
    text=['// Frozen decoded-linear-1, selected by owner after reserved comparison.',
          '// Exported by tools/publication_routing_export_linear.py; do not hand-edit.',
          '// Selection SHA256: '+study.sha(freeze),'// Model SHA256: '+spec['sha256']]
    def array(name,values):
        values=np.asarray(values).reshape(-1)
        text.append(f'pub(super) const {name}: [f64; {len(values)}] = [')
        text.extend('    '+repr(float(v))+',' for v in values);text.append('];')
    array('MEAN',scaler.mean_);array('SCALE',scaler.scale_)
    array('GAIN',gain_model.coef_);array('RISK',risk_model.coef_)
    for name,value in [('GAIN_BIAS',gain_model.intercept_),('RISK_BIAS',risk_model.intercept_[0]),('PENALTY',point['penalty']),('THRESHOLD',point['threshold'])]:
        text.append(f'pub(super) const {name}: f64 = {float(value)!r};')
    text.append('pub(super) const CATEGORIES: [&[&str]; 3] = [')
    text.extend('    &['+','.join(json.dumps(str(v)) for v in values)+'],' for values in categorical.categories_);text.append('];')
    text.append(f'pub(super) const PIXELS: [(&str, &str); {len(study.PIXELS)}] = [')
    for name in study.PIXELS:
        bare=name.removeprefix('pixel_');text.append('    ('+json.dumps(bare)+', '+json.dumps(definitions[bare])+'),')
    text.append('];');source.write_text('\n'.join(text)+'\n')
    rows,digest=read_score_tables(root,'locked_test')
    if digest!=data['test_cells_sha256']:raise ValueError('score data changed')
    x=study.inputs(rows,spec['features']);pg,pr=study.predict(model,x)
    decisions=study.decisions(pg,pr,point)
    fields=list(spec['features'])+list(study.CATEGORICAL)+['predicted_gain','predicted_risk','restore']
    def write(path,indices):
        with path.open('w') as f:
            writer=csv.writer(f,delimiter='\t',lineterminator='\n');writer.writerow(fields)
            for i in indices:writer.writerow([rows[i][k] for k in fields[:-3]]+[repr(float(pg[i])),repr(float(pr[i])),int(decisions[i])])
    write(replay,range(len(rows)))
    indices=sorted(set(map(int,np.linspace(0,len(rows)-1,8)))|set(map(int,np.argsort(abs(pg-point['penalty']*pr-point['threshold']))[:8])))
    write(golden,indices)
    if jpeg_replay is not None:
        with jpeg_replay.open('w') as f:
            writer=csv.writer(f,delimiter='\t',lineterminator='\n');writer.writerow(fields+['jpeg_path'])
            for i in indices:
                writer.writerow([rows[i][k] for k in fields[:-3]]+[repr(float(pg[i])),repr(float(pr[i])),int(decisions[i]),str(root/rows[i]['jpeg_artifact'])])
    if golden.stat().st_size>30000:raise ValueError('golden exceeds inline-data limit')
    print(json.dumps(dict(cells=len(rows),goldens=len(indices),restored=int(decisions.sum()),source_sha256=study.sha(source),replay_sha256=study.sha(replay))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','selection','report','source','golden','replay'):p.add_argument(name,type=Path)
    p.add_argument('--jpeg-replay',type=Path)
    a=p.parse_args();export(a.root,a.selection,a.report,a.source,a.golden,a.replay,a.jpeg_replay)
