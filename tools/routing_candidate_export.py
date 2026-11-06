#!/usr/bin/env python3
"""Export a development-frozen candidate for native replay, without activating it."""
import argparse
import csv
import json
from pathlib import Path
import joblib
import numpy as np
import publication_routing_analysis as study


def export(root, inputs, source, golden, replay):
    freeze = root/'_MANIFEST.json'
    record = json.loads(freeze.read_text())
    if not record.get('complete'):
        raise ValueError('incomplete development run')
    key = 'no_bpp_geometry'
    model_path = root/f'{key}.joblib'
    if study.sha(model_path) != record['outputs_sha256'][model_path.name]:
        raise ValueError('candidate model changed')
    frontiers = root/'frontiers.json'
    if study.sha(frontiers) != record['outputs_sha256'][frontiers.name]:
        raise ValueError('development choices changed')
    point = next(p for p in json.loads(frontiers.read_text())[key] if p['budget']==.005)['selection']
    if point['skip'] or point['penalty']!=32. or point['threshold']!=0.:
        raise ValueError('candidate operating point changed')
    transform,gain,risk,_ = model = joblib.load(model_path)
    numeric = record['numeric_features'][key]
    complete = study.FEATURES['decoded']
    scaler = transform.named_transformers_['numeric']
    categories = transform.named_transformers_['categorical'].categories_
    slots = [complete.index(name) for name in numeric]
    means = np.zeros(len(complete)); scales = np.ones(len(complete))
    means[slots] = scaler.mean_; scales[slots] = scaler.scale_
    expanded = []
    for coefficients in (gain.coef_,risk.coef_.reshape(-1)):
        values = np.zeros(len(complete)+sum(map(len,categories)))
        values[slots] = coefficients[:len(numeric)]
        values[len(complete):] = coefficients[len(numeric):]
        expanded.append(values)
    text = ['// Metadata-independent heads; activation is selected in decoded_router.rs.',
            '// Exported by tools/routing_candidate_export.py.',
            '// Development manifest SHA256: '+study.sha(freeze),
            '// Model SHA256: '+study.sha(model_path)]
    for name,values in [('MEAN',means),('SCALE',scales),('GAIN',expanded[0]),('RISK',expanded[1])]:
        text += [f'pub(super) const {name}: [f64; {len(values)}] = [']
        text += ['    '+repr(float(v))+',' for v in values]
        text += ['];']
    for name,value in [('GAIN_BIAS',gain.intercept_),('RISK_BIAS',risk.intercept_[0]),('PENALTY',point['penalty']),('THRESHOLD',point['threshold'])]:
        text += [f'pub(super) const {name}: f64 = {float(value)!r};']
    text += ['pub(super) const CATEGORIES: [&[&str]; 3] = [']
    text += ['    &['+','.join(json.dumps(str(v)) for v in values)+'],' for values in categories]
    text += ['];']
    source.write_text('\n'.join(text)+'\n')
    with inputs.open() as f:
        reader=csv.DictReader(f,delimiter='\t');fields=reader.fieldnames;rows=list(reader)
    x=study.inputs(rows,numeric);pg,pr=study.predict(model,x)
    decisions=study.decisions(pg,pr,point)
    def write(path,indices):
        with path.open('w') as f:
            writer=csv.DictWriter(f,fields,delimiter='\t',lineterminator='\n');writer.writeheader()
            for i in indices:
                row=dict(rows[i],predicted_gain=repr(float(pg[i])),predicted_risk=repr(float(pr[i])),restore=int(decisions[i]))
                writer.writerow(row)
    write(replay,range(len(rows)))
    indices=sorted(set(map(int,np.linspace(0,len(rows)-1,8)))|set(map(int,np.argsort(abs(pg-32*pr))[:8])))
    write(golden,indices)
    if golden.stat().st_size>30000:
        raise ValueError('inline golden exceeds data limit')
    print(json.dumps(dict(cells=len(rows),goldens=len(indices),restored=int(decisions.sum()),
                          model_sha256=study.sha(model_path),source_sha256=study.sha(source))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','inputs','source','golden','replay'):
        p.add_argument(name,type=Path)
    a=p.parse_args();export(a.root,a.inputs,a.source,a.golden,a.replay)
