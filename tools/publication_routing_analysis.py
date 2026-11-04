#!/usr/bin/env python3
"""Grouped cross-fit selection and one-shot reserved-test routing evaluation.

All inference columns are an explicit allowlist. The reference score enters only
labels and reports. No split is inferred from filenames or random cell sampling.
"""
import argparse
import collections
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BUDGETS=(0.,.005,.01,.02,.05)
LAMBDAS=(0.,1.,2.,4.,8.,16.,32.,64.)
THRESHOLDS=(0.,.1,.25,.5,.75,1.,1.5,2.,3.,4.,6.,10.)
CATEGORICAL=('probe_family','probe_scale','ss')
BASE=('width','height','probe_value','estimated_gain','bpp')
COEF=tuple('coef_'+x for x in ('zero_ac_blocks','zero_ac_coefs','hf_survival','mean_abs_ac','dc_spread','ac_count_spread','gutted_block_frac','gutted_clustering'))
PIXELS=tuple('pixel_'+x for x in ('edge_slope_stdev','patch_fraction_fast','aq_map_std','variance','colourfulness','laplacian_variance','variance_spread','distinct_color_bins','palette_density','luma_histogram_entropy','luma_kurtosis','skin_tone_fraction','chroma_luma_covariance_cb','chroma_luma_covariance_cr','cr_peak_sharpness','orientation_energy_ratio','aq_map_p90','noise_floor_y_p25','noise_floor_y_p90','noise_floor_uv_p50','noise_floor_uv_p75','noise_floor_uv_p90','quant_survival_y','xyb444_color_loss'))
POST=tuple('post_'+x for x in ('clamped_frac','mean_alpha','roundtrip_mae','fallback_blend','projection_y','projection_cb','projection_cr'))
FEATURES={
    'coefficient':BASE+COEF,
    'chooser':BASE+('coef_zero_ac_blocks','chooser_p_graphics'),
    'edge':BASE+('coef_zero_ac_blocks','pixel_edge_slope_stdev'),
    'decoded':BASE+COEF+PIXELS+('chooser_p_graphics',),
    'post':BASE+COEF+PIXELS+('chooser_p_graphics',)+POST,
}
CONFIGS=(('linear',1.),('linear',10.),('hist',2),('hist',3))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_cells(root, expected_role):
    if not (root/'COMPLETE').is_file():raise ValueError(f'incomplete run {root}')
    rows=[];h=hashlib.sha256();keys=set()
    for path in sorted((root/'cells').glob('*.tsv')):
        data=path.read_bytes();h.update(path.name.encode());h.update(data)
        cell=list(csv.DictReader(data.decode().splitlines(),delimiter='\t'))
        if len(cell)!=1:raise ValueError(f'incomplete cell {path}')
        r=cell[0]
        if r['role']!=expected_role:raise ValueError('unexpected dataset role')
        key=tuple(r[c] for c in ('reference_sha256','size','encoder','ss','nominal_q'))
        if key in keys:raise ValueError('duplicate cell')
        keys.add(key);rows.append(r)
    if not rows:raise ValueError('empty study')
    if (root/'COVERAGE.tsv').exists():
        with (root/'COVERAGE.tsv').open() as f:
            expected={(r['file'],r['size'],r['encoder'],r['ss'],r['q']) for r in csv.DictReader(f,delimiter='\t') if r['status']=='required'}
        actual={tuple(r[c] for c in ('file','size','encoder','ss','nominal_q')) for r in rows}
        if actual!=expected:raise ValueError('incomplete coverage')
    audit=root/'AUDITED.json'
    if not audit.is_file():raise ValueError('artifact audit required before fitting or reporting')
    if json.loads(audit.read_text())['cells_sha256']!=h.hexdigest():raise ValueError('cells changed after artifact audit')
    return rows,h.hexdigest()


def labels(rows):
    gain=np.asarray([float(Decimal(r['restored_ssim2'])-Decimal(r['identity_ssim2'])) for r in rows])
    limits=np.asarray([float(Decimal('.5')-Decimal('.04')*min(Decimal(10),max(Decimal(0),Decimal(r['identity_ssim2'])-90))) for r in rows])
    if not np.isfinite(gain).all():raise ValueError('non-finite gains')
    return gain,gain < -limits,limits


def origin_weights(groups):
    _,inverse,counts=np.unique(groups,return_inverse=True,return_counts=True)
    weights=1./counts[inverse]
    return weights/weights.mean()


def inputs(rows,numeric):
    # No wildcard feature inclusion: adding a reference-derived TSV column must
    # never silently make it an inference input.
    forbidden=('identity','restored','nominal','origin','content','encoder','file')
    if any(c.startswith(forbidden) for c in numeric):raise ValueError('reference/label feature requested')
    x=np.empty((len(rows),len(numeric)+len(CATEGORICAL)),dtype=object)
    for j,c in enumerate(numeric):
        vals=np.asarray([float(r[c]) for r in rows])
        if not np.isfinite(vals).all():raise ValueError(f'missing/non-finite feature {c}')
        x[:,j]=vals
    for j,c in enumerate(CATEGORICAL,start=len(numeric)):x[:,j]=[r[c] for r in rows]
    return x


def fit_model(x,gain,harm,groups,config,n_numeric):
    kind,param=config
    numeric=list(range(n_numeric));categorical=list(range(n_numeric,x.shape[1]))
    transform=ColumnTransformer([
        ('numeric',StandardScaler(),numeric),
        ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical)])
    xt=transform.fit_transform(x)
    weights=origin_weights(groups)
    if kind=='linear':
        gain_model=Ridge(alpha=param)
        risk_model=LogisticRegression(C=1./param,max_iter=3000,random_state=20261104)
    else:
        # Disable random row-level early stopping: it would split variants of
        # the same origin between training and internal validation.
        options=dict(max_iter=100,max_depth=param,max_leaf_nodes=2**param,
                     min_samples_leaf=100,l2_regularization=10.,early_stopping=False,
                     learning_rate=.08,random_state=20261104)
        gain_model=HistGradientBoostingRegressor(**options)
        risk_model=HistGradientBoostingClassifier(**options)
    gain_model.fit(xt,gain,sample_weight=weights)
    if len(np.unique(harm))==1:
        risk_model=float(harm[0])
    else:risk_model.fit(xt,harm,sample_weight=weights)
    return transform,gain_model,risk_model


def predict(model,x):
    transform,gain_model,risk_model=model;xt=transform.transform(x)
    pg=gain_model.predict(xt)
    pr=np.full(len(x),risk_model) if isinstance(risk_model,float) else risk_model.predict_proba(xt)[:,1]
    if not (np.isfinite(pg).all() and np.isfinite(pr).all()):raise ValueError('non-finite prediction')
    return pg,pr


def summary(gain,harm,limits,decision,groups):
    w=origin_weights(groups);w=w/w.sum();realized=gain*decision
    harmed=harm&decision;nrest=int(decision.sum())
    values=dict(n=len(gain),origins=len(set(groups)),mean=float(realized.mean()),
        origin_mean=float(w@realized),coverage=float(decision.mean()),
        harm=float(harmed.mean()),origin_harm=float(w@harmed),harm_count=int(harmed.sum()),
        harm_among_restored=float(harmed.sum()/nrest) if nrest else None,
        harmed_origins=len(set(groups[harmed])),worst=float(realized.min()),
        mean_loss=float(-np.minimum(realized,0.).mean()),
        excess_loss=float(np.maximum(-realized-limits,0.).mean()),
        quantiles=dict(zip(('p01','p05','p25','p50','p75','p95'),map(float,np.quantile(realized,[.01,.05,.25,.5,.75,.95])))))
    return values


def threshold_choices(pg,pr,gain,harm,groups,allowed=None):
    w=origin_weights(groups);w/=w.sum()
    allowed=np.ones(len(gain),dtype=bool) if allowed is None else allowed
    choices=[dict(skip=True,penalty=0.,threshold=0.,mean=0.,harm=0.,coverage=0.)]
    for penalty in LAMBDAS:
        score=pg-penalty*pr
        for threshold in THRESHOLDS:
            decision=(score>=threshold)&allowed
            choices.append(dict(skip=False,penalty=penalty,threshold=threshold,
                mean=float(w@(gain*decision)),harm=float(w@(harm&decision)),coverage=float(w@decision)))
    return choices


def choose(candidates,budget):
    eligible=[c for c in candidates if c['harm']<=budget]
    return max(eligible,key=lambda c:(c['mean'],-c['harm'],-c['coverage']))


def decisions(pg,pr,choice):
    return np.zeros(len(pg),dtype=bool) if choice['skip'] else pg-choice['penalty']*pr>=choice['threshold']


def fit(root,out):
    if out.exists():raise ValueError('refusing to overwrite model selection')
    rows,identity=read_cells(root,'router_development')
    out.mkdir(parents=True)
    groups=np.asarray([r['origin'] for r in rows]);gain,harm,limits=labels(rows)
    if len(set(groups))<20:raise ValueError('insufficient independent development origins')
    folds=list(GroupKFold(n_splits=5,shuffle=True,random_state=20261104).split(np.zeros(len(rows)),gain,groups))
    all_models={};selection={};progress=[]
    for family,numeric in FEATURES.items():
        x=inputs(rows,numeric);candidates=[]
        for kind,param in CONFIGS:
            key=f'{family}-{kind}-{param:g}';pg=np.full(len(rows),np.nan);pr=pg.copy()
            for fold,(train,valid) in enumerate(folds):
                assert not set(groups[train])&set(groups[valid])
                model=fit_model(x[train],gain[train],harm[train],groups[train],(kind,param),len(numeric))
                pg[valid],pr[valid]=predict(model,x[valid])
                event=dict(model=key,fold=fold,train_origins=len(set(groups[train])),valid_origins=len(set(groups[valid])))
                progress.append(event);(out/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                print(json.dumps(event),flush=True)
            if not np.isfinite(pg).all():raise ValueError('missing OOF predictions')
            np.savez_compressed(out/f'{key}.oof.npz',gain=pg,risk=pr)
            for c in threshold_choices(pg,pr,gain,harm,groups):candidates.append(dict(c,model=key))
            all_models[key]=dict(family=family,config=[kind,param],features=list(numeric))
        selected=[]
        for budget in BUDGETS:
            choice=choose(candidates,budget);choice=dict(choice,budget=budget)
            selected.append(choice)
        selection[family]=selected
    # Fit only models needed by some frontier point, then freeze their hashes.
    chosen=sorted({c['model'] for choices in selection.values() for c in choices if not c['skip']})
    for key in chosen:
        spec=all_models[key];numeric=spec['features'];x=inputs(rows,numeric)
        model=fit_model(x,gain,harm,groups,spec['config'],len(numeric))
        joblib.dump(model,out/f'{key}.joblib')
        spec['sha256']=sha(out/f'{key}.joblib')
    # Repeated grouped train/calibration/audit splits are a development stability
    # check. Hyperparameters were selected above, so this is not new confirmation.
    stability=[]
    for repetition,(outer_train,audit) in enumerate(GroupShuffleSplit(n_splits=20,test_size=.2,random_state=20261104).split(np.zeros(len(rows)),groups=groups)):
        inner_train,inner_cal=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=20261104+repetition).split(np.zeros(len(outer_train)),groups=groups[outer_train]))
        train=outer_train[inner_train];cal=outer_train[inner_cal]
        assert not (set(groups[train])&set(groups[cal]) or set(groups[train])&set(groups[audit]) or set(groups[cal])&set(groups[audit]))
        for key in chosen:
            spec=all_models[key];x=inputs(rows,spec['features'])
            model=fit_model(x[train],gain[train],harm[train],groups[train],spec['config'],len(spec['features']))
            cg,cr=predict(model,x[cal]);ag,ar=predict(model,x[audit])
            candidates=threshold_choices(cg,cr,gain[cal],harm[cal],groups[cal])
            for budget in BUDGETS:
                choice=choose(candidates,budget);d=decisions(ag,ar,choice)
                stability.append(dict(repetition=repetition,model=key,budget=budget,
                    calibration=choice,audit=summary(gain[audit],harm[audit],limits[audit],d,groups[audit])))
        (out/'stability.json').write_text(json.dumps(stability,indent=2)+'\n')
        print('completed grouped stability split',repetition+1,'/20',flush=True)
    frozen=dict(development_cells_sha256=identity,development_origins=sorted(set(groups)),
                selection=selection,models=all_models,stability_sha256=sha(out/'stability.json'),python=platform.python_version(),
                numpy=np.__version__,sklearn=sklearn.__version__,
                harm_curve='0.5 - 0.04*clamp(identity_ssim2-90,0,10)',
                methodology='Five grouped cross-fit folds for development selection; no random-row early stopping. Origin-balanced budget and objective. OOF results are selection data, not final test evidence.')
    freeze=out/'selection.freeze.json';freeze.write_text(json.dumps(frozen,indent=2)+'\n')
    print('frozen selection',sha(freeze),flush=True)


def bootstrap(gain,harm,decision,groups,iters=2000):
    unique,inverse=np.unique(groups,return_inverse=True);counts=np.bincount(inverse)
    means=np.bincount(inverse,weights=gain*decision)/counts
    risks=np.bincount(inverse,weights=harm&decision)/counts
    rng=np.random.default_rng(20261104)
    indices=rng.integers(0,len(unique),size=(iters,len(unique)))
    return dict(origin_gain_95=list(map(float,np.quantile(means[indices].mean(axis=1),[.025,.975]))),
                origin_harm_95=list(map(float,np.quantile(risks[indices].mean(axis=1),[.025,.975]))))


def paired_interval(gain,harm,decision,baseline,groups,iters=2000):
    unique,inverse=np.unique(groups,return_inverse=True);counts=np.bincount(inverse)
    delta=decision.astype(float)-baseline.astype(float)
    means=np.bincount(inverse,weights=gain*delta)/counts
    risks=np.bincount(inverse,weights=harm*delta)/counts
    rng=np.random.default_rng(20261104)
    indices=rng.integers(0,len(unique),size=(iters,len(unique)))
    return dict(mean_gain_difference=float(means.mean()),harm_difference=float(risks.mean()),
                gain_difference_95=list(map(float,np.quantile(means[indices].mean(axis=1),[.025,.975]))),
                harm_difference_95=list(map(float,np.quantile(risks[indices].mean(axis=1),[.025,.975]))))


def evaluate(root,selection,out):
    if out.exists():raise ValueError('refusing to overwrite reserved-test report')
    freeze=selection/'selection.freeze.json';frozen=json.loads(freeze.read_text())
    rows,identity=read_cells(root,'locked_test');groups=np.asarray([r['origin'] for r in rows])
    if set(groups)&set(frozen['development_origins']):raise ValueError('origin leakage into reserved test')
    gain,harm,limits=labels(rows)
    for row in rows:
        quality=float(row['identity_ssim2'])
        row['identity_quality_band']='below90' if quality<90 else ('90to95' if quality<95 else '95plus')
    result=dict(test_cells_sha256=identity,selection_sha256=sha(freeze),policies={})
    predicted={}
    for key,spec in frozen['models'].items():
        if 'sha256' not in spec:continue
        path=selection/f'{key}.joblib'
        if sha(path)!=spec['sha256']:raise ValueError('model artifact changed')
        predicted[key]=predict(joblib.load(path),inputs(rows,spec['features']))
    choices={}
    base=np.asarray([float(r['estimated_gain']) for r in rows])
    gate=np.asarray([r['gate']=='true' for r in rows]);cons=np.asarray([r['conservative_gate']=='true' for r in rows])
    choices['unchanged']=np.zeros(len(rows),bool);choices['ungated']=np.ones(len(rows),bool)
    choices['current_fidelity']=(base>=.25)&~gate
    choices['current_conservative']=(base>=.25)&~cons
    choices['oracle_positive']=gain>0
    for family,frontier in frozen['selection'].items():
        for c in frontier:
            pg,pr=predicted[c['model']] if not c['skip'] else (np.zeros(len(rows)),np.zeros(len(rows)))
            choices[f'{family}/budget-{c["budget"]:g}']=decisions(pg,pr,c)
    for name,d in choices.items():
        entry=summary(gain,harm,limits,d,groups);entry.update(bootstrap(gain,harm,d,groups))
        entry['paired_vs_fidelity']=paired_interval(gain,harm,d,choices['current_fidelity'],groups)
        entry['paired_vs_conservative']=paired_interval(gain,harm,d,choices['current_conservative'],groups)
        strata={}
        for column in ('encoder','ss','size','nominal_q','content','ref_kind','identity_quality_band'):
            strata[column]={}
            for value in sorted({r[column] for r in rows}):
                mask=np.asarray([r[column]==value for r in rows]);strata[column][value]=summary(gain[mask],harm[mask],limits[mask],d[mask],groups[mask])
        entry['content_macro_gain']=float(np.mean([v['origin_mean'] for v in strata['content'].values()]))
        entry['content_macro_harm']=float(np.mean([v['origin_harm'] for v in strata['content'].values()]))
        entry['strata']=strata;result['policies'][name]=entry
    out.write_text(json.dumps(result,indent=2)+'\n')
    for name,v in result['policies'].items():print(name,{k:v[k] for k in ('origin_mean','origin_harm','coverage','worst')},flush=True)


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('fit');p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    p=sub.add_parser('evaluate');p.add_argument('root',type=Path);p.add_argument('selection',type=Path);p.add_argument('output',type=Path)
    a=ap.parse_args()
    if a.command=='fit':fit(a.root,a.output)
    else:evaluate(a.root,a.selection,a.output)

if __name__=='__main__':main()
