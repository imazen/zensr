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
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import time
import warnings

import joblib
import numpy as np
import sklearn
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from publication_routing_cells import required_coverage
from publication_routing_tables import cell_bytes

STUDY_SIZES=(32,40,48,64,80,96,128,160,192,256,320,384,512,640,768,1024,1280,1536,2048,4096)
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
    'decoded':BASE+COEF+PIXELS,
    'decoded_legacy_chooser':BASE+COEF+PIXELS+('chooser_p_graphics',),
    'post':BASE+COEF+PIXELS+POST,
}
CONFIGS=(('linear',1.),('linear',10.),('hist',2),('hist',3))
MODE_ENCODERS=('zenjpeg',)+tuple('zenjpeg:'+name for name in (
    'baseline','progressive','auto','hybrid','hybridmax','mozbase','mozprog',
    'trellis','trellisdc','noaq','nodering','noaq_nodering','blur'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment_contract():
    versions={}
    for line in Path(__file__).with_name('publication-routing-requirements.txt').read_text().splitlines():
        if not line or line.startswith('#'):continue
        package,expected=line.split('==')
        actual=importlib.metadata.version(package)
        if actual!=expected:raise ValueError(f'{package}: expected {expected}, installed {actual}')
        versions[package]=actual
    return dict(python=platform.python_version(),packages=versions)


def study_contract(root):
    identity=dict(line.split('\t',1) for line in (root/'IDENTITY.tsv').read_text().splitlines())
    # Roles intentionally differ; every other primary-grid input must agree.
    identity.pop('role')
    return dict(identity=identity,features_sha256=sha(root/'FEATURES.tsv'),
                encoders_sha256=sha(root/'ENCODERS.tsv'),
                encoder_files=dict(line.split('\t',1) for line in (root/'ENCODERS.tsv').read_text().splitlines()))


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def checkpoint_predictions(path,calculate):
    cached=path.exists()
    if cached:
        with np.load(path,allow_pickle=False) as data:
            gain,risk,domain=data['gain'],data['risk'],data['in_domain']
    else:
        gain,risk,domain=calculate()
    if (gain.ndim!=1 or risk.shape!=gain.shape or domain.shape!=gain.shape
            or domain.dtype!=np.dtype(bool)):
        raise ValueError('invalid prediction checkpoint shape or domain type')
    if not (np.isfinite(gain).all() and np.isfinite(risk).all()):
        raise ValueError('non-finite prediction checkpoint')
    if ((risk<0)|(risk>1)).any():raise ValueError('checkpoint risk outside probability range')
    if not cached:
        temporary=path.with_suffix('.partial')
        with temporary.open('wb') as f:np.savez_compressed(f,gain=gain,risk=risk,in_domain=domain)
        temporary.replace(path)
    return gain,risk,domain


def read_cells(root, expected_role):
    if not (root/'COMPLETE').is_file():raise ValueError(f'incomplete run {root}')
    rows=[];h=hashlib.sha256();keys=set()
    for name,data in cell_bytes(root):
        h.update(name.encode());h.update(data)
        cell=list(csv.DictReader(data.decode().splitlines(),delimiter='\t'))
        if len(cell)!=1:raise ValueError(f'incomplete cell {root}: {name}')
        r=cell[0]
        if r['role']!=expected_role:raise ValueError('unexpected dataset role')
        key=tuple(r[c] for c in ('reference_sha256','size','encoder','ss','nominal_q'))
        if key in keys:raise ValueError('duplicate cell')
        keys.add(key);rows.append(r)
    if not rows:raise ValueError('empty study')
    expected=required_coverage(root)
    actual={tuple(r[c] for c in ('file','size','encoder','ss','nominal_q')) for r in rows}
    if actual!=expected:raise ValueError('incomplete coverage')
    audit=root/'AUDITED.json'
    if not audit.is_file():raise ValueError('artifact audit required before fitting or reporting')
    audited=json.loads(audit.read_text())
    if audited['cells_sha256']!=h.hexdigest():raise ValueError('cells changed after artifact audit')
    if audited.get('coverage_sha256')!=sha(root/'COVERAGE.tsv'):raise ValueError('coverage changed after artifact audit')
    if audited.get('audit_source_sha256')!=sha(Path(__file__).with_name('publication_routing_cells.py')):raise ValueError('audit tool changed; reaudit artifacts')
    return rows,h.hexdigest()



def development_cells(root,size_root):
    """Join audited primary and size-dense data, without counting replays twice."""
    rows,primary_digest=read_cells(root,'router_development');primary_contract=study_contract(root)
    extra,size_digest=read_cells(size_root,'router_development');size_contract=study_contract(size_root)
    if primary_contract['identity'].get('sizes')!='512':raise ValueError('primary development must use size 512')
    if size_contract['identity'].get('sizes')!=','.join(map(str,STUDY_SIZES)):raise ValueError('dense development size inventory differs')
    contracts=json.loads(json.dumps([primary_contract,size_contract]))
    for contract in contracts:
        for key in ('sizes','manifest_sha256'):contract['identity'].pop(key,None)
    if contracts[0]!=contracts[1]:raise ValueError('dense development changed frozen scientific inputs')
    selection_path=size_root/'selection.json';selection=json.loads(selection_path.read_text())
    if selection['input_cells_sha256']!=primary_digest:raise ValueError('dense representatives were selected from different primary cells')
    if selection['source_manifest_sha256']!=primary_contract['identity']['manifest_sha256']:raise ValueError('dense selector used a different source manifest')
    if selection['features']!=list(BASE+COEF+PIXELS):raise ValueError('dense selector feature protocol differs')
    if selection['selected_manifest_sha256']!=size_contract['identity']['manifest_sha256']:raise ValueError('dense representative manifest changed')
    if selection['selector_sha256']!=sha(Path(__file__).with_name('publication_routing_sizes.py')):raise ValueError('dense representative selector changed')
    if selection['sizes']!=list(STUDY_SIZES) or selection['clusters_per_content']!=2 or selection['seed']!=20261104:raise ValueError('dense representative protocol differs')
    from publication_routing_sizes import NATIVE_RULE
    if selection.get('native_rule')!=NATIVE_RULE:raise ValueError('dense native-resolution protocol differs')
    selected={c['representative_sha256'] for c in selection['clusters']}
    references={r['reference_sha256']:tuple(r[k] for k in ('file','origin','content','ref_kind')) for r in rows}
    if {r['reference_sha256'] for r in extra}!=selected:raise ValueError('dense representative membership differs')
    def key(r):return tuple(r[c] for c in ('reference_sha256','size','encoder','ss','nominal_q'))
    joined={key(r):r for r in rows};replays=0
    for r in extra:
        if references.get(r['reference_sha256'])!=tuple(r[k] for k in ('file','origin','content','ref_kind')):
            raise ValueError('dense reference outside primary development cohort')
        k=key(r)
        if k in joined:
            if r!=joined[k]:raise ValueError('replayed development cell differs')
            replays+=1
        else:joined[k]=r
    provenance=dict(primary_cells_sha256=primary_digest,size_cells_sha256=size_digest,
        size_selection_sha256=sha(selection_path),size_study_contract=size_contract,
        primary_cells=len(rows),size_cells=len(extra),identical_replays=replays,combined_cells=len(joined),
        weighting='Equal total weight per origin across its observed primary and dense-size cells; repeated 512 cells counted once.')
    identity=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    return list(joined.values()),identity,provenance

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
    allowed=set(BASE+COEF+PIXELS+POST+('chooser_p_graphics',))
    if not set(numeric)<=allowed:raise ValueError('feature outside inference allowlist')
    x=np.empty((len(rows),len(numeric)+len(CATEGORICAL)),dtype=object)
    for j,c in enumerate(numeric):
        vals=np.asarray([float(r[c]) for r in rows])
        if not np.isfinite(vals).all():raise ValueError(f'missing/non-finite feature {c}')
        x[:,j]=vals
    for j,c in enumerate(CATEGORICAL,start=len(numeric)):x[:,j]=[r[c] for r in rows]
    return x


class OriginScaler(StandardScaler):
    """Translate before moment accumulation so constant features have zero variance."""
    def fit(self,x,y=None,sample_weight=None):
        values=np.asarray(x,dtype=float)
        offset=values[0].copy()
        super().fit(values-offset,y,sample_weight=sample_weight)
        self.mean_+=offset
        return self


def fit_model(x,gain,harm,groups,config,n_numeric):
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        warnings.simplefilter('error',RuntimeWarning)
        return _fit_model(x,gain,harm,groups,config,n_numeric)


def _fit_model(x,gain,harm,groups,config,n_numeric):
    kind,param=config
    numeric=list(range(n_numeric));categorical=list(range(n_numeric,x.shape[1]))
    transform=ColumnTransformer([
        ('numeric',OriginScaler(),numeric),
        ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical)])
    weights=origin_weights(groups)
    transform.fit(x)
    # Keep dense-size origins from dominating the numeric scaling statistics.
    # Category presence is unchanged; only the fitted numeric transformer is
    # refit with the same origin-balanced weights used by the predictors.
    transform.named_transformers_['numeric'].fit(x[:,numeric].astype(float),sample_weight=weights)
    xt=transform.transform(x)
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
    numbers=x[:,:n_numeric].astype(float)
    domain=(numbers.min(axis=0),numbers.max(axis=0),
            [tuple(sorted(set(x[:,j]))) for j in range(n_numeric,x.shape[1])])
    return transform,gain_model,risk_model,domain


def predict(model,x):
    transform,gain_model,risk_model,_=model;xt=transform.transform(x)
    pg=gain_model.predict(xt)
    pr=np.full(len(x),risk_model) if isinstance(risk_model,float) else risk_model.predict_proba(xt)[:,1]
    if not (np.isfinite(pg).all() and np.isfinite(pr).all()):raise ValueError('non-finite prediction')
    return pg,pr


def in_domain(model,x):
    low,high,categories=model[3];n=len(low);numbers=x[:,:n].astype(float)
    valid=((numbers>=low)&(numbers<=high)).all(axis=1)
    for j,known in enumerate(categories,start=n):valid &= np.asarray([v in known for v in x[:,j]])
    return valid


def summary(gain,harm,limits,decision,groups):
    w=origin_weights(groups);w=w/w.sum();realized=gain*decision
    harmed=harm&decision;nrest=int(decision.sum());origin_restored=float(w@decision)
    values=dict(n=len(gain),origins=len(set(groups)),mean=float(realized.mean()),
        origin_mean=float(w@realized),coverage=float(decision.mean()),origin_coverage=origin_restored,
        harm=float(harmed.mean()),origin_harm=float(w@harmed),harm_count=int(harmed.sum()),
        harm_among_restored=float(harmed.sum()/nrest) if nrest else None,
        origin_harm_among_restored=float(w@harmed)/origin_restored if origin_restored else None,
        harmed_origins=len(set(groups[harmed])),worst=float(realized.min()),
        mean_loss=float(-np.minimum(realized,0.).mean()),
        origin_mean_loss=float(w@(-np.minimum(realized,0.))),
        excess_loss=float(np.maximum(-realized-limits,0.).mean()),
        origin_excess_loss=float(w@np.maximum(-realized-limits,0.)),
        quantiles=dict(zip(('p01','p05','p25','p50','p75','p95'),map(float,np.quantile(realized,[.01,.05,.25,.5,.75,.95])))))
    return values




def psnr_changes(rows,decision,groups):
    # The historical measurement helper encodes exact matches as PSNR=99.
    # Use retained SSE to avoid treating that sentinel as a finite score.
    errors=[]
    for arm in ('identity','restored'):
        values=np.asarray([sum(float(r[f'{arm}_sse_{c}']) for c in ('r','g','b')) for r in rows])
        if not np.isfinite(values).all() or (values<0).any():raise ValueError('invalid squared-error sum')
        errors.append(values)
    before,restored=errors;after=np.where(decision,restored,before)
    both_positive=(before>0)&(after>0);both_exact=(before==0)&(after==0)
    finite=both_positive|both_exact
    change=np.zeros(len(rows));change[both_positive]=10*np.log10(before[both_positive]/after[both_positive])
    positive=(before>0)&(after==0);negative=(before==0)&(after>0)
    w=origin_weights(groups);w/=w.sum();mass=float(w@finite)
    worst_kind='negative_infinity' if negative.any() else ('finite' if finite.any() else 'positive_infinity')
    return dict(origin_mean_improvement=float(w@change) if finite.all() else None,
        origin_weighted_mean_finite_change=float(w@change/mass) if mass else None,
        finite_change_origin_weight_mass=mass,finite_change_count=int(finite.sum()),
        positive_infinite_change_count=int(positive.sum()),negative_infinite_change_count=int(negative.sum()),
        exact_to_exact_count=int(both_exact.sum()),origin_fraction_worse=float(w@(after>before)),
        worst_kind=worst_kind,worst_improvement=float(change[finite].min()) if worst_kind=='finite' else None,
        convention='Equal exact reconstructions have zero change. Exact/nonexact transitions have infinite PSNR change; a full mean containing them is not reported. Finite conditional means retain original origin weights and renormalize their finite mass.')

def worst_cases(rows,gain,harm,decision,limit=10):
    """Trace losses to retained artifacts, with one worst cell per origin."""
    order=sorted(np.flatnonzero(decision & (gain<0)),key=lambda i:(gain[i],rows[i]['origin'],rows[i]['file'],rows[i]['encoder'],rows[i]['ss'],rows[i]['nominal_q']))
    seen=set();cases=[]
    columns=('file','origin','content','ref_kind','reference_sha256','size','width','height',
             'encoder','ss','nominal_q','identity_ssim2','restored_ssim2',
             'jpeg_artifact','reference_artifact','decoded_artifact','restored_artifact',
             'identity_butter_map','restored_butter_map')
    for i in order:
        if rows[i]['origin'] in seen:continue
        seen.add(rows[i]['origin'])
        cases.append(dict({c:rows[i][c] for c in columns},gain=float(gain[i]),harm=bool(harm[i])))
        if len(cases)==limit:break
    return cases

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


def decisions(pg,pr,choice,domain=None):
    decision=np.zeros(len(pg),dtype=bool) if choice['skip'] else pg-choice['penalty']*pr>=choice['threshold']
    if choice.get('domain_guard',False):
        if domain is None:raise ValueError('missing fitted-domain decision')
        decision &= domain
    return decision


def runtime_gates():
    source=Path(__file__).resolve().parents[1]/'crates/zensr-zenjpeg/src/calibration.rs'
    gates={name:float(value) for name,value in re.findall(r'const GATE_(\w+): f32 = ([0-9.]+);',source.read_text())}
    if len(gates)!=6:raise ValueError('runtime gate schema changed')
    return gates


def threshold_inputs(rows,gates):
    scale=np.asarray([r['probe_scale'] for r in rows])
    distance=scale=='ButteraugliDistance'
    quality=np.isin(scale,['IjgQuality','MozjpegQuality'])
    names=[('DISTANCE' if d else ('MOZJPEG' if r['probe_family']=='Mozjpeg' else 'IJG'))+'_'+r['ss'] for r,d in zip(rows,distance)]
    return (np.asarray([r['probe_value'] for r in rows],dtype=np.float32),
            np.asarray([gates[name] for name in names],dtype=np.float32),distance,quality,
            np.asarray([r['estimated_gain'] for r in rows],dtype=np.float32))


def threshold_from_inputs(values,choice):
    value,boundary,distance,quality,gain=values
    if choice['skip']:return np.zeros(len(value),bool)
    gated=(distance & (value<=boundary+np.float32(choice['distance_margin']))) | (quality & (value>=boundary-np.float32(choice['quality_margin'])))
    return (gain>=np.float32(choice['min_gain'])) & ~gated


def threshold_decisions(rows,choice,gates):
    return threshold_from_inputs(threshold_inputs(rows,gates),choice)


def threshold_frontier(rows,gain,harm,groups,gates):
    # Reuse the existing threshold-only study's fixed grid, not its labels.
    from routing_frontier import MIN_GAINS, QUALITY_MARGINS, DISTANCE_MARGINS
    import itertools
    weights=origin_weights(groups);weights/=weights.sum()
    for margin,column in ((0.,'gate'),(6.,'conservative_gate')):
        control=dict(skip=False,min_gain=-float('inf'),quality_margin=margin,distance_margin=margin)
        expected=np.asarray([r[column]=='false' for r in rows])
        if not np.array_equal(threshold_decisions(rows,control,gates),expected):
            raise ValueError(f'threshold baseline disagrees with recorded runtime {column}')
    values=threshold_inputs(rows,gates)
    candidates=[dict(skip=True,min_gain=0.,quality_margin=0.,distance_margin=0.,mean=0.,harm=0.,coverage=0.)]
    for minimum,qmargin,dmargin in itertools.product(MIN_GAINS,QUALITY_MARGINS,DISTANCE_MARGINS):
        c=dict(skip=False,min_gain=minimum,quality_margin=qmargin,distance_margin=dmargin)
        d=threshold_from_inputs(values,c)
        candidates.append(dict(c,mean=float(weights@(gain*d)),harm=float(weights@(harm&d)),coverage=float(weights@d)))
    return [dict(choose(candidates,budget),budget=budget) for budget in BUDGETS]


def fit(root,out,size_root):
    rows,identity,development_sources=development_cells(root,size_root)
    contract=study_contract(root)
    fit_contract=dict(cells_sha256=identity,development_sources=development_sources,study=contract,environment=environment_contract(),analysis_sha256=sha(__file__),
        threshold_source_sha256=sha(Path(__file__).with_name('routing_frontier.py')),
        selector_source_sha256=sha(Path(__file__).with_name('publication_routing_sizes.py')),
        audit_source_sha256=sha(Path(__file__).with_name('publication_routing_cells.py')),
        table_snapshot_source_sha256=sha(Path(__file__).with_name('publication_routing_tables.py')),
        requirements_sha256=sha(Path(__file__).with_name('publication-routing-requirements.txt')),
        features=FEATURES,configs=CONFIGS,budgets=BUDGETS)
    # JSON normalization makes tuples compare identically after a restart.
    fit_contract=json.loads(json.dumps(fit_contract))
    if out.exists():
        if json.loads((out/'fit.contract.json').read_text())!=fit_contract:
            raise ValueError('fit resume contract changed; preserve this run and use a new directory')
        if (out/'selection.freeze.json').exists():raise ValueError('selection already frozen; do not refit')
    else:
        out.mkdir(parents=True)
        atomic_json(out/'fit.contract.json',fit_contract)
    groups=np.asarray([r['origin'] for r in rows]);gain,harm,limits=labels(rows)
    if len(set(groups))<20:raise ValueError('insufficient independent development origins')
    folds=list(GroupKFold(n_splits=5,shuffle=True,random_state=20261104).split(np.zeros(len(rows)),gain,groups))
    gates=runtime_gates()
    threshold_selection=threshold_frontier(rows,gain,harm,groups,gates)
    all_models={};selection={};progress=[];all_candidates={}
    for family,numeric in FEATURES.items():
        x=inputs(rows,numeric);candidates=[]
        for kind,param in CONFIGS:
            key=f'{family}-{kind}-{param:g}';pg=np.full(len(rows),np.nan);pr=pg.copy();domain=np.zeros(len(rows),bool)
            for fold,(train,valid) in enumerate(folds):
                assert not set(groups[train])&set(groups[valid])
                def calculate():
                    model=fit_model(x[train],gain[train],harm[train],groups[train],(kind,param),len(numeric))
                    return (*predict(model,x[valid]),in_domain(model,x[valid]))
                fg,fr,fd=checkpoint_predictions(out/f'{key}.fold-{fold}.npz',calculate)
                if any(v.shape!=(len(valid),) for v in (fg,fr,fd)):raise ValueError('invalid fold checkpoint')
                pg[valid],pr[valid],domain[valid]=fg,fr,fd
                event=dict(model=key,fold=fold,train_origins=len(set(groups[train])),valid_origins=len(set(groups[valid])))
                progress.append(event);(out/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                print(json.dumps(event),flush=True)
            if not np.isfinite(pg).all():raise ValueError('missing OOF predictions')
            np.savez_compressed(out/f'{key}.oof.npz',gain=pg,risk=pr,in_domain=domain)
            for guard in (False,True):
                for c in threshold_choices(pg,pr,gain,harm,groups,domain if guard else None):candidates.append(dict(c,model=key,domain_guard=guard))
            all_models[key]=dict(family=family,config=[kind,param],features=list(numeric))
        selected=[]
        for budget in BUDGETS:
            choice=choose(candidates,budget);choice=dict(choice,budget=budget)
            selected.append(choice)
        selection[family]=selected
        all_candidates[family]=candidates
        atomic_json(out/'development-candidates.json',all_candidates)
    # Fit only models needed by some frontier point, then freeze their hashes.
    chosen=sorted({c['model'] for choices in selection.values() for c in choices if not c['skip']})
    for key in chosen:
        spec=all_models[key];numeric=spec['features'];x=inputs(rows,numeric)
        model=fit_model(x,gain,harm,groups,spec['config'],len(numeric))
        joblib.dump(model,out/f'{key}.joblib.partial')
        (out/f'{key}.joblib.partial').replace(out/f'{key}.joblib')
        spec['sha256']=sha(out/f'{key}.joblib')
        spec['serialized_bytes']=(out/f'{key}.joblib').stat().st_size
    # Repeated grouped train/calibration/audit splits are a development stability
    # check. Hyperparameters were selected above, so this is not new confirmation.
    stability=[]
    for repetition,(outer_train,audit) in enumerate(GroupShuffleSplit(n_splits=20,test_size=.2,random_state=20261104).split(np.zeros(len(rows)),groups=groups)):
        checkpoint=out/f'stability-{repetition:02}.json'
        if checkpoint.exists():
            stability.extend(json.loads(checkpoint.read_text()))
            continue
        repetition_results=[]
        inner_train,inner_cal=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=20261104+repetition).split(np.zeros(len(outer_train)),groups=groups[outer_train]))
        train=outer_train[inner_train];cal=outer_train[inner_cal]
        assert not (set(groups[train])&set(groups[cal]) or set(groups[train])&set(groups[audit]) or set(groups[cal])&set(groups[audit]))
        calibration_rows=[rows[i] for i in cal];audit_rows=[rows[i] for i in audit]
        for choice in threshold_frontier(calibration_rows,gain[cal],harm[cal],groups[cal],gates):
            d=threshold_decisions(audit_rows,choice,gates)
            repetition_results.append(dict(repetition=repetition,model='threshold',budget=choice['budget'],
                calibration=choice,audit=summary(gain[audit],harm[audit],limits[audit],d,groups[audit])))
        for key in chosen:
            spec=all_models[key];x=inputs(rows,spec['features'])
            model=fit_model(x[train],gain[train],harm[train],groups[train],spec['config'],len(spec['features']))
            cg,cr=predict(model,x[cal]);ag,ar=predict(model,x[audit])
            candidates=[]
            for guard in (False,True):
                candidates.extend(dict(c,domain_guard=guard) for c in threshold_choices(cg,cr,gain[cal],harm[cal],groups[cal],in_domain(model,x[cal]) if guard else None))
            for budget in BUDGETS:
                choice=choose(candidates,budget);d=decisions(ag,ar,choice,in_domain(model,x[audit]))
                repetition_results.append(dict(repetition=repetition,model=key,budget=budget,
                    calibration=choice,audit=summary(gain[audit],harm[audit],limits[audit],d,groups[audit])))
        atomic_json(checkpoint,repetition_results)
        stability.extend(repetition_results)
        print('completed grouped stability split',repetition+1,'/20',flush=True)
    atomic_json(out/'stability.json',stability)
    frozen=dict(fit_contract_sha256=sha(out/'fit.contract.json'),development_candidates_sha256=sha(out/'development-candidates.json'),development_cells_sha256=identity,development_origins=sorted(set(groups)),study_contract=contract,development_sources=development_sources,
                selection=selection,threshold_selection=threshold_selection,runtime_gates=gates,models=all_models,stability_sha256=sha(out/'stability.json'),python=platform.python_version(),
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


def harmed_origin_interval(harm,decision,groups):
    """A distinct estimand: any harmed cell in an origin's tested grid."""
    total=len(set(groups));affected=len(set(groups[harm & decision]))
    interval=binomtest(affected,total).proportion_ci(confidence_level=.95,method='exact')
    return dict(origins=total,harmed_origins=affected,fraction=affected/total,
        interval_95=[float(interval.low),float(interval.high)],
        interpretation="Two-sided pointwise Clopper-Pearson interval for any-harm origin incidence under an independent, identically sampled origin model. Not an interval for cell harm or origin-average cell harm. Corpus selection and grid coverage limit population interpretation; this is not a deployment guarantee.")


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


def evaluate(root,selection,out,primary_report=None,mode_report=None):
    if primary_report is not None and mode_report is not None:raise ValueError('choose one generalization axis')
    if out.exists():raise ValueError('refusing to overwrite reserved-test report')
    freeze=selection/'selection.freeze.json';frozen=json.loads(freeze.read_text())
    fit_contract=json.loads((selection/'fit.contract.json').read_text())
    if sha(selection/'fit.contract.json')!=frozen['fit_contract_sha256']:raise ValueError('fit contract changed after selection')
    if sha(__file__)!=fit_contract['analysis_sha256']:raise ValueError('analysis source changed after selection')
    if sha(Path(__file__).with_name('publication_routing_cells.py'))!=fit_contract['audit_source_sha256']:raise ValueError('audit source changed after selection')
    if sha(Path(__file__).with_name('publication_routing_sizes.py'))!=fit_contract['selector_source_sha256']:raise ValueError('representative selector changed after fitting')
    if environment_contract()!=fit_contract['environment']:raise ValueError('evaluation environment changed after fitting')
    if sha(Path(__file__).with_name('publication_routing_tables.py'))!=fit_contract['table_snapshot_source_sha256']:raise ValueError('table snapshot reader changed after fitting')
    actual_contract=study_contract(root)
    expected_contract=json.loads(json.dumps(frozen['study_contract']))
    primary=None;size_selection=None
    generalization_report=primary_report or mode_report
    if generalization_report is not None:
        primary=json.loads(generalization_report.read_text())
        if primary['selection_sha256']!=sha(freeze) or primary['evaluation_scope']!='primary':raise ValueError('size audit requires this selection primary report')
        from publication_routing_sizes import NATIVE_RULE
        if primary_report is not None:
            if actual_contract['identity'].get('sizes')!=','.join(map(str,STUDY_SIZES)):
                raise ValueError('reserved size inventory differs from fixed protocol')
        size_selection=json.loads((root/'selection.json').read_text())
        expected_selection=dict(input_cells_sha256=primary['test_cells_sha256'],
            source_manifest_sha256=expected_contract['identity']['manifest_sha256'],
            selected_manifest_sha256=actual_contract['identity']['manifest_sha256'],
            selector_sha256=sha(Path(__file__).with_name('publication_routing_sizes.py')),
            features=list(BASE+COEF+PIXELS),sizes=list(STUDY_SIZES),
            clusters_per_content=2,seed=20261104,native_rule=NATIVE_RULE)
        if any(size_selection.get(k)!=v for k,v in expected_selection.items()):
            raise ValueError('reserved representative selection differs from fixed protocol')
        # Explicit size-generalization mode changes only image-size inventory
        # and the representative subset; codecs/features/model stay identical.
        changed_fields=('sizes','manifest_sha256') if primary_report is not None else ('encoders','encoder_dependencies_sha256','manifest_sha256')
        if mode_report is not None:
            encoders=actual_contract['identity'].get('encoders','').split(',')
            if encoders!=list(MODE_ENCODERS):raise ValueError('mode audit encoder inventory differs from fixed protocol')
            expected_exclusions=','.join(e for e in expected_contract['identity'].get('exclusions','').split(',') if e and e.split('=')[0] in encoders)
            if actual_contract['identity'].get('exclusions','')!=expected_exclusions:raise ValueError('mode audit has unexpected exclusions')
            actual_contract['identity']['exclusions']=expected_contract['identity'].get('exclusions','')
            expected_contract['identity'].setdefault('exclusions','')
            if not actual_contract['encoder_files'] or any(expected_contract['encoder_files'].get(p)!=h for p,h in actual_contract['encoder_files'].items()):
                raise ValueError('mode audit changed encoder binaries or libraries')
            for field in ('encoder_files','encoders_sha256'):actual_contract[field]=expected_contract[field]
        for c in (actual_contract,expected_contract):
            for field in changed_fields:c['identity'].pop(field,None)
    if actual_contract!=expected_contract:raise ValueError('reserved inputs differ from frozen study contract')
    rows,identity=read_cells(root,'locked_test');groups=np.asarray([r['origin'] for r in rows])
    if primary is not None and not {r['reference_sha256'] for r in rows}<=set(primary['test_reference_sha256s']):raise ValueError('size audit reference outside primary reserved cohort')
    if size_selection is not None and {r['reference_sha256'] for r in rows}!={c['representative_sha256'] for c in size_selection['clusters']}:
        raise ValueError('reserved size representative membership differs')
    if set(groups)&set(frozen['development_origins']):raise ValueError('origin leakage into reserved test')
    gain,harm,limits=labels(rows)
    for row in rows:
        quality=float(row['identity_ssim2'])
        row['identity_quality_band']='below90' if quality<90 else ('90to95' if quality<95 else '95plus')
    result=dict(test_cells_sha256=identity,selection_sha256=sha(freeze),policies={},
                evaluation_scope='size_generalization' if primary_report is not None else ('mode_generalization' if mode_report is not None else 'primary'),
                primary_report_sha256=sha(generalization_report) if generalization_report is not None else None,
                test_reference_sha256s=sorted({r['reference_sha256'] for r in rows}),
                uncertainty='Pointwise paired origin-bootstrap intervals, not simultaneous policy-selection guarantees. Zero observed harm can give a degenerate interval and is not proof of zero population risk.',
                legacy_chooser='Chooser-containing policies retain unresolved historical training/feature-version provenance.')
    result['development_sources']=frozen['development_sources']
    result['representative_selection_sha256']=sha(root/'selection.json') if size_selection is not None else None
    result['size_selection_sha256']=sha(root/'selection.json') if size_selection is not None else None
    result['router_size_scope']='Serialized Python joblib artifacts include fitted transforms and gain/risk heads; these are not measured Rust deployment binary or resident-memory sizes. Restoration weights are separate.'
    result['training_provenance']='Reserved origins are excluded against the documented canonical model-source policy. Historical training pairs lack source-image IDs and a complete generation snapshot; per-pair training provenance is not reconstructed.'
    predicted={};domains={}
    for key,spec in frozen['models'].items():
        if 'sha256' not in spec:continue
        path=selection/f'{key}.joblib'
        if sha(path)!=spec['sha256']:raise ValueError('model artifact changed')
        model=joblib.load(path);x=inputs(rows,spec['features'])
        predicted[key]=predict(model,x);domains[key]=in_domain(model,x)
    choices={};operating_points={}
    base=np.asarray([float(r['estimated_gain']) for r in rows])
    gate=np.asarray([r['gate']=='true' for r in rows]);cons=np.asarray([r['conservative_gate']=='true' for r in rows])
    choices['unchanged']=np.zeros(len(rows),bool);choices['ungated']=np.ones(len(rows),bool)
    choices['current_fidelity']=(base>=.25)&~gate
    choices['current_conservative']=(base>=.25)&~cons
    choices['oracle_positive']=gain>0
    for c in frozen['threshold_selection']:
        name=f'threshold/budget-{c["budget"]:g}'
        choices[name]=threshold_decisions(rows,c,frozen['runtime_gates'])
        operating_points[name]=dict(selection=c,router_artifact_bytes=None,features=None)
    for family,frontier in frozen['selection'].items():
        for c in frontier:
            pg,pr=predicted[c['model']] if not c['skip'] else (np.zeros(len(rows)),np.zeros(len(rows)))
            name=f'{family}/budget-{c["budget"]:g}'
            choices[name]=decisions(pg,pr,c,domains.get(c['model'],np.zeros(len(rows),bool)))
            spec=frozen['models'][c['model']]
            operating_points[name]=dict(selection=c,router_artifact_bytes=spec.get('serialized_bytes') if not c['skip'] else None,features=spec['features'],model_config=spec['config'])
    for name,d in choices.items():
        entry=summary(gain,harm,limits,d,groups);entry.update(bootstrap(gain,harm,d,groups))
        entry['harmed_origin_incidence']=harmed_origin_interval(harm,d,groups)
        weights=origin_weights(groups);weights/=weights.sum()
        secondary={'psnr':psnr_changes(rows,d,groups)}
        for metric,sign in (('butter_max',-1.),('butter_pnorm3',-1.)):
            delta=sign*np.asarray([float(r['restored_'+metric])-float(r['identity_'+metric]) for r in rows])*d
            if not np.isfinite(delta).all():raise ValueError('non-finite secondary metric')
            secondary[metric]=dict(origin_mean_improvement=float(weights@delta),
                origin_fraction_worse=float(weights@(delta<0)),worst_improvement=float(delta.min()))
        entry['operating_point']=operating_points.get(name)
        entry['secondary_metrics']=secondary
        entry['worst_origin_cases']=worst_cases(rows,gain,harm,d)
        entry['paired_vs_fidelity']=paired_interval(gain,harm,d,choices['current_fidelity'],groups)
        entry['paired_vs_conservative']=paired_interval(gain,harm,d,choices['current_conservative'],groups)
        strata={}
        for column in ('encoder','ss','size','nominal_q','content','ref_kind','identity_quality_band'):
            strata[column]={}
            for value in sorted({r[column] for r in rows}):
                mask=np.asarray([r[column]==value for r in rows]);strata[column][value]=summary(gain[mask],harm[mask],limits[mask],d[mask],groups[mask])
        entry['content_macro_gain']=float(np.mean([v['origin_mean'] for v in strata['content'].values()]))
        entry['content_macro_harm']=float(np.mean([v['origin_harm'] for v in strata['content'].values()]))
        entry['harm_sensitivity']={str(limit):summary(gain,gain < -limit,np.full(len(rows),limit),d,groups) for limit in (.1,.5)}
        entry['strata']=strata;result['policies'][name]=entry
    out.write_text(json.dumps(result,indent=2)+'\n')
    for name,v in result['policies'].items():print(name,{k:v[k] for k in ('origin_mean','origin_harm','coverage','worst')},flush=True)


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('fit');p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--size-development',type=Path,required=True)
    p=sub.add_parser('evaluate');p.add_argument('root',type=Path);p.add_argument('selection',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--size-generalization-of',type=Path)
    p.add_argument('--mode-generalization-of',type=Path)
    a=ap.parse_args()
    if a.command=='fit':fit(a.root,a.output,a.size_development)
    else:evaluate(a.root,a.selection,a.output,a.size_generalization_of,a.mode_generalization_of)

if __name__=='__main__':main()
