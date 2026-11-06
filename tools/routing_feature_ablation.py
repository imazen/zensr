#!/usr/bin/env python3
"""Development-only grouped feature ablation; never consumes reserved outcomes.

Compare the frozen linear architecture with bpp removed, and with bpp plus
image dimensions removed. No production model or original study is overwritten.
OOF frontier selection scores are development diagnostics, not held-out claims.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
import publication_routing_analysis as study


def run(root, original, output):
    if output.exists():
        raise ValueError('preserve existing output; choose a new directory')
    output.mkdir(parents=True)
    environment = study.environment_contract()
    rows, identity, provenance = study.development_cells(root, primary_only=True)
    contract = json.loads((original/'fit.contract.json').read_text())
    if identity != contract['cells_sha256']:
        raise ValueError('development inputs differ from frozen study')
    if study.study_contract(root) != contract['study']:
        raise ValueError('scientific input identities differ')
    groups = np.asarray([r['origin'] for r in rows])
    gain, harm, limits = study.labels(rows)
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=20261104)
                 .split(np.zeros(len(rows)), gain, groups))
    variants = {'baseline': study.FEATURES['decoded'],
                'no_bpp': tuple(c for c in study.FEATURES['decoded'] if c != 'bpp'),
                'no_bpp_geometry': tuple(c for c in study.FEATURES['decoded']
                                         if c not in ('bpp', 'width', 'height'))}
    manifest = dict(build_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        scope='primary_512_development_only', development_identity=identity,
        provenance=provenance, environment=environment,
        frozen_contract_sha256=study.sha(original/'fit.contract.json'),
        source_sha256={str(p): study.sha(p) for p in (
            Path(__file__), Path(study.__file__),
            Path(study.__file__).with_name('publication_routing_scaler.py'))},
        numeric_features=variants, config=['linear', 1.],
        domain_guard=False, folds=5, seed=20261104,
        interpretation='OOF frontier selection diagnostics; no reserved outcomes, no size-generalization claim')
    study.atomic_json(output/'_MANIFEST.json', manifest)
    report = {}
    progress = []
    for name, numeric in variants.items():
        x = study.inputs(rows, numeric)
        pg = np.full(len(rows), np.nan)
        pr = pg.copy()
        for fold, (train, valid) in enumerate(folds):
            assert not set(groups[train]) & set(groups[valid])
            model = study.fit_model(x[train], gain[train], harm[train], groups[train],
                                    ('linear', 1.), len(numeric))
            pg[valid], pr[valid] = study.predict(model, x[valid])
            progress.append(dict(variant=name, fold=fold,
                                 train_origins=len(set(groups[train])),
                                 validation_origins=len(set(groups[valid]))))
            study.atomic_json(output/'progress.json', progress)
            print(json.dumps(progress[-1]), flush=True)
        if not (np.isfinite(pg).all() and np.isfinite(pr).all()):
            raise ValueError('incomplete/nonfinite OOF predictions')
        np.savez_compressed(output/f'{name}.oof.npz', gain=pg, risk=pr)
        candidates = study.threshold_choices(pg, pr, gain, harm, groups)
        points = []
        for budget in study.BUDGETS:
            choice = study.choose(candidates, budget)
            decision = study.decisions(pg, pr, choice)
            points.append(dict(budget=budget, selection=choice,
                               result=study.summary(gain, harm, limits, decision, groups)))
        report[name] = points
        study.atomic_json(output/'frontiers.json', report)
        model = study.fit_model(x, gain, harm, groups, ('linear', 1.), len(numeric))
        joblib.dump(model, output/f'{name}.joblib')
    baseline = np.load(original/'decoded-linear-1.oof.npz')
    fresh = np.load(output/'baseline.oof.npz')
    difference = {k: float(np.max(np.abs(baseline[k]-fresh[k]))) for k in ('gain', 'risk')}
    # Fresh fits have the same input ordering/folds/configuration; confirm the
    # replay before interpreting any feature-removal comparison.
    if difference['gain'] > 1e-8 or difference['risk'] > 1e-8:
        raise ValueError(f'frozen OOF baseline differs: {difference}')
    manifest['baseline_oof_max_absolute_difference'] = difference
    manifest['outputs_sha256'] = {p.name: study.sha(p) for p in output.iterdir()
                                  if p.is_file() and p.name != '_MANIFEST.json'}
    manifest['complete'] = True
    study.atomic_json(output/'_MANIFEST.json', manifest)
    print(json.dumps(dict(stage='complete', cells=len(rows), origins=len(set(groups)),
                         baseline_oof_difference=difference)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'original', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.root, args.original, args.output)
