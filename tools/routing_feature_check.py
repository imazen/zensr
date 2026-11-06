#!/usr/bin/env python3
"""Secondary check of development-frozen candidates on an already-inspected cohort.

This is not a new untouched publication test. No fitting or threshold selection
uses reserved outcomes. The owner-cancelled pixel audit remains cancelled.
"""
import argparse
import json
from pathlib import Path
import subprocess
import joblib
import numpy as np
import publication_routing_analysis as study
from publication_routing_scores_only import read_score_tables


def check(development, candidates, reserved, original_report, output):
    if output.exists():
        raise ValueError('preserve prior report; use a new output')
    frozen = json.loads((candidates/'_MANIFEST.json').read_text())
    if not frozen.get('complete'):
        raise ValueError('candidate development run incomplete')
    if study.environment_contract() != frozen['environment']:
        raise ValueError('candidate numerical environment differs')
    for name, expected in frozen['outputs_sha256'].items():
        if study.sha(candidates/name) != expected:
            raise ValueError('candidate changed after development freeze')
    dev, identity, _ = study.development_cells(development, primary_only=True)
    if identity != frozen['development_identity']:
        raise ValueError('development identity differs')
    development_origins = {r['origin'] for r in dev}
    del dev
    rows, digest = read_score_tables(reserved, 'locked_test')
    previous = json.loads(original_report.read_text())
    if digest != previous['test_cells_sha256']:
        raise ValueError('reserved tables differ from prior report')
    if development_origins & {r['origin'] for r in rows}:
        raise ValueError('origin overlap between fitting and reserved cohort')
    gain, harm, limits = study.labels(rows)
    groups = np.asarray([r['origin'] for r in rows])
    frontiers = json.loads((candidates/'frontiers.json').read_text())
    report = dict(build_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  interpretation='secondary check on reused reserved cohort; not a fresh held-out claim',
                  artifact_verification='incomplete_owner_cancelled; score tables only',
                  candidate_manifest_sha256=study.sha(candidates/'_MANIFEST.json'),
                  original_report_sha256=study.sha(original_report),
                  test_cells_sha256=digest, environment=frozen['environment'],
                  source_sha256=study.sha(__file__), policies={})
    baseline = None
    for name in ('baseline','no_bpp','no_bpp_geometry'):
        numeric = frozen['numeric_features'][name]
        model = joblib.load(candidates/f'{name}.joblib')
        pg, pr = study.predict(model, study.inputs(rows, numeric))
        # The 0.5% development point was fixed before this secondary check.
        point = next(p for p in frontiers[name] if p['budget']==.005)
        decision = study.decisions(pg, pr, point['selection'])
        result = study.summary(gain,harm,limits,decision,groups)
        result['selection'] = point['selection']
        result['model_sha256'] = frozen['outputs_sha256'][f'{name}.joblib']
        if name == 'baseline':
            baseline = decision
            for key in ('origin_mean','origin_harm','coverage','worst'):
                if abs(result[key]-previous['policies']['decoded/budget-0.005'][key])>1e-10:
                    raise ValueError('frozen reserved baseline does not replay')
        result['paired_vs_baseline'] = study.paired_interval(gain,harm,decision,baseline,groups)
        result['changed_decisions'] = int(np.count_nonzero(decision!=baseline))
        result['worst_origin_cases'] = study.worst_cases(rows,gain,harm,decision)
        report['policies'][name] = result
        print(json.dumps(dict(policy=name,**{k:result[k] for k in ('origin_mean','origin_harm','coverage','worst','changed_decisions')})),flush=True)
    study.atomic_json(output,report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('development','candidates','reserved','original_report','output'):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    check(a.development,a.candidates,a.reserved,a.original_report,a.output)
