#!/usr/bin/env python3
"""Compare frozen canon4 routers under empirical calibration harm budgets.

Select on calibration images only. Validation is a reused audit set, not new
independent confirmation. No model, projection, or gain curves are refitted.
"""
import argparse
import itertools
import json
import statistics
from fit_canon4 import load, gate_group
from gate_recalibrate import interp


MIN_GAINS = (.25, .5, .75, 1., 1.5, 2., 3., 4., 6., 10.)
QUALITY_MARGINS = (0., 2., 4., 6., 8., 10.)
DISTANCE_MARGINS = (0., .1, .25, .5, 1., 2., 6.)
BUDGETS = (0., .005, .01, .02, .05)


def prediction(row, fit, coefficient=True):
    content = ('graphic' if row['z'] >= fit['threshold'] else 'photo') if coefficient else 'pooled'
    value = interp(fit['curves'][f'{row["axis"]}_{row["ss"]}_{content}'], row['x'])
    if row['encoder'] == 'zenjpeg:noaq':
        value += interp(fit['aq_deltas'][row['ss']], row['x'])
    return value


def restore(row, predicted, gates, policy):
    if policy['skip_all']:
        return False
    distance = row['axis'] == 'distance'
    boundary = gates[gate_group(row)] + (policy['distance_margin'] if distance else -policy['quality_margin'])
    gated = row['x'] <= boundary if distance else row['x'] >= boundary
    return predicted >= policy['min_gain'] and not gated


def policy(min_gain=.25, quality_margin=0., distance_margin=0., skip_all=False):
    return dict(min_gain=min_gain, quality_margin=quality_margin,
                distance_margin=distance_margin, skip_all=skip_all)


def harm_limit(identity_ssim2, mode='curve'):
    """Evaluation-only tolerance; the deployment router never sees the reference."""
    if mode == 'legacy':
        return .1
    if mode == 'flat05':
        return .5
    if mode != 'curve':
        raise ValueError(f'unknown harm definition {mode}')
    return round(.5 - .04 * min(10., max(0., float(identity_ssim2) - 90.)), 12)


def summarize(rows, decisions, harm_mode='curve'):
    values = [r['gain'] if yes else 0. for r, yes in zip(rows, decisions, strict=True)]
    n = len(values)
    if not n:
        raise ValueError('empty evaluation stratum')
    restored = sum(decisions)
    harmed = sum(v < -harm_limit(r['identity_ssim2'], harm_mode)
                 for r, v in zip(rows, values, strict=True))
    harmed_images = {r['file'] for r, v in zip(rows, values, strict=True)
                     if v < -harm_limit(r['identity_ssim2'], harm_mode)}
    return dict(n=n, images=len({r['file'] for r in rows}), harmed_images=len(harmed_images), mean=statistics.fmean(values),
                restored=restored/n, harmed=harmed/n, harmed_count=harmed,
                harmed_among_restored=harmed/restored if restored else None,
                worst=min(values), total_loss=-sum(min(v, 0.) for v in values)/n)


def score(rows, predictions, gates, choice, harm_mode='curve'):
    return summarize(rows, [restore(r, p, gates, choice) for r, p in zip(rows, predictions, strict=True)], harm_mode)


def select(candidates, budget):
    eligible = [(p, s) for p, s in candidates if s['harmed'] <= budget]
    # Stable tie break: fewer harmful cases, then less work. Skip-all is always available.
    return max(eligible, key=lambda ps: (ps[1]['mean'], -ps[1]['harmed'], -ps[1]['restored']))


def frontier(rows, fit, coefficient=True, harm_mode='curve'):
    cal = [r for r in rows if r['cal']]
    val = [r for r in rows if not r['cal']]
    cp = [prediction(r, fit, coefficient) for r in cal]
    vp = [prediction(r, fit, coefficient) for r in val]
    choices = [policy(*args) for args in itertools.product(MIN_GAINS, QUALITY_MARGINS, DISTANCE_MARGINS)]
    choices.append(policy(skip_all=True))
    candidates = [(p, score(cal, cp, fit['gates'], p, harm_mode)) for p in choices]
    results = []
    for budget in BUDGETS:
        choice, calibration = select(candidates, budget)
        validation = score(val, vp, fit['gates'], choice, harm_mode)
        groups = {}
        for enc in sorted({r['encoder'] for r in val}):
            indices = [i for i, r in enumerate(val) if r['encoder'] == enc]
            groups[enc] = score([val[i] for i in indices], [vp[i] for i in indices], fit['gates'], choice, harm_mode)
        results.append(dict(budget=budget, policy=choice, calibration=calibration,
                            validation=validation, validation_by_encoder=groups))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', nargs='+', required=True)
    ap.add_argument('--metadata', nargs='+', required=True)
    ap.add_argument('--fit', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = load(a.scores, a.metadata)
    with open(a.fit) as f:
        fit = json.load(f)
    result = {'coefficient': frontier(rows, fit), 'pooled': frontier(rows, fit, False)}
    result['sensitivity'] = {mode: frontier(rows, fit, harm_mode=mode) for mode in ('legacy', 'flat05')}
    val = [r for r in rows if not r['cal']]
    vp = [prediction(r, fit) for r in val]
    result['current'] = {name: score(val, vp, fit['gates'], p) for name, p in
                         [('fidelity', policy()), ('conservative', policy(.25, 6., 6.))]}
    result['harm_definition'] = 'Loss >0.5 at baseline SSIM2 <=90; linear taper to >0.1 at 100; clamped outside [90,100]. Baseline reference score is evaluation-only.'
    result['limitations'] = 'Empirical cell harm, correlated encodes; 33 fit/30 reused audit images. Not a population risk guarantee. Frozen curves previously selected on calibration images only.'
    with open(a.out, 'w') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    for kind in ('coefficient', 'pooled'):
        for r in result[kind]:
            print(kind, r['budget'], r['policy'], 'cal', r['calibration'], 'validation', r['validation'])


if __name__ == '__main__':
    main()
