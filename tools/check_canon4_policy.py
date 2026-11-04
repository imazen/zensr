#!/usr/bin/env python3
"""Check the Rust metadata-only router against the frozen Python calibration."""
import argparse
import json
from fit_canon4 import load, gated
from gate_recalibrate import interp
from routing_frontier import MIN_GAINS


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scores',nargs='+',required=True)
    ap.add_argument('--runtime',required=True)
    ap.add_argument('--fit',required=True)
    a=ap.parse_args()
    data=load(a.scores,[a.runtime])
    fit=json.load(open(a.fit)); max_error=0.; decisions=0
    for r in data:
        cls='graphic' if r['z']>=fit['threshold'] else 'photo'
        predicted=interp(fit['curves'][f'{r["axis"]}_{r["ss"]}_{cls}'],r['x'])
        if r['encoder']=='zenjpeg:noaq':predicted+=interp(fit['aq_deltas'][r['ss']],r['x'])
        actual=float(r['estimated_gain']); error=abs(actual-predicted)
        max_error=max(max_error,error)
        if error>1e-4:raise ValueError(f'estimate mismatch: {r["file"]} {r["encoder"]} {r["ss"]} {r["q"]}: {actual} vs {predicted}')
        checks = [(0., r['gate']), (6., r['conservative_gate'])]
        if 'margin_gates' in r:
            checks.extend((float(m), value) for m, value in
                          (entry.split(':') for entry in r['margin_gates'].split(',')))
        for margin,gate_value in checks:
            gates={k:v+(margin if k.startswith('distance') else -margin) for k,v in fit['gates'].items()}
            expected_gate=gated(r,gates)
            actual_gate=gate_value=='true'
            if expected_gate!=actual_gate:raise ValueError(f'gate mismatch: {r} margin={margin}')
            for min_gain in MIN_GAINS:
                if (predicted>=min_gain and not expected_gate)!=(actual>=min_gain and not actual_gate):
                    raise ValueError(f'decision mismatch: {r} min_gain={min_gain}')
                decisions+=1
    print(f'{len(data)} identity/probe joins verified; {decisions} routing decisions agree; max estimate error {max_error:.9g}')


if __name__=='__main__': main()
