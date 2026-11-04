#!/usr/bin/env python3
"""Check the Rust metadata-only router against the frozen Python calibration."""
import argparse
import json
from fit_canon4 import load, gated
from gate_recalibrate import interp


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
        for margin,col in [(0.,'gate'),(6.,'conservative_gate')]:
            gates={k:v+(margin if k.startswith('distance') else -margin) for k,v in fit['gates'].items()}
            expected_gate=gated(r,gates)
            actual_gate=r[col]=='true'
            if expected_gate!=actual_gate:raise ValueError(f'gate mismatch: {r} margin={margin}')
            if (predicted>=.25 and not expected_gate)!=(actual>=.25 and not actual_gate):
                raise ValueError(f'decision mismatch: {r}')
            decisions+=1
    print(f'{len(data)} identity/probe joins verified; {decisions} routing decisions agree; max estimate error {max_error:.9g}')


if __name__=='__main__': main()
