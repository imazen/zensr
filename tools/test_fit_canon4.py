"""Calibration integrity tests: held-out files cannot change fitted rules."""
import copy
import unittest
from fit_canon4 import fit, fit_gates, isotonic


class CalibrationIntegrity(unittest.TestCase):
    def test_validation_gains_cannot_change_fit(self):
        rows=[]
        for axis,encoder in [('quality','turbo'),('distance','jpegli')]:
            for ss in ('420','444'):
                for x in [1.,2.,3.]:
                    for z in [0.,1.]:
                        for cal in [False,True]:
                            rows.append(dict(axis=axis,encoder=encoder,ss=ss,x=x,z=z,
                                             cal=cal,gain=x if axis=='distance' else -x,
                                             routing_family='CjpegliYcbcr' if axis=='distance' else 'ImageMagick'))
        changed=copy.deepcopy(rows)
        for r in changed:
            if not r['cal']: r['gain']=1000000.
        self.assertEqual(fit(rows,.2),fit(changed,.2))
        self.assertEqual(fit_gates(rows),fit_gates(changed))

    def test_isotonic_preserves_monotone_measurements(self):
        points=[(1.,1.,3),(2.,2.,4),(3.,3.,2)]
        self.assertEqual(isotonic(points,True),[(1.,1.),(2.,2.),(3.,3.)])

    def test_isotonic_pools_only_violating_neighbors(self):
        self.assertEqual(isotonic([(1.,3.,1),(2.,1.,1),(3.,4.,1)],True),
                         [(1.,2.),(2.,2.),(3.,4.)])

if __name__=='__main__': unittest.main()
