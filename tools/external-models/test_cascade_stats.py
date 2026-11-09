"""Paired repair effects must retain skips, tail losses and exact boundaries."""
import unittest
from cascade_stats import paired


def cell(key, baseline, repaired, butter='1', repaired_butter='1'):
    return dict(row=dict(id=key),direct=dict(ssim2=baseline,butter_pnorm3=butter),
        repaired=dict(ssim2=repaired,butter_pnorm3=repaired_butter))


class CascadeStatsTests(unittest.TestCase):
    def test_skips_and_changed_subset_have_different_denominators(self):
        rows=[cell('skip','80','80'),cell('better','80','82'),cell('loss','80','79')]
        full=paired(rows,'repaired','direct');changed=paired(rows[1:],'repaired','direct')
        self.assertEqual(full['n'],3);self.assertEqual(full['ties'],1)
        self.assertEqual(changed['n'],2);self.assertEqual(changed['ties'],0)
        self.assertEqual(full['worst_gain'],'-1')

    def test_half_point_is_not_harm_and_mean_keeps_pairing(self):
        result=paired([cell('boundary','80','79.5'),cell('over','80','79.49999999999999999')],'repaired','direct')
        self.assertEqual(result['harm'],1)
        self.assertEqual(result['median_gain'],'-0.500000000000000005')

    def test_butter_direction_and_tail(self):
        result=paired([cell('gain','80','82','3','2'),cell('loss','80','79','1','4')],'repaired','direct')
        self.assertEqual(result['butter_wins'],1);self.assertEqual(result['butter_worse'],1)
        self.assertEqual(result['worst_butter_delta'],'3')

    def test_no_restored_cells_is_explicit(self):
        result=paired([],'repaired','direct')
        self.assertEqual(result['n'],0);self.assertIsNone(result['mean_gain']);self.assertIsNone(result['harm_percent'])
