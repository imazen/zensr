"""Boundary regressions for the published paired external-model summaries."""
import unittest
from summarize import band, summary


def row(baseline, score, key='case'):
    return dict(id=key,baseline=baseline,score=score,butter='1',baseline_butter='1')


class SummaryTests(unittest.TestCase):
    def test_half_point_boundary_is_exact(self):
        result=summary([row('80','79.5'),row('80','79.49999999999999999')])
        self.assertEqual(result['harm'],1)
        self.assertEqual(result['worse'],2)

    def test_high_quality_curve(self):
        result=summary([row('95','94.7'),row('95','94.69999999999999999')])
        self.assertEqual(result['harm'],1)

    def test_pristine_tenth_boundary(self):
        result=summary([row('100','99.9'),row('100','99.89999999999999999')])
        self.assertEqual(result['harm'],1)

    def test_tail_is_not_hidden_by_mean(self):
        result=summary([row('50','60','gain'),row('50','45','loss')])
        self.assertEqual(result['mean_gain'],'2.5')
        self.assertEqual(result['worst_gain'],'-5')
        self.assertEqual(result['worst_id'],'loss')
        self.assertEqual(result['harm'],1)

    def test_quality_bands_cover_both_ends(self):
        self.assertEqual([band(q) for q in [0,5,20,25,60,65,100]],
                         ['clean','05-20','05-20','25-60','25-60','65-100','65-100'])


if __name__=='__main__':unittest.main()
