import copy
import unittest
from routing_frontier import frontier, policy, restore, select, summarize, harm_limit, prediction


class RoutingRiskTests(unittest.TestCase):
    def test_quality_dependent_harm_boundary(self):
        for baseline, limit in [(50., .5), (89., .5), (90., .5), (95., .3), (100., .1)]:
            self.assertEqual(harm_limit(baseline), limit)
            row = dict(file='a', gain=-limit, identity_ssim2=baseline)
            self.assertEqual(summarize([row], [True])['harmed_count'], 0)
            row['gain'] -= .001
            self.assertEqual(summarize([row], [True])['harmed_count'], 1)
        self.assertEqual(harm_limit(101.), .1)
        self.assertEqual(harm_limit(-10.), .5)

    def test_reference_score_is_not_an_inference_input(self):
        row = dict(axis='quality', ss='420', x=75., z=.5, encoder='turbo',
                   routing_family='ImageMagick', identity_ssim2=70.)
        fit = dict(threshold=.2, curves={'quality_420_graphic': [(0., 2.), (100., 2.)]})
        before = prediction(row, fit)
        row['identity_ssim2'] = 99.
        self.assertEqual(before, prediction(row, fit))
        self.assertTrue(restore(row, before, {'ijg_420': 96.}, policy()))

    def test_budget_selection_prefers_gain_within_limit(self):
        candidates = [(policy(skip_all=True), dict(mean=0., harmed=0., restored=0.)),
                      (policy(1.), dict(mean=2., harmed=.01, restored=.5)),
                      (policy(.25), dict(mean=3., harmed=.02, restored=.8))]
        self.assertTrue(select(candidates, 0.)[0]['skip_all'])
        self.assertEqual(select(candidates, .01)[0]['min_gain'], 1.)

    def test_harm_denominator_includes_skipped_cells(self):
        rows = [dict(file='a', gain=-1., identity_ssim2=80.), dict(file='b', gain=-2., identity_ssim2=80.), dict(file='c', gain=3., identity_ssim2=80.)]
        result = summarize(rows, [True, False, True])
        self.assertEqual(result['harmed'], 1/3)
        self.assertEqual(result['harmed_among_restored'], .5)
        self.assertEqual(result['mean'], 2/3)

    def test_gate_boundaries_and_independent_units(self):
        quality = dict(axis='quality', ss='420', routing_family='Mozjpeg', x=94.)
        distance = dict(axis='distance', ss='444', x=.5)
        gates = dict(mozjpeg_420=96., distance_444=.25)
        choice = policy(.5, 2., .25)
        self.assertFalse(restore(quality, 10., gates, choice))
        self.assertFalse(restore(distance, 10., gates, choice))
        quality['x'] = 93.9
        distance['x'] = .501
        self.assertTrue(restore(quality, .5, gates, choice))
        self.assertTrue(restore(distance, .5, gates, choice))
        self.assertFalse(restore(distance, .499, gates, choice))

    def test_validation_changes_cannot_select_policy(self):
        rows = [dict(file=str(i), cal=cal, axis='quality', ss='420',
                     identity_ssim2=80., routing_family='ImageMagick', encoder='turbo', x=75., z=.5, gain=g)
                for i, (cal, g) in enumerate([(True, 2.), (True, -.2), (False, 1.)])]
        fit = dict(threshold=.2, curves={'quality_420_graphic': [(0., 2.), (100., 2.)]},
                   gates={'ijg_420': 96.}, aq_deltas={})
        changed = copy.deepcopy(rows)
        changed[-1]['gain'] = -1000.
        before, after = frontier(rows, fit), frontier(changed, fit)
        self.assertEqual([r['policy'] for r in before], [r['policy'] for r in after])
        self.assertEqual([r['calibration'] for r in before], [r['calibration'] for r in after])


if __name__ == '__main__':
    unittest.main()
