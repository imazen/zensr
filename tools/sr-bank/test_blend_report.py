"""Independent dominance and exact-match controls for the study report."""
import math
import unittest
from blend_report import nondominated, pooled_psnr


class Frontiers(unittest.TestCase):
    def test_dominated_and_equivalent_choices_do_not_expand_frontier(self):
        rows = [dict(zensim_a=a, mse=e, intensity_quarter=4, repair='direct', filter='triangle')
                for a, e in [(100., 4.), (99., 3.), (98., 2.), (97., 5.), (99., 4.), (100., 4.)]]
        expected = [(100., 4.), (99., 3.), (98., 2.)]
        for order in (rows, list(reversed(rows))):
            self.assertEqual([(r['zensim_a'], r['mse']) for r in nondominated(order)], expected)

    def test_exact_matches_stay_exact_and_known_mse_has_known_psnr(self):
        self.assertTrue(math.isinf(pooled_psnr(0)))
        self.assertEqual(pooled_psnr(650.25), 20.)
        row = dict(zensim_a=100., mse=0., intensity_quarter=0, repair='direct', filter='triangle')
        self.assertEqual(nondominated([row]), [row])


if __name__ == '__main__':
    unittest.main()
