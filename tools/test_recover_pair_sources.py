"""Exactness and RNG failures must not turn into asserted source identities."""
import random
import unittest

import numpy as np

from recover_pair_sources import replay_crops, verify_crop, verify_recipe


class RecoveryTests(unittest.TestCase):
    def test_exact_crop_and_coordinates(self):
        image = np.random.default_rng(3).integers(0, 256, (21, 23, 3), dtype=np.uint8)
        rows = list(replay_crops(['a'], 8, random.Random(11), 8, lambda _: image))
        self.assertEqual(len(rows), 8)
        for path, x, y, rgb, _ in rows:
            self.assertEqual(path, 'a')
            np.testing.assert_array_equal(rgb, image[y:y + 8, x:x + 8, ::-1])

    def test_missing_and_tiny_sources_do_not_draw_coordinates(self):
        image = np.random.default_rng(5).integers(0, 256, (17, 19, 3), dtype=np.uint8)
        sources = {'missing': None, 'tiny': image[:3, :3], 'good': image}
        a = list(replay_crops(list(sources), 6, random.Random(7), 8, sources.get))
        b = list(replay_crops(['good'], 6, random.Random(7), 8, sources.get))
        self.assertEqual([(r[1], r[2]) for r in a], [(r[1], r[2]) for r in b])

    def test_all_unreadable_is_a_failure(self):
        with self.assertRaisesRegex(ValueError, 'no readable'):
            list(replay_crops(['missing'], 1, random.Random(7), 8, lambda _: None))

    def test_one_bit_error_rejected(self):
        target = np.zeros((8, 8, 3), dtype=np.uint8)
        changed = target.copy()
        changed[7, 7, 2] = 1
        with self.assertRaisesRegex(ValueError, 'pair 17'):
            verify_crop(changed, target, 17)

    def test_saved_recipe_checks_quality_clean_and_encoder(self):
        meta = dict(encoders=['turbo'], gen2=0, gen3=0)
        # Frozen values from the historical draw order at seed 1.
        row = dict(encoder='turbo', ss='420', q='68', clean='0', gens='1')
        verify_recipe([row], meta, random.Random(1))
        for key, bad in [('encoder', 'zenjpeg'), ('q', '67'), ('clean', '1')]:
            with self.assertRaisesRegex(ValueError, 'recipe replay'):
                verify_recipe([{**row, key: bad}], meta, random.Random(1))


if __name__ == '__main__':
    unittest.main()
