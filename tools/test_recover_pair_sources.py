"""Exactness and RNG failures must not turn into asserted source identities."""
import random
from pathlib import Path
import tempfile
import unittest

import numpy as np

from recover_pair_sources import canonical_rows, replay_crops, verify_crop, verify_recipe
from verify_pair_sources import coordinate_crop


class RecoveryTests(unittest.TestCase):
    def test_coordinate_verifier_edges_and_bounds(self):
        image = np.arange(11 * 13 * 3, dtype=np.uint8).reshape(11, 13, 3)
        row = dict(x='5', y='3', width='8', height='8')
        np.testing.assert_array_equal(coordinate_crop(image, row), image[3:11, 5:13, ::-1])
        for bad in [dict(x='6'), dict(y='4'), dict(x='-1'), dict(width='0')]:
            with self.assertRaisesRegex(ValueError, 'coordinate'):
                coordinate_crop(image, {**row, **bad})

    def test_join_small_identity_map_to_hashed_bucket_tables(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as td:
            corpus = Path(td)
            (corpus / 'manifests').mkdir()
            (corpus / 'manifests/split_map.tsv').write_text('id\tsplit\tpath\n1000\ttrain\ta.jpg\n')
            for bucket in ['train', 'validate', 'test']:
                (corpus / 'manifests' / (bucket + '.tsv')).write_text(
                    'id\tsplit\tpath\tsha256\n' +
                    ('1000\ttrain\ta.jpg\t' + 'a' * 64 + '\n' if bucket == 'train' else ''))
            self.assertEqual(canonical_rows(corpus)['a.jpg']['sha256'], 'a' * 64)
            (corpus / 'manifests/split_map.tsv').write_text('id\tsplit\tpath\n1000\ttest\ta.jpg\n')
            with self.assertRaisesRegex(ValueError, 'disagree'):
                canonical_rows(corpus)

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
