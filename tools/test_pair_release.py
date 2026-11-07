import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pair_release import ARRAYS, FIELDS, assemble, build, digest, json_file, split_rows, write_rows


class PairReleaseTests(unittest.TestCase):
    def test_roundtrip_canonical_selection_and_original_indices(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as td:
            root = Path(td)
            dataset, manifest, output = (root / n for n in ['data', 'manifest', 'release'])
            dataset.mkdir()
            manifest.mkdir()
            images = np.arange(6 * 4 * 4 * 3, dtype=np.uint8).reshape(6, 4, 4, 3)
            for name in ARRAYS:
                np.save(dataset / name, images if name in ['lr_u8.npy', 'hr_u8.npy'] else np.arange(6))
            meta = dict(n=6, val_tail=2, crop=4, seed=7)
            json_file(dataset / 'meta.json', meta)
            rows = []
            for idx, (source, bucket, partition) in enumerate([
                ('1000', 'train', 'model_train'), ('1000', 'train', 'model_train'),
                ('1002', 'train', 'model_train'), ('1001', 'validate', 'model_train'),
                ('1004', 'train', 'internal_validation'), ('1007', 'test', 'internal_validation')]):
                sha = hashlib.sha256(images[idx].tobytes()).hexdigest()
                rows.append(dict(idx=idx, imazen_id=source, canonical_split=bucket, partition=partition,
                                 hr_rgb_sha256=sha, lr_rgb_sha256=sha, source_sha256='a' * 64,
                                 encoder='turbo', ss='420', q=30, clean=1, severity=0, gens=1))
            write_rows(manifest / 'pairs.tsv', rows, FIELDS)
            write_rows(dataset / 'pairs.tsv', rows, FIELDS)
            src = [dict(imazen_id=r['imazen_id'], license='PD') for r in rows]
            write_rows(manifest / 'sources.tsv', src, ['imazen_id', 'license'])
            write_rows(manifest / 'sources-with-links.tsv', src, ['imazen_id', 'license'])
            summary = dict(dataset_meta=meta, outputs={n:digest(manifest/n) for n in ['pairs.tsv','sources.tsv']},
                           dataset_inputs={p.name:dict(sha256=digest(p)) for p in dataset.iterdir()})
            json_file(manifest / 'summary.json', summary)
            json_file(manifest / 'verification.json', dict(status='complete',
                reconstruction_summary_sha256=digest(manifest/'summary.json'),
                sources_with_links_sha256=digest(manifest/'sources-with-links.tsv')))
            build(dataset, manifest, output)
            assembled = root / 'assembled'
            assemble(output / 'bundle-index.json', assembled, 'canonical-train')
            np.testing.assert_array_equal(np.load(assembled/'hr_u8.npy'), images[[0,1,2,4]])
            self.assertEqual(json.loads((assembled/'meta.json').read_text())['val_tail'], 1)
            with (assembled/'pairs.tsv').open() as f: recovered=list(csv.DictReader(f,delimiter='\t'))
            self.assertEqual([int(r['original_idx']) for r in recovered], [0,1,2,4])
            self.assertTrue(all(r['canonical_split']=='train' for r in recovered))
            historical = root / 'historical'
            assemble(output/'bundle-index.json', historical, 'historical')
            np.testing.assert_array_equal(np.load(historical/'hr_u8.npy'), images)
            archive=output/(json.loads((output/'bundle-index.json').read_text())['bundles'][0]['name']+'.tar.zst')
            with archive.open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'archive hash'):
                assemble(output/'bundle-index.json',root/'corrupt','canonical-train')

    def test_shards_preserve_sources_and_full_inventory(self):
        rows=[dict(idx=i,imazen_id=str(i//3),canonical_split='train' if i<9 else 'test') for i in range(12)]
        a,b,other=split_rows(rows)
        self.assertFalse({r['imazen_id'] for r in a}&{r['imazen_id'] for r in b})
        self.assertEqual([r['idx'] for r in other],[9,10,11])


if __name__ == '__main__':unittest.main()
