"""Destructive retention keeps source inputs, scores and unrelated files intact."""
import csv
import hashlib
from pathlib import Path
import tempfile
import unittest

import publication_routing_retention as retention


class ScoredOutputRetention(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'study'
        self.root.mkdir()
        (self.root / 'cells').mkdir()
        self.output = self.root / 'retention-plan'
        self.row = dict(file='source', size='32', encoder='zenjpeg', ss='444', nominal_q='5')
        for arm in ['identity', 'restored']:
            for score in ['ssim2', 'butter_max', 'butter_pnorm3']:
                self.row[arm + '_' + score] = '70'
        for column in retention.KEEP + retention.OUTPUTS:
            suffix = '.jpg' if column == 'jpeg_artifact' else (
                '.butter-f32le.zst' if column.endswith('_butter_map') else '.rgb8.zst')
            data = column.encode()
            digest = hashlib.sha256(data).hexdigest()
            value = f'artifacts/{digest[:2]}/{digest}{suffix}'
            path = self.root / value
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            self.row[column] = value
        self.row['decoded_artifact'] = self.row['reference_artifact']
        self.write_cell()
        with (self.root / 'COVERAGE.tsv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=['file', 'size', 'encoder', 'ss', 'q', 'status'], delimiter='\t')
            writer.writeheader()
            writer.writerow(dict(file='source', size='32', encoder='zenjpeg', ss='444', q='5', status='required'))
        for name in retention.METADATA:
            if not (self.root / name).exists():
                (self.root / name).write_text('frozen identity\n')
        (self.root / 'AUDITED.json').write_text('{"historical": true}\n')

    def write_cell(self):
        with (self.root / 'cells' / 'a.tsv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.row), delimiter='\t')
            writer.writeheader()
            writer.writerow(self.row)

    def plan(self):
        return retention.plan(self.root, self.root, self.output)

    def test_shared_reference_inputs_and_scores_survive_and_apply_is_idempotent(self):
        before = {p: p.read_bytes() for p in [self.root / self.row[c] for c in retention.KEEP]}
        before[self.root / 'cells' / 'a.tsv'] = (self.root / 'cells' / 'a.tsv').read_bytes()
        orphan = self.root / 'artifacts' / 'unreferenced.rgb8.zst'
        orphan.write_bytes(b'unreferenced data remains protected')
        result = self.plan()
        self.assertEqual(result['output_files'], 3)
        done = retention.apply(self.root, self.output)
        self.assertEqual(done['removed_this_attempt'], 3)
        for p, data in before.items():
            self.assertEqual(p.read_bytes(), data)
        self.assertTrue(orphan.exists())
        self.assertFalse((self.root / 'AUDITED.json').exists())
        self.assertEqual(len(list((self.root / 'audit-history').glob('*.json'))), 1)
        resumed = retention.apply(self.root, self.output)
        self.assertEqual(resumed['removed_this_attempt'], 0)
        self.assertEqual(resumed['absent_this_attempt'], 3)

    def test_incomplete_run_cannot_be_planned(self):
        (self.root / 'COMPLETE').unlink()
        with self.assertRaisesRegex(ValueError, 'completed grid'):
            self.plan()

    def test_missing_input_nonfinite_score_and_incomplete_coverage_are_rejected(self):
        for change in ['input', 'score', 'coverage']:
            with self.subTest(change=change):
                old = self.row.copy()
                if change == 'input':
                    self.row['jpeg_artifact'] = 'missing.jpg'
                elif change == 'score':
                    self.row['identity_ssim2'] = 'NaN'
                else:
                    self.row['nominal_q'] = '10'
                self.write_cell()
                with self.assertRaises(ValueError):
                    self.plan()
                self.row = old
        self.assertFalse(self.output.exists())

    def test_traversal_and_symlink_output_paths_are_rejected(self):
        original = self.row['restored_artifact']
        self.row['restored_artifact'] = '../outside.rgb8.zst'
        self.write_cell()
        with self.assertRaisesRegex(ValueError, 'invalid output'):
            self.plan()
        self.row['restored_artifact'] = original
        self.write_cell()
        target = self.root / original
        target.unlink()
        target.symlink_to(self.root / self.row['reference_artifact'])
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.plan()

    def test_changed_output_aborts_before_any_deletion(self):
        self.plan()
        target = self.root / self.row['restored_artifact']
        target.write_bytes(b'changed after plan')
        with self.assertRaisesRegex(ValueError, 'output changed'):
            retention.apply(self.root, self.output)
        self.assertTrue((self.root / self.row['identity_butter_map']).exists())
        self.assertTrue((self.root / 'AUDITED.json').exists())

    def test_manifest_tampering_and_different_root_are_rejected(self):
        self.plan()
        other = self.root.parent / 'other'
        other.mkdir()
        with self.assertRaisesRegex(ValueError, 'different run'):
            retention.apply(other, self.output)
        with (self.output / 'outputs.jsonl').open('a') as f:
            f.write('{}\n')
        with self.assertRaisesRegex(ValueError, 'plan changed'):
            retention.apply(self.root, self.output)
        self.assertTrue((self.root / self.row['restored_artifact']).exists())


if __name__ == '__main__':
    unittest.main()
