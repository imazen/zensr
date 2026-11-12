"""Split admission controls, independent of GPU/model availability."""
import csv
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('sr_bank', Path(__file__).with_name('bank.py'))
bank = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bank)


class Splits(unittest.TestCase):
    def test_verifier_cannot_be_shadowed_by_another_report_module(self):
        import sys
        with patch.dict(sys.modules, {'report': SimpleNamespace(validate=lambda *_: None)}):
            validator = bank.record_validator()
            with self.assertRaises(KeyError):
                validator({}, {}, {}, 'missing')

    def test_export_ids_and_disagreeing_roles_cannot_enter_training(self):
        scratch = Path.home() / 'tmp'
        scratch.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=scratch, prefix='sr-bank-split-') as d:
            root = Path(d)
            (root / 'manifests').mkdir()
            members = [('1000', 'train'), ('1001', 'validate'), ('1002', 'train'), ('1007', 'test')]
            for split in ('train', 'validate', 'test'):
                with (root / 'manifests' / (split + '.tsv')).open('w') as f:
                    w = csv.DictWriter(f, ['id', 'split', 'content_class', 'path'], delimiter='\t')
                    w.writeheader()
                    for rid, role in members:
                        if role == split:
                            w.writerow(dict(id=rid, split=role, content_class='photos', path=rid + '.jpg'))
            with (root / 'manifests/split_map_family_v2.tsv').open('w') as f:
                w = csv.DictWriter(f, ['id', 'split', 'family'], delimiter='\t')
                w.writeheader()
                for rid, role in members:
                    w.writerow(dict(id=rid, split='validate' if rid in ('1002', '1007') else role, family=''))
            bank.save(root / 'SOURCE_INVENTORY.json', dict(archives=[dict(split='train', path='train', entries=1, sha256='a'), dict(split='validate', path='validate', entries=3, sha256='b')]))

            class Worker:
                def __init__(self, *_): pass
                def close(self): pass
                def send(self, action, archive):
                    ids = ['11000'] if archive == 'train' else ['11001', '11002', '11007']
                    return dict(entries=[dict(key=int(rid) * 10, width=400, height=300, channels=3, sample=0) for rid in ids])
            args = SimpleNamespace(output=root, corpus=root, binary=Path('unused'), corpus_revision='frozen', train_per_category=8, eval_per_category=4)
            with patch.object(bank, 'Worker', Worker), patch.object(bank, 'local_space'):
                bank.freeze(args)
            plan = json.loads((root / 'PLAN.json').read_text())
            self.assertEqual([(r['id'], r['export_id'], r['split']) for r in plan['sources']], [('1000', '11000', 'train'), ('1001', '11001', 'validate')])
            self.assertEqual({r['id'] for r in plan['exclusions']}, {'1002', '1007'})


if __name__ == '__main__':
    unittest.main()
