import hashlib
import tempfile
import unittest
from pathlib import Path

from publication_routing_scores_only import read_score_tables


class ScoresOnlyTests(unittest.TestCase):
    def test_complete_scores_require_coverage_and_role_without_audit_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'cells').mkdir()
            (root/'COMPLETE').write_text('1/1 required cells completed\n')
            (root/'COVERAGE.tsv').write_text('file\tsize\tencoder\tss\tq\tstatus\na\t512\tturbo\t420\t50\trequired\n')
            with self.assertRaisesRegex(ValueError,'empty'):read_score_tables(root,'locked_test')
            data=b'file\treference_sha256\trole\tsize\tencoder\tss\tnominal_q\na\th\tlocked_test\t512\tturbo\t420\t50\n'
            (root/'cells'/'a.tsv').write_bytes(data)
            rows,digest=read_score_tables(root,'locked_test')
            self.assertEqual(len(rows),1)
            self.assertEqual(digest,hashlib.sha256(b'a.tsv'+data).hexdigest())
            self.assertFalse((root/'AUDITED.json').exists())
            with self.assertRaisesRegex(ValueError,'role'):read_score_tables(root,'router_development')
            (root/'COVERAGE.tsv').write_text((root/'COVERAGE.tsv').read_text().replace('\t50\t','\t51\t'))
            with self.assertRaisesRegex(ValueError,'coverage'):read_score_tables(root,'locked_test')


if __name__=='__main__':unittest.main()
