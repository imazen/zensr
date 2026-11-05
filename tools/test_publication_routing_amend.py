import csv
import hashlib
from pathlib import Path
import tempfile
import unittest

from publication_routing_amend import finalize


class AmendmentTests(unittest.TestCase):
    def test_explicit_exclusion_and_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory); root=p/'original'; recovery=p/'recovery'
            for r in (root,recovery):
                r.mkdir(); (r/'cells').mkdir()
                for name in ('FEATURES.tsv','ENCODERS.tsv'):(r/name).write_text('frozen\n')
            fields=['reference_sha256','file','role','width','height']
            refs=[dict(zip(fields,values)) for values in [('a','large','locked_test','16320','7612'),('b','small','locked_test','512','512')]]
            manifests=[]
            for name,rows in [('manifest.tsv',refs),('subset.tsv',refs[1:])]:
                m=p/name
                with m.open('w') as f:
                    w=csv.DictWriter(f,fields,delimiter='\t');w.writeheader();w.writerows(rows)
                manifests.append(m)
            for r,m in zip((root,recovery),manifests):
                (r/'IDENTITY.tsv').write_text('binary_sha256\tfrozen\nmanifest_sha256\t'+hashlib.sha256(m.read_bytes()).hexdigest()+'\n')
            coverage='file\tsize\tencoder\tss\tq\tstatus\nlarge\t512\tturbo\t420\t50\trequired\nsmall\t512\tturbo\t420\t50\trequired\n'
            (root/'COVERAGE.tsv').write_text(coverage)
            (recovery/'COMPLETE').write_text('1/1 required cells completed\n')
            args=(root,recovery,*manifests,'a','owner-approved oversized input exclusion')
            with self.assertRaisesRegex(ValueError,'incomplete'):finalize(*args)
            self.assertFalse((root/'COMPLETE').exists())
            data='reference_sha256\tfile\trole\tsize\tencoder\tss\tnominal_q\nb\tsmall\tlocked_test\t512\tturbo\t420\t50\n'
            (recovery/'cells'/'cell.tsv').write_text(data)
            saved=(recovery/'IDENTITY.tsv').read_text()
            (recovery/'IDENTITY.tsv').write_text(saved.replace('frozen','changed'))
            with self.assertRaisesRegex(ValueError,'scientific inputs'):finalize(*args)
            (recovery/'IDENTITY.tsv').write_text(saved)
            finalize(*args)
            self.assertEqual((root/'cells'/'cell.tsv').read_text(),data)
            self.assertEqual((root/'reference-exclusion-history'/'COVERAGE.tsv').read_text(),coverage)
            self.assertEqual((root/'COMPLETE').read_text(),'1/1 required cells completed\n')


if __name__=='__main__':unittest.main()
