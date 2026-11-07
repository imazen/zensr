import json
from pathlib import Path
import tempfile
import unittest

from model_release import build, portable_metadata


class ModelReleaseTests(unittest.TestCase):
    def test_metadata_keeps_architecture_without_local_commands(self):
        meta=dict(arch='compact',source='public-model.pth',repro=dict(host='internal',
                  dataset_dir='/home/example/private',env={'TOKEN':'secret'},seed_torch=7,commit='abcd'))
        result=portable_metadata(meta)
        self.assertEqual(result['source'],'public-model.pth')
        text=json.dumps(result)
        for value in ['internal','/home/','TOKEN','secret']:self.assertNotIn(value,text)

    def test_binary_capture_is_complete_and_deduplicated(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as td:
            root=Path(td);inputs=root/'inputs'
            for host in ['a','b']:
                model=inputs/host/'models/adopted/example';model.mkdir(parents=True)
                (model/'weights_f16.raw').write_bytes(b'unchanged-binary')
                (model/'meta.json').write_text(json.dumps(dict(arch='compact',nf=24,nc=8)))
                (model/'repro.sh').write_text('local command')
            build(inputs,root/'output')
            summary=json.loads((root/'output/summary.json').read_text())
            self.assertEqual(summary['adopted_model_names'],1)
            self.assertEqual(summary['captured_binary_files'],2)
            self.assertEqual(summary['unique_binary_objects'],1)
            self.assertEqual(summary['original_text_files_not_uploaded'],4)


if __name__=='__main__':unittest.main()
