"""Public checkpoint acquisition must work from a fresh root and reject drift."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from acquire_sr import model_metadata


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        scratch=Path.home()/'tmp'
        scratch.mkdir(exist_ok=True)
        self.directory=tempfile.TemporaryDirectory(dir=scratch,prefix='zensr-acquire-test-')
        self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)

    def test_fresh_root_uses_pinned_revision_and_reuses_record(self):
        data=dict(sha='fixed-revision',siblings=[])
        with patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(data).encode())) as request:
            self.assertEqual(model_metadata(self.root,'author/model','fixed-revision'),data)
            self.assertEqual(model_metadata(self.root,'author/model','fixed-revision'),data)
        request.assert_called_once_with('https://huggingface.co/api/models/author/model/revision/fixed-revision?blobs=true',timeout=60)

    def test_changed_remote_revision_is_not_persisted(self):
        with patch('urllib.request.urlopen',return_value=io.BytesIO(b'{"sha":"other"}')):
            with self.assertRaisesRegex(ValueError,'revision differs'):
                model_metadata(self.root,'author/model','fixed-revision')
        self.assertEqual(list(self.root.iterdir()),[])

    def test_changed_existing_revision_is_rejected(self):
        path=self.root/'author-model-metadata.json'
        path.write_text('{"sha":"other"}')
        with patch('urllib.request.urlopen') as request:
            with self.assertRaisesRegex(ValueError,'revision differs'):
                model_metadata(self.root,'author/model','fixed-revision')
        request.assert_not_called()


if __name__=='__main__':unittest.main()
