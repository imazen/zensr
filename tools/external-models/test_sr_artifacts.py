"""Missing or corrupt retained SR images must fail admission."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from sr_artifacts import validate_png


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        scratch = Path.home()/'tmp'
        scratch.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=scratch, prefix='zensr-image-test-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        data = b'encoded image fixture'
        digest = hashlib.sha256(data).hexdigest()
        self.path = self.root/(digest+'.png')
        self.path.write_bytes(data)
        self.score = dict(png=dict(path=str(self.path), sha256=digest, bytes=len(data),
            width=7, height=5, format='PNG RGB8 sRGB lossless'))

    def test_identity_and_dimensions(self):
        validate_png(self.score, 7, 5, self.root)
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            validate_png(self.score, 8, 5, self.root)

    def test_missing_image_fails(self):
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            validate_png(self.score, 7, 5, self.root)

    def test_changed_image_fails(self):
        self.path.write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            validate_png(self.score, 7, 5, self.root)

    def test_escape_fails(self):
        with self.assertRaisesRegex(ValueError, 'escapes'):
            validate_png(self.score, 7, 5, self.root/'nested')
