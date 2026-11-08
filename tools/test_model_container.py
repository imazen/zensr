"""Cross-language container fixtures use the committed canon4 source artifact."""
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import model_container as m

MODEL = m.ROOT / "models/adopted/dejpeg_rt24i_canon4"
SCRATCH = Path(os.environ.get("TMPDIR", Path.home() / "tmp" / "zensr-container-tests"))
SCRATCH.mkdir(parents=True, exist_ok=True)


class ContainerTests(unittest.TestCase):
    def test_deterministic_and_weight_preserving(self):
        for policy in ("manual", "canon4", "canon4-coefficients"):
            data = m.pack(MODEL, policy, "realtime")
            self.assertEqual(data, m.pack(MODEL, policy, "realtime"))
            info = m.inspect_bytes(data)
            self.assertEqual(info["weights_sha256"], m.CANON4)
            self.assertEqual(info["weights_bytes"], 86406)
            self.assertEqual(data[-86406:], (MODEL / "weights_f16.raw").read_bytes())
            self.assertEqual(info["name"], MODEL.name)

    def test_weight_binding_and_architecture(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as scratch:
            path = Path(scratch)
            meta = json.loads((MODEL / "meta.json").read_text())
            (path / "meta.json").write_text(json.dumps(meta))
            weights = bytearray((MODEL / "weights_f16.raw").read_bytes())
            weights[0] ^= 1
            (path / "weights_f16.raw").write_bytes(weights)
            with self.assertRaisesRegex(ValueError, "exact canon4"):
                m.pack(path, "canon4", "realtime")
            m.pack(path, "manual", "quality")
            weights[-2:] = struct.pack("<H", 0x7e00)
            (path / "weights_f16.raw").write_bytes(weights)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                m.pack(path, "manual", "quality")
            meta["nf"] = 64
            (path / "meta.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "geometry"):
                m.pack(path, "manual", "quality")

    def test_f32_export_convention_and_span_layout(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as scratch:
            path = Path(scratch)
            meta = dict(arch="compact", nf=4, nc=0, scale=1, space="rgb", f16_goldens=False)
            (path / "meta.json").write_text(json.dumps(meta))
            (path / "weights.raw").write_bytes(struct.pack("<227f", *([0.0] * 227)))
            info = m.inspect_bytes(m.pack(path, "manual", "quality"))
            self.assertEqual(info["encoding"], "f32")
            self.assertEqual(info["weights_bytes"], 908)
            meta.update(arch="span48", nf=48, nc=6, scale=2)
            count = 28*48 + 19*(9*48*48+48) + 4*48*48+48 + 27*48*4+3*4
            (path / "meta.json").write_text(json.dumps(meta))
            (path / "weights.raw").write_bytes(bytes(count*4))
            info = m.inspect_bytes(m.pack(path, "manual", "quality"))
            self.assertEqual((info["architecture"], info["scale"]), ("span48",2))

    def test_metadata_is_part_of_calibration_identity(self):
        data = m.pack(MODEL, "canon4", "realtime")
        _, _, meta_size, policy_size, _ = struct.unpack_from("<8sIIQQ", data)
        meta = data[64:64+meta_size]
        policy = data[64+meta_size:64+meta_size+policy_size]
        self.assertEqual(policy[62:94], hashlib.sha256(meta+data[-86406:]).digest())

    def test_checksum_and_truncation(self):
        data = m.pack(MODEL, "canon4", "realtime")
        for n in (0, 1, 63, 64, len(data)-1):
            with self.assertRaises(ValueError):
                m.inspect_bytes(data[:n])
        data = bytearray(data)
        data[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            m.inspect_bytes(data)

    def test_missing_source_constant_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            m.numbers("", "MEAN")
        with patch.object(m, "CANON4", "wrong"):
            with self.assertRaisesRegex(ValueError, "exact canon4"):
                m.pack(MODEL, "canon4", "realtime")


if __name__ == "__main__":
    unittest.main()
