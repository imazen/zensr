import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from qat import canon4 as q

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get('TMPDIR', Path.home() / 'tmp' / 'zensr-qat-tests'))
SCRATCH.mkdir(parents=True, exist_ok=True)

class QATTests(unittest.TestCase):
    def baseline(self):
        return q.load_baseline(ROOT / 'models/adopted/dejpeg_rt24i_canon4/weights_f16.raw')

    def test_native_fake_quant_ste_and_output_channel_axis(self):
        w = torch.tensor([[[[0.125, 0.5, 1.]]], [[[0.01, 0.1, 0.2]]]], requires_grad=True)
        quant, scale = q.quantized(w)
        expected = torch.round(w.detach() / scale[:,None,None,None]).clamp(-127,127) * scale[:,None,None,None]
        torch.testing.assert_close(quant, expected, atol=0, rtol=0)
        quant.sum().backward()
        torch.testing.assert_close(w.grad, torch.ones_like(w), atol=0, rtol=0)
        self.assertNotEqual(scale[0], scale[1])

    def test_baseline_matches_committed_pytorch_golden(self):
        m = self.baseline().eval()
        d = ROOT / 'models/adopted/dejpeg_rt24i_canon4'
        x = torch.from_numpy(np.fromfile(d/'gold_in_17x13.raw', dtype='<f4').reshape(1,3,17,13))
        y = np.fromfile(d/'gold_out_17x13.raw', dtype='<f4').reshape(1,3,17,13)
        with torch.no_grad():
            torch.testing.assert_close(m(x), torch.from_numpy(y), atol=2e-6, rtol=2e-6)

    def test_qat_is_active_and_packed_roundtrip_is_exact(self):
        m = q.enable_qat(self.baseline())
        packed = q.pack(m)
        decoded = q.unpack(packed)
        np.testing.assert_array_equal(decoded.view('u4'), q.flat(m).view('u4'))
        self.assertFalse(np.array_equal(q.flat(m), q.flat(self.baseline())))
        x = torch.rand(1,3,17,13)
        m(x).abs().mean().backward()
        grad = m.body[0].parametrizations.weight.original.grad
        self.assertTrue(torch.isfinite(grad).all() and grad.abs().sum() > 0)
        with tempfile.TemporaryDirectory(dir=SCRATCH) as d:
            q.export(m, Path(d)/'export', True, dict(test_fixture=True))

    def test_packed_bounds_are_checked(self):
        packed = q.pack(self.baseline())
        for bad in (b'',packed[:23],packed[:-1],packed+b'\0'):
            with self.assertRaises(ValueError): q.unpack(bad)
        bad = bytearray(packed); bad[24:28] = np.float32(float('nan')).tobytes()
        with self.assertRaisesRegex(ValueError,'scales'): q.unpack(bad)

if __name__ == '__main__': unittest.main()
