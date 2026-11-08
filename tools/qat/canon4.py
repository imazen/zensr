"""Canon4 W8A32 QAT primitives and portable per-output-channel weight storage.

Reuse train_people.Student and PyTorch's native fake-quantization/STE operator.
This is weight-only quantization, not an integer convolution implementation.
"""
import csv
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_people as original

BASELINE_SHA = 'c3db9373791270bbe5493d3ee98cd4f62296510fe6931073e2966f433e9a5cf4'
COUNT = 43203


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def model():
    # Pin the original trainer's global scale in this dedicated x1 process.
    original.SCALE = 1
    return original.Student(24, 8)


def load_baseline(path):
    if sha(path) != BASELINE_SHA:
        raise ValueError('requires the exact deployed canon4 f16 weights')
    values = np.fromfile(path, dtype='<f2').astype('<f4')
    return load_flat(values)


def load_flat(values):
    if len(values) != COUNT or not np.isfinite(values).all():
        raise ValueError('invalid canon4 weights')
    result = model()
    off = 0
    with torch.no_grad():
        for parameter in result.parameters():
            n = parameter.numel()
            parameter.copy_(torch.from_numpy(values[off:off+n].reshape(parameter.shape).copy()))
            off += n
    assert off == COUNT
    return result


def quantized(weight):
    scale = weight.detach().abs().amax(dim=(1, 2, 3)).clamp_min(torch.finfo(torch.float32).eps) / 127
    zero = torch.zeros(scale.shape, dtype=torch.int32, device=weight.device)
    return torch.fake_quantize_per_channel_affine(weight, scale, zero, 0, -127, 127), scale


class WeightQAT(nn.Module):
    def forward(self, weight):
        return quantized(weight)[0]


def enable_qat(result):
    for layer in result.body:
        if isinstance(layer, nn.Conv2d):
            parametrize.register_parametrization(layer, 'weight', WeightQAT())
    return result


def layers(result):
    return [layer for layer in result.body if isinstance(layer, (nn.Conv2d, nn.PReLU))]


def flat(result, quantize=False):
    values = []
    for layer in layers(result):
        if isinstance(layer, nn.Conv2d):
            weight = layer.weight
            if quantize and not parametrize.is_parametrized(layer):
                weight = quantized(weight)[0]
            values.extend([weight.detach().cpu().numpy().reshape(-1), layer.bias.detach().cpu().numpy()])
        else:
            values.append(layer.weight.detach().cpu().numpy())
    return np.concatenate(values).astype('<f4')


def pack(result):
    """Q8PC v1: header then conv scale[f32], weights[i8], bias[f32], PReLU[f32]."""
    out = bytearray(struct.pack('<8sIIII', b'ZSRQ8PC\0', 1, 24, 8, 1))
    for layer in layers(result):
        if isinstance(layer, nn.Conv2d):
            # Use original shadow weights, not a second quantization of dequantized values.
            weight = layer.parametrizations.weight.original if parametrize.is_parametrized(layer) else layer.weight
            dequantized, scale = quantized(weight)
            # Native fake quantization rounds weight * reciprocal(scale), which
            # can differ from weight / scale at ties. Recover its integer codes.
            integer = (dequantized.detach() / scale[:, None, None, None]).round().clamp(-127, 127).to(torch.int8)
            out += scale.detach().cpu().numpy().astype('<f4').tobytes()
            out += integer.cpu().numpy().tobytes()
            out += layer.bias.detach().cpu().numpy().astype('<f4').tobytes()
        else:
            out += layer.weight.detach().cpu().numpy().astype('<f4').tobytes()
    return bytes(out)


def unpack(data):
    if len(data) < 24 or struct.unpack_from('<8sIIII', data) != (b'ZSRQ8PC\0', 1, 24, 8, 1):
        raise ValueError('unsupported quantized weight header')
    off = 24
    values = []

    def take(dtype, n):
        nonlocal off
        size = np.dtype(dtype).itemsize * n
        if off + size > len(data):
            raise ValueError('truncated quantized weights')
        v = np.frombuffer(data, dtype=dtype, count=n, offset=off).copy()
        off += size
        return v

    for i in range(10):
        cin, cout = (3 if i == 0 else 24), (3 if i == 9 else 24)
        scale = take('<f4', cout)
        if not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('invalid channel scales')
        q = take('i1', cout * cin * 9)
        if (q == -128).any():
            raise ValueError('out-of-range quantized weight')
        values.extend([(q.reshape(cout, -1).astype('<f4') * scale[:, None]).reshape(-1), take('<f4', cout)])
        if i != 9:
            values.append(take('<f4', cout))
    if off != len(data):
        raise ValueError('trailing quantized weights')
    values = np.concatenate(values).astype('<f4')
    if not np.isfinite(values).all():
        raise ValueError('non-finite decoded weights')
    return values


def export(result, root, quantize, metadata):
    root.mkdir()
    if quantize:
        data = pack(result)
        decoded = unpack(data)
        expected = flat(result, quantize=True)
        if not np.array_equal(decoded.view('u4'), expected.view('u4')):
            raise ValueError('packed weights differ from QAT/PTQ forward')
        (root / 'weights_q8pc.bin').write_bytes(data)
        decoded.tofile(root / 'weights.raw')
    else:
        flat(result).tofile(root / 'weights.raw')
    meta = dict(arch='compact', nf=24, nc=8, scale=1, space='rgb', total_floats=COUNT,
                f16_goldens=False, quantization='symmetric_per_output_channel_w8a32' if quantize else 'none',
                **metadata)
    meta['weights_sha256'] = sha(root / 'weights.raw')
    if quantize:
        meta['packed_weights_bytes'] = (root / 'weights_q8pc.bin').stat().st_size
        meta['packed_weights_sha256'] = sha(root / 'weights_q8pc.bin')
    (root / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    # Independent CPU check: exported float kernel sees the identical operands.
    loaded = load_flat(np.fromfile(root / 'weights.raw', dtype='<f4')).eval()
    reference = load_flat(flat(result, quantize=quantize)).eval()
    generator = torch.Generator().manual_seed(991)
    x = torch.rand((1, 3, 17, 13), generator=generator)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), reference(x), atol=0, rtol=0)
    x.numpy().astype('<f4').tofile(root / 'gold_in_17x13.raw')
    with torch.no_grad():
        loaded(x).numpy().astype('<f4').tofile(root / 'gold_out_17x13.raw')
    return meta


def checked_pairs(dataset, split_map):
    splits = {r['id']: r['split'] for r in csv.DictReader(Path(split_map).open(), delimiter='\t')}
    rows = list(csv.DictReader((dataset / 'pairs.tsv').open(), delimiter='\t'))
    meta = json.loads((dataset / 'meta.json').read_text())
    n, val = int(meta['n']), int(meta['val_tail'])
    if len(rows) != n or not 0 < val < n:
        raise ValueError('invalid pair count or validation tail')
    roles = {}
    for i, row in enumerate(rows):
        if int(row['idx']) != i or splits.get(row['imazen_id']) != 'train' or row['canonical_split'] != 'train':
            raise ValueError('pair violates published imazen-26 training split')
        expected = 'model_train' if i < n - val else 'internal_validation'
        if row['partition'] != expected:
            raise ValueError('invalid internal pair partition')
        roles.setdefault(expected, set()).add(row['imazen_id'])
    if roles['model_train'] & roles['internal_validation']:
        raise ValueError('source overlap between train and internal validation')
    for name in ('lr_u8.npy', 'hr_u8.npy'):
        array = np.load(dataset / name, mmap_mode='r')
        if array.shape != (n, 128, 128, 3) or array.dtype != np.uint8:
            raise ValueError('invalid training array geometry')
    return rows, n - val
