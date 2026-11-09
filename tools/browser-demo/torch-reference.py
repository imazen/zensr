"""Frozen author-PyTorch controls for browser SR (no training or IQA maps)."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'external-models'))
from sr_models import CPUDeviceLoader, load


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(a):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261109)
    a.output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in a.inputs.read_text().splitlines()]
    cases = []
    # Select by source identity and geometry, without examining any output score.
    for identity in ('1017', '7109', '8157'):
        row = next(r for r in rows if r['source_id'] == identity and r['q'] == 0)
        path = Path(row['reference'])
        assert sha(path) == row['reference_sha256']
        image = np.fromfile(path, np.uint8).reshape(row['height'], row['width'], 3)
        image = image[:263, :259].copy()
        target = a.output / (identity + '.rgb8')
        image.tofile(target)
        cases.append(dict(id=identity, width=259, height=263, rgb=str(target),
                          sha256=sha(target), source_sha256=sha(path), content=row['content'], crop=[0, 0, 259, 263]))
    tiny = np.arange(17*13*3, dtype=np.uint32).reshape(13, 17, 3)
    tiny = ((tiny*71 + 19) % 256).astype(np.uint8)
    target = a.output / 'tiny.rgb8'
    tiny.tofile(target)
    cases.append(dict(id='tiny', width=17, height=13, rgb=str(target), sha256=sha(target), content='synthetic RGB pattern'))
    (a.output/'cases.json').write_text(json.dumps(cases, indent=2)+'\n')
    records = []
    for name in ('spanf', 'dscf'):
        if name == 'spanf':
            source = a.spanf / 'team24_SPANF.py'
            spec = importlib.util.spec_from_loader('author_spanf_cpu', CPUDeviceLoader(source))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model = module.SPANF(3, 3, feature_channels=32, upscale=4)
            checkpoint = a.spanf/'team24_spanf.pth'
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        else:
            source = a.dscf/'source/models/team23_DSCF.py'
            checkpoint = a.dscf/'team23_DSCF.pth'
            model = load(a.dscf, name, device='cpu').model
        model.eval().requires_grad_(False)
        for precision in (('author-f32', 'browser-mixed') if name == 'spanf' else ('author-f32',)):
            if precision == 'browser-mixed':
                state = model.state_dict()
                order = ['conv_near.weight']
                for b in range(1, 6):
                    for c in ('c1_r', 'c2_r', 'c3_r'):
                        order += [f'block_{b}.{c}.eval_conv.weight', f'block_{b}.{c}.eval_conv.bias']
                order += ['conv_cat.weight', 'conv_cat.bias', 'conv_2.eval_conv.weight', 'conv_2.eval_conv.bias']
                assert b''.join(state[k].numpy().astype('<f4').tobytes() for k in order) == a.spanf_f32.read_bytes()
                offset = 0
                data = a.spanf_mixed.read_bytes()
                for key in order:
                    dtype = '<f2' if key.endswith('.weight') else '<f4'
                    values = np.frombuffer(data, dtype=dtype, count=state[key].numel(), offset=offset).astype('<f4')
                    offset += values.size*np.dtype(dtype).itemsize
                    state[key].copy_(torch.from_numpy(values.reshape(state[key].shape)))
                assert offset == len(data)
            for case in cases:
                image = np.fromfile(case['rgb'], np.uint8).reshape(case['height'], case['width'], 3)
                low = torch.from_numpy(image.transpose(2,0,1).copy()).float()[None]/255
                with torch.inference_mode():
                    output = model(low)
                assert tuple(output.shape) == (1, 3, case['height']*4, case['width']*4)
                assert torch.isfinite(output).all()
                for scale in (4, 2):
                    resized = output if scale == 4 else torch.nn.functional.avg_pool2d(output, 2, 2)
                    pixels = (resized[0].permute(1,2,0)*255+.5).clamp(0,255).to(torch.uint8).contiguous().numpy()
                    path = a.output/f"{case['id']}-{name}-{precision}-x{scale}.rgb8"
                    pixels.tofile(path)
                    records.append(dict(case=case['id'], model=name, precision=precision, scale=scale,
                                        width=pixels.shape[1], height=pixels.shape[0], rgb=str(path), sha256=sha(path)))
                print(case['id'], name, precision, flush=True)
    (a.output/'reference.json').write_text(json.dumps(dict(torch=torch.__version__, device='cpu', threads=4,
        constructor_adaptation='CUDA device selections only; unchanged author forward',
        x2='full x4 float output averaged over aligned 2x2 cells before RGB8',
        assets={str(p):sha(p) for p in [a.spanf/'team24_SPANF.py', a.spanf/'team24_spanf.pth',
            a.dscf/'source/models/team23_DSCF.py', a.dscf/'team23_DSCF.pth', a.spanf_f32, a.spanf_mixed]}, records=records), indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'spanf', 'dscf', 'spanf-f32', 'spanf-mixed', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    main(p.parse_args())
