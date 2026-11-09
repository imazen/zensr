#!/usr/bin/env python3
"""Bounded inference feasibility probe; this is not an image-quality benchmark."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch


def main(a):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261109)
    spec = importlib.util.spec_from_file_location('author_jpeg_model', a.root / 'jpeg_model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.load_jpeg_model(a.root / 'meanflow_jpeg_qf5_20.pth', a.device)
    model.eval()
    x = torch.linspace(0, 1, 3 * a.height * a.width, device=a.device).reshape(1, 3, a.height, a.width)
    records = []
    with torch.inference_mode():
        for steps in [1, 2]:
            torch.manual_seed(20261109)
            y = module.restore(model, x, sample_steps=steps)
            if y.shape != x.shape or not torch.isfinite(y).all():
                raise ValueError('author inference shape/finiteness failed')
            b = y.cpu().contiguous().numpy().astype('<f4').tobytes()
            records.append(dict(steps=steps, shape=list(y.shape), minimum=float(y.min()),
                                maximum=float(y.max()), f32_sha256=hashlib.sha256(b).hexdigest()))
    result = dict(device=a.device, torch=torch.__version__, parameters=sum(p.numel() for p in model.parameters()),
                  records=records, scope='Synthetic inference feasibility only; no IQA or latency claim.')
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--height', type=int, default=32)
    p.add_argument('--width', type=int, default=48)
    main(p.parse_args())
