"""Export pinned author DSCF to static FP32 ONNX graphs for CPU evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from sr_models import load


def main(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_available():
        raise ValueError('export requires explicit CPU-only execution')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = load(a.root, 'dscf', device='cpu')
    a.output.mkdir(exist_ok=True)
    records = []
    for n in [64, 128, 256]:
        path = a.output/f'DSCF_x4_{n}x{n}.onnx'
        if path.exists():raise ValueError('refusing to overwrite a frozen export')
        torch.manual_seed(20261109)
        image = torch.rand(1, 3, n, n)
        torch.onnx.export(model.model, image, str(path), opset_version=17,
            input_names=['rgb'], output_names=['upscaled'], dynamo=False)
        records.append(dict(input_side=n, path=str(path), bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    with (a.output/'DSCF_EXPORT.json').open('x') as f:
        f.write(json.dumps(dict(torch=torch.__version__, parameters=model.parameters,
            author_source_sha256=hashlib.sha256((a.root/'source/models/team23_DSCF.py').read_bytes()).hexdigest(),
            checkpoint_sha256=hashlib.sha256((a.root/'team23_DSCF.pth').read_bytes()).hexdigest(),
            precision='FP32, no quantization', records=records), indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
