"""Separate inherited chroma from learned RGB/edge behavior on retained witnesses.

Default CUDA outputs must replay the frozen bank exactly before pad/crop controls
are interpreted. Controls are research candidates; no runtime padding changes.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from bank import Worker, save, sha

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'tools/external-models'), str(REPO / 'tools/qat')]
from gpu_8k_worker import spanf
from sr_models import load
from canon4 import load_baseline


def chroma(p):
    spread = p.max(axis=2).astype(np.float64) - p.min(axis=2).astype(np.float64)
    h, w = spread.shape
    y, x = np.indices((h, w))
    distance = np.minimum(np.minimum(x, w - 1 - x), np.minimum(y, h - 1 - y))
    return dict(max_spread=float(spread.max()), mean_spread=float(spread.mean()),
                non_gray=int(np.count_nonzero(spread)), over_one=int(np.count_nonzero(spread > 1)),
                rings=[dict(distance=d, pixels=int(np.count_nonzero(distance == d)),
                            max_spread=float(spread[distance == d].max()),
                            mean_spread=float(spread[distance == d].mean()))
                       for d in range(min(9, (min(w, h) + 1) // 2))])


@torch.inference_mode()
def main(a):
    if not torch.cuda.is_available() or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Requires deterministic CUDA environment')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.25)
    a.output.mkdir(exist_ok=False)
    bank = json.loads(a.shard.read_text())
    witnesses = json.loads((a.witnesses / 'WITNESSES.json').read_text())
    inputs = {}
    for c in witnesses['cases']:
        for role, image in [('clean', c['clean_lr']), ('jpeg', c['decoded_lr'])]:
            rid = f"e{c['edge']}/q0/clean" if role == 'clean' else c['id'].split('/', 1)[1]
            path = a.witnesses / image['path']
            with Image.open(path) as im:
                p = np.array(im.convert('RGB'), dtype=np.uint8)
            if sha(path) != image['png_sha256'] or __import__('hashlib').sha256(p.tobytes()).hexdigest() != image['pixels_sha256']:
                raise ValueError('Witness PNG or decoded pixels changed')
            inputs[rid] = p
    identities = json.loads(a.identity.read_text())
    assets = identities['files']
    def asset(tail):
        p = Path(next(k for k in assets if k.endswith(tail)))
        if sha(p) != assets[str(p)]:
            raise ValueError('Frozen model/source identity changed')
        return p
    span_root = asset('/team24_SPANF.py').parent
    dscf_root = asset('/team23_DSCF.pth').parent
    options = SimpleNamespace(spanf_root=span_root,
        spanf_f32=asset('/spanf_weights.raw'), spanf_mixed=asset('/spanf_weights_f16.raw'))
    models = [('canon4', load_baseline(asset('/dejpeg_rt24i_canon4/weights_f16.raw')), 1),
              ('spanf', spanf(options), 4), ('dscf', load(dscf_root, 'dscf').model, 4)]
    worker = Worker(a.binary, a.output / 'worker.log')
    source = bank['source']
    archive = a.output / 'candidates.imijset'
    worker.send('source', archive=source['archive'], key=source['entry']['key'], output=str(archive))
    rows, replay = [], []
    failed = False
    required = len(inputs) * 2 * len(models) * 3
    try:
        for name, model, scale in models:
            model = model.cuda().eval().requires_grad_(False)
            for rid, original in sorted(inputs.items()):
                for transpose in [False, True]:
                    pixels = original.transpose(1, 0, 2).copy() if transpose else original
                    h, w = pixels.shape[:2]
                    # Match the frozen bank's CPU f32 division before CUDA transfer.
                    values = pixels.transpose(2, 0, 1).copy().astype('<f4') / np.float32(255)
                    x = torch.from_numpy(values[None]).cuda()
                    for halo in [0, 16, 32]:
                        padded = F.pad(x, (halo, halo, halo, halo), mode='replicate') if halo else x
                        output = model(padded)
                        if halo:
                            begin = halo * scale
                            output = output[:, :, begin:begin + h * scale, begin:begin + w * scale]
                        floats = output[0].permute(1, 2, 0).cpu().numpy()
                        if not np.isfinite(floats).all():
                            raise ValueError('Non-finite model result')
                        out = np.clip(floats * np.float32(255) + np.float32(.5), 0, 255).astype('u1')
                        expected = None
                        if halo == 0 and not transpose and name != 'canon4':
                            cell = next(c for c in bank['cells'] if c['id'] == source['id'] + '/' + rid)
                            expected = next(r for r in cell['sr'] if r['model'] == name and r['repair'] == 'direct')
                        encoded = worker.send('put', payload=out.tobytes(), width=w*scale, height=h*scale,
                                              label=f'{rid}/{name}/transpose-{transpose}/halo-{halo}', score=False)
                        if expected is not None:
                            if encoded['pixels_sha256'] != expected['pixels_sha256']:
                                raise ValueError(f'Default SR does not replay the frozen bank: {name} {rid} expected {expected["pixels_sha256"]}, got {encoded["pixels_sha256"]}')
                            replay.append(dict(id=rid, model=name, pixels_sha256=encoded['pixels_sha256']))
                        rows.append(dict(id=rid, model=name, transpose=transpose, halo=halo,
                                         width=w, height=h, input_chroma=chroma(pixels),
                                         output_chroma=chroma(out), raw_output_chroma=chroma(floats * 255),
                                         artifact=encoded))
                        save(a.output / 'PROGRESS.json', dict(stage='running', completed=len(rows), required=required,
                                                            pid=os.getpid(), worker_pid=worker.process.pid))
            del model
        worker.send('finish')
    except BaseException as error:
        failed = True
        save(a.output / 'PROGRESS.json', dict(stage='failed', completed=len(rows), required=required, pid=os.getpid(), error=repr(error)))
        raise
    finally:
        worker.close(check=not failed)
    if len(rows) != required or len(replay) != 18:
        raise ValueError('Incomplete control grid or default replay')
    save(a.output / 'RESULT.json', dict(rows=rows, replay=replay, archive_sha256=sha(archive),
         identity_sha256=sha(a.identity), shard_sha256=sha(a.shard), script_sha256=sha(Path(__file__)),
         torch=torch.__version__, device=torch.cuda.get_device_name(0),
         padding='author model unchanged; replicate input halo then crop; experimental, no default change'))
    save(a.output / 'PROGRESS.json', dict(stage='complete', completed=len(rows), required=required, pid=os.getpid()))
    print(json.dumps(dict(comparisons=len(rows), default_replays=len(replay), archive_bytes=archive.stat().st_size)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['shard', 'witnesses', 'identity', 'binary', 'output']:
        p.add_argument('--' + name, required=True, type=Path)
    main(p.parse_args())
