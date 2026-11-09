"""Frozen direct/production-canon4 × native-SPANF/author-DSCF comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import threading
import sys
import hashlib

import numpy as np
import torch
from evaluate_sr import Scorer, save, sha
from sr_models import load
from cascade_contract import validate


class CascadeScorer(Scorer):
    def __init__(self, a, index):
        self.log = (a.output/f'scorer-{index}.log').open('a')
        env = dict(os.environ, RAYON_NUM_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        self.process = subprocess.Popen([str(a.scorer), 'cascade', str(a.canon4), str(a.spanf), str(a.artifacts)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, env=env)
        if self.receive() != dict(ready=True, canon4_sha256=sha(a.canon4), spanf_sha256=sha(a.spanf)):
            raise ValueError('cascade model identity differs')

    def prepare(self, row):
        header = self.send(dict(action='prepare', row=row))
        n = 12*row['low_width']*row['low_height']
        if header['input_bytes'] != n or header['id'] != row['id']:
            raise ValueError('cascade transport geometry differs')
        arrays = []
        for field in ['plain_f32_sha256', 'repair_f32_sha256']:
            payload = bytearray(n)
            view = memoryview(payload)
            while view:
                count = self.process.stdout.readinto(view)
                if not count:raise RuntimeError('incomplete cascade tensor pipe')
                view = view[count:]
            if hashlib.sha256(payload).hexdigest() != header[field]:
                raise ValueError('cascade transport bytes differ')
            image = np.frombuffer(payload, dtype='<f4').reshape(1, 3, row['low_height'], row['low_width']).copy()
            if not np.isfinite(image).all():raise ValueError('nonfinite cascade input')
            arrays.append(image)
        return header, arrays


def main(a):
    if not 1 <= a.jobs <= 6:raise ValueError('invalid worker budget')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    rows = [json.loads(line) for line in a.inputs.read_text().splitlines()]
    if len(rows) != 697 or len({r['id'] for r in rows}) != 697:
        raise ValueError('cascade cohort coverage differs')
    if a.pilot:
        rows = [r for r in rows if r['source_id'] == '1017' and r['q'] in (0, 10, 50, 90)]
        if len(rows) != 7:raise ValueError('missing pilot controls')
    a.output.mkdir(exist_ok=True);(a.output/'cells').mkdir(exist_ok=True)
    source = Path(__file__).parent
    identity = dict(schema='zensr-sr-cascade-v1', source_revision=a.revision, inputs_sha256=sha(a.inputs),
        scorer_sha256=sha(a.scorer), canon4_sha256=sha(a.canon4), spanf_sha256=sha(a.spanf),
        dscf_acquisition_sha256=sha(a.model_root/'ACQUISITION.json'),
        tool_sources={p.name:sha(p) for p in [Path(__file__), source/'evaluate_sr.py', source/'sr_models.py', source/'cascade_contract.py', source/'sr_artifacts.py']},
        python=sys.version, packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        torch_native_sha256=sha(Path(torch._C.__file__)), cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), device=torch.cuda.get_device_name(0),
        cublas=os.environ['CUBLAS_WORKSPACE_CONFIG'], tf32=False, deterministic=True,
        artifact_root=str(a.artifacts.resolve()), input_ids=[r['id'] for r in rows], jobs=a.jobs,
        pilot=a.pilot, repair='public Restorer default Fidelity/routing; Realtime canon4; one worker; clean passthrough',
        precision='native SPANF mixed f16 weights/f32 biases; canon4 f16 expanded; GPU DSCF FP32; unrounded f32 between stages',
        retention='exact scored RGB8 PNG plus LR previews and f32 transport hashes; no IQA maps')
    path = a.output/'IDENTITY.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError('resumed cascade identity differs')
    if not path.exists():save(path, identity)
    completed = 0;pending = []
    for row in rows:
        path = a.output/'cells'/(row['id']+'.json')
        if path.exists():validate(json.loads(path.read_text()), row, a.artifacts);completed += 1
        else:pending.append(row)
    scorers = [];lock = threading.Lock();gpu_lock = threading.Lock();halt = threading.Event()
    def progress(stage, error=None):
        save(a.output/'PROGRESS.json', dict(stage=stage, completed=completed, required=len(rows), pid=os.getpid(),
            scorer_pids=[s.process.pid for s in scorers], checked=datetime.datetime.now(datetime.timezone.utc).isoformat(), error=error))
    progress('loading')
    model = load(a.model_root, 'dscf')
    iterator = iter(pending)
    try:
        scorers = [CascadeScorer(a, i) for i in range(a.jobs)]
        def worker(index):
            nonlocal completed
            try:
                while not halt.is_set():
                    with lock:row = next(iterator, None)
                    if row is None:return
                    header, (plain, repaired) = scorers[index].prepare(row)
                    seed = (int(row['id'][:16], 16)^20261109)&((1<<63)-1)
                    with gpu_lock:
                        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                        direct = model(torch.from_numpy(plain).cuda()).cpu().float().contiguous()
                        if header['routing']['applied']:
                            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                            repaired_output = model(torch.from_numpy(repaired).cuda()).cpu().float().contiguous()
                        else:repaired_output = direct
                    shape = [1, 3, row['height'], row['width']]
                    if list(direct.shape) != shape or list(repaired_output.shape) != shape:
                        raise ValueError('cascade output geometry differs')
                    direct_wire = direct.numpy().astype('<f4', copy=False).tobytes()
                    repaired_wire = repaired_output.numpy().astype('<f4', copy=False).tobytes()
                    result = scorers[index].send(dict(action='score', id=row['id'], bytes=len(direct_wire)), direct_wire+repaired_wire)
                    if result['id'] != row['id']:raise ValueError('cascade result id differs')
                    cell = dict(row=row, seed=seed, **{k:v for k,v in header.items() if k not in ('id','input_bytes')},
                        dscf_direct=result['dscf_direct'], dscf_repair=result['dscf_repair'])
                    validate(cell, row, a.artifacts)
                    save(a.output/'cells'/(row['id']+'.json'), cell)
                    with lock:
                        completed += 1
                        if completed%16 == 0 or completed == len(rows):progress('evaluating');print(f'completed {completed}/{len(rows)}', flush=True)
            except BaseException:
                halt.set();raise
        progress('evaluating')
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for future in as_completed([pool.submit(worker, i) for i in range(a.jobs)]):future.result()
        if completed != len(rows):raise ValueError('incomplete cascade comparison')
        save(a.output/'COMPLETE.json', dict(inputs=completed, identity_sha256=sha(a.output/'IDENTITY.json')))
        progress('complete')
    except BaseException as error:
        progress('failed', repr(error));raise
    finally:
        for scorer in scorers:scorer.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['inputs', 'scorer', 'canon4', 'spanf', 'model-root', 'output', 'artifacts']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--pilot', action='store_true')
    main(p.parse_args())
