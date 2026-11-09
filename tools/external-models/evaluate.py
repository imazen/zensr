#!/usr/bin/env python3
"""Score fixed author MeanFlow schedules through a scalar-only Rust pipe."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import datetime
from decimal import Decimal
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix(f'.partial-{os.getpid()}-{threading.get_ident()}')
    with temporary.open('x') as f:
        json.dump(value, f, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


class Scorer:
    def __init__(self, a, index):
        self.log = (a.output / f'scorer-{index}.log').open('a')
        env = dict(os.environ, RAYON_NUM_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        self.process = subprocess.Popen([str(a.scorer), str(a.quality_weights)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, env=env)
        self.ready = self.receive()
        expanded = np.fromfile(a.quality_weights, dtype='<f2').astype('<f4').tobytes()
        if self.ready != dict(ready=True, weights_sha256=sha(a.quality_weights),
                              expanded_weights_sha256=hashlib.sha256(expanded).hexdigest()):
            raise ValueError('scorer model identity differs')

    def receive(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'scorer {self.process.pid} ended; inspect its retained log')
        return json.loads(line)

    def send(self, value, payload=None):
        self.process.stdin.write(json.dumps(value).encode() + b'\n')
        if payload is not None:
            self.process.stdin.write(payload)
        self.process.stdin.flush()
        return self.receive()

    def prepare(self, row):
        value = {k: row[k] for k in ['id', 'jpeg', 'jpeg_sha256', 'reference_rgb8', 'reference_rgb8_sha256']}
        value.update(action='prepare', width=int(row['width']), height=int(row['height']))
        header = self.send(value)
        count = header['input_bytes']
        if count != 12 * value['width'] * value['height'] or header['id'] != row['id']:
            raise ValueError('scorer input geometry/identity differs')
        payload = bytearray(count)
        view = memoryview(payload)
        while view:
            n = self.process.stdout.readinto(view)
            if not n:
                raise RuntimeError('truncated scorer pixel pipe')
            view = view[n:]
        return header, np.frombuffer(payload, dtype='<f4').reshape(1, 3, value['height'], value['width']).copy()

    def score(self, key, pixels):
        return self.send(dict(action='score', id=key, bytes=len(pixels)), pixels)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write(b'{"action":"quit"}\n')
            self.process.stdin.flush()
            self.process.stdin.close()
        code = self.process.wait()
        self.log.close()
        if code:
            raise RuntimeError(f'scorer {self.process.pid} exit {code}')


def check_result(result, key):
    if result['id'] != key:
        raise ValueError('scored result belongs to a different input')
    for stage in ['raw', 'projected']:
        for metric in ['ssim2', 'butter_max', 'butter_pnorm3']:
            if not Decimal(result[stage][metric]).is_finite():
                raise ValueError('nonfinite external score')
        digest = result[stage]['rgb8_sha256']
        if len(digest) != 64 or any(v not in '0123456789abcdef' for v in digest):
            raise ValueError('invalid output digest')


def main(a):
    if not 1 <= a.jobs <= 8:
        raise ValueError('worker count must be between one and eight')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    rows = list(csv.DictReader(a.inputs.open(), delimiter='\t'))
    if len(rows) != 2960 or len({r['id'] for r in rows}) != 2960:
        raise ValueError('frozen input coverage differs')
    native = {r['id']: r for r in csv.DictReader(a.native_results.open(), delimiter='\t') if r['arm'] == 'native'}
    if set(native) != {r['id'] for r in rows}:
        raise ValueError('native comparison coverage differs')
    for row in rows:
        for key in ['source_id', 'origin', 'content', 'cohort', 'size', 'encoder', 'ss', 'q', 'width', 'height']:
            if row[key] != native[row['id']][key]:
                raise ValueError('paired input metadata differs')
    controls = [r for r in rows if r['cohort'] == 'primary' and r['source_id'] == '1017' and r['q'] == '10']
    if len(controls) != 8:
        raise ValueError('fixed encoder/sampling replay controls missing')
    if a.pilot:
        rows = controls
    a.output.mkdir(exist_ok=True)
    (a.output / 'cells').mkdir(exist_ok=True)
    identity = dict(schema='zensr-external-jpeg-v1', inputs_sha256=sha(a.inputs),
        native_results_sha256=sha(a.native_results), checkpoint_sha256=sha(a.model_root/'meanflow_jpeg_qf5_20.pth'),
        source_sha256=sha(a.model_root/'jpeg_model.py'), scorer_sha256=sha(a.scorer),
        quality_weights_sha256=sha(a.quality_weights), script_sha256=sha(Path(__file__)),
        torch=torch.__version__, numpy=np.__version__, python=sys.version,
        torch_native_sha256=sha(Path(torch._C.__file__)), cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), device=torch.cuda.get_device_name(0),
        capability=torch.cuda.get_device_capability(0), jobs=a.jobs,
        tf32=False, deterministic=True, cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        pilot=a.pilot, input_ids=[r['id'] for r in rows],
        noise='fresh device Generator per input and arm; seed=(input id prefix xor 20261109) mod 2^63',
        decoded_input='zenjpeg deblock off', precision='float32',
        retention='scalar scores and output hashes; no RGB/map files')
    identity_path = a.output / 'IDENTITY.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError('resumed scientific/numerical identity differs')
    else:
        save(identity_path, identity)
    spec = importlib.util.spec_from_file_location('author_jpeg_model', a.model_root/'jpeg_model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.load_jpeg_model(a.model_root/'meanflow_jpeg_qf5_20.pth', 'cuda').eval()
    scorers = [Scorer(a, index) for index in range(a.jobs)]
    try:
        controls_result = []
        # Admission compares exact RGB8, not a relaxed pixel tolerance. Metric
        # scalars allow 1e-9 numerical serialization/dispatch noise, declared here.
        for i, row in enumerate(controls):
            scorer = scorers[i % a.jobs]
            header, _ = scorer.prepare(row)
            baseline = native[row['id']]
            if abs(Decimal(header['identity_ssim2']) - Decimal(baseline['identity_ssim2'])) > Decimal('1e-9'):
                raise ValueError('decoded identity score differs from frozen comparison')
            result = scorer.send(dict(action='native-control', id=row['id']))
            check_result(result, row['id'])
            for stage in ['raw', 'projected']:
                if result[stage]['rgb8_sha256'] != baseline[stage+'_rgb8_sha256']:
                    raise ValueError(f'native {stage} pixels differ from frozen study on {row["id"]}')
                if abs(Decimal(result[stage]['ssim2']) - Decimal(baseline[stage+'_ssim2'])) > Decimal('1e-9'):
                    raise ValueError('native scalar score differs from frozen study')
            controls_result.append(result)
        save(a.output / 'CONTROL.json', controls_result)
    except BaseException as error:
        save(a.output/'PROGRESS.json',dict(stage='failed-admission',pid=os.getpid(),error=repr(error)))
        for scorer in scorers:
            scorer.close()
        raise
    halt = threading.Event()
    gpu = threading.Lock()
    progress_lock = threading.Lock()
    completed = 0

    def progress(stage, error=None):
        save(a.output/'PROGRESS.json', dict(stage=stage, completed=completed, required=len(rows),
            pid=os.getpid(), scorer_pids=[s.process.pid for s in scorers],
            checked=datetime.datetime.now(datetime.timezone.utc).isoformat(), error=error))

    def work_items(index):
        nonlocal completed
        scorer = scorers[index]
        for row in rows[index::a.jobs]:
            if halt.is_set():
                return
            target = a.output/'cells'/(row['id']+'.json')
            if target.exists():
                cell = json.loads(target.read_text())
                if cell['input'] != row or set(cell['schedules']) != {'1', '2'}:
                    raise ValueError('resumed result membership differs')
                for result in cell['schedules'].values():
                    check_result(result, row['id'])
            else:
                header, planes = scorer.prepare(row)
                if abs(Decimal(header['identity_ssim2'])-Decimal(native[row['id']]['identity_ssim2'])) > Decimal('1e-9'):
                    raise ValueError('input identity differs from frozen scores')
                seed = (int(row['id'][:16], 16) ^ 20261109) & ((1 << 63)-1)
                schedules = {}
                payloads = {}
                states = []
                with gpu, torch.inference_mode():
                    x = torch.from_numpy(planes).to('cuda')
                    for steps in [1, 2]:
                        generator = torch.Generator(device='cuda').manual_seed(seed)
                        y, _ = model.sample(x, generator=generator, sample_steps=steps)
                        if y.shape != x.shape or not torch.isfinite(y).all():
                            raise ValueError('external inference shape/finiteness failed')
                        payloads[steps] = y.cpu().contiguous().numpy().astype('<f4').tobytes()
                        states.append(generator.get_state())
                    if not torch.equal(states[0], states[1]):
                        raise ValueError('one/two-step schedules consumed different noise')
                    del x, y
                for steps in [1, 2]:
                    result = scorer.score(row['id'], payloads[steps])
                    check_result(result, row['id'])
                    result.update(seed=seed, steps=steps)
                    schedules[str(steps)] = result
                del payloads
                cell = dict(input=row, identity=header, schedules=schedules)
                save(target, cell)
            with progress_lock:
                completed += 1
                if completed % 16 == 0 or completed == len(rows):
                    progress('evaluating')
                    print(f'completed {completed}/{len(rows)} inputs', flush=True)

    def work(index):
        try:
            work_items(index)
        except BaseException:
            halt.set()
            raise

    progress('evaluating')
    try:
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            futures = [pool.submit(work, index) for index in range(a.jobs)]
            for future in as_completed(futures):
                future.result()
        if completed != len(rows):
            raise ValueError('incomplete external evaluation')
        save(a.output/'COMPLETE.json', dict(inputs=completed, schedules=2, identity_sha256=sha(identity_path)))
        progress('complete')
    except BaseException as error:
        progress('failed', repr(error))
        raise
    finally:
        for scorer in scorers:
            scorer.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['model-root', 'scorer', 'inputs', 'native-results', 'quality-weights', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--pilot', action='store_true')
    main(p.parse_args())
