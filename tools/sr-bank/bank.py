"""Generate a split-preserving, lossless small-image restoration bank on CUDA.

Source archives arrive over HTTP, never a mounted remote filesystem. Models and
codecs are existing implementations; this tool coordinates their offline use.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import datetime
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import urllib.request

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
QUALITIES = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
EDGES = [32, 64, 128]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    # One owner per path; completed records replace only this process's status.
    pending = path.with_name(path.name + '.pending')
    with pending.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    pending.replace(path)


def local_space(root, minimum_gib=80):
    fs = subprocess.check_output(['findmnt', '-T', str(root), '-n', '-o', 'FSTYPE'], text=True).strip()
    if fs not in ('xfs', 'ext4', 'btrfs', 'apfs'):
        raise ValueError(f'Candidate bank requires local disk, found {fs}')
    if shutil.disk_usage(root).free < minimum_gib * 2**30:
        raise ValueError('Local SSD reserve reached; preserving all completed work')


class Worker:
    def __init__(self, binary, log):
        self.log = log.open('ab')
        env = dict(os.environ, RAYON_NUM_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        self.process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.log, env=env)
        if self.receive() != {'ready': True}:
            raise ValueError('Unexpected worker handshake')

    def receive(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'Worker {self.process.pid} ended: {self.process.poll()}')
        return json.loads(line)

    def send(self, action, payload=None, **values):
        self.process.stdin.write(json.dumps(dict(action=action, **values)).encode() + b'\n')
        if payload is not None:
            self.process.stdin.write(payload)
        self.process.stdin.flush()
        return self.receive()

    def read(self, n):
        result = bytearray(n)
        view = memoryview(result)
        while view:
            count = self.process.stdout.readinto(view)
            if not count:
                raise RuntimeError('Truncated pixel transport')
            view = view[count:]
        return result

    def close(self, check=True):
        self.process.stdin.close()
        code = self.process.wait(timeout=30)
        self.log.close()
        if code and check:
            raise RuntimeError(f'Worker {self.process.pid} exited {code}')


def fetch(a):
    a.output.mkdir(parents=True, exist_ok=True)
    local_space(a.output)
    with urllib.request.urlopen(a.archive_base.rstrip('/') + '/output/imij-sets/MANIFEST.tsv', timeout=30) as r:
        inventory = r.read()
    rows = list(csv.DictReader(io.StringIO(inventory.decode()), delimiter='\t'))
    selected = []
    for split in ('train', 'validate'):
        matches = [r for r in rows if r['key'].endswith(f'imazen-26-imi-srgb-{split}.imiset')
                   and r['status'] == 'current']
        if len(matches) != 1:
            raise ValueError(f'Ambiguous source archive for {split}')
        row = matches[0]
        relative = row['tower_path'].removeprefix('/mnt/tower/')
        if relative == row['tower_path']:
            raise ValueError('Unrecognized inventory path mapping')
        selected.append(dict(split=split, bytes=int(row['bytes']), sha256=row['sha256'],
                             url=a.archive_base.rstrip('/') + '/' + relative,
                             path=str(a.output / f'source-{split}.imiset'), entries=int(row['files'])))
    save(a.output / 'SOURCE_INVENTORY.json', dict(inventory_sha256=hashlib.sha256(inventory).hexdigest(), archives=selected))

    def one(row):
        path = Path(row['path'])
        if path.exists():
            if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
                raise ValueError(f'Existing archive differs: {path}')
            return
        pending = path.with_name(path.name + '.partial')
        # Fail visibly on an earlier incomplete download rather than overwrite it.
        with urllib.request.urlopen(row['url'], timeout=60) as response, pending.open('xb') as out:
            digest = hashlib.sha256()
            count = 0
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                out.write(chunk)
                count += len(chunk)
                if count % (256 * 1024 * 1024) == 0:
                    local_space(a.output)
                    print(f"fetch {row['split']} {count}/{row['bytes']}", flush=True)
            out.flush()
            os.fsync(out.fileno())
        if count != row['bytes'] or digest.hexdigest() != row['sha256']:
            raise ValueError('Downloaded source archive identity differs')
        pending.rename(path)
        print(f"verified {row['split']} {count} bytes", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(one, r) for r in selected]):
            f.result()


def freeze(a):
    local_space(a.output)
    inventory = json.loads((a.output / 'SOURCE_INVENTORY.json').read_text())
    membership = {}
    snapshots = {}
    for split in ('train', 'validate', 'test'):
        path = a.corpus / 'manifests' / f'{split}.tsv'
        snapshots[split] = sha(path)
        for row in csv.DictReader(io.StringIO(path.read_text()), delimiter='\t'):
            if row['id'] in membership or row['split'] != split:
                raise ValueError('Corpus split membership is ambiguous')
            membership[row['id']] = row
    family_path = a.corpus / 'manifests/split_map_family_v2.tsv'
    if not family_path.exists():
        family_path = a.corpus / 'manifests/split_map_family.tsv'
    family_rows = list(csv.DictReader(io.StringIO(family_path.read_text()), delimiter='\t'))
    families = {r['id']: r['family'] for r in family_rows}
    family_roles = {r['id']: r['split'] for r in family_rows}
    family_splits = {}
    for r in family_rows:
        if r['family']:
            family_splits.setdefault(r['family'], set()).add(r['split'])
    worker = Worker(a.binary, a.output / 'catalog.log')
    groups = {}
    exclusions = []
    try:
        for archive in inventory['archives']:
            rows = worker.send('catalog', archive=archive['path'])['entries']
            if len(rows) != archive['entries']:
                raise ValueError('Source archive inventory count differs')
            save(a.output / f"catalog-{archive['split']}.json", rows)
            for entry in rows:
                export_id = str(entry['key'] // 10)
                if len(export_id) != 5 or not export_id.startswith('1'):
                    raise ValueError('Unexpected render-v4 export ID')
                rid = export_id[1:]
                row = membership.get(rid)
                reason = None
                if not row:
                    reason = 'not in frozen membership'
                elif family_roles.get(rid) != archive['split']:
                    raise ValueError(f'Archive disagrees with frozen successor family split for {rid}')
                elif row['split'] != archive['split']:
                    reason = 'legacy membership and successor family role disagree'
                elif entry['key'] % 10:
                    reason = 'HDR companion'
                elif entry['channels'] not in (1, 3) or entry['sample'] != 0:
                    reason = 'initial bank requires rendered RGB8'
                elif entry['width'] * entry['height'] > 64 * 1024 * 1024:
                    reason = 'source exceeds 64 MP admission'
                elif families.get(rid) and len(family_splits[families[rid]]) != 1:
                    reason = 'declared source family crosses official splits'
                if reason:
                    exclusions.append(dict(id=rid, reason=reason))
                    continue
                category = row['content_class']
                groups.setdefault((row['split'], category), []).append(dict(
                    id=rid, export_id=export_id, split=row['split'], category=category,
                    family=families.get(rid) or 'id:' + rid, source=row,
                    archive=archive['path'], archive_sha256=archive['sha256'], entry=entry))
    finally:
        worker.close()
    selected = []
    for (split, category), rows in sorted(groups.items()):
        # Hash order avoids quality-based selection and numerical-ID adjacency.
        rows.sort(key=lambda r: hashlib.sha256(('zensr-bank-v1:' + r['id']).encode()).digest())
        target = a.train_per_category if split == 'train' else a.eval_per_category
        used = set()
        for row in rows:
            if row['family'] not in used:
                selected.append(row)
                used.add(row['family'])
                if len(used) == target:
                    break
    if not selected or {r['split'] for r in selected} != {'train', 'validate'}:
        raise ValueError('Selection did not retain train and evaluation')
    origin_sets = [{r['family'] for r in selected if r['split'] == s} for s in ('train', 'validate')]
    if origin_sets[0] & origin_sets[1]:
        raise ValueError('Family leakage in selected cohort')
    plan = dict(schema='zensr-small-bank-v1', corpus_revision=a.corpus_revision or subprocess.check_output(
        ['git', '-C', str(a.corpus), 'rev-parse', 'HEAD'], text=True).strip(),
        export_id_mapping=dict(rule='render-v4 prefixes the four-digit corpus ID with 1', source='https://github.com/imazen/imazen-26/blob/bcea0abc1210ac589ada62c80df9dca177f13ff7/scripts/render_v4_list.py'),
        membership_sha256=snapshots, family_manifest_sha256=sha(family_path),
        selection='intersection of published ID split and successor family-v2 role; hash-ordered within split/category; one member per family; no quality scores',
        train_per_category=a.train_per_category, eval_per_category=a.eval_per_category,
        edges=EDGES, qualities=QUALITIES, sampling=['420', '444'], sources=selected,
        exclusions=exclusions, test='reserved; no image entry accessed',
        framing='whole source frame; nearest integer aspect-fit LR dimensions; HR exactly 4x LR; no crop',
        color='rendered sRGB RGB8; linear-light CatmullRom source->HR->LR via zenresize',
        feature_reuse='existing zenanalyze-api qualified Request/satisfies/reuse_for; pixels/stage/geometry bound externally',
        repair='offline forced canon4 CNN, native guarded_merge and coefficient projection; slack 0.5x/1x/2x plus guard-only; no learned runtime routing claim',
        sr='native-matching SPANF weights and author DSCF FP32 CUDA; direct, half repair, full repair; RGB8 between stages',
        retention='lossless IMIJ RGB8 and raw CNN F32, per-origin imij-set archives; score-only IQA; no maps')
    path = a.output / 'PLAN.json'
    if path.exists():
        raise ValueError('A frozen plan already exists; preserve it')
    save(path, plan)
    print(json.dumps(dict(origins=len(selected), train=sum(r['split'] == 'train' for r in selected),
                          validate=sum(r['split'] == 'validate' for r in selected),
                          categories=len({r['category'] for r in selected}), exclusions=len(exclusions))), flush=True)


def generate(a):
    import numpy as np
    import torch
    from types import SimpleNamespace
    sys.path[:0] = [str(REPO / 'tools/external-models'), str(REPO / 'tools/qat')]
    from gpu_8k_worker import spanf
    from sr_models import load
    from canon4 import load_baseline
    if not torch.cuda.is_available() or not 1 <= a.jobs <= 4:
        raise ValueError('Requires CUDA and one to four bounded workers')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(0.65)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Requires CUBLAS_WORKSPACE_CONFIG=:4096:8')
    plan = json.loads((a.output / 'PLAN.json').read_text())
    sources = plan['sources']
    if a.pilot:
        tokens = ('photos-general', 'plots', 'web-screenshots', 'manuscript-text')
        sources = [next(r for r in sources if token in r['category']) for token in tokens[:a.pilot]]
    run = a.output / a.run
    run.mkdir(exist_ok=True)
    (run / 'shards').mkdir(exist_ok=True)
    local_space(run)
    identities = [a.canon4, a.spanf_root / 'team24_SPANF.py', a.spanf_root / 'team24_spanf.pth',
                  a.spanf_f32, a.spanf_mixed, a.dscf_root / 'source/models/team23_DSCF.py',
                  a.dscf_root / 'team23_DSCF.pth', HERE / 'bank.py',
                  REPO / 'tools/qat/canon4.py', REPO / 'tools/train_people.py',
                  REPO / 'tools/external-models/gpu_8k_worker.py',
                  REPO / 'tools/external-models/sr_models.py']
    identity = dict(plan_sha256=sha(a.output / 'PLAN.json'), worker_sha256=sha(a.binary),
                    source_revision=a.revision, files={str(p): sha(p) for p in identities},
                    packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
                    torch_binary_sha256=sha(Path(torch._C.__file__)), python=sys.version,
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    device=torch.cuda.get_device_name(0), tf32=False, deterministic=True,
                    jobs=a.jobs, pilot=a.pilot, source_ids=[r['id'] for r in sources])
    if a.reuse_from:
        prior = a.output / a.reuse_from
        prior_identity = json.loads((prior / 'IDENTITY.json').read_text())
        for k in identity:
            if k in ('source_revision', 'files'):
                continue
            if identity[k] != prior_identity[k]:
                raise ValueError(f'Incompatible prior generation identity: {k}')
        for path, digest in identity['files'].items():
            if path != str(HERE / 'bank.py') and prior_identity['files'].get(path) != digest:
                raise ValueError('Prior model or generation dependency changed')
        identity['reused_prior_identity_sha256'] = sha(prior / 'IDENTITY.json')
    identity_path = run / 'IDENTITY.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Resume identity differs')
    if not identity_path.exists():
        save(identity_path, identity)
    per_source = len(plan['edges']) * (1 + 2 * (len(plan['qualities']) - 1))
    lock = threading.Lock()
    gpu_lock = threading.Lock()
    stop = threading.Event()
    archive_paths = []
    progress_state = dict(stage='loading-models', completed_origins=0, required_origins=len(sources),
                          completed_cells=0, required_cells=per_source * len(sources), pid=os.getpid(),
                          worker_pids=[], run_directory=str(run), source_revision=a.revision,
                          identity_sha256=sha(identity_path), error=None)

    def progress(stage=None, error=None):
        if stage:
            progress_state['stage'] = stage
        progress_state['error'] = error
        progress_state['archive_bytes'] = sum(p.stat().st_size for p in archive_paths if p.exists())
        if progress_state['archive_bytes'] > 40 * 2**30:
            raise ValueError('Candidate-bank 40 GiB storage budget reached')
        progress_state['checked_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(run / 'PROGRESS.json', progress_state)

    progress()
    repair_model = load_baseline(a.canon4).cuda().eval().requires_grad_(False)
    models = dict(spanf=spanf(SimpleNamespace(spanf_root=a.spanf_root, spanf_f32=a.spanf_f32,
                                            spanf_mixed=a.spanf_mixed)).cuda(),
                  dscf=load(a.dscf_root, 'dscf').model)
    for m in [repair_model, *models.values()]:
        if any(p.dtype != torch.float32 or p.device.type != 'cuda' for p in m.parameters()):
            raise ValueError('Models must retain FP32 CUDA parameters')
    if a.reuse_from:
        # The frozen executable, inputs and models must agree above. Only the
        # Python orchestration differs; source records remain hash-bound.
        from report import validate
        prior = a.output / a.reuse_from
        prior_schema = json.loads((prior / 'FEATURE_SCHEMA.json').read_text())
        for name in ('FEATURE_SCHEMA.json', f"FEATURE_SCHEMA-{prior_schema['schema_hash']}.json"):
            target = run / name
            if not target.exists():
                save(target, prior_schema)
        for source in sources:
            original_record = prior / 'shards' / (source['id'] + '.json')
            target_record = run / 'shards' / (source['id'] + '.json')
            if not original_record.exists() or target_record.exists():
                continue
            record = json.loads(original_record.read_text())
            validate(record, source, plan, identity['reused_prior_identity_sha256'])
            target_archive = run / 'shards' / (source['id'] + '.imijset')
            if target_archive.exists():
                raise ValueError('Unrecorded reuse target exists')
            os.link(record['archive'], target_archive)
            record['archive'] = str(target_archive)
            record['generation_identity_sha256'] = record['identity_sha256']
            record['identity_sha256'] = sha(identity_path)
            record['reused_source_record_sha256'] = sha(original_record)
            for cell in record['cells']:
                cell['features']['schema_hash'] = prior_schema['schema_hash']
            save(target_record, record)
    pending = []
    for source in sources:

        record = run / 'shards' / (source['id'] + '.json')
        if record.exists():
            r = json.loads(record.read_text())
            archive = Path(r['archive'])
            if (r['identity_sha256'] != sha(identity_path) or len(r['cells']) != per_source
                    or sha(archive) != r['archive_sha256']):
                raise ValueError('Completed origin does not match this run')
            archive_paths.append(archive)
            progress_state['completed_origins'] += 1
            progress_state['completed_cells'] += per_source
        else:
            pending.append(source)
    iterator = iter(pending)

    def tensor(p, w, h):
        values = np.frombuffer(p, dtype='u1').reshape(h, w, 3).transpose(2, 0, 1).copy().astype('<f4') / np.float32(255)
        return torch.from_numpy(values[None]).cuda()

    def bytes8(t):
        values = t.detach().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite CUDA result')
        return np.clip(values[0].transpose(1, 2, 0) * np.float32(255) + np.float32(.5), 0, 255).astype('u1').tobytes()

    def worker(index):
        worker = Worker(a.binary, run / f'worker-{index}.log')
        with lock:
            progress_state['worker_pids'].append(worker.process.pid)
            progress('generating')
        try:
            while not stop.is_set():
                with lock:
                    source = next(iterator, None)
                if source is None:
                    break
                local_space(run)
                path = run / 'shards' / (source['id'] + '.imijset')
                temporary = path.with_name(path.name + '.partial')
                if path.exists() or temporary.exists():
                    raise ValueError(f'Unrecorded prior artifact is protected: {path}')
                with lock:
                    archive_paths.append(temporary)
                source_info = worker.send('source', archive=source['archive'], key=source['entry']['key'], output=str(temporary))
                cells = []
                references = []
                for edge in plan['edges']:
                    for q in plan['qualities']:
                        for sampling in ['clean'] if q == 0 else plan['sampling']:
                            if stop.is_set():
                                raise RuntimeError('Peer failure; preserving partial origin')
                            case_id = f"{source['id']}/e{edge}/q{q}/{sampling}"
                            prep = worker.send('prepare', edge=edge, q=q, sampling=sampling, id=case_id)
                            w, h = prep['width'], prep['height']
                            plain = bytes(worker.read(prep.pop('bytes')))
                            feature = prep['features']
                            schema = {k: feature[k] for k in ('names', 'schema_hash', 'config_hash', 'descriptor_hash', 'analyzer_version')}
                            with lock:
                                schema_path = run / 'FEATURE_SCHEMA.json'
                                if not schema_path.exists():
                                    save(schema_path, schema)
                                master = json.loads(schema_path.read_text())
                                if (any(schema[k] != master[k] for k in ('config_hash', 'descriptor_hash', 'analyzer_version'))
                                        or not set(schema['names']) <= set(master['names'])):
                                    raise ValueError('Feature definitions/configuration changed')
                                variant = run / f"FEATURE_SCHEMA-{schema['schema_hash']}.json"
                                if variant.exists() and json.loads(variant.read_text()) != schema:
                                    raise ValueError('Feature-schema digest collision')
                                if not variant.exists():
                                    save(variant, schema)
                            for k in schema:
                                if k != 'schema_hash':
                                    feature.pop(k)
                            if q == 0:
                                references.append(worker.send('references', hr_label=f'e{edge}/hr', lr_label=f'e{edge}/clean'))
                            decoded = worker.send('put', plain, width=w, height=h, label=case_id + '/decoded', score=False)
                            variants = dict(direct=plain)
                            repair_records = []
                            if q:
                                with gpu_lock, torch.inference_mode():
                                    raw = repair_model(tensor(plain, w, h)).cpu().numpy().astype('<f4', copy=False)
                                raw_bytes = raw.tobytes()
                                if a.pilot and edge == plan['edges'][0] and q in (5, 50, 100):
                                    n = worker.send('native', weights=str(a.canon4))['bytes']
                                    native = np.frombuffer(worker.read(n), dtype='<f4').reshape(raw.shape)
                                    native8 = np.clip(native[0].transpose(1, 2, 0) * np.float32(255) + np.float32(.5), 0, 255).astype('u1').tobytes()
                                    gpu8 = np.clip(raw[0].transpose(1, 2, 0) * np.float32(255) + np.float32(.5), 0, 255).astype('u1').tobytes()
                                    delta = int(np.abs(np.frombuffer(native8, 'u1').astype('i2') - np.frombuffer(gpu8, 'u1')).max())
                                    control = worker.send('compare', native8 + gpu8, width=w, height=h)
                                    if delta > 1 or control['ssim2'] < 99.99:
                                        raise ValueError(f'GPU canon4 differs from native: {delta}, {control}')
                                    prep['native_control'] = dict(max_rgb8_delta=delta, max_f32_delta=float(np.abs(native - raw).max()), **control)
                                repair = worker.send('raw', raw_bytes, id=case_id)
                                full = bytes(worker.read(repair['bytes']))
                                half = ((np.frombuffer(plain, 'u1').astype('u2') + np.frombuffer(full, 'u1').astype('u2') + 1) // 2).astype('u1').tobytes()
                                variants.update(half=half, repaired=full)
                                repair_records = repair['records']
                                repair_records.append(worker.send('put', half, width=w, height=h, label=case_id + '/half-repair', score=True))
                            sr = []
                            for name, model in models.items():
                                for stage, pixels in variants.items():
                                    with gpu_lock, torch.inference_mode():
                                        result = model(tensor(pixels, w, h))
                                        if tuple(result.shape) != (1, 3, h * 4, w * 4):
                                            raise ValueError('SR output dimensions differ')
                                        encoded = bytes8(result)
                                    r = worker.send('put', encoded, width=w * 4, height=h * 4,
                                                    label=case_id + f'/{name}/{stage}', score=True)
                                    sr.append(dict(model=name, repair=stage, **r))
                            cells.append(dict(id=case_id, edge=edge, q=q, sampling=sampling, **prep,
                                              decoded=decoded, repair=repair_records, sr=sr))
                            with lock:
                                progress_state['completed_cells'] += 1
                                if progress_state['completed_cells'] % 16 == 0:
                                    local_space(run)
                                    progress()
                finish = worker.send('finish')
                temporary.rename(path)
                with lock:
                    archive_paths.remove(temporary)
                    archive_paths.append(path)
                record = dict(source=source, source_info=source_info, identity_sha256=sha(identity_path),
                              archive=str(path), archive_sha256=sha(path), archive_bytes=path.stat().st_size,
                              entries=finish['entries'], references=references, cells=cells)
                save(run / 'shards' / (source['id'] + '.json'), record)
                with lock:
                    progress_state['completed_origins'] += 1
                    progress()
                    print(f"origins {progress_state['completed_origins']}/{len(sources)} cells {progress_state['completed_cells']}/{progress_state['required_cells']}", flush=True)
        except BaseException:
            stop.set()
            raise
        finally:
            worker.close(check=not stop.is_set())

    try:
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for f in as_completed([pool.submit(worker, i) for i in range(a.jobs)]):
                f.result()
        if progress_state['completed_cells'] != progress_state['required_cells']:
            raise ValueError('Incomplete bank coverage')
        progress('complete')
        save(run / 'COMPLETE.json', dict(**progress_state,
                                        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    except BaseException as error:
        progress('failed', repr(error))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    f = commands.add_parser('fetch')
    f.add_argument('--archive-base', required=True)
    f.add_argument('--output', type=Path, required=True)
    f.set_defaults(function=fetch)
    f = commands.add_parser('freeze')
    f.add_argument('--output', type=Path, required=True)
    f.add_argument('--corpus', type=Path, required=True)
    f.add_argument('--corpus-revision')
    f.add_argument('--binary', type=Path, required=True)
    f.add_argument('--train-per-category', type=int, default=8)
    f.add_argument('--eval-per-category', type=int, default=4)
    f.set_defaults(function=freeze)
    f = commands.add_parser('generate')
    for name in ('output', 'binary', 'canon4', 'spanf-root', 'spanf-f32', 'spanf-mixed', 'dscf-root'):
        f.add_argument('--' + name, type=Path, required=True)
    f.add_argument('--run', required=True)
    f.add_argument('--revision', required=True)
    f.add_argument('--jobs', type=int, default=4)
    f.add_argument('--pilot', type=int, default=0)
    f.add_argument('--reuse-from')
    f.set_defaults(function=generate)
    a = p.parse_args()
    a.function(a)


if __name__ == '__main__':
    main()
