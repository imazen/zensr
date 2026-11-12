"""Run the resident blend evaluator against every retained bank image.

Stream scalar results directly into Parquet+zstd, with a complete-grid and
stored-SR endpoint comparison before marking each origin complete.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading

import pyarrow as pa
import pyarrow.parquet as pq
from bank import local_space, save, sha


def main(a):
    if not 1 <= a.jobs <= 8:
        raise ValueError('Expected one to eight bounded workers')
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    root = a.root / a.bank_run
    complete = json.loads((root / 'COMPLETE.json').read_text())
    if complete['stage'] != 'complete' or complete['identity_sha256'] != sha(root / 'IDENTITY.json'):
        raise ValueError('Input bank is incomplete')
    bank_identity = json.loads((root / 'IDENTITY.json').read_text())
    ids = bank_identity['source_ids'][:a.pilot] if a.pilot else bank_identity['source_ids']
    out = a.root / a.run
    out.mkdir(exist_ok=True)
    identity = dict(bank_identity_sha256=sha(root / 'IDENTITY.json'), binary_sha256=sha(a.binary),
                    source_revision=a.revision, script_sha256=sha(Path(__file__)),
                    pyarrow=pa.__version__, python=sys.version, sources=ids, jobs=a.jobs,
                    settings=dict(filters=['triangle', 'mitchell', 'catmull_rom', 'lanczos2'],
                                  intensity_quarters=list(range(5)), scales=[2, 4]),
                    baseline='same post-repair LR as the corresponding SR path',
                    blend_space='encoded sRGB f32; stored RGB8 SR; Box x4->x2; RGB8 final rounding')
    ipath = out / 'IDENTITY.json'
    if ipath.exists() and json.loads(ipath.read_text()) != identity:
        raise ValueError('Resumed blend identity differs')
    if not ipath.exists():
        save(ipath, identity)
    identity_sha = sha(ipath)
    lock = threading.Lock()
    stop = threading.Event()
    progress = dict(stage='scoring', pid=os.getpid(), active_children=[], completed_origins=0,
                    required_origins=len(ids), completed_scores=0, identity_sha256=identity_sha)

    def update():
        progress['checked_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(out / 'PROGRESS.json', progress)

    def worker(rid):
        source_record = root / 'shards' / (rid + '.json')
        source = json.loads(source_record.read_text())
        expected = {(c['id'], r['model'], r['repair'], scale, kernel, intensity)
                    for c in source['cells'] for r in c['sr'] for scale in [2, 4]
                    for kernel in identity['settings']['filters'] for intensity in range(5)}
        record_path = out / (rid + '.json')
        parquet = out / (rid + '.parquet')
        if record_path.exists():
            r = json.loads(record_path.read_text())
            if r['identity_sha256'] != identity_sha or r['parquet_sha256'] != sha(parquet) or r['rows'] != len(expected):
                raise ValueError('Completed blend record differs')
            return r
        partial = parquet.with_name(parquet.name + '.partial')
        if parquet.exists() or partial.exists():
            raise ValueError('Unrecorded prior blend output is protected')
        local_space(out)
        env = dict(os.environ, RAYON_NUM_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        log = (out / (rid + '.log')).open('ab')
        process = subprocess.Popen([str(a.binary), str(source_record), '-'], stdout=subprocess.PIPE, stderr=log, env=env)
        with lock:
            progress['active_children'].append(process.pid)
            update()
        stored = {(c['id'], r['model'], r['repair']): r for c in source['cells'] for r in c['sr']}
        observed = set()
        shared_endpoints = {}
        buffer = []
        writer = None
        try:
            for line in process.stdout:
                row = json.loads(line)
                key = tuple(row[k] for k in ['id', 'model', 'repair', 'scale', 'filter', 'intensity_quarter'])
                if key not in expected or key in observed:
                    raise ValueError('Unexpected or duplicate blend cell')
                observed.add(key)
                if not math.isfinite(row['zensim_a']):
                    raise ValueError('Nonfinite zensim-A score')
                row['psnr'] = float(row['psnr'])
                if math.isnan(row['psnr']):
                    raise ValueError('NaN PSNR score')
                endpoint = (row['rgb8_sha256'], row['psnr'], row['zensim_a'])
                if row['intensity_quarter'] in (0, 4):
                    shared = (row['id'], row['repair'], row['scale'], row['filter']) if row['intensity_quarter'] == 0 else (row['id'], row['repair'], row['scale'], row['model'])
                    shared = (row['intensity_quarter'],) + shared
                    if shared in shared_endpoints and shared_endpoints[shared] != endpoint:
                        raise ValueError('Equivalent endpoint paths disagree')
                    shared_endpoints[shared] = endpoint
                if row['intensity_quarter'] == 4 and row['scale'] == 4:
                    reference = stored[(row['id'], row['model'], row['repair'])]
                    if endpoint != (reference['pixels_sha256'], float(reference['scores']['psnr']), reference['scores']['zensim_a']):
                        raise ValueError('Pure-SR endpoint differs from stored generation pixels/scores')
                buffer.append(row)
                if len(buffer) == 4096:
                    table = pa.Table.from_pylist(buffer)
                    if writer is None:
                        writer = pq.ParquetWriter(partial, table.schema, compression='zstd', compression_level=3)
                    writer.write_table(table)
                    buffer.clear()
            if process.wait(timeout=30) != 0 or observed != expected:
                raise ValueError('Blend worker failed or coverage is incomplete')
            if buffer:
                table = pa.Table.from_pylist(buffer)
                if writer is None:
                    writer = pq.ParquetWriter(partial, table.schema, compression='zstd', compression_level=3)
                writer.write_table(table)
            writer.close()
            writer = None
            with partial.open('rb') as f:
                os.fsync(f.fileno())
            if pq.read_metadata(partial).num_rows != len(expected):
                raise ValueError('Parquet row count differs')
            partial.rename(parquet)
            result = dict(source_id=rid, source_record_sha256=sha(source_record), identity_sha256=identity_sha,
                          parquet_sha256=sha(parquet), bytes=parquet.stat().st_size, rows=len(observed),
                          equivalent_endpoint_groups=len(shared_endpoints))
            save(record_path, result)
            return result
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            process.stdout.close()
            if writer is not None:
                writer.close()
            log.close()
            with lock:
                progress['active_children'].remove(process.pid)
                update()

    def supervised(rid):
        if stop.is_set():
            raise RuntimeError('Peer failure; no new source admitted')
        try:
            return worker(rid)
        except BaseException as error:
            stop.set()
            with lock:
                progress.update(stage='failed', error=repr(error))
                update()
            raise

    update()
    results = []
    try:
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for future in as_completed([pool.submit(supervised, rid) for rid in ids]):
                result = future.result()
                results.append(result)
                with lock:
                    progress['completed_origins'] += 1
                    progress['completed_scores'] += result['rows']
                    update()
                    print(f"origins {progress['completed_origins']}/{len(ids)} scores {progress['completed_scores']}", flush=True)
        progress['stage'] = 'complete'
        update()
        save(out / 'COMPLETE.json', dict(**progress, parquet_bytes=sum(r['bytes'] for r in results),
                                        equivalent_endpoint_groups=sum(r['equivalent_endpoint_groups'] for r in results)))
    except BaseException as error:
        progress.update(stage='failed', error=repr(error))
        update()
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--bank-run', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--pilot', type=int, default=0)
    main(p.parse_args())
