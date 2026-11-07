#!/usr/bin/env python3
"""Reconstruct v4 pair provenance by seeded replay and exact saved-target checks.

Reads existing arrays; never regenerates JPEGs or changes training artifacts.
An identity is emitted only after its RGB crop equals the saved target exactly.
Recipe verification and immutable input hashes must pass before publication of
the final manifest names. This is reconstructed provenance, not an original log
or a uniqueness proof against other sources with identical pixels.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

import cv2
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pixel_digest(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def replay_crops(files, want, rng, crop, reader):
    """The historical gen_crops order, RNG calls, contrast test and RGB layout."""
    if not files:
        raise ValueError('empty source pool')
    produced = visits = stalled = 0
    while produced < want:
        path = files[visits % len(files)]
        visits += 1
        image = reader(path)
        if image is None or min(image.shape[:2]) < crop:
            stalled += 1
            if stalled >= len(files):
                raise ValueError('no readable source large enough for this crop')
            continue
        stalled = 0
        for _ in range(6):
            if produced >= want:
                break
            y = rng.randrange(0, image.shape[0] - crop + 1)
            x = rng.randrange(0, image.shape[1] - crop + 1)
            bgr = image[y:y + crop, x:x + crop]
            if cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).std() < 8.0:
                continue
            yield path, x, y, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), image
            produced += 1


def verify_crop(actual, saved, idx):
    if actual.shape != saved.shape or not np.array_equal(actual, saved):
        raise ValueError(f'pair {idx}: replay crop differs from saved HR; identity unresolved')


def verify_recipe(rows, meta, rng):
    """Replay the encoder draws that follow ALL crop draws, without encoding."""
    for idx, row in enumerate(rows):
        clean = rng.random() < 0.05
        r = rng.random()
        n = 3 if r < meta['gen3'] else 2 if r < meta['gen3'] + meta['gen2'] else 1
        for _ in range(n):
            encoder = rng.choice(meta['encoders'])
            subsampling = '420' if rng.random() < 0.6 else '444'
            quality = rng.randrange(5, 97)
        actual = (encoder, subsampling, quality, int(clean), n)
        recorded = (row['encoder'], row['ss'], int(row['q']), int(row['clean']), int(row['gens']))
        if actual != recorded:
            raise ValueError(f'pair {idx}: recipe replay {actual} != recorded {recorded}')


def recover(dataset, output, corpus, model_meta):
    output.mkdir(parents=True, exist_ok=False)
    state = dict(stage='initializing', completed=0, required=None,
                 pid=os.getpid(), failure=None, output=str(output))

    def progress(**updates):
        state.update(updates)
        from datetime import datetime, timezone
        state['last_check'] = datetime.now(timezone.utc).isoformat()
        (output / 'progress.json').write_text(json.dumps(state, indent=2) + '\n')

    progress()
    try:
        meta = json.loads((dataset / 'meta.json').read_text())
        with (dataset / 'pairs.tsv').open() as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
        n, tail, crop = meta['n'], meta['val_tail'], meta['crop']
        if len(rows) != n or [int(r['idx']) for r in rows] != list(range(n)):
            raise ValueError('pair ledger must have every index exactly once in order')
        if not 0 < tail < n:
            raise ValueError('invalid internal validation boundary')
        hr = np.load(dataset / 'hr_u8.npy', mmap_mode='r', allow_pickle=False)
        lr = np.load(dataset / 'lr_u8.npy', mmap_mode='r', allow_pickle=False)
        for array in (hr, lr):
            if array.shape != (n, crop, crop, 3) or array.dtype != np.uint8:
                raise ValueError('unexpected RGB uint8 array shape/type')
        input_paths = [dataset / name for name in
                       ('meta.json', 'pairs.tsv', 'hr_u8.npy', 'lr_u8.npy',
                        'cond_scalar_f32.npy', 'dmap_u16.npy')]
        inputs = {p.name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p in input_paths}
        # Reuse the existing corpus adapter and decoder, with explicit settings.
        os.environ['IMAZEN26_REPO'] = str(corpus)
        os.environ['ZENSR_ROOT'] = str(corpus)
        os.environ.pop('ZENSR_CLEAN_GT', None)
        os.environ.pop('ZENSR_SUBS', None)
        os.environ['ZENSR_FOLDER_CAP'] = '0'
        import make_distill_data as historical
        from corpus_split import split_map
        from imazen26_canonical import split_rows
        canonical = {r['path']: r for r in split_rows()}
        effective = split_map()
        rng = random.Random(meta['seed'])
        files = historical.build_pool(subs=meta['subs'], rng=rng)
        rng.shuffle(files)
        nv = max(16, len(files) // 20)
        partitions = [('model_train', files[:-nv], n - tail),
                      ('internal_validation', files[-nv:], tail)]
        sources = {}
        counts = collections.defaultdict(collections.Counter)
        fields = ['idx', 'partition', 'imazen_id', 'source_path', 'canonical_split',
                  'effective_split', 'x', 'y', 'width', 'height', 'source_sha256',
                  'hr_rgb_sha256', 'lr_rgb_sha256', 'encoder', 'ss', 'q', 'clean',
                  'severity', 'gens', 'verification']
        progress(stage='replaying_crops', required=n,
                 source_pool=len(files), train_pool=len(files) - nv, validation_pool=nv)
        idx = 0
        with (output / 'pairs.partial.tsv').open('x') as f:
            writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
            writer.writeheader()
            for partition, pool, want in partitions:
                for path, x, y, rgb, image in replay_crops(
                        pool, want, rng, crop, historical.read_image_bgr):
                    verify_crop(rgb, hr[idx], idx)
                    source = Path(path).relative_to(corpus).as_posix()
                    origin = canonical[source]
                    if source not in sources:
                        sources[source] = dict(imazen_id=origin['id'], source_path=source,
                            canonical_split=origin['split'], effective_split=effective[source],
                            source_sha256=digest(path), manifest_sha256=origin['sha256'],
                            width=image.shape[1], height=image.shape[0],
                            train_pairs=0, validation_pairs=0)
                    src = sources[source]
                    src['train_pairs' if partition == 'model_train' else 'validation_pairs'] += 1
                    row = rows[idx]
                    if int(row['clean']):
                        verify_crop(lr[idx], hr[idx], idx)
                    writer.writerow(dict(idx=idx, partition=partition,
                        imazen_id=origin['id'], source_path=source,
                        canonical_split=origin['split'], effective_split=effective[source],
                        x=x, y=y, width=crop, height=crop, source_sha256=src['source_sha256'],
                        hr_rgb_sha256=pixel_digest(hr[idx]), lr_rgb_sha256=pixel_digest(lr[idx]),
                        **{k: row[k] for k in ('encoder', 'ss', 'q', 'clean', 'severity', 'gens')},
                        verification='exact_saved_hr_crop_replay'))
                    counts[partition][origin['split']] += 1
                    idx += 1
                    if idx % 1000 == 0 or idx == n:
                        f.flush()
                        progress(completed=idx, sources=len(sources))
                        print(f'exact crop matches {idx}/{n}; sources {len(sources)}', flush=True)
            f.flush()
            os.fsync(f.fileno())
        progress(stage='verifying_recipe')
        verify_recipe(rows, meta, rng)
        if any(digest(p) != inputs[p.name]['sha256'] for p in input_paths):
            raise ValueError('dataset input bytes changed during reconstruction')
        for source, row in sources.items():
            if digest(corpus / source) != row['source_sha256']:
                raise ValueError(f'source changed during reconstruction: {source}')
        overlap = {s for s, r in sources.items() if r['train_pairs'] and r['validation_pairs']}
        if overlap:
            raise ValueError('internal train/validation source overlap')
        source_fields = list(next(iter(sources.values())))
        with (output / 'sources.tsv').open('x') as f:
            writer = csv.DictWriter(f, fieldnames=source_fields, delimiter='\t')
            writer.writeheader()
            writer.writerows(sources[s] for s in sorted(sources))
        (output / 'pairs.partial.tsv').rename(output / 'pairs.tsv')
        source_counts = {part: dict(collections.Counter(
            r['canonical_split'] for r in sources.values() if r[key]))
            for part, key in [('model_train', 'train_pairs'), ('internal_validation', 'validation_pairs')]}
        summary = dict(schema=1, status='complete', pairs=n, exact_target_matches=idx,
            recipe_matches=n, sources=len(sources), pair_counts_by_canonical_split=dict(counts),
            source_counts_by_canonical_split=source_counts, internal_source_overlap=len(overlap),
            source_bytes_match_manifest=sum(r['source_sha256'] == r['manifest_sha256'] for r in sources.values()),
            dataset_inputs=inputs, dataset_meta=meta, model_metadata_sha256=digest(model_meta),
            corpus_revision=subprocess.check_output(['git', '-C', str(corpus), 'rev-parse', 'HEAD'], text=True).strip(),
            corpus_inputs={name: digest(corpus / name) for name in ['manifests/split_map.tsv', 'CORPUS-MANIFEST.tsv']},
            tool_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            tool_sources={name: digest(Path(__file__).parent / name) for name in
                ['recover_pair_sources.py', 'make_distill_data.py', 'corpus_split.py', 'imazen26_canonical.py']},
            environment=dict(numpy=np.__version__, opencv=cv2.__version__),
            outputs={name: digest(output / name) for name in ['pairs.tsv', 'sources.tsv']},
            limitation='Reconstructed by exact target/seeded-recipe replay, not contemporaneous source logging. '
                'Does not prove uniqueness against pixel-identical sources or reproduce JPEG encoder binaries. '
                'Only clean LR pairs are checked against HR; other LR bytes are bound by hashes, not regenerated.')
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        progress(stage='complete', completed=n, sources=len(sources), summary_sha256=digest(output / 'summary.json'))
        print(json.dumps({k: summary[k] for k in ['status', 'pairs', 'sources',
            'pair_counts_by_canonical_split', 'source_counts_by_canonical_split']}, indent=2), flush=True)
    except Exception as error:
        progress(stage='failed', failure=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--model-meta', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    recover(args.dataset.resolve(), args.output.resolve(), args.corpus.resolve(), args.model_meta.resolve())
