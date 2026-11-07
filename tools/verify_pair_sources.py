#!/usr/bin/env python3
"""Check recovered identities directly by coordinates, independently of RNG replay.

Also retain source URLs and licence labels from the canonical manifests.
Never overwrites the reconstruction or changes pair arrays.
"""
import argparse
import collections
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np

from recover_pair_sources import canonical_rows, digest, pixel_digest, verify_crop


def read_tsv(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def coordinate_crop(image, row):
    x, y, width, height = (int(row[k]) for k in ['x', 'y', 'width', 'height'])
    if min(x, y) < 0 or min(width, height) < 1 or x + width > image.shape[1] or y + height > image.shape[0]:
        raise ValueError('crop coordinate outside source image')
    return image[y:y + height, x:x + width, ::-1]


def verify(manifest, dataset, corpus, output):
    output.mkdir(parents=True, exist_ok=False)
    state = dict(stage='checking_inputs', pid=os.getpid(), sources_checked=0, pairs_checked=0)

    def progress(**updates):
        from datetime import datetime, timezone
        state.update(updates, last_check=datetime.now(timezone.utc).isoformat())
        (output / 'progress.json').write_text(json.dumps(state, indent=2) + '\n')

    progress()
    try:
        summary = json.loads((manifest / 'summary.json').read_text())
        if summary['status'] != 'complete':
            raise ValueError('reconstruction is not complete')
        for name, sha in summary['outputs'].items():
            if digest(manifest / name) != sha:
                raise ValueError('reconstructed manifest hash mismatch: ' + name)
        for name, record in summary['dataset_inputs'].items():
            if digest(dataset / name) != record['sha256']:
                raise ValueError('dataset hash mismatch: ' + name)
        for name, sha in summary['corpus_inputs'].items():
            if digest(corpus / name) != sha:
                raise ValueError('corpus manifest hash mismatch: ' + name)
        os.environ['IMAZEN26_REPO'] = str(corpus)
        from make_distill_data import read_image_bgr
        canonical = canonical_rows(corpus)
        content = {r['path']: r for r in read_tsv(corpus / 'CORPUS-MANIFEST.tsv')}
        pairs = read_tsv(manifest / 'pairs.tsv')
        sources = read_tsv(manifest / 'sources.tsv')
        meta = summary['dataset_meta']
        if [int(r['idx']) for r in pairs] != list(range(meta['n'])):
            raise ValueError('pair indices missing, duplicated or reordered')
        hr = np.load(dataset / 'hr_u8.npy', mmap_mode='r', allow_pickle=False)
        lr = np.load(dataset / 'lr_u8.npy', mmap_mode='r', allow_pickle=False)
        grouped = collections.defaultdict(list)
        for row in pairs:
            idx = int(row['idx'])
            expected = 'model_train' if idx < meta['n'] - meta['val_tail'] else 'internal_validation'
            if row['partition'] != expected:
                raise ValueError('incorrect training/validation partition')
            grouped[row['source_path']].append(row)
        if len({r['source_path'] for r in sources}) != len(sources) or set(grouped) != {r['source_path'] for r in sources}:
            raise ValueError('source and pair inventories disagree')
        enriched = []
        progress(stage='checking_crops', required_pairs=len(pairs), required_sources=len(sources))
        verified = 0
        for count, source in enumerate(sources, 1):
            path = source['source_path']
            origin = canonical[path]
            if source['imazen_id'] != origin['id'] or source['canonical_split'] != origin['split']:
                raise ValueError('canonical ID or split mismatch')
            if digest(corpus / path) != source['source_sha256']:
                raise ValueError('source bytes mismatch')
            image = read_image_bgr(str(corpus / path))
            if image is None:
                raise ValueError('source no longer decodes')
            partition_counts = collections.Counter()
            for row in grouped[path]:
                idx = int(row['idx'])
                if row['imazen_id'] != origin['id'] or row['canonical_split'] != origin['split'] or row['source_sha256'] != source['source_sha256']:
                    raise ValueError('pair source identity mismatch')
                rgb = coordinate_crop(image, row)
                verify_crop(rgb, hr[idx], idx)
                if pixel_digest(hr[idx]) != row['hr_rgb_sha256'] or pixel_digest(lr[idx]) != row['lr_rgb_sha256']:
                    raise ValueError('pair RGB hash mismatch')
                partition_counts[row['partition']] += 1
                verified += 1
            if partition_counts['model_train'] != int(source['train_pairs']) or partition_counts['internal_validation'] != int(source['validation_pairs']):
                raise ValueError('per-source pair count mismatch')
            if partition_counts['model_train'] and partition_counts['internal_validation']:
                raise ValueError('internal training/validation source overlap')
            # Preserve all canonical source hyperlinks, including empty slots.
            links = {k: v for k, v in origin.items() if k.endswith('_url')}
            enriched.append(dict(source, **links, license=content[path]['license']))
            if count % 100 == 0 or count == len(sources):
                progress(sources_checked=count, pairs_checked=verified)
                print(f'coordinate verification: {count}/{len(sources)} sources, {verified}/{len(pairs)} pairs', flush=True)
        with (output / 'sources-with-links.tsv').open('x') as f:
            writer = csv.DictWriter(f, fieldnames=list(enriched[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(enriched)
        report = dict(status='complete', pairs=verified, sources=len(sources),
            reconstruction_summary_sha256=digest(manifest / 'summary.json'),
            sources_with_links_sha256=digest(output / 'sources-with-links.tsv'),
            verification_source_sha256=digest(Path(__file__)),
            method='Exact coordinate-based RGB comparison of every saved HR target; every LR/HR pixel hash checked. No RNG or encoder replay.',
            limitation=summary['limitation'])
        (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
        progress(stage='complete')
        print(json.dumps(report, indent=2), flush=True)
    except Exception as error:
        progress(stage='failed', failure=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['manifest', 'dataset', 'corpus', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    verify(args.manifest.resolve(), args.dataset.resolve(), args.corpus.resolve(), args.output.resolve())
