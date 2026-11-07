#!/usr/bin/env python3
"""Build ID-based pair assets; assemble canonical-only data for the trainer."""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

import numpy as np

from recover_pair_sources import digest

ARRAYS = ['lr_u8.npy', 'hr_u8.npy', 'cond_scalar_f32.npy', 'dmap_u16.npy']
FIELDS = ['idx', 'encoder', 'ss', 'q', 'clean', 'severity', 'gens', 'partition',
          'imazen_id', 'canonical_split', 'effective_split', 'x', 'y', 'width',
          'height', 'source_sha256', 'hr_rgb_sha256', 'lr_rgb_sha256', 'verification']
CORPUS_REV = '187fbf338ce08e8e6654db7f04ddae58d5263da2'


def read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write_rows(path, rows, fields):
    with path.open('x') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def json_file(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def split_rows(rows):
    """Balance two canonical shards by whole source; retain all other rows."""
    grouped = collections.defaultdict(list)
    other = []
    for row in rows:
        if row['canonical_split'] == 'train':
            grouped[row['imazen_id']].append(row)
        else:
            other.append(row)
    bins = [[], []]
    for source in sorted(grouped):
        target = min(range(2), key=lambda i: len(bins[i]))
        bins[target].extend(grouped[source])
    result = [sorted(b, key=lambda r: int(r['idx'])) for b in bins] + [other]
    if sorted(int(r['idx']) for b in result for r in b) != list(range(len(rows))):
        raise ValueError('incomplete, duplicate or unordered original pair indices')
    return result


def copy_selection(source, destination, indices):
    target = np.lib.format.open_memmap(destination, mode='w+', dtype=source.dtype,
                                     shape=(len(indices), *source.shape[1:]))
    for start in range(0, len(indices), 64):
        target[start:start + 64] = source[indices[start:start + 64]]
    target.flush()
    del target


def make_archive(folder, destination, names):
    if destination.exists():
        raise ValueError('archive already exists')
    subprocess.run(['tar', '-I', 'zstd -T1 -3', '-cf', str(destination),
                    '-C', str(folder), *names], check=True)
    if destination.stat().st_size >= 2 ** 31:
        raise ValueError('asset exceeds GitHub per-file limit')
    return dict(name=destination.name, bytes=destination.stat().st_size,
                sha256=digest(destination), members={n: digest(folder / n) for n in names})


def extract_checked(archive, destination, members):
    """Stream safe named members and verify them; no tar metadata is applied."""
    destination.mkdir(parents=True, exist_ok=False)
    proc = subprocess.Popen(['zstd', '-dc', str(archive)], stdout=subprocess.PIPE)
    found = set()
    try:
        with tarfile.open(fileobj=proc.stdout, mode='r|') as tar:
            for member in tar:
                if member.name not in members or member.name in found or not member.isfile() or '/' in member.name:
                    raise ValueError('unexpected or unsafe archive member')
                found.add(member.name)
                h = hashlib.sha256()
                with tar.extractfile(member) as src, (destination / member.name).open('xb') as dst:
                    for block in iter(lambda: src.read(1024 * 1024), b''):
                        dst.write(block)
                        h.update(block)
                if h.hexdigest() != members[member.name]:
                    raise ValueError('archive member hash mismatch')
        if proc.wait() != 0 or found != set(members):
            raise ValueError('incomplete archive')
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
            proc.wait()


def build(dataset, manifest, output):
    output.mkdir(parents=True, exist_ok=False)
    def progress(stage, **fields):
        from datetime import datetime, timezone
        (output/'progress.json').write_text(json.dumps(dict(stage=stage, pid=os.getpid(),
            last_check=datetime.now(timezone.utc).isoformat(), **fields), indent=2)+'\n')
    progress('checking_inputs')
    summary = json.loads((manifest / 'summary.json').read_text())
    checked = json.loads((manifest / 'verification.json').read_text())
    if checked['status'] != 'complete' or digest(manifest / 'summary.json') != checked['reconstruction_summary_sha256']:
        raise ValueError('coordinate verification not bound to reconstruction')
    for name, record in summary['dataset_inputs'].items():
        if digest(dataset / name) != record['sha256']:
            raise ValueError('dataset bytes changed: ' + name)
    for name, sha in summary['outputs'].items():
        if digest(manifest / name) != sha:
            raise ValueError('source manifest bytes changed')
    rows = read_rows(manifest / 'pairs.tsv')
    sources = read_rows(manifest / 'sources-with-links.tsv')
    if digest(manifest / 'sources-with-links.tsv') != checked['sources_with_links_sha256']:
        raise ValueError('source links inventory changed')
    public = output / 'manifest'
    public.mkdir()
    write_rows(public / 'pairs.tsv', rows, FIELDS)
    source_fields = ['imazen_id', 'canonical_split', 'effective_split', 'source_sha256',
                     'manifest_sha256', 'width', 'height', 'train_pairs', 'validation_pairs', 'license']
    write_rows(public / 'sources.tsv', sources, source_fields)
    arrays = {name: np.load(dataset / name, mmap_mode='r', allow_pickle=False) for name in ARRAYS}
    bundles = []
    for number, selected in enumerate(split_rows(rows)):
        name = f'canon4-canonical-train-{number + 1:02d}' if number < 2 else 'canon4-canonical-heldout'
        folder = output / name
        folder.mkdir()
        indices = np.array([int(r['idx']) for r in selected], dtype=np.int64)
        for array_name, source in arrays.items():
            progress('copying_array', bundle=name, array=array_name, rows=len(selected), completed_bundles=len(bundles))
            copy_selection(source, folder / array_name, indices)
        np.save(folder / 'original_indices.npy', indices, allow_pickle=False)
        write_rows(folder / 'pairs.tsv', selected, FIELDS)
        details = dict(name=name, kind='canonical-train' if number < 2 else 'canonical-heldout',
                       rows=len(selected), partitions=dict(collections.Counter(r['partition'] for r in selected)),
                       canonical_buckets=dict(collections.Counter(r['canonical_split'] for r in selected)))
        # Verify every copied LR/HR row against its source manifest before packaging.
        for array_name, field in [('hr_u8.npy', 'hr_rgb_sha256'), ('lr_u8.npy', 'lr_rgb_sha256')]:
            copied = np.load(folder / array_name, mmap_mode='r', allow_pickle=False)
            for i, row in enumerate(selected):
                if hashlib.sha256(copied[i].tobytes()).hexdigest() != row[field]:
                    raise ValueError('copied pixel mismatch')
            del copied
        json_file(folder / 'bundle.json', details)
        progress('compressing', bundle=name, completed_bundles=len(bundles))
        bundle = make_archive(folder, output / (name + '.tar.zst'),
                              [*ARRAYS, 'original_indices.npy', 'pairs.tsv', 'bundle.json'])
        bundle.update(details)
        bundles.append(bundle)
        json_file(output / (name + '.asset.json'), bundle)
        print(json.dumps({k: bundle[k] for k in ['name', 'rows', 'bytes', 'sha256']}) , flush=True)
    index = dict(schema=1, dataset='canon4-existing-v4-pairs', original_rows=len(rows),
        reconstruction_summary_sha256=digest(manifest / 'summary.json'),
        coordinate_verification_sha256=digest(manifest / 'verification.json'),
        corpus_revision=CORPUS_REV,
        canonical_indexes=[f'https://github.com/imazen/imazen-26/blob/{CORPUS_REV}/manifests/{n}.tsv'
                           for n in ['train', 'validate', 'test']],
        bundles=bundles, manifest_files={n: digest(public / n) for n in ['pairs.tsv', 'sources.tsv']},
        dataset_recipe=summary['dataset_meta'],
        limitations=['Reconstructed identities, not contemporaneous source logs.',
            'Source-byte drift versus older corpus hashes remains unreconciled.',
            'ID-based metadata omits descriptive paths and machine logs. Image pixels have not been privacy-reviewed.',
            'Licence labels preserve corpus declarations; screenshot-unverified sources remain identified.'])
    json_file(output / 'bundle-index.json', index)
    progress('complete', completed_bundles=len(bundles))
    print('build complete', flush=True)


def assemble(index_path, output, policy):
    index = json.loads(index_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for bundle in index['bundles']:
        if policy == 'canonical-train' and bundle['kind'] != policy:
            continue
        # bundle name is a logical identifier, archive filename has .tar.zst.
        archive = index_path.parent / (bundle['name'] + '.tar.zst')
        if digest(archive) != bundle['sha256']:
            raise ValueError('archive hash mismatch')
        folder = output / 'input-shards' / bundle['name']
        extract_checked(archive, folder, bundle['members'])
        rows = read_rows(folder / 'pairs.tsv')
        original = np.load(folder / 'original_indices.npy', allow_pickle=False)
        if list(original) != [int(r['idx']) for r in rows] or len(rows) != bundle['rows']:
            raise ValueError('shard row/index mismatch')
        arrays = {n: np.load(folder / n, mmap_mode='r', allow_pickle=False) for n in ARRAYS}
        records.extend((int(r['idx']), r, arrays, i) for i, r in enumerate(rows))
    records.sort(key=lambda item: item[0])
    if not records or len({r[0] for r in records}) != len(records):
        raise ValueError('missing or duplicated pair rows')
    if policy == 'canonical-train' and any(r[1]['canonical_split'] != 'train' for r in records):
        raise ValueError('canonical held-out source admitted to training dataset')
    # Preserve the original internal split, with the validation rows at the tail.
    records.sort(key=lambda r: (r[1]['partition'] != 'model_train', r[0]))
    train_ids = {r[1]['imazen_id'] for r in records if r[1]['partition'] == 'model_train'}
    val_ids = {r[1]['imazen_id'] for r in records if r[1]['partition'] == 'internal_validation'}
    if train_ids & val_ids:
        raise ValueError('internal source overlap')
    tail = sum(r[1]['partition'] == 'internal_validation' for r in records)
    if not 0 < tail < len(records):
        raise ValueError('invalid internal validation tail')
    for name in ARRAYS:
        source = records[0][2][name]
        target = np.lib.format.open_memmap(output / name, mode='w+', dtype=source.dtype,
                                          shape=(len(records), *source.shape[1:]))
        for i, (_, _, arrays, local) in enumerate(records):
            target[i] = arrays[name][local]
        target.flush()
        del target
    fields = [*FIELDS, 'original_idx']
    write_rows(output / 'pairs.tsv', [dict(r[1], idx=i, original_idx=r[0]) for i, r in enumerate(records)], fields)
    meta = dict(index['dataset_recipe'], n=len(records), val_tail=tail,
                source_policy=policy, bundle_index_sha256=digest(index_path),
                reconstructed_from_original_indices=True)
    json_file(output / 'meta.json', meta)
    print(json.dumps(dict(rows=len(records), training_rows=len(records) - tail, validation_rows=tail,
                          source_policy=policy)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    for name in ['dataset', 'manifest', 'output']:
        b.add_argument('--' + name, type=Path, required=True)
    a = sub.add_parser('assemble')
    a.add_argument('--index', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--policy', choices=['canonical-train', 'historical'], default='canonical-train')
    args = parser.parse_args()
    if args.command == 'build':
        build(args.dataset.resolve(), args.manifest.resolve(), args.output.resolve())
    else:
        assemble(args.index.resolve(), args.output.resolve(), args.policy)
