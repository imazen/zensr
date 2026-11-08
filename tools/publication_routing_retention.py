#!/usr/bin/env python3
"""Plan and explicitly apply removal of scored output pixels/maps, keeping inputs."""

import argparse
import csv
import datetime
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat

from publication_routing_cells import digest, required_coverage, write_record
from publication_routing_tables import cell_bytes

OUTPUTS = ('decoded_artifact', 'restored_artifact',
           'identity_butter_map', 'restored_butter_map')
KEEP = ('jpeg_artifact', 'reference_artifact')
METADATA = ('COMPLETE', 'COVERAGE.tsv', 'IDENTITY.tsv', 'FEATURES.tsv', 'ENCODERS.tsv')
PATTERN = re.compile(r'artifacts/([0-9a-f]{2})/([0-9a-f]{64})\.(rgb8|butter-f32le)\.zst')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def persist(path, record):
    write_record(path, record)
    sync_directory(path.parent)


def stopped():
    # Global refusal also covers writers using a different bind-mount alias.
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = p.read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if args and Path(os.fsdecode(args[0])).name == 'routing_study':
            raise ValueError('a routing_study writer is active; stop it before retention changes')


def identity(root):
    return {name: digest(root / name) for name in METADATA}


def artifact(root, value):
    match = PATTERN.fullmatch(value)
    if not match or match[1] != match[2][:2]:
        raise ValueError('invalid output artifact path')
    p = root / value
    if any(parent.is_symlink() for parent in (root / 'artifacts', p.parent)):
        raise ValueError('symlink artifact directory')
    if p.is_symlink():
        raise ValueError('symlink output artifact')
    return p


def signature(path):
    s = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(s.st_mode):
        raise ValueError('output artifact is not a regular file')
    return dict(dev=s.st_dev, ino=s.st_ino, size=s.st_size, mtime=s.st_mtime_ns,
                allocated=s.st_blocks * 512, links=s.st_nlink)


def plan(root, tables, output):
    stopped()
    root = root.resolve(strict=True)
    if not (root / 'COMPLETE').is_file():
        raise ValueError('completed grid required before dropping outputs')
    if (root / 'ARTIFACT_RETENTION.json').exists():
        raise ValueError('retention record already exists; use its original plan to resume')
    original = identity(root)
    if identity(tables) != original:
        raise ValueError('table source metadata differs from completed run')
    required = required_coverage(root)
    covered = set()
    protected = set()
    candidates = set()
    cell_hash = hashlib.sha256()
    count = 0
    for name, data in cell_bytes(tables):
        cell_hash.update(name.encode())
        cell_hash.update(data)
        rows = list(csv.DictReader(io.StringIO(data.decode()), delimiter='\t'))
        if len(rows) != 1:
            raise ValueError('incomplete scored cell')
        row = rows[0]
        key = tuple(row[c] for c in ('file', 'size', 'encoder', 'ss', 'nominal_q'))
        if key in covered:
            raise ValueError('duplicate scored cell')
        covered.add(key)
        for arm in ('identity', 'restored'):
            for metric in ('ssim2', 'butter_max', 'butter_pnorm3'):
                if not Decimal(row[arm + '_' + metric]).is_finite():
                    raise ValueError('nonfinite score')
        for column in KEEP:
            p = root / row[column]
            if not row[column] or not p.is_file() or p.resolve().is_relative_to(root) is False:
                raise ValueError('missing or escaping retained input')
            protected.add(str(p.resolve().relative_to(root)))
        for column in OUTPUTS:
            if row[column]:
                artifact(root, row[column])
                candidates.add(row[column])
        count += 1
        if count % 8192 == 0:
            print(f'planned roles from {count} scored cells', flush=True)
    if covered != required:
        raise ValueError('scored cells differ from required coverage')
    if identity(root) != original:
        raise ValueError('run metadata changed during planning')
    candidates -= protected
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / 'outputs.jsonl'
    allocated = logical = 0
    with manifest.open('x') as f:
        for value in sorted(candidates):
            s = signature(artifact(root, value))
            f.write(json.dumps(dict(path=value, **s)) + '\n')
            allocated += s['allocated']
            logical += s['size']
        f.flush()
        os.fsync(f.fileno())
    record = dict(schema='zensr-scored-output-retention-v1', created=now(),
                  root=str(root), tool_sha256=digest(Path(__file__)), metadata=original,
                  cells=count, cells_sha256=cell_hash.hexdigest(),
                  protected_inputs=len(protected), output_files=len(candidates),
                  output_allocated_bytes=allocated, output_logical_bytes=logical,
                  outputs_sha256=digest(manifest), policy='keep references, JPEGs, scores and identities; drop scored RGB outputs and metric maps')
    persist(output / 'PLAN.json', record)
    sync_directory(output.parent)
    print(json.dumps(record), flush=True)
    return record


def entries(output):
    with (output / 'outputs.jsonl').open() as f:
        for line in f:
            yield json.loads(line)


def apply(root, output):
    stopped()
    root = root.resolve(strict=True)
    record = json.loads((output / 'PLAN.json').read_text())
    plan_hash = digest(output / 'PLAN.json')
    if record['schema'] != 'zensr-scored-output-retention-v1' or record['root'] != str(root):
        raise ValueError('plan belongs to a different run')
    if digest(output / 'outputs.jsonl') != record['outputs_sha256']:
        raise ValueError('output plan changed')
    if identity(root) != record['metadata']:
        raise ValueError('run metadata changed since plan')
    marker = root / 'ARTIFACT_RETENTION.json'
    previous = json.loads(marker.read_text()) if marker.exists() else None
    if previous and previous['plan_sha256'] != plan_hash:
        raise ValueError('different retention plan already started')
    # Validate the entire plan before the first mutation. Resume only a plan
    # whose intent was durably recorded before any of its files were removed.
    count = 0
    for entry in entries(output):
        p = artifact(root, entry['path'])
        if not p.exists():
            if previous:
                count += 1
                continue
            raise ValueError('planned output is missing before first apply')
        expected = {k: entry[k] for k in signature(p)}
        if signature(p) != expected:
            raise ValueError('planned output changed')
        count += 1
    if count != record['output_files']:
        raise ValueError('planned output count differs')
    stopped()
    state = dict(stage='dropping-scored-outputs', checked=now(),
                 plan_sha256=plan_hash, plan=str(output),
                 original_artifact_audit_is_current=False, **record)
    persist(marker, state)
    audited = root / 'AUDITED.json'
    if audited.exists():
        history = root / 'audit-history'
        history.mkdir(exist_ok=True)
        target = history / ('before-retention-' + digest(audited) + '.json')
        if target.exists():
            if target.read_bytes() != audited.read_bytes():
                raise ValueError('audit-history collision')
            audited.unlink()
        else:
            audited.rename(target)
        sync_directory(history)
        sync_directory(root)
    removed = absent = allocated = logical = 0
    for entry in entries(output):
        p = artifact(root, entry['path'])
        if not p.exists():
            absent += 1
            continue
        if signature(p) != {k: entry[k] for k in signature(p)}:
            raise ValueError('planned output changed while applying')
        p.unlink()
        removed += 1
        logical += entry['size']
        if entry['links'] == 1:
            allocated += entry['allocated']
        if removed % 4096 == 0:
            state.update(removed_this_attempt=removed, absent_this_attempt=absent,
                         unlinked_allocated_bytes_this_attempt=allocated, checked=now())
            persist(marker, state)
            print(f'dropped {removed} scored output files', flush=True)
    for prefix in (root / 'artifacts').iterdir():
        if prefix.is_dir() and not prefix.is_symlink():
            sync_directory(prefix)
    if identity(root) != record['metadata']:
        raise ValueError('run metadata changed during apply')
    state.update(stage='complete', checked=now(), removed_this_attempt=removed,
                 absent_this_attempt=absent, unlinked_logical_bytes_this_attempt=logical,
                 unlinked_allocated_bytes_this_attempt=allocated)
    persist(marker, state)
    print(json.dumps(state), flush=True)
    return state


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['plan', 'apply'])
    p.add_argument('root', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--tables', type=Path)
    a = p.parse_args()
    os.nice(19)
    if a.action == 'plan':
        plan(a.root, a.tables or a.root, a.output)
    else:
        apply(a.root, a.output)
