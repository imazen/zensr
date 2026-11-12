"""Verify completed bank records and report split/category/quality comparisons."""
import argparse
from collections import defaultdict
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def validate(record, source, plan, identity_sha):
    if record['source'] != source or record['identity_sha256'] != identity_sha:
        raise ValueError('Source/run identity mismatch')
    archive = Path(record['archive'])
    if archive.stat().st_size != record['archive_bytes'] or sha(archive) != record['archive_sha256']:
        raise ValueError('Retained archive differs')
    grid = {(e, q, s) for e in plan['edges'] for q in plan['qualities']
            for s in (['clean'] if q == 0 else plan['sampling'])}
    if len(record['cells']) != len(grid) or {(c['edge'], c['q'], c['sampling']) for c in record['cells']} != grid:
        raise ValueError('Incomplete or duplicate input grid')
    records = []
    for r in record['references']:
        records.extend([r['hr'], r['lr']])
    for c in record['cells']:
        if not c['id'].startswith(source['id'] + '/'):
            raise ValueError('Cell source identity mismatch')
        if c['features']['width'] != c['width'] or c['features']['height'] != c['height']:
            raise ValueError('Feature geometry mismatch')
        if not all(math.isfinite(x) for x in c['features']['values']):
            raise ValueError('Invalid feature vector')
        if c['features']['pixels_sha256'] != c['decoded']['pixels_sha256']:
            raise ValueError('Feature pixel identity mismatch')
        if c['jpeg']:
            records.append(c['jpeg'])
        records.append(c['decoded'])
        records.extend(c['repair'])
        records.extend(c['sr'])
        expected = {(m, r) for m in ('spanf', 'dscf')
                    for r in (['direct'] if c['q'] == 0 else ['direct', 'half', 'repaired'])}
        if {(r['model'], r['repair']) for r in c['sr']} != expected or len(c['sr']) != len(expected):
            raise ValueError('SR model/repair coverage mismatch')
        for r in c['repair'] + c['sr']:
            for name, x in r.get('scores', {}).items():
                if x is not None and x != 'inf' and not math.isfinite(x):
                    raise ValueError(f'Nonfinite {name}')
    if {r['key'] for r in records} != set(range(record['entries'])) or len(records) != record['entries']:
        raise ValueError('Archive entry ledger differs')


def differences(scores, baseline, metrics):
    result = {k: scores[k] - baseline[k] for k in metrics}
    a, b = scores['psnr'], baseline['psnr']
    result['psnr'] = a - b if a != 'inf' and b != 'inf' else (0. if a == b else None)
    result['psnr_from_exact'] = a != 'inf' and b == 'inf'
    result['psnr_to_exact'] = a == 'inf' and b != 'inf'
    return result


def main(a):
    plan = json.loads((a.root / 'PLAN.json').read_text())
    run = a.root / a.run
    identity = json.loads((run / 'IDENTITY.json').read_text())
    identity_sha = sha(run / 'IDENTITY.json')
    if identity['plan_sha256'] != sha(a.root / 'PLAN.json'):
        raise ValueError('Frozen plan changed')
    members = {r['id']: r for r in plan['sources']}
    groups = defaultdict(list)
    all_records = []
    controls = []
    images = 0
    sr_images = 0
    stored_bytes = 0
    pixel_bytes = 0
    for rid in identity['source_ids']:
        record = json.loads((run / 'shards' / (rid + '.json')).read_text())
        validate(record, members[rid], plan, identity_sha)
        all_records.append(record)
        stored_bytes += record['archive_bytes']
        for ref in record['references']:
            images += 2
            pixel_bytes += ref['hr']['pixel_bytes'] + ref['lr']['pixel_bytes']
        for c in record['cells']:
            images += 1 + len(c['repair']) + len(c['sr'])
            sr_images += len(c['sr'])
            pixel_bytes += c['decoded']['pixel_bytes'] + sum(r['pixel_bytes'] for r in c['repair'] + c['sr'])
            if 'native_control' in c:
                controls.append(c['native_control'])
            common = (record['source']['split'], record['source']['category'], c['q'], c['sampling'], c['edge'])
            for r in c['repair']:
                if 'scores' in r:
                    stage = r['label'].rsplit('/', 1)[-1]
                    delta = differences(r['scores'], c['baseline'], ('zensim_a', 'ssim2'))
                    groups[common + ('artifact', stage)].append(delta)
            for r in c['sr']:
                direct = next(x for x in c['sr'] if x['model'] == r['model'] and x['repair'] == 'direct')
                delta = differences(r['scores'], direct['scores'], ('zensim_a',))
                groups[common + (r['model'], r['repair'])].append(delta)
    rows = []
    for key, values in sorted(groups.items()):
        row = dict(zip(['split', 'category', 'q', 'sampling', 'edge', 'model', 'stage'], key))
        row['count'] = len(values)
        for metric in ('zensim_a', 'ssim2', 'psnr'):
            v = [x[metric] for x in values if metric in x and x[metric] is not None]
            if v:
                row[metric + '_finite_count'] = len(v)
                row[metric + '_mean_delta'] = statistics.fmean(v)
                row[metric + '_worst_delta'] = min(v)
                row[metric + '_win_fraction'] = sum(x > 0 for x in v) / len(v)
        row['psnr_from_exact'] = sum(x['psnr_from_exact'] for x in values)
        row['psnr_to_exact'] = sum(x['psnr_to_exact'] for x in values)
        rows.append(row)
    a.output.mkdir(exist_ok=False)
    (a.output / 'category-quality.json').write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
    summary = dict(identity_sha256=identity_sha, plan_sha256=sha(a.root / 'PLAN.json'),
                   origins=len(all_records), train_origins=sum(r['source']['split'] == 'train' for r in all_records),
                   validate_origins=sum(r['source']['split'] == 'validate' for r in all_records),
                   categories=len({r['source']['category'] for r in all_records}),
                   input_cases=sum(len(r['cells']) for r in all_records), images=images, sr_images=sr_images,
                   archives_bytes=stored_bytes, uncompressed_pixel_bytes=pixel_bytes,
                   native_controls=len(controls),
                   native_max_rgb8_delta=max((r['max_rgb8_delta'] for r in controls), default=None),
                   native_min_ssim2=min((r['ssim2'] for r in controls), default=None),
                   limitation='Offline forced repair candidates. Base CNN has historical corpus exposure; no fresh base-model or production-router generalization claim.',
                   checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (a.output / 'VERIFIED.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
