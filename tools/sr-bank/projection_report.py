"""Report a complete fixed-projection replay without selecting a policy."""
import argparse
from collections import defaultdict
from decimal import Decimal
import json
import math
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from bank import sha, save

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
from routing_frontier import harm_limit


def summarize(rows):
    return dict(cases=len(rows), origins=len({r['source_id'] for r in rows}),
        changed=sum(not r['same'] for r in rows), aligned=sum(r['aligned'] for r in rows),
        mean_a_delta=math.fsum(r['a_delta'] for r in rows) / len(rows),
        mean_ssim2_delta=math.fsum(r['ssim2_delta'] for r in rows) / len(rows),
        losses_over_half=sum(r['ssim2_delta'] < -.5 for r in rows),
        old_harmed=sum(r['old_harmed'] for r in rows), new_harmed=sum(r['new_harmed'] for r in rows),
        newly_harmed=sum(r['new_harmed'] and not r['old_harmed'] for r in rows),
        resolved_harm=sum(r['old_harmed'] and not r['new_harmed'] for r in rows),
        worst_a=min(r['a_delta'] for r in rows), worst_ssim2=min(r['ssim2_delta'] for r in rows))


def main(a):
    identity = json.loads((a.run / 'IDENTITY.json').read_text())
    done = json.loads((a.run / 'PROGRESS.json').read_text())
    if done['stage'] != 'complete' or done['completed_origins'] != len(identity['sources']):
        raise ValueError('Projection replay is incomplete')
    rows = []
    for source in identity['sources']:
        path = Path(source['file'])
        if sha(path) != source['sha256']:
            raise ValueError('Original source record changed')
        bank = json.loads(path.read_text())
        directory = a.run / path.stem
        result = json.loads((directory / 'RESULT.json').read_text())
        if result['source_record_sha256'] != source['sha256']:
            raise ValueError('Replay source differs')
        cells = {r['id']: r for r in bank['cells'] if r['q'] != 0}
        old = {r['id']: r for r in result['rows'] if r['padding'] == 'replicated-padding'}
        new = {r['id']: r for r in result['rows'] if r['padding'] == 'runtime-fixed'}
        if len(result['rows']) != 2 * len(cells) or set(old) != set(cells) or set(new) != set(cells):
            raise ValueError('Required replay grid differs')
        for rid, cell in cells.items():
            before, after = old[rid], new[rid]
            artifact = after['artifact']
            if sha(directory / artifact['path']) != artifact['encoded_sha256']:
                raise ValueError('Fixed candidate bytes changed')
            baseline = cell['baseline']['ssim2']
            def harmed(score):
                return Decimal(str(baseline)) - Decimal(str(score)) > Decimal(str(harm_limit(baseline)))
            same = before['artifact']['pixels_sha256'] == artifact['pixels_sha256']
            if after['aligned'] and not same:
                raise ValueError('Aligned output changed')
            rows.append(dict(source_id=bank['source']['id'], split=bank['source']['split'],
                category=bank['source']['category'], id=rid, edge=cell['edge'], q=cell['q'], sampling=cell['sampling'],
                aligned=after['aligned'], same=same, old_a=before['zensim_a'], new_a=after['zensim_a'],
                a_delta=after['zensim_a']-before['zensim_a'], old_ssim2=before['ssim2'], new_ssim2=after['ssim2'],
                ssim2_delta=after['ssim2']-before['ssim2'], baseline_ssim2=baseline,
                old_harmed=harmed(before['ssim2']), new_harmed=harmed(after['ssim2'])))
    if 2 * len(rows) != done['rows']:
        raise ValueError('Completion row count differs')
    a.output.mkdir(exist_ok=False)
    rows.sort(key=lambda r: r['id'])
    pq.write_table(pa.Table.from_pylist(rows), a.output / 'paired-cells.parquet', compression='zstd')
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in ['split', 'category', 'q', 'sampling', 'edge'])].append(r)
    grouped = [dict(zip(['split', 'category', 'q', 'sampling', 'edge'], key), **summarize(value))
               for key, value in sorted(groups.items())]
    pq.write_table(pa.Table.from_pylist(grouped), a.output / 'category-quality.parquet', compression='zstd')
    result = dict(overall=summarize(rows),
        by_split={s:summarize([r for r in rows if r['split']==s]) for s in sorted({r['split'] for r in rows})},
        worst_ssim2=sorted(rows, key=lambda r:r['ssim2_delta'])[:10],
        worst_a=sorted(rows, key=lambda r:r['a_delta'])[:10],
        replay_identity_sha256=sha(a.run / 'IDENTITY.json'), report_script_sha256=sha(Path(__file__)),
        harm_definition_source_sha256=sha(TOOLS / 'routing_frontier.py'),
        scope='forced repair of frozen small-bank inputs; no router/default-policy selection')
    save(a.output / 'SUMMARY.json', result)
    print(json.dumps(result['overall']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    main(p.parse_args())
