"""Report train-side Pareto choices and their untouched validation comparisons.

No setting is selected using validation scores and no runtime default changes.
PSNR selection uses mean MSE so exact matches remain mathematically exact.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from bank import save, sha

GROUP = ['category', 'q', 'sampling', 'edge', 'model', 'scale']
KNOBS = ['repair', 'filter', 'intensity_quarter']


def pooled_psnr(mse):
    return math.inf if mse == 0 else 10 * math.log10(255**2 / mse)


def nondominated(rows):
    # Sort ties deterministically; equivalent endpoint aliases need one entry.
    ordered = sorted(rows, key=lambda r: (-r['zensim_a'], r['mse'],
                     r['intensity_quarter'] != 0, r['repair'] != 'direct',
                     r['repair'], r['filter'], r['intensity_quarter']))
    frontier = []
    best_mse = math.inf
    for row in ordered:
        if row['mse'] < best_mse:
            frontier.append(row)
            best_mse = row['mse']
    return frontier


def main(a):
    run = a.root / a.run
    done = json.loads((run / 'COMPLETE.json').read_text())
    identity_sha = sha(run / 'IDENTITY.json')
    identity = json.loads((run / 'IDENTITY.json').read_text())
    if done['stage'] != 'complete' or done['identity_sha256'] != identity_sha:
        raise ValueError('Blend sweep is incomplete')
    files = []
    for rid in identity['sources']:
        r = json.loads((run / (rid + '.json')).read_text())
        p = run / (rid + '.parquet')
        if r['identity_sha256'] != identity_sha or sha(p) != r['parquet_sha256']:
            raise ValueError('Scalar shard identity differs')
        files.append(p)
    raw = pl.scan_parquet(files)
    summary = (raw.with_columns((pl.lit(255.**2) * pl.lit(10.).pow(-pl.col('psnr') / 10)).alias('mse'))
        .group_by(['split', *GROUP, *KNOBS])
        .agg(pl.col('source_id').n_unique().alias('origins'),
             pl.col('zensim_a').alias('a_values'), pl.col('mse').alias('mse_values'),
             pl.col('zensim_a').min().alias('min_zensim_a'),
             pl.col('psnr').is_infinite().sum().alias('exact_matches'))
        .collect(engine='streaming'))
    # Correctly rounded sums make equivalent endpoints and selection ties
    # independent of hash-group/parallel reduction order.
    summary = (summary.with_columns(
        pl.col('a_values').map_elements(lambda v: math.fsum(v) / len(v), return_dtype=pl.Float64).alias('zensim_a'),
        pl.col('mse_values').map_elements(lambda v: math.fsum(v) / len(v), return_dtype=pl.Float64).alias('mse'))
        .drop(['a_values', 'mse_values']).sort(['split', *GROUP, *KNOBS]))
    a.output.mkdir(exist_ok=False)
    pq.write_table(summary.to_arrow(), a.output / 'all-settings.parquet', compression='zstd', compression_level=3)
    training = defaultdict(list)
    validation = {}
    baselines = {}
    for r in summary.to_dicts():
        group = tuple(r[k] for k in GROUP)
        if r['split'] == 'train':
            training[group].append(r)
        elif r['split'] == 'validate':
            validation[group + tuple(r[k] for k in KNOBS)] = r
            if r['repair'] == 'direct' and r['filter'] == 'triangle' and r['intensity_quarter'] == 4:
                baselines[group] = r
        else:
            raise ValueError('Unexpected split in scalar data')
    frontier = []
    anchors = []
    for group, rows in sorted(training.items()):
        baseline = baselines[group]
        selected = nondominated(rows)
        best_a = selected[0]
        best_mse = selected[-1]
        for row in selected:
            val = validation[group + tuple(row[k] for k in KNOBS)]
            frontier.append(dict(**dict(zip(GROUP, group)), **{k: row[k] for k in KNOBS},
                train_origins=row['origins'], validate_origins=val['origins'],
                train_zensim_a=row['zensim_a'], train_mse=row['mse'],
                validate_zensim_a=val['zensim_a'], validate_mse=val['mse'],
                validate_a_delta=val['zensim_a'] - baseline['zensim_a'],
                baseline_validate_mse=baseline['mse']))
        for objective, row in [('zensim_a', best_a), ('mse', best_mse)]:
            val = validation[group + tuple(row[k] for k in KNOBS)]
            anchors.append(dict(objective=objective, **dict(zip(GROUP, group)),
                **{k: row[k] for k in KNOBS}, origins=val['origins'],
                validate_a_delta=val['zensim_a'] - baseline['zensim_a'],
                validate_mse=val['mse'], baseline_mse=baseline['mse']))
    pq.write_table(pa.Table.from_pylist(frontier), a.output / 'train-frontier-on-validate.parquet', compression='zstd', compression_level=3)
    pq.write_table(pa.Table.from_pylist(anchors), a.output / 'train-selected-anchors.parquet', compression='zstd', compression_level=3)
    overall = []
    for objective in ['zensim_a', 'mse']:
        for model in ['spanf', 'dscf']:
            for scale in [2, 4]:
                rows = [r for r in anchors if (r['objective'], r['model'], r['scale']) == (objective, model, scale)]
                n = sum(r['origins'] for r in rows)
                before = sum(r['baseline_mse'] * r['origins'] for r in rows) / n
                after = sum(r['validate_mse'] * r['origins'] for r in rows) / n
                overall.append(dict(objective=objective, model=model, scale=scale,
                    validation_comparisons=n, validate_mean_a_delta=sum(r['validate_a_delta'] * r['origins'] for r in rows) / n,
                    validate_psnr_delta=pooled_psnr(after) - pooled_psnr(before)))
    selected = pl.DataFrame(anchors).select(['objective', *GROUP, *KNOBS])
    val = raw.filter(pl.col('split') == 'validate')
    baseline = (val.filter((pl.col('repair') == 'direct') & (pl.col('filter') == 'triangle') & (pl.col('intensity_quarter') == 4))
        .select(['id', 'model', 'scale', pl.col('zensim_a').alias('baseline_a'), pl.col('psnr').alias('baseline_psnr')]))
    paired = (val.join(selected.lazy(), on=[*GROUP, *KNOBS], how='inner')
        .join(baseline, on=['id', 'model', 'scale'], how='left', validate='m:1')
        .with_columns((pl.col('zensim_a') - pl.col('baseline_a')).alias('a_delta'),
            (pl.col('baseline_psnr').is_infinite() & pl.col('psnr').is_finite()).alias('psnr_from_exact'),
            (pl.col('baseline_psnr').is_finite() & pl.col('psnr').is_infinite()).alias('psnr_to_exact'),
            pl.when(pl.col('baseline_psnr').is_infinite() & pl.col('psnr').is_infinite()).then(0.)
              .otherwise(pl.col('psnr') - pl.col('baseline_psnr')).alias('psnr_delta'))
        .collect(engine='streaming').sort(['objective', 'id', 'model', 'scale']))
    if paired.height != sum(r['validation_comparisons'] for r in overall):
        raise ValueError('Paired validation comparison count differs')
    pq.write_table(paired.to_arrow(), a.output / 'anchor-validation-cells.parquet', compression='zstd', compression_level=3)
    tails = paired.group_by(['objective', 'model', 'scale']).agg(
        pl.col('source_id').n_unique().alias('validation_origins'),
        pl.col('a_delta').min().alias('worst_a_delta'),
        (pl.col('a_delta') < 0).mean().alias('lower_a_fraction'),
        pl.col('psnr_delta').filter(pl.col('psnr_delta').is_finite()).min().alias('worst_finite_psnr_delta'),
        pl.col('psnr_from_exact').sum(), pl.col('psnr_to_exact').sum())
    tail_map = {(r['objective'], r['model'], r['scale']): r for r in tails.to_dicts()}
    for row in overall:
        row.update(tail_map[(row['objective'], row['model'], row['scale'])])
    result = dict(report_script_sha256=sha(Path(__file__)), polars_version=pl.__version__, pyarrow_version=pa.__version__, numerical_threads=pl.thread_pool_size(), blend_identity_sha256=identity_sha, scores=done['completed_scores'],
                  source_origins=len(identity['sources']), groups=len(training),
                  pareto_rows=len(frontier), anchors=overall,
                  selection='frontier and two objective anchors chosen on train only, separately per category/q/sampling/edge/model/scale; all reported on validate',
                  limitation='Ground-truth category and encoder q buckets are offline strata, not a deployed classifier. Three small sizes; base CNN historical corpus exposure; no default change.')
    save(a.output / 'SUMMARY.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
