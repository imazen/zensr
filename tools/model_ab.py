#!/usr/bin/env python3
"""Compare two models measured by separate dejpeg_eval runs, per file.

Both runs must share corpus, selection, encoders and q grid; only the model
differs. Reports the per-file delta of B minus A on the same arm, because a
model that wins on the cell mean can still lose on most individual files —
that has happened here before.

Usage: model_ab.py <A.tsv> <B.tsv> [--arm model_proj] [--metric ssim2]
"""
import argparse
import csv
import statistics
import sys
from collections import defaultdict


def load(path, arm, metric):
    out = {}
    for r in csv.DictReader(open(path), delimiter="\t"):
        if r["arm"] == arm:
            out[(r["encoder"], r.get("ss", ""), int(r["q"]), r["file"])] = float(r[metric])
    if not out:
        sys.exit(f"{path}: no rows for arm {arm!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--arm", default="model_proj")
    ap.add_argument("--metric", default="ssim2")
    args = ap.parse_args()
    A, B = load(args.a, args.arm, args.metric), load(args.b, args.arm, args.metric)
    shared = set(A) & set(B)
    if not shared:
        sys.exit("the two runs share no (encoder, ss, q, file) cell — different grids?")
    if len(shared) != len(A) or len(shared) != len(B):
        print(f"# note: {len(A)} vs {len(B)} rows, {len(shared)} shared", file=sys.stderr)

    cells = defaultdict(list)
    for k in shared:
        cells[(k[0], k[1], k[2])].append(B[k] - A[k])
    print(f"# B({args.b}) minus A({args.a}), arm {args.arm}, {args.metric}, per file")
    print("encoder\tss\tq\tn\tmedian\tmean\twin_frac")
    alld = []
    for k in sorted(cells):
        d = cells[k]
        alld += d
        print(f"{k[0]}\t{k[1]}\t{k[2]}\t{len(d)}\t{statistics.median(d):+.3f}\t"
              f"{statistics.fmean(d):+.3f}\t{sum(1 for x in d if x > 0) / len(d):.2f}")
    print(f"\n# overall n={len(alld)} median {statistics.median(alld):+.3f} "
          f"mean {statistics.fmean(alld):+.3f} "
          f"win_frac {sum(1 for x in alld if x > 0) / len(alld):.2f}")


if __name__ == "__main__":
    main()
