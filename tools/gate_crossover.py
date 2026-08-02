#!/usr/bin/env python3
"""Where does restoration stop helping? Re-derives the high-q identity gate.

Reads a dejpeg_eval TSV and reports, per (encoder, subsampling, q), the
PER-FILE delta of an arm against the plain decode. Per-file, not cell-mean:
a cell mean once hid that a quarter of individual files regressed while the
average looked positive (2026-07-30), and the gate is a per-file decision.

Reported per cell:
  n            files compared
  median       median delta (the gate should follow this, not the mean)
  mean         cell mean, shown only so the two can be compared
  p10 / p90    spread
  win_frac     fraction of files strictly improved
  harm_frac    fraction of files made worse by more than --harm-eps

The crossover is the lowest q at or above which the median delta is <= 0 for
every higher q in the grid — "stops helping and stays that way", rather than
the first q that happens to dip.

Usage: gate_crossover.py <tsv> [--arm model_proj] [--base identity_off]
                         [--metric ssim2] [--harm-eps 0.1]
"""
import argparse
import csv
import statistics
import sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("--arm", default="model_proj")
    ap.add_argument("--base", default="identity_off")
    ap.add_argument("--metric", default="ssim2")
    ap.add_argument("--harm-eps", type=float, default=0.1)
    ap.add_argument("--absolute", metavar="ARM",
                    help="instead of a delta, report the median ABSOLUTE metric for "
                         "this arm per cell — how damaged the input actually is, which "
                         "is what a quality-keyed gate is really trying to proxy")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.tsv), delimiter="\t"))
    if not rows:
        sys.exit(f"{a.tsv}: no rows — a probe that produces nothing must fail loudly")
    for col in ("arm", "q", "encoder", "file", a.metric):
        if col not in rows[0]:
            sys.exit(f"{a.tsv}: missing column {col!r} (have: {', '.join(rows[0])})")

    if a.absolute:
        cells = defaultdict(list)
        for r in rows:
            if r["arm"] == a.absolute:
                cells[(r["encoder"], r.get("ss", ""), int(r["q"]))].append(float(r[a.metric]))
        if not cells:
            sys.exit(f"no rows for arm {a.absolute!r}")
        print(f"# median absolute {a.metric} for arm {a.absolute}")
        print("encoder\tss\tq\tn\tmedian")
        for k in sorted(cells):
            v = cells[k]
            print(f"{k[0]}\t{k[1]}\t{k[2]}\t{len(v)}\t{statistics.median(v):.2f}")
        return

    # (encoder, ss, q, file) -> {arm: value}
    by = defaultdict(dict)
    for r in rows:
        key = (r["encoder"], r.get("ss", ""), int(r["q"]), r["file"])
        by[key][r["arm"]] = float(r[a.metric])

    cells = defaultdict(list)
    missing = 0
    for (enc, ss, q, _f), arms in by.items():
        if a.arm in arms and a.base in arms:
            cells[(enc, ss, q)].append(arms[a.arm] - arms[a.base])
        else:
            missing += 1
    if not cells:
        sys.exit(f"no file had both {a.base!r} and {a.arm!r} — nothing to compare")
    if missing:
        print(f"# {missing} file-cells lacked one of the two arms and were skipped",
              file=sys.stderr)

    print(f"# {a.arm} minus {a.base}, {a.metric}, per file")
    print("encoder\tss\tq\tn\tmedian\tmean\tp10\tp90\twin_frac\tharm_frac")
    for (enc, ss, q) in sorted(cells):
        d = sorted(cells[(enc, ss, q)])
        n = len(d)
        p = lambda f: d[min(n - 1, int(f * n))]  # noqa: E731
        print(f"{enc}\t{ss}\t{q}\t{n}\t{statistics.median(d):+.3f}\t"
              f"{statistics.fmean(d):+.3f}\t{p(0.10):+.3f}\t{p(0.90):+.3f}\t"
              f"{sum(1 for x in d if x > 0) / n:.2f}\t"
              f"{sum(1 for x in d if x < -a.harm_eps) / n:.2f}")

    print("\n# crossover: lowest q whose median delta is <= 0 and stays <= 0 above it")
    for enc, ss in sorted({(e, s) for (e, s, _q) in cells}):
        qs = sorted(q for (e, s, q) in cells if (e, s) == (enc, ss))
        med = {q: statistics.median(cells[(enc, ss, q)]) for q in qs}
        cross = next((q for i, q in enumerate(qs)
                      if all(med[q2] <= 0 for q2 in qs[i:])), None)
        detail = " ".join(f"{q}:{med[q]:+.2f}" for q in qs)
        print(f"{enc}\t{ss}\t"
              + (f"crossover q{cross}" if cross is not None
                 else f"no crossover in grid (still helping at q{qs[-1]})")
              + f"\t[{detail}]")


if __name__ == "__main__":
    main()
