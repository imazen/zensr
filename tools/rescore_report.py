#!/usr/bin/env python3
"""Per-file gain over the plain decode, from dejpeg_eval TSVs.

PAIRED statistics only: the median of per-file differences and the win fraction,
never the difference of two per-method medians. A model that wins on the cell
mean can still lose on most individual files — that has happened in this repo.

Also splits by `gt_src`, the reference provenance. A JPEG-sourced reference is
itself compressed, so a "gain" measured against one partly rewards REPRODUCING
the reference's own artifacts. The 2026-07 defect happened for want of this
column; reporting an aggregate over both kinds hides it again.

**Do not read the png/jpg gap as a pure reference-quality effect.** In the
canonical corpus the two are nearly collinear with CONTENT: only 6 of 207
photographic files are natively PNG, so "png reference" is mostly screenshots,
plots and AI renders, and "jpg reference" is mostly photographs. Graphic content
is also where dejpeg legitimately gains most — flat regions and hard edges make
ringing both visible and removable. So a gap between the two columns mixes
reference cleanliness with content type and cannot be attributed to either alone.
Separating them needs clean PHOTOGRAPHIC references, i.e. the
downscale-to-pristine treatment (handoff §5 step 2). Until that exists, the two
columns are two different populations, not a controlled comparison.

Usage: rescore_report.py <model.tsv> [more.tsv ...] [--arm model_proj]
"""
import argparse
import collections
import csv
import os
import statistics


def load(path, arm):
    """{(enc, ss, q, file): (gain, gt_src)} for `arm` minus identity_off."""
    base, cur, src = {}, {}, {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            k = (r["encoder"], r["ss"], int(r["q"]), r["file"])
            if r["arm"] == "identity_off":
                base[k] = float(r["ssim2"])
                src[k] = r.get("gt_src", "?")
            elif r["arm"] == arm:
                cur[k] = float(r["ssim2"])
    return {k: (cur[k] - base[k], src.get(k, "?")) for k in cur if k in base}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", nargs="+")
    ap.add_argument("--arm", default="model_proj")
    a = ap.parse_args()
    print(f"arm: {a.arm}  (gain in ssim2 over identity_off, per file)\n")
    for path in a.tsv:
        d = load(path, a.arm)
        if not d:
            print(f"{os.path.basename(path)}: no rows for arm {a.arm!r}\n")
            continue
        byq = collections.defaultdict(list)
        byqs = collections.defaultdict(list)
        for (enc, ss, q, fn), (g, s) in d.items():
            byq[q].append(g)
            byqs[(q, s)].append(g)
        qs = sorted(byq)
        name = os.path.basename(path).replace(".tsv", "")
        print(f"=== {name} ===")
        print(f"{'q':>4} {'n':>4} {'median gain':>12} {'win%':>6}   "
              f"{'png n/med':>16} {'jpg n/med':>16}")
        for q in qs:
            v = byq[q]
            win = 100 * sum(1 for x in v if x > 0) / len(v)
            cells = []
            for s in ("png", "jpg"):
                w = byqs.get((q, s), [])
                cells.append(f"{len(w):>4}/{statistics.median(w):+7.3f}" if w else f"{0:>4}/{'--':>7}")
            print(f"{q:>4} {len(v):>4} {statistics.median(v):>+12.3f} {win:>5.0f}%   "
                  f"{cells[0]:>16} {cells[1]:>16}")
        print()


if __name__ == "__main__":
    main()
