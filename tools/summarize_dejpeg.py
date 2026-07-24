#!/usr/bin/env python3
"""Summarize dejpeg_eval TSV: per (encoder, ss, q) cell medians + worse-rates,
the 2x2 deblock-arm verdict, probe-family accuracy, and universality gaps.

Usage: summarize_dejpeg.py <tsv>
"""
import csv
import statistics as st
import sys
from collections import defaultdict

rows = []
with open(sys.argv[1]) as f:
    r = csv.reader(f, delimiter="\t")
    header = next(r)
    for c in r:
        if len(c) >= 11:
            rows.append(c)

# (enc, ss, q, arm) -> {(sub,file): (ssim2, butter, psnr)}
cells = defaultdict(dict)
probes = defaultdict(lambda: defaultdict(int))  # enc -> family -> count
for sub, fn, enc, ss, q, arm, psnr, ssim2, butter, pf, pq in (c[:11] for c in rows):
    cells[(enc, ss, int(q), arm)][(sub, fn)] = (float(ssim2), float(butter), float(psnr))
    if arm == "identity_off":
        probes[(enc, ss)][pf] += 1

ARMS = ["identity_off", "identity_auto", "model_off", "model_auto"]
print("encoder\tss\tq\t" + "\t".join(f"{a}_ssim2" for a in ARMS) +
      "\tmodel_off_worse%\tmodel_auto_worse%\tbest_arm(butter)")
encs = sorted({k[0] for k in cells})
qs = sorted({k[2] for k in cells})
gaps = []
for enc in encs:
    for ss in ["420", "444"]:
        for q in qs:
            base = cells.get((enc, ss, q, "identity_off"), {})
            if not base:
                continue
            meds, buts = {}, {}
            for a in ARMS:
                m = cells.get((enc, ss, q, a), {})
                common = set(m) & set(base)
                meds[a] = st.median(m[k][0] for k in common) if common else float("nan")
                buts[a] = st.median(m[k][1] for k in common) if common else float("nan")
            def worse(a):
                m = cells.get((enc, ss, q, a), {})
                common = set(m) & set(base)
                if not common:
                    return "-"
                w = sum(1 for k in common if m[k][0] < base[k][0])
                return f"{100*w/len(common):.0f}"
            best = min(ARMS, key=lambda a: buts[a])
            if meds["model_off"] < meds["identity_off"]:
                gaps.append((enc, ss, q, meds["model_off"] - meds["identity_off"]))
            print(f"{enc}\t{ss}\t{q}\t" + "\t".join(f"{meds[a]:.1f}" for a in ARMS) +
                  f"\t{worse('model_off')}\t{worse('model_auto')}\t{best}")

print("\n== 2x2 verdict (mean ssim2 median across all cells) ==")
for a in ARMS:
    vals = []
    for enc in encs:
        for ss in ["420", "444"]:
            for q in qs:
                m = cells.get((enc, ss, q, a), {})
                if m:
                    vals.append(st.median(v[0] for v in m.values()))
    print(f"{a}: {st.mean(vals):.2f}")

print("\n== universality gaps (cells where model_off median < identity_off) ==")
if not gaps:
    print("NONE — universal activation holds on this grid")
for g in gaps:
    print(f"  {g[0]} {g[1]} q{g[2]}: {g[3]:+.2f}")

print("\n== probe family by encoder x ss ==")
for k in sorted(probes):
    total = sum(probes[k].values())
    top = sorted(probes[k].items(), key=lambda x: -x[1])
    print(f"  {k[0]} {k[1]}: " + ", ".join(f"{f}={c}({100*c/total:.0f}%)" for f, c in top[:3]))
