#!/usr/bin/env python3
"""How much is left on the table by quality-only routing?

`gate_recalibrate.py` asks whether the shipped rule is the best rule *of its
kind*. This asks a different question: what is the ceiling for ANY rule, and how
much of the remaining headroom needs information the current estimator does not
have (content, not just quality)?

The decomposition:

  q-only ceiling   restore iff the (subsampling, quality) cell median is
                   positive — the best a curve keyed on quality alone can do,
                   given perfect knowledge of the curve.
  per-image ceiling restore iff THIS image's actual delta is positive — perfect
                   foresight, unreachable, but it bounds every possible router.

The difference between them is the value of knowing anything about the image
beyond its quality. If that gap is small, routing is finished and effort belongs
elsewhere. If it is large, a content-aware router is worth building.

Also here: bootstrap confidence on the crossover point (how identifiable is it,
really), a threshold sweep showing the quality/work/harm tradeoff, and a
decision-level test of whether content type predicts gain.

Usage: routing_headroom.py <ladder.tsv> [more.tsv ...]
"""
import collections
import random
import statistics
import sys

MIN_GAIN = 0.25


def load(paths, arm="model_policy", base="identity_off"):
    rows = {}
    for path in paths:
        with open(path) as f:
            hdr = f.readline().rstrip("\n").split("\t")
            ix = {k: i for i, k in enumerate(hdr)}
            for line in f:
                c = line.rstrip("\n").split("\t")
                if len(c) < len(hdr):
                    continue
                key = (c[ix["sub"]], c[ix["file"]], c[ix["encoder"]],
                       c[ix["ss"]], float(c[ix["q"]]))
                rows.setdefault(key, {})[c[ix["arm"]]] = float(c[ix["ssim2"]])
    out = []
    for (sub, fn, enc, ss, q), arms in rows.items():
        if arm in arms and base in arms:
            out.append((sub, fn, enc, ss, q, arms[arm] - arms[base]))
    return out


def med(xs):
    return statistics.median(xs) if xs else 0.0


def section(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def headroom(data):
    section("1. CEILINGS — what any router could achieve")
    cellmed = collections.defaultdict(list)
    for sub, fn, enc, ss, q, d in data:
        cellmed[(ss, q)].append(d)
    cellmed = {k: med(v) for k, v in cellmed.items()}

    n = len(data)
    always = sum(d for *_, d in data) / n
    qonly = sum(d for _, _, _, ss, q, d in data if cellmed[(ss, q)] > 0) / n
    perimg = sum(d for *_, d in data if d > 0) / n
    # Restricting the per-image oracle to cells the q-only rule already takes
    # separates "restore in more places" from "restore in the right places".
    within = sum(d for _, _, _, ss, q, d in data
                 if cellmed[(ss, q)] > 0 and d > 0) / n

    print(f"{'always restore':<42} {always:+7.4f}")
    print(f"{'q-only ceiling (perfect cell curve)':<42} {qonly:+7.4f}")
    print(f"{'per-image ceiling (perfect foresight)':<42} {perimg:+7.4f}")
    print(f"{'  ...of which inside q-only-accepted cells':<42} {within:+7.4f}")
    print()
    print(f"headroom above the q-only ceiling: {perimg - qonly:+.4f} ssim2 "
          f"({(perimg / qonly - 1) * 100:.0f}% more)")
    print(f"  reachable by better per-image decisions in cells we already\n"
          f"  restore: {within - qonly:+.4f}; the rest needs restoring in cells\n"
          f"  the curve currently rejects: {perimg - within:+.4f}")


def sweep(data):
    section("2. THRESHOLD SWEEP — quality vs work vs harm")
    cellmed = collections.defaultdict(list)
    for sub, fn, enc, ss, q, d in data:
        cellmed[(ss, q)].append(d)
    cellmed = {k: med(v) for k, v in cellmed.items()}
    n = len(data)
    print("min_gain  mean_ssim2  restored  harmed  harm_when_restored  mean_harm")
    for mg in (0.0, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 1.00, 2.00):
        tot = r = h = 0
        hsum = 0.0
        for sub, fn, enc, ss, q, d in data:
            if cellmed[(ss, q)] >= mg:
                r += 1
                tot += d
                if d < 0:
                    h += 1
                    hsum += d
        print(f"{mg:8.2f}  {tot / n:+10.4f}  {r / n:8.2f}  {h / n:6.2f}"
              f"  {(h / r if r else 0):18.2f}  {(hsum / h if h else 0):+9.2f}")
    print("\nharmed = share of ALL cells that were restored and got worse.")
    print("mean_harm = average damage on those, in ssim2 points.")


def bootstrap_crossover(data, iters=2000, seed=20260803):
    section("3. CROSSOVER — bootstrap confidence over images")
    rng = random.Random(seed)
    files = sorted({fn for _, fn, *_ in data})
    by = collections.defaultdict(lambda: collections.defaultdict(dict))
    for sub, fn, enc, ss, q, d in data:
        by[(enc, ss)][q][fn] = d

    def cross(per_q, sample):
        qs = sorted(per_q)
        m = {q: med([per_q[q][f] for f in sample if f in per_q[q]]) for q in qs}
        for i, q in enumerate(qs):
            if all(m[q2] <= 0 for q2 in qs[i:]):
                return q
        return qs[-1] + 1

    print("enc/ss            point  p05   p50   p95   width  n_distinct")
    for k in sorted(by):
        per_q = by[k]
        point = cross(per_q, files)
        draws = sorted(cross(per_q, [rng.choice(files) for _ in files])
                       for _ in range(iters))
        p05, p50, p95 = (draws[int(iters * 0.05)], draws[iters // 2],
                         draws[int(iters * 0.95)])
        print(f"{k[0] + '/' + k[1]:<16} q{point:<5.0f} q{p05:<4.0f} q{p50:<4.0f}"
              f" q{p95:<4.0f} {p95 - p05:5.0f}  {len(set(draws))}")
    print("\nA wide p05..p95 means the crossover is not identifiable at this")
    print("sample size — the point estimate is one draw from that spread.")


# Subcorpora whose content is synthetic: flat regions, sharp edges, limited
# palette. JPEG damage there is ringing and mosquito noise, which is both very
# visible and very removable. The rest is photographic — stochastic detail,
# grain, stippling — where the "artifacts" are entangled with real texture.
GRAPHIC_SUBS = {"documents", "maps", "screen"}


def content(data):
    section("4. CONTENT TYPE — does it predict gain beyond quality?")
    subs = sorted({s for s, *_ in data})
    qs = sorted({q for *_, q, _ in data})
    print("Median gain by subcorpus and quality (4:2:0 and 4:4:4 pooled):\n")
    print(f"{'subcorpus':<12}" + "".join(f"{'q' + str(int(q)):>8}" for q in qs))
    per = collections.defaultdict(list)
    for sub, fn, enc, ss, q, d in data:
        per[(sub, q)].append(d)
    for s in subs:
        print(f"{s:<12}" + "".join(f"{med(per[(s, q)]):+8.2f}" for q in qs))
    print(f"{'ALL':<12}" + "".join(
        f"{med([d for sub, fn, e, ss, qq, d in data if qq == q]):+8.2f}"
        for q in qs))

    print("\nSpread across subcorpora at each quality (max - min of the above):")
    print(f"{'':<12}" + "".join(f"{'q' + str(int(q)):>8}" for q in qs))
    print(f"{'spread':<12}" + "".join(
        f"{max(med(per[(s, q)]) for s in subs) - min(med(per[(s, q)]) for s in subs):8.2f}"
        for q in qs))

    # Decision-level: does a content-aware rule beat the pooled one on held-out
    # images? Same calibrate/validate discipline as everywhere else.
    #
    # READ THE CAVEAT BELOW BEFORE QUOTING THESE NUMBERS. Content class here is
    # the ground-truth subcorpus directory, so both content-aware rows are
    # ORACLE-LABEL CEILINGS. A real classifier misclassifies and lands lower;
    # zensr's shipped chooser measures P=0.950 / R=0.594 at its 0.85 threshold,
    # so recall alone would forfeit a large share of the graphic-side gain.
    # Substitute real classifier output to get an achievable number.
    files = sorted({fn for _, fn, *_ in data})
    calib = {f for f in files if hstem(f) % 2 == 0}
    pooled = collections.defaultdict(list)
    bysub = collections.defaultdict(list)
    bybin = collections.defaultdict(list)
    for sub, fn, enc, ss, q, d in data:
        if fn in calib:
            pooled[(ss, q)].append(d)
            bysub[(sub, ss, q)].append(d)
            bybin[(sub in GRAPHIC_SUBS, ss, q)].append(d)
    pooled = {k: med(v) for k, v in pooled.items()}
    bysub = {k: med(v) for k, v in bysub.items()}
    bybin = {k: med(v) for k, v in bybin.items()}

    rules = {
        "pooled curve (quality only)":
            lambda sub, ss, q: pooled.get((ss, q), 0.0),
        "+ binary graphic/photographic":
            lambda sub, ss, q: bybin.get((sub in GRAPHIC_SUBS, ss, q),
                                         pooled.get((ss, q), 0.0)),
        f"+ subcorpus ({len(subs)} classes)":
            lambda sub, ss, q: bysub.get((sub, ss, q), pooled.get((ss, q), 0.0)),
    }
    val = [x for x in data if x[1] not in calib]
    n = len(val)
    print(f"\nHeld-out decision test at min_gain={MIN_GAIN} (n={n} cells):")
    print(f"{'rule':<38}{'mean_ssim2':>11}{'restored':>10}")
    for name, f in rules.items():
        t = r = 0
        for sub, fn, enc, ss, q, d in val:
            if f(sub, ss, q) >= MIN_GAIN:
                t += d
                r += 1
        print(f"{name:<38}{t / n:>+11.4f}{r / n:>10.2f}")
    print(f"{'per-image oracle (ceiling)':<38}"
          f"{sum(d for *_, d in val if d > 0) / n:>+11.4f}"
          f"{sum(1 for *_, d in val if d > 0) / n:>10.2f}")
    print("\nCAVEAT: the content-aware rows use ground-truth labels and are")
    print("CEILINGS, not achievable gains. See the comment above this test.")

    print("\nBinary curves fitted on the calibrate images (what a router ships):")
    for g, label in ((True, "graphic"), (False, "photographic")):
        for ss in sorted({s for _, _, _, s, _, _ in data}):
            pts = " ".join(f"q{q:g}:{bybin[(g, ss, q)]:+6.2f}"
                           for q in qs if (g, ss, q) in bybin)
            print(f"  {label:<14}{ss}  {pts}")


def hstem(s):
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h


def spread(data):
    section("5. PER-IMAGE SPREAD — why the median hides so much")
    per = collections.defaultdict(list)
    for sub, fn, enc, ss, q, d in data:
        per[q].append(d)
    print("   q      n   p10     p25   median    p75     p90   frac<0")
    for q in sorted(per):
        v = sorted(per[q])
        pc = lambda f: v[min(len(v) - 1, int(len(v) * f))]  # noqa: E731
        print(f"{q:5.0f}  {len(v):5d} {pc(.10):+7.2f} {pc(.25):+7.2f}"
              f" {med(v):+7.2f} {pc(.75):+7.2f} {pc(.90):+7.2f}"
              f"  {sum(1 for x in v if x < 0) / len(v):6.2f}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    data = load(sys.argv[1:])
    print(f"{len(data)} paired cells, "
          f"{len({fn for _, fn, *_ in data})} images, "
          f"{sorted({e for _, _, e, *_ in data})}")
    headroom(data)
    sweep(data)
    bootstrap_crossover(data)
    content(data)
    spread(data)


if __name__ == "__main__":
    main()
