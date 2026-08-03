#!/usr/bin/env python3
"""Recalibrate the restore gate against clean-reference measurements.

Two questions, both answered with a calibrate/validate split BY IMAGE so a
threshold is never scored on the files that chose it:

  1. Is the encoder split real? The clean ladder puts mozjpeg's crossover
     several q below libjpeg-turbo's at the same subsampling. If that holds it
     belongs in the gate, which today keys only on subsampling. Differences up
     here sit near the ssim2 metric floor, so the test is a two-sided sign test
     over paired per-file deltas, not a comparison of medians.

  2. Where does the shipped estimator disagree with clean references? The
     estimator's curve was fit when 39% of eval references were themselves
     JPEGs, which flatters restoration: the model moves output toward something
     already JPEG-like. Any optimism should concentrate at high q, where the
     true gain is small enough for that bias to dominate.

Usage: gate_recalibrate.py <ladder.tsv> [--arm model_policy] [--base identity_off]
"""
import argparse
import collections
import math
import statistics
import sys

# Shipped estimator (crates/zensr-zenjpeg/src/api.rs). Duplicated here rather
# than parsed so this script reports what the library actually does today.
G420 = [(15.0, 5.70), (35.0, 3.07), (55.0, 1.80), (75.0, 0.65), (85.0, 0.46),
        (90.0, 0.26), (94.0, 0.15), (96.0, 0.05), (100.0, -0.17)]
G444 = [(90.0, -0.04), (92.0, -0.13), (94.0, -0.38), (96.0, -0.78),
        (98.0, -1.38), (100.0, -2.04)]
# Below q90 the 4:4:4 curve reads the 4:2:0 curve shifted by this many points:
# 4:4:4 at q90 decodes about as cleanly as 4:2:0 at q94.
S444_SHIFT = 4.0


def interp(curve, q):
    if q <= curve[0][0]:
        return curve[0][1]
    if q >= curve[-1][0]:
        return curve[-1][1]
    for (q0, g0), (q1, g1) in zip(curve, curve[1:]):
        if q0 <= q <= q1:
            t = (q - q0) / (q1 - q0)
            return g0 + t * (g1 - g0)
    return curve[-1][1]


def estimate_gain(ss, q):
    if ss == "444":
        return interp(G444, q) if q >= 90.0 else interp(G420, q - S444_SHIFT)
    return interp(G420, q)


def sign_test(deltas, eps=0.0):
    """Two-sided exact binomial on signs. Zeros (|d|<=eps) are dropped, which
    is the conservative choice: they can only weaken a claimed difference."""
    pos = sum(1 for d in deltas if d > eps)
    neg = sum(1 for d in deltas if d < -eps)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return pos, neg, min(1.0, 2.0 * tail / (2.0 ** n))


def load(path, arm, base):
    rows = {}
    with open(path) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        ix = {k: i for i, k in enumerate(hdr)}
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < len(hdr):
                continue
            key = (c[ix["file"]], c[ix["encoder"]], c[ix["ss"]], float(c[ix["q"]]))
            rows.setdefault(key, {})[c[ix["arm"]]] = float(c[ix["ssim2"]])
    out = []
    for (fn, enc, ss, q), arms in rows.items():
        if arm in arms and base in arms:
            out.append((fn, enc, ss, q, arms[arm] - arms[base]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", nargs="+")
    ap.add_argument("--arm", default="model_policy")
    ap.add_argument("--base", default="identity_off")
    a = ap.parse_args()

    data = [row for p in a.tsv for row in load(p, a.arm, a.base)]
    if not data:
        sys.exit(f"no paired rows for {a.arm} vs {a.base}")
    files = sorted({d[0] for d in data})
    # Split by image so a threshold is never validated on the files that set it.
    # Deterministic and independent of listing order.
    calib = {f for f in files if hash_stem(f) % 2 == 0}
    print(f"# {len(files)} images: {len(calib)} calibrate / "
          f"{len(files) - len(calib)} validate (split by image)")

    print(f"\n## 1. is the encoder split real?  ({a.arm} - {a.base}, per file)")
    print("ss\tq\tn\tmoz_med\tturbo_med\tdiff\tsign(+/-)\tp")
    by = collections.defaultdict(dict)
    for fn, enc, ss, q, d in data:
        by[(ss, q)].setdefault(enc, {})[fn] = d
    for ss, q in sorted(by, key=lambda k: (k[0], k[1])):
        m, t = by[(ss, q)].get("mozjpeg", {}), by[(ss, q)].get("turbo", {})
        common = sorted(set(m) & set(t))
        if not common:
            continue
        paired = [t[f] - m[f] for f in common]  # >0 => turbo gains more
        pos, neg, p = sign_test(paired)
        print(f"{ss}\t{q:g}\t{len(common)}\t{statistics.median(m[f] for f in common):+.3f}"
              f"\t{statistics.median(t[f] for f in common):+.3f}"
              f"\t{statistics.median(paired):+.3f}\t{pos}/{neg}\t{p:.2g}")

    print(f"\n## 2. shipped estimator vs clean references (validate images only)")
    print("enc\tss\tq\tn\tmeasured\testimated\terror")
    val = [d for d in data if d[0] not in calib]
    cells = collections.defaultdict(list)
    for fn, enc, ss, q, d in val:
        cells[(enc, ss, q)].append(d)
    for (enc, ss, q) in sorted(cells):
        med = statistics.median(cells[(enc, ss, q)])
        est = estimate_gain(ss, q)
        print(f"{enc}\t{ss}\t{q:g}\t{len(cells[(enc,ss,q)])}\t{med:+.3f}\t{est:+.3f}\t{est-med:+.3f}")

    print(f"\n## 3. crossover per (encoder, ss): calibrate vs validate")
    print("enc\tss\tcalib_cross\tvalid_cross")
    for enc in sorted({d[1] for d in data}):
        for ss in sorted({d[2] for d in data}):
            print(f"{enc}\t{ss}\t{cross(data, enc, ss, calib, True)}"
                  f"\t{cross(data, enc, ss, calib, False)}")

    main_part4(data, calib, files)


def fit_curve(data, calib, enc, ss, qs):
    """Per-cell median gain on the calibrate images — the curve the library
    would interpolate. Median, not mean: a few images with huge gain at low q
    would otherwise drag the whole curve up."""
    cells = collections.defaultdict(list)
    for fn, e, s, q, d in data:
        if e == enc and s == ss and fn in calib:
            cells[q].append(d)
    return [(q, statistics.median(cells[q])) for q in qs if cells[q]]


def simulate(data, val_files, decide):
    """Realized quality and work under a routing rule.

    Sums the ACTUAL measured delta on every image the rule chose to restore.
    A rule that restores nothing scores 0.0 and does no work; a rule only earns
    its cycles by clearing that bar."""
    tot, n_restored, n = 0.0, 0, 0
    for fn, enc, ss, q, d in data:
        if fn not in val_files:
            continue
        n += 1
        if decide(enc, ss, q):
            tot += d
            n_restored += 1
    return tot / n, n_restored / n


def main_part4(data, calib, files):
    val_files = {f for f in files if f not in calib}
    qs = sorted({d[3] for d in data})
    fitted = {(e, s): fit_curve(data, calib, e, s, qs)
              for e in sorted({d[1] for d in data})
              for s in sorted({d[2] for d in data})}

    print("\n## 4. routing rules scored on the validate images only")
    print("# mean_ssim2 = realized ssim2 per image-cell (higher better)")
    print("# restored   = fraction of cells that paid for a restore pass")
    print("rule\tmean_ssim2\trestored")

    MIN_GAIN = 0.25

    # Paired offset (turbo - mozjpeg on the SAME image), fit on calibrate only.
    paired = collections.defaultdict(dict)
    for fn, e, s, q, d in data:
        if fn in calib:
            paired[(s, q)].setdefault(fn, {})[e] = d
    off_tbl = {}
    for (s, q), per_file in paired.items():
        both = [v["turbo"] - v["mozjpeg"] for v in per_file.values()
                if "turbo" in v and "mozjpeg" in v]
        if both:
            off_tbl[(s, q)] = statistics.median(both)

    def off(e, s, q):
        qs_ = sorted({k[1] for k in off_tbl if k[0] == s})
        if not qs_:
            return 0.0
        cur = [(qq, off_tbl[(s, qq)]) for qq in qs_]
        half = interp(cur, q) / 2.0
        return half if e == "turbo" else -half

    # Pooled-over-encoder curve per subsampling, calibrate images only.
    pooled_cells = collections.defaultdict(list)
    for fn, e, s, q, d in data:
        if fn in calib:
            pooled_cells[(s, q)].append(d)
    pooled = {}
    for s in sorted({k[0] for k in pooled_cells}):
        pooled[s] = [(q, statistics.median(pooled_cells[(s, q)]))
                     for q in sorted(qq for ss_, qq in pooled_cells if ss_ == s)]

    rules = [
        ("always restore", lambda e, s, q: True),
        ("never restore", lambda e, s, q: False),
        ("shipped gate (ss thresholds)",
         lambda e, s, q: not (q >= 88.0 if s == "444" else q >= 94.5)),
        (f"shipped estimator, min_gain={MIN_GAIN}",
         lambda e, s, q: estimate_gain(s, q) >= MIN_GAIN),
        (f"encoder-aware estimator, min_gain={MIN_GAIN}",
         lambda e, s, q: interp(fitted[(e, s)], q) >= MIN_GAIN),
        ("oracle (restore iff cell median > 0)",
         lambda e, s, q: interp(fitted[(e, s)], q) > 0.0),
        # The statistically sound way to use the encoder effect. §1 measures it
        # PAIRED (same image, both encoders), which cancels the image-to-image
        # variance that swamps an independent per-encoder curve fit. So keep the
        # shipped curve — it is the two-encoder average — and split the measured
        # paired offset around it rather than refitting each encoder alone.
        (f"shipped estimator + paired encoder offset, min_gain={MIN_GAIN}",
         lambda e, s, q: estimate_gain(s, q) + off(e, s, q) >= MIN_GAIN),
        # Candidate replacement: one curve per subsampling, both measured
        # directly, pooled over encoders (per §4, per-encoder curves hurt).
        # This drops the "4:4:4 ~= 4:2:0 shifted 4 points" approximation, which
        # over-predicts 4:4:4 gain by 0.3-0.6 across the whole range now that
        # 4:4:4 is measured rather than inferred.
        (f"measured curve (calibrate-fit, both ss direct), min_gain={MIN_GAIN}",
         lambda e, s, q: interp(pooled[s], q) >= MIN_GAIN),
    ]
    for name, dec in rules:
        mq, fr = simulate(data, val_files, dec)
        print(f"{name}\t{mq:+.4f}\t{fr:.2f}")

    print("\n# candidate shipping curve (calibrate images, pooled over encoders)")
    for s, cur in sorted(pooled.items()):
        print(f"G{s} = [" + ", ".join(f"({q:g}, {g:+.2f})" for q, g in cur) + "]")

    print("\n# encoder-aware curves fitted on the calibrate images")
    print("enc\tss\t" + "\t".join(f"q{q:g}" for q in qs))
    for (e, s), cur in sorted(fitted.items()):
        print(f"{e}\t{s}\t" + "\t".join(f"{g:+.2f}" for _, g in cur))


def hash_stem(s):
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h


def cross(data, enc, ss, calib, in_calib):
    """Lowest q whose median delta is <=0 and stays <=0 for every higher q."""
    cells = collections.defaultdict(list)
    for fn, e, s, q, d in data:
        if e == enc and s == ss and ((fn in calib) == in_calib):
            cells[q].append(d)
    qs = sorted(cells)
    meds = {q: statistics.median(cells[q]) for q in qs}
    for i, q in enumerate(qs):
        if all(meds[q2] <= 0 for q2 in qs[i:]):
            return f"q{q:g}"
    return ">q%g" % qs[-1]


if __name__ == "__main__":
    main()
