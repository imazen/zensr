#!/usr/bin/env python3
"""zenjpeg knob-sensitivity report (zenjpeg#189): does an encode-side knob
change what the restoration model gets, at equal quality dial?

Reads a dejpeg_eval TSV whose encoders include `zenjpeg` (default) and
`zenjpeg:<variant>` rows. Per variant x ss x q, paired per file against the
`zenjpeg` default: identity ssim2 (how damaged the input is), model gain
(model_off - identity_off), delta of that gain vs default, win fraction, and
the probe's family/quality so we can see whether the fingerprinter can tell.
Also prints the model A/B when a second TSV is given (same grid).

Usage: knob_report.py <A.tsv> [B.tsv]
"""
import collections
import csv
import statistics as st
import sys


def load(path):
    arms = collections.defaultdict(dict)
    for r in csv.DictReader(open(path), delimiter="\t"):
        arms[(r["encoder"], r["ss"], int(r["q"]), r["file"])][r["arm"]] = r
    return arms


def gain(a, arm="model_off", base="identity_off", m="ssim2"):
    return float(a[arm][m]) - float(a[base][m])


def med(xs):
    return st.median(xs) if xs else float("nan")


def main():
    A = load(sys.argv[1])
    encs = sorted({k[0] for k in A})
    qs = sorted({k[2] for k in A})
    print("== probe view per encoder (family, median probe_q by dial q) ==")
    for e in encs:
        fams = collections.Counter(a["identity_off"]["probe_family"] for k, a in A.items() if k[0] == e)
        pq = {q: med([float(a["identity_off"]["probe_q"]) for k, a in A.items() if k[0] == e and k[2] == q]) for q in qs}
        print(f"{e:24s} {dict(fams)}  probe_q@{qs}: {[round(pq[q], 1) for q in qs]}")
    print("\n== per variant, paired vs zenjpeg default (ss=420+444 pooled): "
          "median identity ssim2 | median model gain | delta gain vs default (median, win%) | n ==")
    for e in encs:
        if e == "zenjpeg":
            continue
        line = [f"{e:24s}"]
        for q in qs:
            idn, g, dg = [], [], []
            for (enc, ss, qq, f), a in A.items():
                if enc != e or qq != q:
                    continue
                b = A.get(("zenjpeg", ss, qq, f))
                if not b:
                    continue
                idn.append(float(a["identity_off"]["ssim2"]))
                g.append(gain(a))
                dg.append(gain(a) - gain(b))
            win = sum(1 for x in dg if x > 0) / len(dg) if dg else 0
            line.append(f"q{q}: {med(idn):5.1f} | {med(g):+5.2f} | {med(dg):+5.2f} ({win:4.0%})")
        print("  ".join(line))
    print("\n== zenjpeg default and turbo, for scale (median identity ssim2 | model gain) ==")
    for e in ("zenjpeg", "turbo"):
        if e not in encs:
            continue
        line = [f"{e:24s}"]
        for q in qs:
            sel = [a for k, a in A.items() if k[0] == e and k[2] == q]
            line.append(f"q{q}: {med([float(a['identity_off']['ssim2']) for a in sel]):5.1f} | {med([gain(a) for a in sel]):+5.2f}")
        print("  ".join(line))
    if len(sys.argv) > 2:
        B = load(sys.argv[2])
        print(f"\n== model A/B: {sys.argv[2]} minus {sys.argv[1]}, model_off arm, paired per file ==")
        for e in encs:
            line = [f"{e:24s}"]
            for q in qs:
                d = [gain(B[k]) - gain(A[k]) for k in A if k[0] == e and k[2] == q and k in B]
                win = sum(1 for x in d if x > 0) / len(d) if d else 0
                neg = sum(1 for k in A if k[0] == e and k[2] == q and k in B and gain(B[k]) < 0)
                line.append(f"q{q}: {med(d):+5.2f} ({win:4.0%}) negB={neg}")
            print("  ".join(line))
        for src in ("native-png", "heic", "jpeg"):
            d = [gain(B[k]) - gain(A[k]) for k in A if k in B and A[k]["identity_off"]["gt_src"] == src]
            if d:
                print(f"  by ref {src:11s} n={len(d):5d} median delta {med(d):+5.2f} win {sum(x > 0 for x in d)/len(d):4.0%}")


if __name__ == "__main__":
    main()
