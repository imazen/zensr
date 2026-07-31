#!/usr/bin/env python3
"""Train + evaluate the JPEG multi-generation detector.

Input: features TSV(s) emitted by crates/zensr-bench/src/bin/gen_detect.rs
(one row per generated JPEG, split column = train|eval; eval rows come from
the pinned eval_split/imazen26_eval_files.tsv sources and are never used for
fitting or threshold selection). Multiple TSVs are concatenated (e.g. a big
train-only run + an eval run).

Outputs (stdout + summary TSV):
  - 3-way confusion matrix (gen1 / gen2a=aligned / gen2r=resized) on eval
  - collapsed gen1-vs-gen2+ binary matrix (what gates the S10 projection)
  - conservative operating point: threshold tau chosen from SOURCE-GROUPED
    out-of-fold train probabilities (never from eval); eval reported at tau
  - ablations (coeff-only / dq-only / pixel-only / gate-lean) + per-feature
    AUCs + permutation importances

Usage: python3 tools/gen_detect_train.py ~/tmp/gendet/train40.tsv \
           ~/tmp/gendet/full2.tsv [--out benchmarks/gen_detect_YYYY-MM-DD.tsv]
"""

import argparse
import csv
import datetime
import subprocess

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold

META = ["split", "sub", "src", "srcfmt", "cls", "sp", "enc1", "q1", "enc2", "q2", "scale", "filt"]
# w/h/bytes are size-correlated (gen2r rows are smaller by construction) and
# t_* are timings — none are legitimate detector features.
EXCLUDE = {"w", "h", "bytes", "t_coeff_ms", "t_pix_ms", "t_feat_ms"}
CLASSES = ["gen1", "gen2a", "gen2r"]

PIXEL_FEATS = {
    "blk_v", "blk_h", "ghost_v_snr", "ghost_v_per", "ghost_h_snr", "ghost_h_per",
    "ghost_match", "ghost_match_s", "ghost_min_snr", "ghost_per_agree",
}
DQ_FEATS = {
    "dq_chi_max", "dq_chi_top3", "dq_chi_wmean", "dq_gapc", "dq_nbands",
    "dc_fill", "dc_expfill", "dc_chi", "dc_width", "dc_deficit",
    "c_chi_max", "c_gapc", "c_dc_fill",
}
# physics-first subset for the production gate (drops raw content-energy
# features that invite source memorization)
GATE_FEATS = {
    "dq_chi_max", "dq_chi_wmean", "dq_gapc", "dc_fill", "dc_chi", "dc_deficit",
    "c_chi_max", "c_gapc", "c_dc_fill", "ghost_match", "ghost_min_snr", "blk_v",
    "blk_h", "q_claim", "bpp", "nz_mid", "nz_hi", "r_hm", "r_ml",
}


def load(paths):
    rows = []
    for path in paths:
        with open(path) as fh:
            lines = [l for l in fh if not l.startswith("#")]
        rows.extend(csv.DictReader(lines, delimiter="\t"))
    return rows


def featurize(rows):
    names = [k for k in rows[0] if k not in META and k not in EXCLUDE]
    x = np.array([[float(r[k]) for k in names] for r in rows])
    cols = {k: x[:, i] for i, k in enumerate(names)}
    eps = 1e-9
    derived = {
        "ghost_min_snr": np.minimum(cols["ghost_v_snr"], cols["ghost_h_snr"]),
        "ghost_per_agree": np.abs(cols["ghost_v_per"] - cols["ghost_h_per"])
        / np.maximum(cols["ghost_v_per"], cols["ghost_h_per"]).clip(min=eps),
        "dc_deficit": cols["dc_expfill"] - cols["dc_fill"],
        "r_hm": cols["e_hi"] / (cols["e_mid"] + eps),
        "r_ml": cols["e_mid"] / (cols["e_lo"] + eps),
    }
    for k, v in derived.items():
        names.append(k)
        x = np.column_stack([x, v])
    y = np.array([CLASSES.index(r["cls"]) for r in rows])
    return x, y, names


def make_clf(seed=0):
    return HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=3,
        learning_rate=0.08,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )


def balanced_w(y):
    w = np.ones(len(y))
    for c in np.unique(y):
        m = y == c
        w[m] = len(y) / (len(np.unique(y)) * m.sum())
    return w


def oof_gen1_probs(x, y, groups):
    """Source-grouped out-of-fold P(gen1) on the train split."""
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5)
    for trn, val in gkf.split(x, y, groups):
        c = make_clf()
        c.fit(x[trn], y[trn], sample_weight=balanced_w(y[trn]))
        oof[val] = c.predict_proba(x[val])[:, 0]
    return oof


def report_confusion(tag, ytrue, ypred, out_rows):
    cm = confusion_matrix(ytrue, ypred, labels=[0, 1, 2])
    acc = (ytrue == ypred).mean()
    print(f"\n== {tag}: 3-way confusion (rows=true, cols=pred {CLASSES}) acc={acc:.4f}")
    for i, c in enumerate(CLASSES):
        rec = cm[i, i] / max(1, cm[i].sum())
        print(f"  {c:6s} {cm[i].tolist()}  recall={rec:.4f}")
        out_rows.append([tag, "confusion3", c] + [str(v) for v in cm[i]] + [f"{rec:.4f}"])
    bt = (ytrue != 0).astype(int)
    bp = (ypred != 0).astype(int)
    bcm = confusion_matrix(bt, bp, labels=[0, 1])
    fg1 = bcm[1, 0] / max(1, bcm[1].sum())
    g1rec = bcm[0, 0] / max(1, bcm[0].sum())
    print(f"  binary gen1-vs-gen2+: [[{bcm[0,0]},{bcm[0,1]}],[{bcm[1,0]},{bcm[1,1]}]]  "
          f"false-gen1={fg1:.4f} gen1-recall={g1rec:.4f}")
    out_rows.append([tag, "binary", "false_gen1", f"{fg1:.4f}", "gen1_recall", f"{g1rec:.4f}",
                     str(bcm.tolist()), ""])
    return acc


def sweep(tag, p1, y, tr_mask, ev_mask, oof, out_rows):
    """Threshold sweep + conservative point chosen on OOF train probs."""
    g2_tr = tr_mask & (y != 0)
    g1_tr = tr_mask & (y == 0)
    print(f"\n== {tag}: P(gen1)>=tau sweep (oof = source-grouped train CV)")
    tau_star = None
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]:
        fg1_oof = (oof[g2_tr] >= t).mean()
        rec_oof = (oof[g1_tr] >= t).mean()
        fg1_ev = (p1[ev_mask & (y != 0)] >= t).mean()
        rec_ev = (p1[ev_mask & (y == 0)] >= t).mean()
        print(f"  tau={t:0.3f} oof:fg1={fg1_oof:.4f},g1rec={rec_oof:.3f}  "
              f"eval:fg1={fg1_ev:.4f},g1rec={rec_ev:.3f}")
        out_rows.append([tag, f"tau={t}", f"oof_fg1={fg1_oof:.4f}", f"oof_g1rec={rec_oof:.4f}",
                         f"eval_fg1={fg1_ev:.4f}", f"eval_g1rec={rec_ev:.4f}", "", ""])
        if tau_star is None and fg1_oof == 0.0:
            tau_star = t
    if tau_star is None:
        tau_star = 0.99
        print("  (no tau reached oof fg1=0; using 0.99)")
    fg1_ev = (p1[ev_mask & (y != 0)] >= tau_star).mean()
    rec_ev = (p1[ev_mask & (y == 0)] >= tau_star).mean()
    n_fg1 = int((p1[ev_mask & (y != 0)] >= tau_star).sum())
    print(f"  CONSERVATIVE tau*={tau_star} (oof fg1=0): eval false-gen1={fg1_ev:.4f} "
          f"({n_fg1} rows), eval gen1-recall={rec_ev:.4f}")
    out_rows.append([tag + "_conservative", f"tau={tau_star}", f"eval_fg1={fg1_ev:.4f}",
                     f"eval_fg1_n={n_fg1}", f"eval_g1rec={rec_ev:.4f}", "", "", ""])
    return tau_star


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = load(args.tsv)
    x, y, names = featurize(rows)
    split = np.array([r["split"] for r in rows])
    tr, ev = split == "train", split == "eval"
    src = np.array([r["src"] for r in rows])
    q1 = np.array([int(r["q1"]) for r in rows])
    q2 = np.array([int(r["q2"]) for r in rows])
    print(f"rows: train={tr.sum()} eval={ev.sum()}  features={len(names)}  "
          f"train_sources={len(set(src[tr]))} eval_sources={len(set(src[ev]))}")
    print(f"train class counts: { {c: int((y[tr]==i).sum()) for i,c in enumerate(CLASSES)} }")
    print(f"eval  class counts: { {c: int((y[ev]==i).sum()) for i,c in enumerate(CLASSES)} }")

    out_rows = []
    # ---- main 3-way model ----
    clf = make_clf()
    clf.fit(x[tr], y[tr], sample_weight=balanced_w(y[tr]))
    ypred = clf.predict(x[ev])
    report_confusion("full", y[ev], ypred, out_rows)
    p1 = clf.predict_proba(x)[:, 0]
    oof3 = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5)
    xi = np.where(tr)[0]
    for trn, val in gkf.split(x[tr], y[tr], src[tr]):
        c = make_clf()
        c.fit(x[xi[trn]], y[xi[trn]], sample_weight=balanced_w(y[xi[trn]]))
        oof3[xi[val]] = c.predict_proba(x[xi[val]])[:, 0]
    tau3 = sweep("full3way", p1, y, tr, ev, oof3, out_rows)

    # leaks at the operating point
    leak = np.where((p1 >= tau3) & ev & (y != 0))[0]
    print(f"  {len(leak)} eval gen2 rows over tau* ({tau3}):")
    for i in leak[:25]:
        r = rows[i]
        print(f"    LEAK {r['cls']} {r['sub']}/{r['src'].split('/')[-1]} "
              f"{r['enc1']}q{r['q1']}->{r['enc2']}q{r['q2']} scale={r['scale']} p1={p1[i]:.3f}")
        out_rows.append(["leak", r["cls"], r["src"], r["enc1"], r["q1"], r["enc2"], r["q2"],
                         f"{p1[i]:.3f}"])

    # subgroup: q-pair buckets x encoder pairing on eval
    same_enc = np.array([r["enc1"] == r["enc2"] for r in rows])
    for cls_i, cls in [(1, "gen2a"), (2, "gen2r")]:
        for cond, cname in [(q1 < q2, "q1<q2"), (q1 == q2, "q1==q2"), (q1 > q2, "q1>q2")]:
            for econd, ename in [(np.ones(len(rows), bool), "all"), (same_enc, "same-enc"),
                                 (~same_enc, "cross-enc")]:
                m = ev & (y == cls_i) & cond & econd
                if m.sum():
                    pred_m = clf.predict(x[m])
                    rec = (pred_m == cls_i).mean()
                    as_g1 = (pred_m == 0).mean()
                    print(f"  eval {cls} {cname} [{ename}]: n={m.sum()} recall={rec:.4f} "
                          f"pred-as-gen1={as_g1:.4f}")
                    out_rows.append(["subgroup", cls, f"{cname}[{ename}]", str(int(m.sum())),
                                     f"recall={rec:.4f}", f"as_gen1={as_g1:.4f}", "", ""])
    scale_col = np.array([r["scale"] for r in rows])
    for sc in sorted(set(scale_col[ev & (y == 2)])):
        m = ev & (y == 2) & (scale_col == sc)
        if m.sum():
            as_g1 = (clf.predict(x[m]) == 0).mean()
            print(f"  eval gen2r scale={sc}: n={m.sum()} pred-as-gen1={as_g1:.4f}")
            out_rows.append(["gen2r_scale", sc, str(int(m.sum())), f"as_gen1={as_g1:.4f}",
                             "", "", "", ""])
    srcfmt = np.array([r["srcfmt"] for r in rows])
    for fmt in ("png", "jpg"):
        m = ev & (y == 0) & (srcfmt == fmt)
        if m.sum():
            as_g2 = (clf.predict(x[m]) != 0).mean()
            print(f"  eval gen1 srcfmt={fmt}: n={m.sum()} pred-as-gen2={as_g2:.4f}")
            out_rows.append(["srcfmt", fmt, str(int(m.sum())), f"as_gen2={as_g2:.4f}", "", "", "",
                             ""])

    # ---- dedicated binary gate (gate-lean physics features) ----
    gate_idx = [i for i, n in enumerate(names) if n in GATE_FEATS]
    yb = (y != 0).astype(int)
    bclf = make_clf()
    bclf.fit(x[tr][:, gate_idx], yb[tr], sample_weight=balanced_w(yb[tr]))
    pb1 = bclf.predict_proba(x[:, gate_idx])[:, 0]
    oofb = np.full(len(y), np.nan)
    for trn, val in gkf.split(x[tr], yb[tr], src[tr]):
        c = make_clf()
        c.fit(x[xi[trn]][:, gate_idx], yb[xi[trn]], sample_weight=balanced_w(yb[xi[trn]]))
        oofb[xi[val]] = c.predict_proba(x[xi[val]][:, gate_idx])[:, 0]
    tau_b = sweep("gate-lean-binary", pb1, y, tr, ev, oofb, out_rows)
    for q in (35, 55, 75, 92):
        mg2 = ev & (y != 0) & (q2 == q)
        mg1 = ev & (y == 0) & (q1 == q)
        f2 = (pb1[mg2] >= tau_b).mean() if mg2.sum() else float("nan")
        r1 = (pb1[mg1] >= tau_b).mean() if mg1.sum() else float("nan")
        print(f"    q={q}: eval fg1(q2={q})={f2:.4f} (n={mg2.sum()})  "
              f"gen1-recall(q1={q})={r1:.4f} (n={mg1.sum()})")
        out_rows.append(["gate_q", str(q), f"fg1={f2:.4f}", str(int(mg2.sum())),
                         f"g1rec={r1:.4f}", str(int(mg1.sum())), "", ""])

    # ---- aligned-threat-only gate ----
    # If the slack physics shows resampled chains do NOT break the +-Q/2 box
    # the way aligned chains do (their last encode is honest for the resized
    # signal), the production gate only needs gen1+gen2r vs gen2a. Fit and
    # sweep that gate; also report how many gen2r rows it waves through (by
    # design, under this threat model that is acceptable).
    m2a = tr & (y != 2)
    ya = (y == 1).astype(int)  # gen2a = positive threat
    aclf = make_clf()
    aclf.fit(x[m2a][:, gate_idx], ya[m2a], sample_weight=balanced_w(ya[m2a]))
    pa = aclf.predict_proba(x[:, gate_idx])[:, 0]  # P(not-gen2a)
    me = ev & (y != 2)
    auc_a = roc_auc_score(ya[me], 1.0 - pa[me])
    print(f"\n== aligned-threat-only gate (gen1 vs gen2a, eval AUC={auc_a:.4f})")
    out_rows.append(["aligned_gate_auc", f"{auc_a:.4f}", "", "", "", "", "", ""])
    oofa = np.full(len(y), np.nan)
    xa = np.where(m2a)[0]
    for trn, val in GroupKFold(n_splits=5).split(x[m2a], ya[m2a], src[m2a]):
        c = make_clf()
        c.fit(x[xa[trn]][:, gate_idx], ya[xa[trn]], sample_weight=balanced_w(ya[xa[trn]]))
        oofa[xa[val]] = c.predict_proba(x[xa[val]][:, gate_idx])[:, 0]
    g2a_tr = m2a & (y == 1)
    g1_tr2 = m2a & (y == 0)
    print("  tau sweep (positive threat = gen2a only):")
    tau_a = None
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]:
        fga_oof = (oofa[g2a_tr] >= t).mean()
        rec_oof = (oofa[g1_tr2] >= t).mean()
        fga_ev = (pa[ev & (y == 1)] >= t).mean()
        rec_ev = (pa[ev & (y == 0)] >= t).mean()
        g2r_pass = (pa[ev & (y == 2)] >= t).mean()
        print(f"  tau={t:0.3f} oof:g2a-missed={fga_oof:.4f},g1rec={rec_oof:.3f}  "
              f"eval:g2a-missed={fga_ev:.4f},g1rec={rec_ev:.3f},g2r-waved={g2r_pass:.3f}")
        out_rows.append(["asweep", f"tau={t}", f"oof_g2a_missed={fga_oof:.4f}",
                         f"oof_g1rec={rec_oof:.4f}", f"eval_g2a_missed={fga_ev:.4f}",
                         f"eval_g1rec={rec_ev:.4f}", f"eval_g2r_waved={g2r_pass:.4f}", ""])
        if tau_a is None and fga_oof == 0.0:
            tau_a = t
    if tau_a is not None:
        fga_ev = (pa[ev & (y == 1)] >= tau_a).mean()
        rec_ev = (pa[ev & (y == 0)] >= tau_a).mean()
        na = int((pa[ev & (y == 1)] >= tau_a).sum())
        # the hard aligned case specifically
        mh = ev & (y == 1) & (q1 < q2)
        fga_h = (pa[mh] >= tau_a).mean() if mh.sum() else float("nan")
        print(f"  ALIGNED-CONSERVATIVE tau*={tau_a}: eval gen2a-missed={fga_ev:.4f} "
              f"({na} rows; q1<q2 missed={fga_h:.4f}), eval gen1-recall={rec_ev:.4f}")
        out_rows.append(["aligned_conservative", f"tau={tau_a}", f"eval_g2a_missed={fga_ev:.4f}",
                         f"n={na}", f"q1lt_missed={fga_h:.4f}", f"eval_g1rec={rec_ev:.4f}", "", ""])

    # ---- trivial runtime rule: dq_gapc threshold ----
    gapc = x[:, names.index("dq_gapc")]
    dcf = x[:, names.index("dc_fill")]
    print("\n== plain-rule check on eval (no model): flag gen2a if dq_gapc>=g or dc_fill<=f")
    for g, fthr in [(0.25, 0.35), (0.5, 0.3), (1.0, 0.25)]:
        flag = (gapc >= g) | (dcf <= fthr)
        fp = flag[ev & (y == 0)].mean()
        tp_a = flag[ev & (y == 1)].mean()
        tp_h = flag[ev & (y == 1) & (q1 < q2)].mean()
        tp_r = flag[ev & (y == 2)].mean()
        print(f"  gapc>={g} | dc_fill<={fthr}: gen1-FP={fp:.4f} gen2a-caught={tp_a:.4f} "
              f"gen2a-hard-caught={tp_h:.4f} gen2r-caught={tp_r:.4f}")
        out_rows.append(["plain_rule", f"gapc>={g}|dcfill<={fthr}", f"gen1_fp={fp:.4f}",
                         f"g2a={tp_a:.4f}", f"g2a_hard={tp_h:.4f}", f"g2r={tp_r:.4f}", "", ""])

    # ---- ablations ----
    def run_ablation(tag, idx):
        c2 = make_clf()
        c2.fit(x[tr][:, idx], y[tr], sample_weight=balanced_w(y[tr]))
        report_confusion(tag, y[ev], c2.predict(x[ev][:, idx]), out_rows)

    coeff_only = [i for i, n in enumerate(names) if n not in PIXEL_FEATS]
    run_ablation("coeff-only", coeff_only)
    run_ablation("dq-only", [i for i, n in enumerate(names) if n in DQ_FEATS])
    run_ablation("pixel-only", [i for i, n in enumerate(names) if n in PIXEL_FEATS])

    # ---- single-feature AUCs ----
    print("\n== single-feature AUC (eval; vs gen1)")
    m_a = ev & (y != 2)
    m_r = ev & (y != 1)
    for i, n in enumerate(names):
        try:
            auc_a = roc_auc_score((y[m_a] != 0).astype(int), x[m_a][:, i])
            auc_r = roc_auc_score((y[m_r] != 0).astype(int), x[m_r][:, i])
        except ValueError:
            continue
        if max(abs(auc_a - 0.5), abs(auc_r - 0.5)) > 0.15:
            print(f"  {n:16s} gen2a-auc={auc_a:.3f}  gen2r-auc={auc_r:.3f}")
        out_rows.append(["auc", n, f"{auc_a:.4f}", f"{auc_r:.4f}", "", "", "", ""])

    # ---- permutation importance (train, main model) ----
    pi = permutation_importance(clf, x[tr], y[tr], n_repeats=3, random_state=0, n_jobs=4)
    order = np.argsort(-pi.importances_mean)
    print("\n== permutation importance (train, top 12)")
    for i in order[:12]:
        print(f"  {names[i]:16s} {pi.importances_mean[i]:.4f}")
        out_rows.append(["perm_importance", names[i], f"{pi.importances_mean[i]:.4f}",
                         "", "", "", "", ""])

    if args.out:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        with open(args.out, "w") as fh:
            fh.write(f"# gen_detect_train commit={commit} date={datetime.date.today()} "
                     f"inputs={args.tsv} train_n={tr.sum()} eval_n={ev.sum()} "
                     f"train_sources={len(set(src[tr]))}\n")
            fh.write("section\tk1\tk2\tk3\tk4\tk5\tk6\tk7\n")
            for r in out_rows:
                fh.write("\t".join(str(v) for v in r) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
