# Restore ladder on clean references — 2026-08-03

The first quality ladder measured against a corpus with **no JPEG-sourced
ground truth**. Every previous ladder scored some fraction of images against
references that were themselves JPEGs, which flatters restoration: the model
moves output toward something already JPEG-like, so the metric rewards it for
reproducing the very artifacts it is supposed to remove. This run replaces
those numbers.

It is also the run that decides whether the shipped routing estimator survives
contact with clean references. It does — and two changes I expected to make
turned out to be unsupported. Those negative results are recorded below,
because both are the kind that get re-derived expensively.

## Provenance

| | |
|---|---|
| corpus | `/mnt/v/imazen-26-clean` — 974 refs across 8 subcorpora, **100% PNG**, 0 JPEG-sourced |
| files scored | the 64 pinned eval files (`eval_split/imazen26_eval_files.tsv`), 8 per subcorpus |
| leakage check | all 64 scored stems ∈ pin list, set difference = 0 (training excludes pin ∪ first-8-sorted) |
| grid | q ∈ {15,35,55,75,85,90,94} × {turbo, mozjpeg} × {4:2:0, 4:4:4} × 3 arms = 5,376 rows |
| model | `dejpeg_rt24g` (realtime tier) |
| arms | `identity_off` (decode only), `model_policy`, `model_proj` |
| metric | ssim2, per file; `gt_src` recorded per row and verified `png` for all 5,376 |
| commit / host | `0a43a70` / `lilith` (WSL2, 28 cores) |
| raw | `~/tmp/clean_ladder_full.tsv`; analysis `tools/gate_recalibrate.py` |

Analysis splits the 64 images into 30 calibrate / 34 validate **by image**, so
no threshold is ever scored on the files that chose it.

## 1. The encoder effect is real and large

Paired per-image comparison (same image, both encoders), two-sided exact
binomial sign test on `model_policy − identity_off`:

| ss | q | mozjpeg | turbo | turbo−moz | signs +/− | p |
|---|---|---|---|---|---|---|
| 420 | 15 | +4.258 | +6.681 | **+1.368** | 62/1 | 1.4e-17 |
| 420 | 35 | +2.734 | +3.348 | +0.515 | 51/12 | 7.5e-07 |
| 420 | 55 | +1.755 | +1.738 | +0.472 | 50/13 | 3.0e-06 |
| 420 | 75 | +0.458 | +0.822 | +0.301 | 53/10 | 3.4e-08 |
| 420 | 85 | +0.117 | +0.243 | +0.234 | 49/14 | 1.1e-05 |
| 420 | 90 | −0.029 | +0.035 | −0.007 | 31/32 | 1.0 |
| 420 | 94 | −0.280 | −0.254 | −0.196 | 20/43 | 0.0052 |
| 444 | 15 | +4.201 | +6.636 | **+1.435** | 63/0 | 2.2e-19 |
| 444 | 35 | +2.608 | +3.441 | +0.577 | 55/8 | 9.8e-10 |
| 444 | 55 | +1.543 | +1.795 | +0.500 | 51/12 | 7.5e-07 |
| 444 | 75 | +0.223 | +0.553 | +0.478 | 56/7 | 1.4e-10 |
| 444 | 85 | −0.058 | +0.171 | +0.280 | 50/13 | 3.0e-06 |
| 444 | 90 | −0.328 | −0.181 | +0.111 | 36/27 | 0.31 |
| 444 | 94 | −0.635 | −0.571 | −0.074 | 28/35 | 0.45 |

**libjpeg-turbo images gain more from restoration than mozjpeg images at the
same nominal quality**, at every q from 15 to 85, in both subsamplings. The
mechanism is not mysterious: mozjpeg's trellis quantization and tuned tables
produce a better image at a given q, so there is less damage left to repair.
Above q90 the effect vanishes into the noise (p = 1.0, 0.31, 0.45) — both
encoders are near-pristine there and neither leaves much to recover.

Sign tests rather than magnitude comparisons because the differences at high q
sit below the ~0.3 ssim2 metric floor.

## 2. NEGATIVE — the per-encoder crossover point is not identifiable at n=64

Lowest q whose median gain is ≤0 and stays ≤0 above it:

| enc | ss | calibrate (30 img) | validate (34 img) |
|---|---|---|---|
| mozjpeg | 420 | q85 | **q94** |
| mozjpeg | 444 | q75 | **q90** |
| turbo | 420 | q94 | q94 |
| turbo | 444 | q90 | q90 |

turbo is stable across the split; mozjpeg swings 9 and 15 q points. Pooling all
64 files gives mozjpeg 444 a crossover of q85, which would have justified
lowering that gate threshold to ~q83 — and the validate half says q90.

Near crossover the median is ≈0, so tiny sampling noise flips its sign, while
per-image spread is large (p10..p90 runs −1.4..+4.2 in a single cell). The
*paired* test in §1 is precise because pairing cancels image-to-image variance;
an unpaired per-cell median does not. **Do not ship per-encoder crossover
thresholds from a 64-image ladder.**

## 3. Shipped estimator vs clean references (validate images only)

`estimated − measured`, positive = estimator optimistic:

| enc | ss | q75 | q85 | q90 | q94 |
|---|---|---|---|---|---|
| mozjpeg | 420 | −0.640 | −0.210 | +0.035 | **+0.330** |
| mozjpeg | 444 | +0.241 | +0.345 | +0.129 | +0.243 |
| turbo | 420 | −1.023 | −0.166 | +0.164 | **+0.316** |
| turbo | 444 | −0.499 | +0.200 | +0.128 | +0.182 |

Two biases, only one of which matters:

- **Mid-range (q55–75): pessimistic** by up to 1.0 ssim2. Harmless — both the
  estimate and the truth are far above any routing threshold, so no decision
  changes.
- **Far tail (q94): optimistic, and wrong in sign.** The curve predicts +0.15
  for 4:2:0 where clean references measure ≈ −0.22. At the default
  `min_gain: 0.25` both values skip, so default routing is unaffected; it bites
  only a caller configuring a lower threshold to chase maximum quality.

The tail above q94 is being measured separately rather than extrapolated —
the pre-existing q96/q98/q100 entries came from the contaminated run.

## 4. NEGATIVE — no routing rule beat the shipped estimator

Each rule scored on the validate images by summing the **actual measured**
delta on every cell it chose to restore:

| rule | mean ssim2 | restored |
|---|---|---|
| always restore | +1.8561 | 1.00 |
| never restore | +0.0000 | 0.00 |
| shipped gate (subsampling thresholds) | +1.9993 | 0.86 |
| **shipped estimator, min_gain=0.25** | **+2.0274** | 0.79 |
| encoder-aware estimator (independent per-encoder curves) | +1.8544 | 0.50 |
| restore iff calibrate-fitted cell median > 0 | +1.9539 | 0.64 |
| shipped estimator + paired encoder offset | +2.0162 | 0.75 |

The shipped estimator wins on quality and beats "always restore" by +0.17
ssim2 while skipping 21% of the work — routing is worth having, and the
version already shipped is the best of the six.

Refitting per-encoder curves is actively harmful (+1.85, and it declines half
the restores) for the §2 reason: unpaired cell medians are too noisy at this
sample size. Folding the encoder effect in as a *paired* offset — the
statistically sound form — costs 0.011 ssim2 and saves 4 points of work. That
difference is an order of magnitude below the metric floor, so it is a wash,
not an improvement.

## Conclusions

1. **Do not change the routing estimator or the gate thresholds.** Both survive
   clean references; nothing tested beats them. (Supersedes the expectation
   that contaminated calibration would have to be redone wholesale.)
2. Fix only the far tail (q ≥ 94) of the 4:2:0 curve, from measurement, once
   the tail run lands. No behavior change at default `min_gain`.
3. The encoder effect is real but belongs in the *magnitude* estimate, not in
   crossover thresholds — and at the decision level it buys nothing today. It
   would start to matter if a caller ran a much lower `min_gain`.
4. Any future ladder that wants per-encoder thresholds needs far more than 64
   images, or a paired design.
