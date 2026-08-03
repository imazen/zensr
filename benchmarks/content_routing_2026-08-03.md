# Content-aware routing — measured and shipped, 2026-08-03

ROADMAP §1.12 measured that content type is the largest routing lever left
after quality, using ground-truth labels — a ceiling, not an achievable gain.
This closes it: two candidate classifiers measured head-to-head on the pinned
split, the cheaper one shipped, and the realized gain confirmed held-out.

**Result: +0.148 ssim2 of realized routing quality for 0.29 ms per image, 82%
of the oracle-label ceiling.**

## The two candidates

| | signal | where it runs | marginal cost |
|---|---|---|---|
| **coefficient** | fraction of luma blocks with no AC coefficients | the file's own DCT coefficients | **0.29 ms** |
| chooser | 21-feature logistic rule over zenanalyze features | centre-512 crop of the decoded image | 4.54 ms |

Costs are marginal and measured on 512-crops (`content_probe` bin, medians over
320 image×quality cells). The chooser's decode is **excluded** — the restore
path decodes pixels regardless, so it is not marginal to this decision.
Including it would have flattered the coefficient path, which does pay for its
own separate parse. The chooser (`crates/zensr-zenjpeg/src/chooser.rs`) was
trained and validated in July but had **zero call sites**; this is the first
measurement of it against an alternative.

## Classifier accuracy (held-out, 34 images)

Truth is the subcorpus: documents/maps/screen graphic, the rest photographic —
the same grouping §1.12 used, so the numbers are directly comparable.

| rule | P | R | F1 |
|---|---|---|---|
| coefficient, single threshold 0.30 | 0.744 | 0.957 | 0.838 |
| coefficient, per-q threshold | 0.767 | 0.943 | 0.846 |
| chooser t=0.85 (shipped for model routing) | 0.947 | 0.771 | 0.850 |
| **chooser t=0.75** | 0.900 | 0.900 | **0.900** |

The chooser is the better classifier. The coefficient signal trades precision
for recall, and its separation is strongly quality-dependent — median zero-AC
fraction is 0.694 graphic vs 0.311 photographic at q15 (2.2x), but 0.515 vs
0.001 at q90 (**515x**). Heavy quantisation flattens photographs too; only at
high quality does a zero-AC block mean the content is genuinely flat. That is
convenient, because high quality is exactly where the routing decision is
hardest and most valuable.

## What actually matters: realized routing quality

Accuracy is not the objective — realized ssim2 is. Threshold selected on the
calibrate half, reported on the held-out half:

| router | ssim2 | restored | harmed | share of ceiling | cost |
|---|---|---|---|---|---|
| quality only (previous behaviour) | +1.2969 | 0.35 | 0.02 | 0% | 0 |
| **coefficient @ 0.25** | **+1.4452** | 0.46 | 0.04 | **82%** | **0.29 ms** |
| chooser t=0.75 | +1.4511 | 0.46 | 0.04 | 85% | 4.54 ms |
| oracle labels | +1.4781 | 0.46 | 0.03 | 100% | — |
| per-image oracle | +1.7049 | 0.58 | 0.00 | — | — |

**The chooser's extra 0.006 ssim2 is an order of magnitude below the ~0.3
metric floor, and costs 16x more.** On a 512-crop realtime restore (~42 ms at
0.16 s/MP) that is 11% overhead versus 0.7%. The coefficient path ships.

Threshold 0.25 was chosen by maximising realized quality on the calibrate half
only. It sits mid-plateau — 0.20 through 0.30 all land within 0.01 ssim2 of
each other held-out — so the choice is not balanced on a peak. It is
deliberately *not* zenjpeg's `SCREENSHOT_ZERO_AC_THRESHOLD` of 0.10, which is
calibrated for picking a deblock strategy: a different decision with a
different loss.

**Honest cost:** restoring more also harms more — 2% of cells to 4%. The
oracle-label router harms 3%, so most of that is inherent to restoring in more
places rather than to misclassification. Mean quality rises by +0.148, well
above the floor, so the trade is clearly worth taking; it is recorded here
rather than buried.

## What shipped

`crates/zensr-zenjpeg/src/api.rs`: `GRAPHIC420`/`GRAPHIC444`/`PHOTO420`/
`PHOTO444` (the pooled `G420`/`G444` split by content, same corpus and method),
`classify_content`, and `estimate_gain_for`. `Routing::Auto` classifies and
routes on the split curves; every other `Routing` mode is unchanged and pays
nothing. An unparseable or truncated file returns `None` and routes exactly as
before.

Public `estimate_gain(&probe)` is unchanged and still quality-only — the
content signal needs the compressed bytes, not a header probe, so it cannot be
offered through that signature. Its docs now say plainly that `Auto` is better
than it and will make decisions it would not.

Where the split changes decisions at `min_gain: 0.25`: graphic content is now
restored up to ~q96, photographic content skipped from ~q75 up. The pooled
curve split the difference and was wrong for both.

## Not done

- The `ButteraugliDistance` curves (cjpegli, zenjpeg) are **not** split by
  content — that ladder measured encoders, not content classes. Those files
  fall back to the pooled distance curve. Measuring the split there is the
  obvious next step.
- Only two classes. §1.12 measured that 8 classes add +0.007 over 2, so there
  is no case for more.
- n=64 images. The split's *magnitude* is large enough to survive this (~4
  ssim2 at q75, against 0.3–0.5 for the encoder effect that did **not**
  survive), but the curve values themselves carry the usual n=64 uncertainty.
