# Reusing the zensim/zenanalyze feature work for routing — 2026-08-04

Prompted by a fair challenge: months of zensim work produced parquets with
encoded variants, precomputed ssim2 and butteraugli, and hundreds of features.
Why hand-roll nine coefficient features and re-measure everything?

Two answers, and the second one is the useful one.

## 1. The canonical datasets cannot be reused here — they are imazen-26

`/mnt/v/output/canonical-picker-2026-06-27/zenjpeg_lossy/` is genuinely rich:
1,484,010 rows (train/validate/test), 469 `feat_*` columns, `score_ssim2`,
`content_class`, encoded-variant R2 URLs, `encode_ms`/`decode_ms`.

It is unusable for dejpeg eval:

- Its README says it is built from "the all-origin clean-picker **imazen-26**
  sweep", and its origin ids sit in 1000–9999 — the imazen-26 range, per the
  sibling nonphoto corpus manifest which documents its own 10000–10501 as
  "disjoint from imazen-26 1000-9999".
- The dejpeg model trains on imazen-26. So these sources are very likely
  training data.
- The manifest's `leakage_check` reports `leaked: false`, but that covers
  **picker-internal train/val/test overlap** — a different question from
  overlap with a different model's training set.
- No `origin_id` → imazen-26 path map exists locally to filter by;
  `origin_split.py` derives the id from the filename and never records a source
  path.

Reusing it would reintroduce exactly the contamination this line of work exists
to remove. Two other details that would have limited it anyway: `content_class`
is `unknown` on every row, and the q grid has only 7 distinct values.

## 2. The precomputed metrics would not have helped much regardless

The stored `score_ssim2` is **(reference, encoded)** — the `identity_off` arm.
The metric this work needs is **(reference, restored)**, which requires running
the dejpeg model and which nobody has precomputed. Reuse would save one arm of
three plus the encode, not the expensive part.

Measured per call at 512² (`metric_cost` bin): ssim2 19.71 ms, butteraugli
16.30 ms, encode 1.99 ms. Model inference and decode dominate beyond those.

## 3. What DOES transfer: the feature set

The features in those parquets are **source-image** features — verified constant
across all 330 encodes of one reference. So they join to any encode at any
quality by filename, and the feature *design* transfers even though the data
does not. `source_features` now extracts them on any corpus:
**101 features on 913 images in 22 seconds**, one row per reference.

Against per-image restore gain on the clean 64-image corpus, the best
zenanalyze feature beats every hand-rolled one:

| feature | q35 | q55 | q75 | q85 | q90 |
|---|---|---|---|---|---|
| **`edge_slope_stdev`** | +0.75 | +0.79 | **+0.81** | +0.75 | +0.68 |
| `patch_fraction_fast` | +0.65 | +0.75 | +0.55 | +0.49 | +0.45 |
| `aq_map_std` | +0.67 | +0.71 | +0.70 | +0.60 | +0.51 |
| `colourfulness` | −0.58 | −0.66 | −0.54 | −0.56 | −0.55 |
| *`mean_abs_ac` (best hand-rolled)* | +0.56 | +0.61 | +0.67 | +0.57 | +0.49 |
| *(ground-truth content label)* | +0.61 | +0.69 | +0.58 | +0.51 | +0.47 |

`edge_slope_stdev` at q75 (+0.81) beats both the best coefficient feature
(+0.67) and the true content label (+0.58).

**And unlike `mean_abs_ac`, it survives the 20-split test:**

| router | mean over 20 splits | beats shipped |
|---|---|---|
| quality only | +1.2119 | 0/20 |
| binary zero-AC (**shipped**) | +1.3569 | — |
| oracle content label | +1.3730 | 16/20 |
| **linear `edge_slope_stdev`** | **+1.4424** | **20/20** |
| linear `edge_slope_stdev` + `aq_map_std` | +1.4339 | 20/20 |
| per-image oracle | +1.5343 | — |

It captures **71% of the per-image oracle headroom** against 45% for the shipped
binary, and beats perfect content labels — a per-image continuous signal
outperforms a perfect two-class one. Adding a second feature makes it slightly
worse, the same overfitting pattern seen everywhere else at this sample size.

## Cost

Marginal, on already-decoded pixels — which is what a router pays, since the
restore path decodes regardless:

| | ms/call |
|---|---|
| coefficient path (shipped) | 0.29 |
| `edge_slope_stdev` alone | **2.15** |
| all 101 zenanalyze features | 4.51 |

7× the shipped path for +0.086 ssim2. On a 512-crop realtime restore (~42 ms)
that is 5% overhead against 0.7%.

## Not shipped yet

20/20 splits at n=64 is strong, but n=64 has produced two false positives
already today (the per-encoder crossover, and `mean_abs_ac` at +1.5346 on one
split). The XL sweep lands 913 images within hours and settles it. Ship after
that confirms, not before.

If it holds, the shape is: replace the binary content class in
`Routing::Auto` with a per-cell linear model on `edge_slope_stdev`, keeping the
coefficient classifier as the fallback when features cannot be computed.
