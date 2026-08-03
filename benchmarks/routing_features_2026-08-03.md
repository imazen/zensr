# Pushing routing past the content split — 2026-08-03

Two questions: split the distance curves by content (the open item from
`content_routing_2026-08-03.md`), and find out whether anything cheap beats the
binary content class.

**Answers: the distance split ships (+0.058 ssim2, 19 of 20 splits). Nothing
beat the binary class — including one candidate that looked like a clear win on
a single split and was not.**

## 1. Distance curves split by content — SHIPPED

The jpegli/zenjpeg ladder measured encoders, not content classes, so
distance-quantised files were falling back to a pooled curve.

The same `GRAPHIC_ZERO_AC_THRESHOLD` of 0.25 transfers to this family unchanged
— on distance-quantised files it measures precision 0.63–0.89 and recall
0.55–0.88 across the range, so no separate threshold is needed. Separation is
again strongest where the file is cleanest (graphic median zero-AC 0.414 vs
photographic 0.001 at distance 0).

Validated over **20 random image splits**, not one:

| router | mean ssim2 | min | max | beats pooled |
|---|---|---|---|---|
| pooled distance curve | +0.7357 | +0.5713 | +0.8906 | — |
| **content-split, classifier** | **+0.7941** | +0.5971 | +1.0107 | **19/20** |
| content-split, oracle labels | +0.8168 | +0.5865 | +1.0083 | 19/20 |

The classifier captures **72%** of what perfect labels would give. The effect is
about a quarter the size of the IJG family's (+0.058 vs +0.15), which is why it
needed 20 splits rather than one: the content *gap* is large here (1.0–3.8 ssim2
between classes) but the routing headroom is small, because the pooled curve
already restores conservatively on this family and most of the grid sits
near-pristine where both classes agree to skip.

Shipped as `GRAPHIC_DIST420`/`GRAPHIC_DIST444`/`PHOTO_DIST420`/`PHOTO_DIST444`.

## 2. Can anything cheap beat the binary content class?

The per-image oracle is +1.7049 against the shipped router's ~+1.45, so a
quarter of the headroom is still on the table. Five more coefficient-domain
features were extracted in the same pass (`coef_features` bin) and tested.

**Correlation with per-image gain at fixed quality** — a feature has to separate
images *within* a quality cell to add anything:

| feature | q15 | q35 | q55 | q75 | q85 | q90 | q94 | q100 |
|---|---|---|---|---|---|---|---|---|
| `zero_ac_blocks` (shipped) | +0.57 | +0.56 | +0.60 | +0.42 | +0.38 | +0.36 | +0.33 | +0.35 |
| **`mean_abs_ac`** | +0.23 | +0.56 | +0.61 | **+0.67** | **+0.57** | **+0.49** | +0.39 | +0.44 |
| `zero_ac_coefs` | +0.36 | +0.22 | +0.16 | +0.04 | +0.03 | +0.03 | +0.03 | +0.21 |
| `hf_survival` | +0.25 | +0.24 | +0.13 | +0.11 | +0.12 | +0.13 | +0.14 | +0.15 |
| `bpp` | −0.32 | −0.26 | −0.24 | −0.15 | −0.17 | −0.18 | −0.20 | −0.32 |
| `dc_spread` | −0.12 | −0.11 | −0.14 | +0.02 | +0.04 | +0.05 | +0.05 | −0.03 |
| *(true content label)* | +0.31 | +0.61 | +0.69 | +0.58 | +0.51 | +0.47 | +0.42 | +0.39 |

`mean_abs_ac` — mean magnitude of *surviving* AC coefficients — beats the
shipped signal from q75 up, and at q75 beats even the ground-truth content
label. The mechanism is sensible: graphic content has few but *large* AC
coefficients at hard edges, photographic content has many small ones spread
everywhere, so it measures the same thing more finely.

### The trap

On the standard calibrate/validate split, a per-cell linear model on
`mean_abs_ac` scored **+1.5346** against **+1.4518** for the shipped binary and
**+1.4692** for oracle content labels — apparently beating perfect labels, and
restoring less while harming less.

**Over 20 random splits it loses**:

| router | mean | min | max | beats shipped |
|---|---|---|---|---|
| quality only | +1.3002 | +1.0720 | +1.4973 | 1/20 |
| **binary zero-AC (shipped)** | **+1.4806** | +1.2257 | +1.8206 | — |
| linear `mean_abs_ac` | +1.4536 | +1.1695 | +1.7108 | 7/20 |
| oracle content label | +1.4798 | +1.2556 | +1.7956 | 7/20 |

The single-split result was a lucky draw. The spread across splits (±0.3) is
four times the effect being claimed, so one split cannot resolve it — the same
lesson as the per-encoder crossover in `clean_ladder_2026-08-03.md`, reached by
a different route. The fitted slope also varies 15× across cells (1.45 at q35 to
0.07 at q100) while `mean_abs_ac`'s range grows from 0–3.9 to 0–58.7, so a
per-cell linear model is 40 parameters fitted on 64 images — fragile by
construction.

### Other negatives

- **More resolution on the shipped signal does not help.** Quantile buckets of
  `zero_ac_blocks`: 3 → +1.4178, 4 → +1.4415, 5 → +1.4450, 6 → +1.4291, against
  +1.4338 for a binary threshold. The signal carries about one bit.
- **More features overfit.** Adding `zero_ac_blocks` to `mean_abs_ac` scored
  worse than `mean_abs_ac` alone (+1.4526 vs +1.5346 on that split); all six
  features together scored +1.4299.
- **Oracle content labels only tie the shipped binary** (+1.4798 vs +1.4806 over
  20 splits). The classifier is already at the ceiling of what a content-class
  router can do. Further gain requires a different *kind* of signal, not a
  better classifier.

## Where that leaves it

Routing now captures roughly 40% of the per-image oracle headroom on the IJG
family and 72% of the content-label headroom on the distance family. The
remaining gap needs something these six features do not carry — plausibly
something spatial (where the damage is, not how much), which none of them
measure.

Raw: `/mnt/v/zensr/features/2026-08-03/` with `SHA256SUMS`.
