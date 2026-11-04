# Canon4 realtime restoration — 2026-11-04

The realtime tool defaults now select `dejpeg_rt24i_canon4`. The library takes
caller-loaded weights; its routing calibration is for this model, not arbitrary
models. This is a source-tree update, not a crates.io release. The prior rt24g
artifact remains available. See [dependencies](../docs/PRODUCTION_DEPENDENCIES.md).

## Evidence and image separation

The original two production TSVs remain untouched under
`/home/lilith/data/zensr/benchmarks/` on i265:
`prod_dejpeg_rt24g_nogate_2026-11-04.tsv` and
`prod_dejpeg_rt24i_canon4_nogate_2026-11-04.tsv`.
The candidate has SHA-256
`704e52b0f32bd526e01fcaa725f9fceb9be501ee85639e4c5326694e9c8ba5d1`.
Each has 63 images × 8 encoder legs × 420/444 × 10 qualities
(15,35,55,75,85,88,90,92,94,96) × 6 arms = 60,480 rows.

The current harness added q98/q100 on the same 63 files and all 8 encoder legs,
gate disabled. `canon4-tail.tsv` contains the additional 12,096 score rows.
The combined grid is 12,096 image-cells, each with six arms. References are PNG,
from the canonical effective held-out split used by the original runs. The
calibration/validation subdivision uses the existing `hash_stem(filename)%2`:
33 calibration images and 30 validation images. All encoder/quality variants of
one image stay in its half. Repeated encodes are not independent images.

The original TSV records the pre-record probe, rounded to one decimal, and
lacks coefficient statistics. The metadata-only harness reproduced every
original identity SSIM2 score and probe field exactly, then recorded the real
record-corrected family/scale/value and zero-AC fraction. The same join check
covers the tail. No nominal quality was substituted for runtime probe quality.
Notably, zenjpeg:mozbase changes from an Unknown fingerprint to a recorded
Mozjpeg-family probe; turbo often fingerprints as ImageMagick.

Additional artifacts are retained under
`/home/lilith/tmp/handoff/canon4-ship/` on i265:

- `probe-metadata.tsv`, `tail-metadata.tsv`, `canon4-tail.tsv`;
- `fit-final.json`: all fitted points, calibration/validation summaries,
  threshold search and per-encoder/ss/q raw/projected medians and harm counts;
- `tail-binaries.sha256`, `tail-eval.log`, `tail-metadata.log`;
- `runtime-policy.tsv`, `runtime-policy.log`: final Rust decision verification;
- `final-tests.log`, `final-clippy-recheck.log`, `golden-check.log`.

The 555-second tail run used run-heavy: peak RSS 0.48 GiB, minimum available
22,359 MiB, peak load 11.15. Metadata for the original grid took 151 seconds:
peak RSS 0.45 GiB, minimum available 22,746 MiB, peak load 8.29. No timing
comparison between model weights is claimed.

## Fit and skip policy

`tools/fit_canon4.py` rejects duplicate/missing joins and any changed identity
score or original probe. Curves use paired `model_proj - identity_off` gains.
`model_off - identity_off` is also retained in the cell report; this is the
unprojected guarded-model arm, not a pure network-only measurement.

The four primary encoders (turbo, mozjpeg, jpegli, zenjpeg) fit the curves;
additional zenjpeg variants do not multiply that encoder's weight in the curve
fit. The runtime quality axes stay separate. Each subsampling and content class
has its own measured curve. Weighted isotonic regression smooths per-cell
medians into monotone curves, with cell sample count as the weight. Thus the
values are monotone fitted medians, not claims that every sample median is
monotone. Interpolation clamps beyond the measured range.

The content class is the actual coefficient statistic, not a directory label.
Thresholds 0.05 through 0.55 at increments of 0.05 are compared on calibration
images only, including the fitted gate and AQ adjustment. Threshold **0.20**
maximizes mean realized calibration gain at the existing min_gain=0.25.
The held-out half does not choose the threshold or curves. Paired AQ-off minus
AQ-on projected gains replace the rt24g AQ correction in `record.rs`.

The high-quality gate uses the historical definition of harm (loss greater
than 0.1 SSIM2). The earlier narrative did not specify a numeric fitting
algorithm; this implementation makes it explicit: a sampled tail cell is
harmful if at least half of its calibration observations are harmed. The
boundary is the first sampled value of a contiguous harmful tail. It uses
family/scale/subsampling, pooling records sharing those runtime inputs.
This is a routing policy, not a guarantee that every gated image would lose.

| gate | old 420 / 444 | canon4 420 / 444 |
|---|---|---|
| IJG quality, including Unknown-family fallback | 94.5 / 88 | 96 / 96 |
| Mozjpeg quality | 94.5 / 88 | 100 / 96 |
| Butteraugli distance (gate at or below) | 0.6 / 1.3 | 0.46 / 0.28 |

Unknown quality **scales** are not treated as IJG quality by the estimator.
Other chroma geometries retain the subsampled gate fallback; full consistency
is not claimed for their unprojected chroma.

## Constants before and after

Full arrays and provenance comments live in `src/calibration.rs` inside
zensr-zenjpeg and are generated from the frozen fit. Representative values
below cover every curve; distance entries are interpolated at d=1.3.

| curve | axis value | before | after |
|---|---:|---:|---:|
| G420 | 94 | -0.2700 | +0.4590 |
| GRAPHIC420 | 75 | +3.7700 | +3.2715 |
| PHOTO420 | 75 | +0.1000 | +0.6770 |
| G444 | 94 | -0.6200 | +0.1050 |
| GRAPHIC444 | 75 | +3.5100 | +2.7580 |
| PHOTO444 | 75 | -0.0900 | +0.5910 |
| DIST420 | 1.3 | -0.1100 | +0.6090 |
| GRAPHIC_DIST420 | 1.3 | +0.4700 | +0.7060 |
| PHOTO_DIST420 | 1.3 | -0.5700 | +0.4240 |
| DIST444 | 1.3 | -0.2600 | +0.4635 |
| GRAPHIC_DIST444 | 1.3 | +0.2400 | +0.7106 |
| PHOTO_DIST444 | 1.3 | -0.7100 | +0.3194 |

Zero-AC threshold: 0.25 → 0.20. The former low-q 444 approximation (420 shifted
by four quality points) is removed; 444 is measured directly. Projection slack
is retained (relative IJG 0.15, Mozjpeg 0.35, Cjpegli/Unknown 0.45; absolute
Cjpegli 0.5, others 1.5). It describes encoder/decoder coefficient mismatch,
not the restoration model. These score grids do not identify an optimal new
slack, and the entire projected arm is measured with the retained values.
This is **not** a fresh slack calibration or proof that its intervals contain
the original coefficients for every file.

## Held-out quality, work and harm

30 images × 8 encoders × 2 subsamplings × 12 qualities = 5,760 observations.
Harm fraction is over all observations, not just restored ones. Skipped cells
receive zero gain. The old-policy comparison uses the same canon4 output
measurements and the old estimator/gates, with the skip implementation repaired;
it is not a model A/B comparison and does not credit the old skip defect.

| policy | mean SSIM2 gain | restored | harmed (>0.1 loss) |
|---|---:|---:|---:|
| old rules, repaired skip | +1.3340 | 30.42% | 1.11% |
| canon4 Auto, Fidelity | +1.6261 | 68.65% | 6.68% |
| canon4 Auto, Conservative (existing margin 6) | +1.0409 | 25.50% | 1.06% |

Default routing buys additional gain at the cost of more per-image regressions.
The primary four-encoder subset gives +1.7744 mean gain, 69.27% restored and
5.90% harmed. These sample results are not a never-worse promise. Near-pristine
420 medians can differ in sign between calibration and validation; the tail
is not a universal physical crossover. The score grid starts at q15; q5–14
and multi-generation inputs were not recalibrated here. Existing low-q decode
policy remains unchanged.

## Runtime correctness and model verification

The review found two defects independent of the fitted values:

- Auto rejection and Routing::Never still ran the model while labeling its
  output Unchanged. They now return before model inference; skipped requests
  also bypass deblocking. Regression tests compare pixels with the decoder.
- Luma-only projection on 422/440 could claim full consistency. Strict requests
  now return decoded pixels, and non-strict requests report the missing chroma
  projection. Projection with nonzero slack is not a bit-exact re-encode
  guarantee; the API documentation now states this limit.

Model weights are **86,406 bytes** (43,203 f16 values), SHA-256
`c3db9373791270bbe5493d3ee98cd4f62296510fe6931073e2966f433e9a5cf4`.
Both saved PyTorch goldens reproduced exactly from the f16-roundtripped
weights. Rust whole/tiled maximum errors were 2.44e-5 and 2.00e-5 for 40×36
and 17×13, with no non-finite values. A permanent runtime test loads the actual
committed artifact, checks both goldens, and requires whole/tiled equality.
The f32 export remains untracked; the original rt24g directory is retained.

The user authorized updating model-dependent tests. The q94-negative
assumption is replaced by positive 420 q94 and negative q98/q100 checks.
The universal 444-below-420 and graphic-above-photo assumptions are restricted
to their measured bands, with measured near-pristine crossings preserved.
Monotonicity, pristine-negative endpoints, actual decoder equality, geometry,
margin direction, and the artifact golden checks remain enforced.

`cargo test -p zensr-zenjpeg -p zensr-micro` passed: 32 micro and 34 zenjpeg
tests, plus doc tests. run-heavy: peak RSS 0.22 GiB, minimum available 23,119 MiB,
peak load 5.22. Clippy with all targets and warnings denied passed. Calibration
integrity tests prove held-out gain changes cannot change fitted curves/gates.

Final runtime verification regenerated all 12,096 image-cells and checked
24,192 Fidelity/Conservative gate and Auto decisions against the frozen fit:
**zero decision mismatches**. All identity/probe joins matched. Maximum
Python-versus-f32 estimate difference was 0.00005965 SSIM2. The 184-second
verification used run-heavy: peak RSS 0.39 GiB, minimum available 22,764 MiB,
peak load 9.01. `tools/check_canon4_policy.py` performs this check.

## Reproduction

```sh
python3 tools/fit_canon4.py \
  --scores /home/lilith/data/zensr/benchmarks/prod_dejpeg_rt24i_canon4_nogate_2026-11-04.tsv /home/lilith/tmp/handoff/canon4-ship/canon4-tail.tsv \
  --metadata /home/lilith/tmp/handoff/canon4-ship/probe-metadata.tsv /home/lilith/tmp/handoff/canon4-ship/tail-metadata.tsv \
  --out /home/lilith/tmp/handoff/canon4-ship/fit-final.json \
  --rust-output crates/zensr-zenjpeg/src/calibration.rs
python3 -m unittest discover -s tools -p test_fit_canon4.py
```

Original production-decision context: [prod_models_2026-11-04.md](prod_models_2026-11-04.md).
