# canon4 realtime model: preparation and calibration blocker — 2026-11-04

**Status: not shipped as the realtime default.** The f16 candidate and its
four goldens are committed, but the old model references, routing curves, and
identity gate remain in force. The requested complete refit conflicts with
existing shape and gate expectations. Those expectations were not weakened.

## Input and method

The production candidate TSV on i265 is
`/home/lilith/data/zensr/benchmarks/prod_dejpeg_rt24i_canon4_nogate_2026-11-04.tsv`
(SHA-256 `704e52b0f32bd526e01fcaa725f9fceb9be501ee85639e4c5326694e9c8ba5d1`).
It contains 60,480 rows: 63 PNG references × 8 encoder legs × 2
subsamplings × 10 qualities (15, 35, 55, 75, 85, 88, 90, 92, 94, 96) ×
6 arms. Each arm has 10,080 rows. `tools/canon4_calibration.py` pairs each
cell with `identity_off`, rejects duplicate and missing relevant arms, and
reports both `model_off` and `model_proj` differences. It uses
`gate_recalibrate.py`'s deterministic hash partition by filename: 33 images
for calibration and 30 for validation. Run:

```sh
just canon4-calibration /home/lilith/data/zensr/benchmarks/prod_dejpeg_rt24i_canon4_nogate_2026-11-04.tsv
```

The TSV stores the original probe family and a rounded probe value. It does
not store the probe scale, the record-corrected probe, or the luma zero-AC
fraction. At encoder q96, turbo probes as `ImageMagick`, mozjpeg as `Mozjpeg`,
jpegli and most zenjpeg legs as `CjpegliYcbcr`, and zenjpeg:mozbase as
`Unknown`. In the last case the original probe value is 93; the verified
record may change family and quality in the actual restore path. Do not fit
record-aware thresholds using the original probe columns as if they were the
values after `record::apply_record`. The TSV also has no slack variants and
no q98/q100 cells, so those quantities cannot be re-estimated from it alone.

## Curve conflict

The existing `api::tests::gain_curve_is_negative_on_near_pristine_input`
requires a negative quality-only estimate for **both** subsamplings at q94,
q96, q98 and q100. For the directly relevant IJG-family legs (turbo and
mozjpeg), pooling the two encoders at q94 gives:

| subsampling | arm minus identity | calibration median, 66 pairs | validation median, 60 pairs |
|---|---|---:|---:|
| 420 | model_off | +0.294 | +0.170 |
| 420 | model_proj | +0.459 | +0.441 |
| 444 | model_off | +0.058 | -0.0025 |
| 444 | model_proj | +0.105 | +0.0065 |

These are per-file paired deltas, pooled over the two encoders; the two
observations for one file are correlated and are not 66 independent images.
A median-fitted 420 q94 anchor would be positive in both halves, contradicting
the established negative-sign test. The unmodified old curve at that point
is -0.27. Fitting a negative value and describing it as this dataset's median
would be false; changing the assertion would weaken a contract the brief
explicitly preserves. This needs a decision on that contract before a faithful
median refit can land.

The gate also has a behavioral constraint: `high_q_gate_is_stricter_at_444`
requires a zenjpeg 444 encode at q90 and q92 to gate. On this grid zenjpeg
444's projected gain medians were +0.376/+0.264 at q90 and +0.186/+0.168 at
q92 (calibration/validation respectively). Its original recorded probe values
are 1.1 and 0.7; the gate currently fires on distance ≤1.3. Before fitting a
replacement, capture the post-record probe for the measured files and decide
how to reconcile those positive cells with the gate test.

For comparison, turbo 444 q96 projected restoration harmed 29 of 33
calibration images and 24 of 30 validation images, with medians -0.353 and
-0.3755 respectively. This is strong evidence for an identity gate at that
cell, but does not identify an exact threshold between the sampled q94 and
q96, or determine the behavior of the other families beyond q96.

## Constants and gate

No calibrated constants or gate thresholds were changed: the current 420/444
IJG thresholds remain 94.5/88, and the distance thresholds remain 0.6/1.3.
G420/G444, DIST420/DIST444, content-specific curves, zero-AC threshold, and
projection slack remain the previous-model values. The existing gate and
estimator can therefore miss gains from the candidate; the candidate must not
be called the new default on this evidence.

To complete the requested fit, recover or reproduce the corresponding encoded
files and record the post-`apply_record` scale/family/value and zero-AC
fraction for each image-cell, with provenance matching this TSV. Measure any
slack choices and the unobserved high-q tail if those constants are to be
refitted. Resolve the sign and gate contracts explicitly, then fit on the 33
calibration images, score the selected rule on the 30 held-out images, rerun
all shape and behavioral tests, and repoint the defaults and documentation.

## Artifact verification

`dejpeg_rt24i_canon4/weights_f16.raw` SHA-256 is
`c3db9373791270bbe5493d3ee98cd4f62296510fe6931073e2966f433e9a5cf4`.
The f16 bytes match conversion of the local f32 training export. Using
`tools/verify_adopted_f16.py`, the saved 40×36 and 17×13 PyTorch outputs both
match an f16-roundtripped forward exactly (max absolute error 0). The Rust
`zensr-adopted-verify` checks both whole and tiled paths: maximum absolute
errors versus these goldens were 2.44e-5 and 2.00e-5, respectively, with no
non-finite outputs. The f32 export is not committed.

Baseline `cargo test -p zensr-zenjpeg -p zensr-micro` passed with the original
routing and gate intact: 31 micro and 32 zenjpeg unit tests passed, including
the shape tests. The run used `run-heavy --mem 16G --jobs 6`: peak RSS 0.73
GiB, minimum available 22,155 MiB, peak load 4.41. This verifies the existing
code, not a refit or a completed model swap.

## Continuation: metadata recovered (2026-11-04)

The user authorized updating model-dependent tests. A metadata-only mode in
`dejpeg_eval` now captures the real API report after record correction and the
luma zero-AC fraction. The restored skip implementation avoids model inference.
`just canon4-metadata` reproduced all 10,080 original cells with identical
rounded identity SSIM2 and original probe fields. Metadata is retained at
`/home/lilith/tmp/handoff/canon4-ship/probe-metadata.tsv` on i265. The 151-second
run used run-heavy: peak RSS 0.45 GiB, minimum available 22,746 MiB, peak load
8.29. These checks establish the join against the original production scores;
the earlier missing-metadata blocker is resolved.
