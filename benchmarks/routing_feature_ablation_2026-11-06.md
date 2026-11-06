# Metadata-independent routing candidates — 2026-11-06

Production default is unchanged pending owner selection. Candidate code is
compiled only for tests. No CNN training, encoder generation or cancelled
pixel-artifact audit was resumed.

## Development comparison

235,224 cells / 404 origins from the original immutable 512-pixel development
snapshot. Same ridge/logistic heads, five origin-separated folds, weighting,
quality-dependent harm curve and operating-point grid as the original fit.
The baseline reproduces both original OOF arrays with maximum absolute
prediction difference zero; its full fitted artifact hash also matches exactly.
Features were removed and models refitted, rather than zeroing inputs in the
existing model. Operating points were selected on development OOF predictions.
These frontier selection scores are development diagnostics, not external
held-out estimates.

At the 0.5% development harm target:

| Inputs | Mean gain | Observed harm | Cell restore coverage |
|---|---:|---:|---:|
| Original 37 numeric inputs | +2.491815 | 0.485139% | 77.0283% |
| Remove file bpp (36 inputs) | +2.490652 | 0.480810% | 76.4752% |
| Remove bpp and dimensions (34 inputs) | +2.494270 | 0.470836% | 76.5300% |

All use predicted gain minus 32 times predicted harm probability >= 0, with
no fitted-range abstention. Removing dimensions avoids direct extrapolation of
width/height coefficients; it does not demonstrate native-window quality across
sizes or remove geometry effects from the pixel statistics.

## Secondary check on the reused reserved cohort

Both feature candidates and their operating points were frozen before this
check. It reuses the already-inspected 140,940-cell / 269-origin cohort. Origin
sets are disjoint from development; this is nevertheless not a fresh untouched
publication evaluation. The original baseline summary replays exactly.

| Policy | Mean gain | Observed harm | Cell restore coverage | Changed decisions |
|---|---:|---:|---:|---:|
| Current default | +2.426027 | 0.548952% | 80.4349% | 0 |
| Remove bpp | +2.418632 | 0.569859% | 79.8680% | 2,689 |
| Remove bpp and dimensions | +2.417327 | 0.566162% | 79.7524% | 2,890 |

Gain and harm are origin-balanced. All three retain the same worst loss,
18.534408 SSIM2. Removing bpp and dimensions changes mean gain by -0.008700
(pointwise paired origin-bootstrap 95% interval [-0.013265, -0.004891]) and harm
by +0.017211 percentage points (interval [-0.006502, +0.048954]). These intervals
carry the cohort and reuse limitations above; they are not deployment bounds.
The observed candidate harm is slightly above the owner's previously accepted
0.55%, so activation requires the owner's choice.

## Actual metadata sensitivity

On the 16 retained JPEG replay inputs, inject opaque APP15 payloads while
requiring exact decoded RGB8, coefficient and record-corrected probe parity.
The inputs include eight near-boundary cases and are not a representative sample.

| Added payload | Changed current-default decisions |
|---|---:|
| 4 KiB | 4 / 16 |
| 16 KiB | 5 / 16 |
| 64 KiB | 9 / 16 |
| 256 KiB | 10 / 16 |

The native 34-input candidate preserves gain/risk bits and decisions in all
64 padded cases. Its 140,940 independent sklearn replay vectors match Rust,
as do 16 built-in goldens. Padding is nonsemantic APP15; metadata that changes
color interpretation, orientation or authoritative encode records is not covered
by this invariance control.

## Spatial selection diagnostic

A test-only selector retains a representative center and chooses additional
native windows using maxima of DC structure, structure weighted by erased AC,
AC population and AC-population variance. It uses integral maps, 16-pixel starts,
and at most five deduplicated windows. Maxima can overlap heavily; this is a
candidate heuristic, not a calibrated risk model or an exhaustive local guard.

A synthetic off-center structure missed by all fixed positions is sampled by the
new selector. Eight retained complete development cells at 1024/2048 longest edge,
turbo 420 q50, change prediction scores but no restoration decisions (six accept,
two reject for both selectors). That diagnostic does not establish a quality
improvement. No production spatial policy was replaced. The recorded dense writer
was absent and its scope inactive when the retained cells were read; generation
was not restarted and all partial results remain preserved.

## Reproduction and evidence

Scripts: `tools/routing_feature_ablation.py`, `routing_feature_check.py` and
`routing_candidate_export.py`. Just recipes expose grouped fitting, secondary
checking, native candidate replay, metadata sensitivity and spatial diagnostics.
Missing inputs fail explicitly; output paths must be new. Frozen study artifacts
and baseline expectations are retained.

Verified on i265. Development artifacts:
`/home/lilith/data/zensr-array-archive/routing-feature-ablation-20261106-v1/`.
Secondary report:
`/home/lilith/data/zensr-array-archive/routing-feature-check-20261106-v1.json`.
Diagnostics, independent vectors, selected cell provenance and logs:
`/home/lilith/data/zensr-array-archive/routing-feature-diagnostics-20261106-v1/`.
Each run has source/model/data hashes; generated arrays and model binaries stay
outside git. Source commits: `c1bcd6ab3c6c`, `374dc5099d06`, `82ee1d8f7e5e`,
`e68f09df46d4`, `91abd132db33`.

The complete development wrapper reported:
`run-heavy: done rc=0 77s | peak-RSS 3.78GiB | min-avail 19195MiB | peak-load 2.79`.
The secondary wrapper reported:
`run-heavy: done rc=0 30s | peak-RSS 3.10GiB | min-avail 19789MiB | peak-load 0.64`.
These are complete analysis commands, not deployment-latency benchmarks.
