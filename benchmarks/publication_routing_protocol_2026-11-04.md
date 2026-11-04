# Publication routing study protocol — 2026-11-04

Status: protocol and corpus audit, before fitting richer routers. No deployment
policy is selected; report the gain/risk frontier. The four-encoder canon4
weights, JPEG dependency revisions, decoder, guards and projection stay frozen.

## Questions and candidate families

1. Does continuous coefficient information improve on the current binary zero-AC
   classifier at matched calibration harm budgets?
2. Do decoded-image features (including edge-slope spread and the existing
   21-feature chooser) add useful predictive information at acceptable cost?
3. Do post-inference guard/projection diagnostics improve accept/reject decisions,
   once their inference cost is charged? This is a distinct budget from routing
   before inference.

Compare unchanged decode, ungated restoration, current Auto Fidelity and
Conservative, calibrated threshold-only policies, coefficient gain/risk
predictors, decoded-feature gain/risk predictors, and post-inference variants.
Include an explicitly reference-dependent oracle as an upper bound only.
Use regularized linear/logistic and shallow tree models, with fixed small
hyperparameter grids and grouped validation; do not infer superiority from a
training score. No reference-image features, nominal encoder dial, directory
label, filename, origin ID or reference quality may enter inference features.
The encoder dial and content labels are evaluation strata only.

## Data separation

The canonical corpus checkout is `187fbf338ce08e8e6654db7f04ddae58d5263da2`.
The existing effective split matches `tools/corpus_split.py` exactly. Model
training origins are excluded, using stronger document-level grouping for
patents, books and the EPA report. Previously inspected files and every related
origin are development-only. The remainder of canonical validate supplies router
development; canonical test supplies reserved test, with any group touching
validate moved entirely into development. No test score may choose features,
hyperparameters, thresholds, harm budgets, or stopping criteria.

The frozen manifest contains 487 new development references (418 origins),
76 previously inspected/development-related references (48 origins), and
314 reserved test references (289 origins). Another 1,228 references are
excluded because their origin touches model training. This is stricter than
page-level separation; it excludes all available pages in some document classes.
Ref-kind strata distinguish native PNG, decoded HEIC and downscaled JPEG:
"pristine" is a construction policy, not proof that the source was lossless.

Use grouped inner folds on development origins, preserving all encodes and sizes
of one origin together. Weight origins equally, and report both origin-balanced
and cell-weighted results so repeated pages/encoder variants do not dominate.
The final test is opened only after candidate parameters, thresholds and input
hashes are frozen. Once inspected, it is an audit set, not fresh confirmation.

## Coverage and limits

The primary grid must include low quality as densely as high quality: q0–100 at
step 5; dense fitted studies additionally use step 2 above 70. Cover 420/444 and
all four encoder families, with encoder mode variants separated. Tiny through
large image sizes are separate axes; the size study uses 16–20 log-spaced sizes
without upscaling, recording native-size exclusions. Choose representative
origins by clustering inference-available features, retaining singleton clusters,
not by selecting images on observed restoration gain. Report all tested knobs,
unsupported combinations and missing cells explicitly.

The available corpus does not establish 50 independent origins in every requested
content class. Do not claim comprehensive publication validation from aggregate
numbers or a 512-crop-only result. Keep exploratory, primary, size-generalization,
and unmeasured cases separate. A pilot measures resource use and artifact output
before scaling; it does not choose quality thresholds from test outcomes.

## Harm, uncertainty and cost

For plain-decoded reference SSIM2 `s`, count harm when restoration loss exceeds
`0.5 - 0.04 * clamp(s - 90, 0, 10)`. This is 0.5 at/below 90, 0.3 at 95 and
0.1 at 100. The above-90 taper is an explicit proposal, not a fitted perceptual
law. Report sensitivity to flat 0.5 and historical flat 0.1 as well.

Report calibration budgets 0, 0.5%, 1%, 2%, 5%; held-out mean gain, coverage,
harm/all cases, harm/restored cases, harmed origins, lower gain quantiles, worst
loss, mean loss and excess loss beyond the curve. Use paired origin-cluster
bootstrap intervals, not binomial intervals treating encodes as independent.
Report encoder, subsampling, size, quality, reference-kind and content strata.
A zero observed count is not a never-worse guarantee. Repeated grouped development
splits test stability; they do not replace the reserved test.

Measure actual end-to-end cost separately from feature extraction, on this host,
with the resource wrapper. Charge model inference to post-inference policies.
Record CPU/binary identity and peak RSS; do not extrapolate costs between sizes.

## Reproducibility and release gate

Persist reference hashes, encoded JPEG bytes, decoded/restored artifacts needed
for metric reproduction, scores and available metric diagnostics before scaling.
Each cell has a stable key and artifact hash. Fail on missing features/cells,
non-finite data, changed input hashes, and split overlap. Preserve full logs and
resumable output. No missing-file early-return tests and no silent feature fallback
in the study. Keep artifacts outside git, with pointer manifests and hashes.

The publication package must include executable analysis, frozen input/model/
feature identities, model-selection records, held-out results, uncertainty,
cost evidence, exclusions and regression tests. A crate release additionally
requires the existing public API/dependency/README approval and release checks;
this study does not publish or merge zenjpeg work.
