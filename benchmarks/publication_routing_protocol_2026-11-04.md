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

## Pre-fit near-duplicate audit

The existing 16×16 luma fingerprint audit (`picker_leakage_audit.thumb`, mean
absolute distance <3 on u8 values) found 61 cross-origin candidate pairs. Treating
these conservatively as related groups leaves **484 development references / 404
origins** and **291 reserved references / 270 origins**. This is a suspected-near-
duplicate quarantine, not proof of semantic identity; white-background documents
can collide. No restoration scores were used. The original manifest is retained;
`references-audited.tsv` owns the updated membership. It also fingerprints decoded
HEIC references, avoiding the old audit's inability to open HEIC originals.
The threshold is inherited from the existing leakage audit, not tuned on outcomes.

## Frozen analysis grid

The primary quality list is the union of q0–100 step 2 and q0–100 step 5 (61
values), so aggressive compression is sampled at least as densely as the upper
range. cjpegli q0 is explicitly excluded: the boundary pilot demonstrated that
this binary rejects it. A coverage ledger distinguishes that exclusion from a
missing result. Primary geometry is full-image aspect-preserving resize with a
512-pixel longest edge, Mitchell filtering, and no upscaling; this differs from
the earlier center-crop study and its mean scores must not be compared directly.
Size-generalization evidence remains a separate required section.

Candidate families are continuous coefficients, chooser probability, decoded
edge slope, the combined decoded feature set, and combined post-inference
signals. Each compares ridge/logistic regularization 1 and 10 against histogram
boosting of depth 2 and 3 (100 iterations, learning rate 0.08, L2 10, minimum
leaf 100 cells). Early stopping is disabled to prevent hidden row-wise splits.
Five shuffled origin-grouped folds use seed 20261104. Threshold selection uses
out-of-fold predictions, gain cutoffs {0,0.1,0.25,0.5,0.75,1,1.5,2,3,4,6,10}
and predicted-risk penalties {0,1,2,4,8,16,32,64}, with an explicit skip-all
candidate. This is selection evidence, not an unbiased test estimate. Selected
model configurations also undergo 20 grouped train/calibration/audit splits as
a development stability check. Only after these records are frozen may the
reserved test run begin.

Budgets constrain origin-balanced empirical harm; report cell-weighted rates
alongside them. Both are grid-dependent observations, not estimates of an
unspecified production traffic mixture. Report content-macro averages and
reference-SSIM2 bands as well as per-content rows. Paired bootstrap comparisons
against both current intents resample origins, with 2,000 replicates and seed
20261104. No policy is chosen for deployment by this study.

Pilot evidence: 380/380 supported cells across five sizes, four encoders and
five boundary qualities; 1,846 artifacts hashed, decompressed and shape-checked,
with JPEGs independently read by Pillow/libjpeg. Runtime 76 s, run-heavy peak
RSS 0.53 GiB, minimum available 22,149 MiB, peak load 3.15. Artifact/oracle audit
6 s, peak RSS 0.04 GiB, minimum available 23,024 MiB, peak load 3.72. A separate
four-cell replay produced byte-identical result rows after the shared-helper
refactor and array-chunk maintenance.
