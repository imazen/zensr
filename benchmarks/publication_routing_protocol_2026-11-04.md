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
The action space is unchanged decode or this fixed restoration. The reference-dependent
oracle chooses the better of those two outputs per cell; its upper bound applies
to that action space.
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

The audited frozen manifest contains 484 new development references (404 origins),
79 previously inspected/development-related references (48 origins), and
291 reserved test references (270 origins). Another 1,251 references are
excluded because their origin touches model training. This is stricter than
page-level separation; it excludes all available pages in some document classes.
Ref-kind strata distinguish native PNG, decoded HEIC and downscaled JPEG:
"pristine" is a construction policy, not proof that the source was lossless.

Use grouped inner folds on development origins, preserving all encodes and sizes
of one origin together. Weight origins equally, and report both origin-balanced
and cell-weighted results so repeated pages/encoder variants do not dominate.
The final test is opened only after candidate parameters, thresholds and input
hashes are frozen. Once inspected, it is an audit set, not fresh confirmation.

Training-provenance limit: the adopted model metadata identifies trainer commit
`4a33019`, the dataset directory, seed and source-class policy. The generator and
split helpers at that commit match their current source and restrict the default
pool to canonical training membership. However, historical `pairs.tsv` records
encoder, quality and conditioning, not each pair's source-image ID, and the
generation environment/corpus snapshot is not fully recorded. Separation here
is verified against that documented source policy and the frozen current
manifest; it is not a complete per-pair reconstruction of model-training inputs.
Do not turn that provenance gap into an unconditional leakage-free claim.

## Coverage and limits

The primary grid includes q0–100 at both step 2 and step 5, including the
low-quality range as densely as the high-quality range. Cover 420/444 and
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

## Baseline and legacy-feature controls

Before fitting, the threshold-only baseline retains the existing fixed minimum-
gain and separate quality/distance-margin grids in `tools/routing_frontier.py`.
Its gate constants are read from the frozen runtime calibration source and stored
in the selection record. All learned policies also report flat 0.1 and flat 0.5
harm sensitivity without reselecting on the reserved set.

The historical chooser fit predates the current corpus split and did not record
feature versions. Its probability is therefore a legacy comparison with unresolved
training-provenance limits. The main decoded and post-inference families exclude
that probability. A separate `decoded_legacy_chooser` family measures its incremental
contribution; it must not inherit an unqualified independent-test claim. The
coefficient-only and edge-only families likewise do not depend on that chooser.
This separation was fixed before any new router fitting or reserved evaluation.

## Size-dense development and reserved selection fixed before fitting

Use the 20 sizes listed in `tools/publication_routing_sizes.py` (32 through
4096, including 512), without upscaling. Choose two k-means clusters per content
stratum, capped by available origins, using standardized inference features from
turbo 420 q50 at the primary 512-pixel geometry. Choose one canonical file per
origin before clustering and the centroid-nearest member from every cluster;
singleton clusters remain represented. Seed is 20261104, with 10 initializations.
No restoration outcome enters this selection. Record cluster memberships,
representative hashes and native-size coverage. This purposeful subset is not a
random sample of production images.

Apply this same outcome-blind selection separately to development and reserved
origins. Before fitting, run the full 20-size, 61-quality, four-encoder 420/444
grid on the development representatives. Join it with the complete primary
512-pixel cohort. Count an identical repeated cell only once and reject any
conflicting replay. Dense references must belong to the primary development
cohort, with unchanged origin/content identity, features, weights and encoder
binaries. Preserve both input audits, selector identity and source digests.

All variants of an origin stay together in every split. Each origin retains equal
total weight across its available primary and dense-size cells. Numeric scaling
and both fitted predictors use the same origin-balanced weights (normalized to
mean one for estimator fitting). Thus representative
origins distribute their weight across sizes while other origins contribute at
512 pixels; the resulting objective is explicitly this designed mixture. Report
primary and size-reserved outcomes separately rather than presenting that mixture
as production traffic. Native-size and aspect-ratio exclusions remain visible.

The reserved size audit reuses the frozen fitted policies. Its separate evaluation
mode permits only changed sizes and a representative manifest subset; all references
must belong to the primary reserved cohort. Executables, encoders, quality grid,
features and weights must still match. The size audit tests new origins across the
training size inventory, with any geometry outside observed training ranges labeled
by the abstention control. It is not comprehensive coverage of every content class.

This corrects the initial plan to fit only at 512 and use other sizes solely for
extrapolation testing. The correction was made before any router fit or reserved
outcome inspection; the primary sweep remains valid and is retained unchanged.

## Encoder-mode generalization

On the same feature-selected reserved representatives, test the 14 existing
`zjtool` configurations at size 512: default, baseline, progressive, auto,
hybrid, hybridmax, mozbase, mozprog, trellis, trellisdc, noaq, nodering,
noaq_nodering and blur. Preserve the full 61-quality grid and both chroma modes.
The list is fixed in `MODE_ENCODERS`; first check boundary support on the old
pilot references. This is a separate generalization audit, not additional fitting.
Encoder executable/shared-library hashes must be a verified subset of those in
the primary run. A different encoder binary is not admitted as a knob change.

This covers the knobs exposed by the frozen harness, not every JPEG mode.
XYB, 4:2:2/4:4:0, grayscale, other decoder upsampling/strictness settings,
multigeneration inputs and separate Cargo-feature builds remain outside this
quality study. Existing correctness tests for those geometries do not establish
their restoration quality. Publication claims must retain these domain limits.

## Training-range abstention control

Before any new router fitting, add guarded and unguarded operating points for
each learned model. The guard rejects a row if any numeric feature lies outside
that fit's observed minimum/maximum, or any categorical value was unseen. Bounds
are learned only from the training origins of each cross-fit fold (and from the
full development cohort for the final model). Threshold and risk-penalty selection
compares both variants under the same budgets. This is an axis-aligned range
check, not a validated distribution-shift detector or a harm guarantee. Geometries and feature combinations outside the observed training ranges may
therefore abstain even when the nominal size belongs to the planned inventory.

Retain every development candidate and all out-of-fold predictions. The reserved
report evaluates the operating points selected at the five fixed budgets plus
fixed baselines. These are sampled gain/risk operating points, not an exhaustive
continuous frontier and not a deployment selection.

## Encoder-mode boundary support measured before fitting

The frozen executable completed all 280 pilot cells: two previously inspected
references, size 32, q0/5/50/95/100, all 14 configurations above, and 420/444.
The current artifact auditor verified 756 unique artifacts, including independent
JPEG reads. No mode/quality exclusion was needed in this pilot. This verifies
boundary support, not quality over the reserved cohort or support for every
possible image. Sweep: run-heavy rc=0, 4 s, peak RSS 0.04 GiB, minimum available
22,173 MiB, peak load 15.96. Audit: rc=0, 1 s, peak RSS 0.02 GiB, minimum
available 22,299 MiB, peak load 15.10. Artifacts: `mode-pilot-v1` under the
[study root](publication_routing_2026-11-04.pointer.md).

## Severity inspection

For every fixed operating point, retain the ten worst affected origins (one
worst cell per origin), ordered by realized SSIM2 loss. Each diagnostic includes
its quality scores, curve-harm flag, content/encoder/geometry, and retained JPEG,
reference, decoded, restored and difference-map paths. Empty lists mean no
negative realized gains in that evaluated grid. These are post-freeze diagnostic
cases, never reserved-set examples for retuning the policies. Synthetic tests
check ordering, origin deduplication, and the empty reference-oracle case.

Each operating point also exports its frozen threshold/risk penalty, domain
abstention flag, feature list, model configuration and serialized router artifact
size. Joblib bytes include Python model/transform serialization; they are not
Rust binary sizes or resident-memory measurements. The restoration model weights
are separate. Skip-all points do not require loading a fitted router.

## Secondary PSNR exact-match convention

The frozen harness's historical PSNR helper stores 99 for exact matches. Analysis
therefore derives PSNR changes from retained RGB squared-error sums, rather than
subtracting those sentinel scores. Equal exact reconstructions have zero change;
exact/nonexact transitions are counted as positive or negative infinite changes.
If any occur, the full mean is left undefined. A separately labeled conditional
finite mean renormalizes the original origin weights over finite-change cases,
with its count and retained weight mass reported. SSIM2 selection labels and the
primary harm curve do not depend on this secondary-metric correction.

## Independent subsampling audit

The JPEG oracle now checks actual chroma sampling against every row's 420/444
label, including deduplicated references, using Pillow's parsed component sampling
factors. The strengthened audit passes on both completed pilots: 280 mode cells /
756 artifacts and 380 primary-encoder cells / 1,846 artifacts. Prior audit records
are retained in each run's `audit-history`; the full development audit will use
this strengthened tool before fitting. The sampling-mismatch regression fails
explicitly instead of accepting a mislabeled cell.

## Large-image resource pilot

A previously inspected document, selected by native geometry without outcomes,
completed the frozen pipeline at 4096×2896, turbo q50, 420 and 444: two cells,
11 verified artifacts. Run-heavy: rc=0, 44 s, peak RSS 2.45 GiB, minimum available
19,982 MiB, peak load 16.23, one worker, 16 GiB cap. Independent artifact audit:
rc=0, 3 s, peak RSS 0.07 GiB, minimum available 22,028 MiB, peak load 13.62.
This is full pipeline resource evidence including encoding, restoration, metrics
and retained artifacts; it is not a runtime-latency benchmark or a measurement
at square 4096×4096. Artifacts are in `resource-4096-v1`; the manifest and selector
identities are in `resource-4096-references.json` under the study root. The 12-worker
512 configuration is not presumed safe for the dense-size grid. Begin that grid
with one worker under the existing cap; any concurrency increase needs its own
measurement at the sizes actually processed.

## Bounded artifact-audit execution

The audit accepts explicit `--workers` (default one), with at most 128 cells'
artifact checks queued per batch. Every row still verifies dimensions and chroma
sampling, even when another row shares verified bytes. The coverage ledger is
hashed into the audit record; fitting rejects a changed ledger as well as changed
cells or auditor source. Regressions cover parallel-worker failures, cached shape
and sampling mismatches, and a changed ledger with unchanged parsed coverage.

A syscall profile of the 380-cell pilot recorded 1,466 waits for decompression
processes; wait4 and futex accounted for 38.95% and 36.43% of traced syscall time.
The implementation resolves the decompressor path once and permits bounded
parallel verification. One-worker and eight-worker runs both verified all 380
cells / 1,846 artifacts with identical cell digest
`eae49fae3a5bb0d6be0a81fde2ec451ee811935623ae0ef74866e7c5546e7b41`.
Run-heavy recorded 20 s / peak RSS 0.04 GiB / minimum available 22,056 MiB /
peak load 16.37 for one worker, and integer-rounded 0 s / peak RSS 0.08 GiB /
minimum available 22,261 MiB / peak load 12.25 for eight. Filesystem caching was
uncontrolled: these observations are not a confirmed speedup. A reverse-order
trial did not start because the primary scope failed to freeze; the scope was
thawed and primary progress verified. Exact profiled sources, commands, syscall
summary and logs are retained in `audit-profile-v1` under the study root.
Complete the remaining audit timing checks after the primary run releases compute.

The auditor atomically updates `AUDIT_PROGRESS.json` after each bounded batch,
recording completed/required cells, verified artifacts, worker count, PID, source
and coverage hashes, and terminal success or failure. `AUDITED.json` is also
atomically published only after all checks pass. Source or coverage changes during
the audit fail the run. Progress records support supervision; the completed audit
record remains the analysis admission gate.

Numeric scaling translates each feature by its first training value before
weighted moment accumulation, then restores the fitted mean's offset. This avoids
roundoff-generated negative variances for constant columns without changing origin
weights or using validation data. A weighted analytic regression and the complete
routing suite pass with numerical runtime warnings treated as errors.

The geometry-only resource selector accepts an explicit reference count for a
concurrency pilot. The two-reference manifest contains the two previously inspected
7016×4961 brochure pages (one origin); its SHA-256 is
`0455e458131961689fde844ad472737be9bfee04469c4769115da7acc7345d5c`.
This is a resource workload, not two independent statistical samples. Its run
remains pending while the primary sweep owns compute.

## Zero-event uncertainty diagnostic

Alongside origin-cluster bootstrap intervals for mean gain and average cell harm,
report the fraction of origins with any harmed tested variant and its two-sided
pointwise 95% Clopper–Pearson interval, using
[SciPy's exact binomial interval](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).
This interval stays nondegenerate when no harmed origins are observed. It is a
separate estimand: an origin counts once regardless of the number of harmful
variants. Its independent, identically sampled origin model does not correct
purposeful corpus selection or establish deployment-population coverage. It does
not replace the primary harm interval or enter policy selection.

## Primary generation completion

The frozen executable completed all 235,224 required primary-development cells
on 2026-11-04. The resumed segment's resource wrapper recorded:
`run-heavy: done rc=0 11281s | peak-RSS 1.06GiB | min-avail 19796MiB | peak-load 19.92`.
This segment includes deliberate suspensions for serialized verification work;
its elapsed time is not a runtime benchmark. The full artifact audit is running.
Generation completion alone does not admit the cells to model fitting.
