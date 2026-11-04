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
Mode evaluation also requires the feature-only representative selection record:
its primary-cell digest, manifests, selector source, fixed parameters and exact
reference membership must match. Missing provenance or altered membership fails
before policy scoring. The reserved size and mode reports retain the selection
record digest. The synthetic selection/evaluation roundtrip covers both failures.
The fit contract also freezes the representative-selector source hash; evaluation
rejects a selector change after fitting, even if newly generated selection records
refer to the changed source. This rejection is covered by the roundtrip test.

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
512 configuration is not presumed safe for the dense-size grid. The subsequent
two- and four-worker pilots below supply the separate large-image concurrency
evidence; the dense grid starts with four workers under the existing cap.

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
Later client-side and storage-local pilot checks are recorded below. No isolated
parallel-speedup claim is made from these cache-uncontrolled runs.

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
This is a resource workload, not two independent statistical samples. Its completed resource and audit results are recorded below.

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
its elapsed time is not a runtime benchmark. The full artifact audit subsequently passed (record below).
Generation completion alone does not admit the cells to model fitting.

Native 4096-edge eligibility in the frozen audited manifest is concentrated in
nature photographs (4 development references, 6 reserved references) and brochures
(11 of 19 development references across 5 of 11 origins; 7 of 12 reserved
references across 4 of 7 origins). The eligible nature references represent
3 development origins and 6 reserved origins. The earlier statement that every
brochure reference reached 4096 confused reference counts with origin counts.
Other content classes have no native 4096-edge references. This is
an input-coverage fact, not a measured large-image quality result; do not generalize
a 4K result from those classes to all content.

A separate four-reference resource manifest uses an explicit router-development
role, geometry-only selection, and no outcomes. All four selected references are
nature images with 4:3 or 3:4 aspect ratios, reaching 4096×3072 or 3072×4096 without
upscaling. Manifest SHA-256:
`12558dadc7ab2978487e601eae8f98c049efaf06a05e5a433491cd5c21173572`.
The selector rejects the reserved role. These resource cells are not additional
training observations; the audited representative-grid contract still controls
fitting membership. The completed resource and audit results are recorded below.

Storage-local verification was tested after finding that the client route used
1 Gbit/s Ethernet and local disk had only 3.8 GiB free. A 15-second whole-interface
sample received 1,349,277,362 bytes; that is not isolated audit throughput.
The offline audit image retains Pillow 13.0.0 and the same auditor source.
A four-worker/two-CPU storage-local pilot took 9.15 seconds; an eight-worker/four-CPU
pilot took 4.86 seconds. Both verified the same 380 cells and 1,846 artifacts.
The latter's GNU-time maximum RSS was 86,744 KiB. Its client-side wrapper reported
`rc=0 6s | peak-RSS 0.01GiB | min-avail 22208MiB | peak-load 11.34`;
that wrapper measures the SSH client, not the container's memory.
The client-side four-worker pilot reported
`rc=0 8s | peak-RSS 0.06GiB | min-avail 22105MiB | peak-load 10.05`.
Cache state and CPU allocations differed, so these do not establish a full-grid
speedup. The partially completed full audit was retained and resumed after each
short serialized test.

## Primary artifact audit completion

The full audit passed all 235,224 required cells and 1,167,767 unique artifacts.
It verified compressed-byte hashes, decompression, buffer sizes, independent JPEG
dimensions/sampling, exact coverage and unchanged source/coverage identities.
Cell digest: `915744b1966b885ab88c8018d1d480de16f9b6911dd0c5d9804564fbfea389ef`.
Resource wrapper:
`rc=0 4466s | peak-RSS 1.08GiB | min-avail 21586MiB | peak-load 13.08`.
Elapsed time includes serialized locality experiments; it is not an isolated
throughput benchmark. The dense development input remains required before fitting.

## Large-image concurrency validation

The two-worker pilot completed four 4096×2896 cells and its 22 artifacts passed
storage-local verification. Generation wrapper:
`rc=0 48s | peak-RSS 4.75GiB | min-avail 18175MiB | peak-load 8.66`.
The four-worker pilot completed eight nature-image cells at 4096×3072 or
3072×4096; all 44 artifacts passed verification. Generation wrapper:
`rc=0 58s | peak-RSS 8.93GiB | min-avail 14340MiB | peak-load 5.04`.
Both used the frozen study executable and a 16 GiB cap. The dense development
run will start with four workers under that cap, based on this measured workload.
These measurements include encoding, restoration, metrics and artifact writes;
they are not isolated restoration latency or square-4096 memory measurements.

## Compressed primary analysis tables

An explicit immutable snapshot retains all 235,224 primary cell tables and the
six audit/contract metadata files. The archive is 172,706,671 bytes, SHA-256
`0ff5a241849430ecad22b9bc911c94d15ed8d10d7844482b80d194d1b4fd4bfb`.
Its canonical cell digest remains
`915744b1966b885ab88c8018d1d480de16f9b6911dd0c5d9804564fbfea389ef`.
The canonical snapshot is `table-snapshots/primary-v1` under the study root;
bulk artifact bytes remain in the original run. Packaging ran beside storage,
taking 4:57.70 with GNU time maximum RSS 348,784 KiB. The surrounding SSH client's
wrapper recorded `rc=0 301s | peak-RSS 0.01GiB | min-avail 22438MiB | peak-load 3.34`;
that RSS describes only the client.

Replaying the original representative selection from a local compressed copy
produced byte-identical `references.tsv` and `selection.json`. The local replay
recorded `rc=0 11s | peak-RSS 3.11GiB | min-avail 19520MiB | peak-load 1.58`,
versus the earlier individual-file selection's
`rc=0 633s | peak-RSS 3.17GiB | min-avail 19578MiB | peak-load 5.04`.
These are observed analysis runs, not controlled storage benchmarks. The original
selection has 31 references/origins, 8 references eligible for 2048 and only one
for 4096. This exposes a native-resolution coverage limitation before any model
fit; it does not support broad 4K conclusions. The 17 routing integrity tests pass,
including snapshot/live-table equality and rejection of changed archive/metadata.

Before fitting, native-size selection now preserves the original per-content
feature representatives and supplements them with two feature clusters per
content among references whose native edge reaches 2048. It also includes one
canonical eligible reference for every origin reaching 4096. Each stratum chooses
an eligible file before deduplicating origins, so a smaller canonical file cannot
hide an origin's larger reference. The selected union is deduplicated by reference
hash; reports retain both reference and origin counts. Selection uses geometry
and decoded features only, never restoration outcomes. All 18 integrity tests
pass, including preservation of the original sample, all eligible large origins,
and invariance to changed restoration scores. Fitting requires this selector
identity and native-resolution rule.

The supplemented development selection contains 45 references from 44 origins,
with 22 references/origins reaching 2048 and all 8 eligible independent origins
reaching 4096. Its manifest SHA-256 is
`63d9097bde99c62b6a7c8ee21070a41361ae3ccf0a1021c156dfe45fa85a1094`.
The original 31-reference selection remains in `development-size-feature-only-v1`;
the supplemented selection occupies `development-size`. Selection recorded
`rc=0 11s | peak-RSS 3.11GiB | min-avail 19566MiB | peak-load 2.73`.
This is selection evidence only: the dense grid has not yet been generated or
used to fit a router, and the reserved cohort has not been evaluated.

Reserved-size evaluation now enforces the same fixed 20-size inventory and
representative-selection protocol as development. It binds the selector source,
native-size rule, feature list, seed, cluster count, primary audited-cell digest,
source manifest, selected manifest and exact reference membership. The report
retains the selection-record digest. A synthetic replay passes; changing the
size inventory or source-selection digest is rejected. This closes an evaluation
admission gap before any real router fit or reserved-set opening.

A six-worker resource pilot used six development references selected by native
geometry only, at edge 4096, turbo q50, both subsampling modes (12 cells).
The frozen executable completed successfully under the 16 GiB cap:
`rc=0 65s | peak-RSS 11.65GiB | min-avail 10601MiB | peak-load 9.92`.
The dense sweep was frozen during this serialized pilot. This is full generation
pipeline resource evidence, not an isolated restoration-latency benchmark. The
four-worker dense segment was then stopped recoverably to resume with six
workers and a shorter executable-search path; scientific inputs are unchanged.
The six-worker pilot's complete audit verified 12 cells and 66 artifacts.
A broad pre-resume diagnostic of the interrupted dense segment was attempted,
then stopped while still reading old cell files on the busy array. It did not
complete and supplies no integrity claim. The harness resumed using its existing
identity, committed-cell schema and byte-equality checks; the full final artifact
audit is still required before fitting. Both diagnostic attempt logs are retained.

## Storage transport validation during dense generation

A bindfs view of the direct-array NFS directory passed all three portable
storage contract tests (conflict preservation, paused publication, concurrent
identical publication). The test wrapper recorded
`rc=0 3s | peak-RSS 0.13GiB | min-avail 22192MiB | peak-load 12.03`.
A six-reference, 512-edge turbo-q50 replay through that view produced 12 cell
tables byte-identical to the already-audited primary run. Its 66 artifacts passed
the full hash/decompression/JPEG audit. The same process and frozen executable
then continued the dense grid through the new view of the same files.

The transport replay recorded
`rc=0 29s | peak-RSS 0.38GiB | min-avail 21789MiB | peak-load 10.40`.
A matching small replay on NVMe-backed storage recorded
`rc=0 1s | peak-RSS 0.45GiB | min-avail 22065MiB | peak-load 13.05`;
all 12 cell tables were again identical. These are small operational observations
with uncontrolled cache state, not a full-grid or restoration-latency speedup.
The 34 MiB array replay and both logs remain with the study artifacts. The owner subsequently approved bounded NVMe staging, with generation stopped
during verified transfers to the array. The native-transfer pilot below validates
that path before the dense sweep switches to it.

Ten-second syscall traces found mean `execve` duration of 95,492 microseconds
on direct NFS and 181 microseconds through bindfs. Explicit `fsync` then dominated
the trace; no end-to-end speedup is inferred from these syscall samples.
Linux's [NFS close path](https://raw.githubusercontent.com/torvalds/linux/master/fs/nfs/nfs4file.c)
can flush pending writes. [bindfs forwards explicit fsync](https://raw.githubusercontent.com/mpartel/bindfs/master/src/bindfs.c)
and does not register a flush callback. The transport process has its own 2 GiB
memory cap and idle I/O priority. Exact mount/export cleanup commands are retained
in the private execution record; no scientific input or artifact path changed.


## Local staging verification

The same six-reference, 512-pixel turbo q50 replay produced 12 cell tables
byte-identical to the earlier array replay. Local generation completed with
`rc=0 1s | peak-RSS 0.45GiB | min-avail 21948MiB | peak-load 9.89`.
The staged transfer retained 66 artifacts (29,467,539 bytes) on the array;
the full artifact audit passed there. These small operational timings do not
establish end-to-end study throughput.

`publication_routing_stage.py` transfers content-addressed files only while the
writer scope is frozen, retaining open files in staging. It validates source
and destination hashes, syncs the destination and directory, then releases the
source copy. Corrupt bytes and conflicting destinations are preserved and fail
closed. The free-space guard requests a drain after 8 GiB of additional disk
use, with a 10 GiB minimum-free threshold; this is polling, not a hard quota.
Any transfer failure leaves the writer frozen. Final draining and the full
study artifact audit remain required before fitting. The frozen generator,
scientific inputs and committed cell records are unchanged.

The 21 analysis integrity tests pass, including synchronization failure,
conflicting destination, corrupt staged data and failure-without-resume cases.


The staged transfer now uses four bounded workers while generation is frozen.
A separate 12-cell / 66-artifact replay passed transfer and full audit with
unchanged bytes. The transfer wrapper reported
`rc=0 2s | peak-RSS 0.03GiB | min-avail 21934MiB | peak-load 2.88`.
This small replay is a correctness check, not evidence of a throughput gain.
Destination parent directories are synchronized before source release as well.
All 22 integrity tests pass, including concurrent identical publication.


The first live staging cycle moved 10,444 artifacts / 8,699,738,814 bytes,
retained zero open files, and automatically resumed the writer. Its log is
`publication-local-stage-watch-v3.log` in the execution record. The four-worker
successor is `publication-local-stage-watch-v4.log`. This verifies one complete
live storage cycle; it does not complete the dense study or its final audit.

Do not scan the live cell directory through NFS for progress. During the dense
run, such a scan left all six writer threads waiting in `start_renaming`;
terminating the read-only scan restored worker activity. Count directory entries
beside storage instead, and check the writer PID/scope separately. Use the full
`publication-routing-cells status` command after generation has stopped.


### Archive durability correction

Staged transfer requires storage that honors synchronization requests. The
study's temporary NFS export initially used `async`; client `fsync` calls alone
therefore did not establish the intended disk durability. Generation and transfer
were frozen, that study-only export was changed to `sync` and verified, and a
storage-side filesystem `syncfs` completed with return 0 / errno 0 before both
workers resumed. Other exports were unchanged. No artifact-loss evidence was
observed; the earlier byte-verification checks remain distinct from durability.
The small transfer timings above precede this correction and must not be used
as synchronous-archive performance evidence. All retained artifacts still
require the final study audit before fitting.


Live staged-copy release now atomically replaces the local pathname with a link
to its verified archive copy. A frozen thread can be between `create_new` returning
AlreadyExists and reading the path without holding an open descriptor; unlinking
would make that resumed lookup fail. This race was identified in source review
and reproduced by the filesystem regression, not observed as a generation failure.
Open descriptors and pending path reads both retain identical bytes. Subsequent
drains validate and skip the retained archive links.

All 23 integrity tests pass. A synchronous-archive pilot transferred 66 artifacts
(29,467,539 bytes), its repeated drain transferred zero, all 66 links remained
readable with matching hashes, and the full 12-cell audit passed. The transfer
wrapper reported `rc=0 21s | peak-RSS 0.03GiB | min-avail 22192MiB | peak-load 5.16`.


The approved NVMe staging path uses storage-local usage checks and transfers,
coordinated by the compute host that owns the writer scope. The coordinator
freezes the writer, records open artifact paths and a unique request nonce,
and accepts only matching transfer acknowledgements and tool hashes before
resuming. Missing observations or failed transfers leave the writer frozen.
Native transfers preserve client-visible archive links and verify/sync bytes
beside storage. The operational triggers are 72 GiB staged or 520 GiB free,
providing margin inside the approved 80 GiB / 500 GiB limits; these polling
triggers are not filesystem quotas. Private execution configuration carries
host paths and command arrays. All 25 integrity tests pass, including excluded
open files, repeated native drains and monitor-failure freezing. The real native-transfer pilot described below passed before switching the sweep.


The NVMe/native-transfer pilot generated the same 12 cell tables as the prior
frozen replay, transferred 66 artifacts / 29,467,539 bytes, and checked every
client-visible archive link. It then preserved the first cell directory and
regenerated all 12 cells through those links: all cell bytes matched again, a
second drain transferred zero artifacts, and the full archive audit passed.
The entire pilot wrapper reported
`rc=0 56s | peak-RSS 0.45GiB | min-avail 21310MiB | peak-load 7.93`.
This combined generation/transfer/replay/audit measurement is not a restoration
runtime benchmark or a controlled storage-speed comparison.


The live dense stage migrated 5,932 artifacts (8,429,265,763 payload bytes)
while generation was frozen. Both source and destination were content-hash
verified; the original directory remains retained. All 256 hash-prefix
directories exist before resumption to preserve pending pathname operations.
The migration wrapper recorded `rc=0 433s | peak-RSS 0.03GiB | min-avail 21569MiB | peak-load 6.27`.
The native usage guard verified its tool identities and available space before
the same six-worker generator resumed. All 27 integrity tests pass, including
migration preservation and rejection of open artifacts before copying.

Final native draining has a distinct ended-writer request: the recorded PID
must be absent, its scope must contain no processes, and the successful-generation
marker must exist. It does not pretend a departed writer is still frozen.
The storage backend requires no excluded open paths for that request. The live
watch keeps its immutable deployed helper identities; an updated helper snapshot
is required before using the final-drain mode. All 29 integrity tests pass,
including refusing a live PID and a missing completion marker and transferring
an artifact that was excluded while the writer was active.

The first native dense transfer was stopped recoverably after observing three
transfer workers waiting on filesystem journal commits. Its partial copies,
verified archive links and logs remain preserved; its interrupted wall time is
not a completed-transfer performance result. The native backend now supports
an explicitly enabled batched path: copy and hash a bounded group of artifacts, synchronize
the archive filesystem, then replace source paths with archive aliases and
synchronize staging. No source path is released before the archive barrier.
The previous per-file path remains available. This changes storage orchestration,
not the frozen scientific executable or numerical outputs.

A separate 66-artifact pilot passed client-link hash checks, a zero-transfer
repeated drain and the full 12-cell artifact audit. Its complete wrapper recorded
`rc=0 122s | peak-RSS 0.02GiB | min-avail 21724MiB | peak-load 4.60`.
That includes preparation and audit and is not a controlled speed comparison.
All 31 integrity tests pass, including barrier failure retaining every source
and preservation of conflicting archive bytes. The Linux storage backend uses
`syncfs` for the batched durability barrier; missing support or errors fail closed.

The initial 256-file live batch completed with verified archive links. Batch
size is now configurable from 1 through 4,096 (default 256), independently of
the four-copy-worker limit. The active run uses 4,096 to reduce repeated
filesystem-wide barriers while retaining all source files until each archive
barrier succeeds. All 32 tests pass, including batch-size bounds and grouping.
The 66-artifact pilot's storage process recorded 29,372 KiB maximum RSS via
GNU time; the wrapper RSS above measures the client process tree only.

Reports retain both origin-balanced and cell-weighted restore coverage,
harm among restored cases, mean loss and excess loss. Origin-balanced conditional
harm is the ratio of origin-balanced harmed-case mass to origin-balanced restored
mass; it is not the unweighted mean of per-origin conditional rates. Empty
restoration sets report null conditional harm. Lower gain quantiles remain
cell-weighted. An unequal-origin fixture verifies that these denominators differ;
all 33 integrity tests pass. No selection objective or runtime policy changed.
