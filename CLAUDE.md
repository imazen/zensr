# zensr project notes

## Known Bugs

- Corrected SR retention scope: the first external ×4 comparisons incorrectly
  interpreted the map-cleanup request as permission to discard upscaled images.
  New scoring retains exact scored RGB8 images as lossless PNG with pixel/file
  hashes; IQA maps stay off. All three earlier external candidates were
  regenerated, matching every original pixel hash and scalar score. Independent
  PNG decoding verifies 4,179 unique retained images including DSCF and controls.
  Historical v1 identities remain unchanged. See `docs/SCORING_RETENTION.md`.

- Fixed binary16 subnormal weight expansion: the normalization shift was one
  bit too large, producing incorrect f32 values for nonzero subnormals. An
  independent mathematical regression fails at `0x0001` before the fix and
  covers all 65,536 binary16 patterns. Existing artifacts contain 889 such
  weights in dejpeg7 and 45 in canon4. External-model comparison admission
  caught differing native pixels. After `a01962d6`, all expanded quality-model
  weights match NumPy, and eight encoder/sampling controls reproduce the retained
  raw/projected pixel hashes and scores. Historical measurements retain their
  original numerical identities.

- Threaded precision timing correctly distinguishes confirmation registration in
  the parent from actual child execution. The first-process replay mode initially
  rejected the registration callback before any timings. Two context regressions
  cover deferral in the parent and mandatory exact checks in child zero.

- Model archive packaging no longer imports pair-image processing dependencies.
  A minimal-environment admission test exposed an unnecessary OpenCV import;
  the existing archive helpers now live in a standard-library-only module.
  All 73 members of the retained QAT model archive verify with the extracted helper.

- Quality-model precision tooling: ONNX int64 counters can become Tract symbolic
  dimension tensors. Range instrumentation now uses bounded int32 counters and
  explicitly casts scalar results; it preserves model outputs bit for bit.
  Full-size independent-model/counter replay gates shared-prefix quality
  evaluation. Timed models remain separate. See `docs/QUALITY_INT8.md`.

- QAT archive transfer initially exceeded its 4 GiB cgroup through dirty NFS
  page-cache charges despite a bounded Python stream. Periodic fdatasync and
  DONTNEED now bound outstanding output pages; partial archives remain retained.
  This orchestration failure does not change completed training or IQA.

- QAT export rounding is fixed: dividing by a scale can disagree with PyTorch's
  native reciprocal quantization at a half-step. Packing now recovers native
  integer codes, with exact operand comparison. CPU/GPU initial scale rounding
  differs slightly; the study retains both PTQ controls. QAT CI uses an explicit
  pinned CPU torch environment. See `docs/CANON4_QAT.md`.

- QAT model packaging uses a nested-member hash verifier. The existing pair
  archive extractor intentionally accepts flat members only; it is unchanged.

- Fixed by the approved 34-input default: the original router used whole-file
  bpp, so nonsemantic APP15
  padding can change its decision while pixels, coefficients and probe match.
  A bounded 16-input diagnostic (including near-boundary cases) reproduces it.
  The 34-input default removes bpp/dimensions and passes all 64 padding checks;
  the owner approved secondary observed harm 0.5662% versus prior 0.5490%. See `benchmarks/routing_feature_ablation_2026-11-06.md`.

- Fixed tiny-image chroma projection: padded MCU block dimensions could make
  420 chroma appear full-resolution, or misclassify 422/440. Sampling geometry
  now follows SOF sampling factors. The regression failed on 8x8 420 before
  the fix and covers tiny/odd 420, 444, 422 and 440 after it. This corrects
  constraint application; it does not establish perceptual quality at tiny sizes.

- The reserved 512-pixel score comparison found a worst 18.5344 SSIM2 loss
  on a turbo 420 q6 line plot, retained by current routing and the 0.5%-budget
  coefficient/decoded candidates. Lower mean harm frequency does not remove
  this tail risk. The owner cancelled pixel-artifact auditing; that limitation
  is explicit in the results. See `benchmarks/publication_primary_reserved_2026-11-05.md`.

- Reserved publication generation stopped on a reference decode panic with
  139,482 completed cells. The pending set included a 16,320 × 7,612 PNG.
  The owner approved excluding oversized inputs; the other three pending
  references run through the unchanged frozen executable. The explicit
  amendment preserves original coverage and measurements. The owner subsequently
  cancelled the artifact audit; the report discloses that limitation. See `tools/publication_routing_amend.py`.

- Publication storage layout: moving retained artifacts to NVMe left encoder
  scratch and per-cell records on the array. Four encoder children were observed
  waiting in `folio_wait_bit_common`. Reserved generation now uses local disk
  for scratch and cell records, preserving the original array data. The frozen
  12-cell replay matches byte for byte and all 66 artifact hashes verify. Retained
  pixel artifacts continue through bounded staging and verified archival.

- Publication monitor recovery: a live usage observation exceeded its 15-second
  deadline and correctly froze generation; an independent retry succeeded.
  The monitor now permits one retry while frozen. Only a valid identity and
  capacity observation can resume a previously running writer; a failed retry,
  changed tool identity or failed drain leaves it frozen. A later regression
  reproduced an ordinary capacity drain resuming a previously frozen writer;
  that path now also preserves the existing freeze. Limits are unchanged.

- Fixed by `3d1bedfc`: binary float subtraction of decimal TSV scores could count
  an exact 0.100 SSIM2 loss as greater than 0.1. Calibration analysis now subtracts
  decimal values before float conversion, with a boundary regression test.
- Quality-dependent routing risk measurements and limitations live in
  `benchmarks/routing_harm_curve_2026-11-04.md`. No routing policy guarantees
  never-worse perceptual quality.
- Fixed during publication preparation: artifact deduplication in the new audit
  could bypass a later row's dimension check. Every reference now checks shape,
  even when byte verification is cached. Coverage is mandatory, audit-tool
  identity is recorded, and previous audit records are retained on re-audit.
  A cached-JPEG dimension regression and the four-cell real pilot pass.
- Fixed during publication preparation: concurrent writers of identical artifacts
  could see an incomplete canonical file and abort. A deterministic paused-writer
  regression reproduced it. The writer now syncs a staging file and publishes via
  an atomic non-overwriting hard link. Existing conflicting bytes are preserved.
  The ongoing frozen study executable retains the earlier fail-fast writer;
  full artifact auditing remains mandatory before fitting.
- Publication analysis derives PSNR changes from retained per-channel SSE. The
  shared measurement helper uses 99 for exact matches, so subtracting its PSNR
  fields could misreport an exact/nonexact transition. Exact transitions are now
  counted separately; infinite changes are not replaced with finite numbers.
  Regression coverage includes both transition directions and unchanged output.
- Publication study design correction before fitting: a 512-only router fit plus
  held-out size tests did not supply size-dense training evidence. Fitting now
  requires audited 20-size development representatives in addition to the full
  primary cohort, with origin-grouped splits and exact replay deduplication.
  The subsequent owner-approved primary-first amendment permits an explicitly
  scoped 512-only frontier; it does not establish full-resolution behavior.
  See `benchmarks/publication_routing_protocol_2026-11-04.md` for the amendment.
- Publication audit records now bind the complete coverage-ledger bytes as well
  as cell bytes and tool identity. A ledger change invalidates the audit even if
  it leaves the parsed required-cell set unchanged. Bounded parallel artifact
  checks preserve each row's dimensions/sampling checks and propagate failures.
- Fixed during publication preparation: weighted scaling of constant numeric
  features emitted invalid-square-root warnings from roundoff in the variance
  calculation. Translating features before moment accumulation makes constant
  columns exactly zero during fitting while preserving weighted means and scales.
  All routing tests pass with numerical runtime warnings promoted to errors.
- Publication reserved-size evaluation now requires the full size inventory and
  feature-only representative-selection provenance, including the audited primary
  cell digest and exact selected membership. Merely changing a run's size or
  subset identity no longer admits it as protocol-compliant generalization data.

- Publication encoder-mode evaluation now enforces the same representative-selection
  provenance as the size study; an arbitrary subset of reserved references is no
  longer admitted merely because it belongs to the primary reserved cohort.

- Publication orchestration: scanning the active cell directory over NFS can
  block cell renames across all generator workers. Even a later storage-local directory count coincided with all six workers
  waiting on filesystem operations; execution resumed after stopping that count.
  Count cells only while generation is already frozen. Use live PID/scope checks
  and guard artifact observations during generation; reserve full scans for stopped runs.

- Publication storage correction: the temporary NFS export used async semantics,
  so client fsync alone did not establish the intended disk durability before
  releasing staged copies. The study-only export now uses sync; a storage-side
  syncfs completed successfully before generation/transfer resumed. Verify
  server-side synchronization semantics before using staged-copy release.

- Publication staging review found a pending-path-read race: freezing a writer
  and checking open descriptors does not exclude a thread paused between
  create_new/AlreadyExists and reading the existing artifact. Live transfers now
  atomically replace local copies with verified archive links. The filesystem
  regression and repeated-drain pilot pass; no run failure was observed.

- Publication fitting now validates resumed prediction checkpoints for finite gain/risk,
  probability bounds, matching vector shapes and boolean domain masks. Previously,
  a non-finite cached risk could silently suppress decisions. Regression coverage
  rejects these corrupt checkpoints before model selection; no real fit existed.

- Publication reporting now retains origin-balanced coverage, conditional harm
  and loss severity alongside cell-weighted values. Previously these supporting
  measures were only cell-weighted despite origin-balanced gain/risk selection.
  An unequal-origin regression verifies the distinct denominators and skip-all
  behavior. No real fit or reserved report existed when this was corrected.

- Publication fitting now rejects estimator non-convergence and numerical
  runtime warnings, rather than merely logging them. A controlled estimator
  warning is covered by regression. This is a pre-fit integrity correction;
  no real router fit had run or emitted such a warning.

- Publication fit resume/evaluation now binds effective numerical thread counts,
  BLAS/OpenMP library bytes and dispatch architecture as well as package versions.
  Previously a changed wrapper job limit could pass the environment check.
  Thread-count and library-byte changes are covered by regression; no real fit
  existed when the contract was strengthened.

- Fixed publication model portability before fitting: the weighted scaler was
  defined in the CLI module, so CLI-produced joblib artifacts referenced
  `__main__.OriginScaler` and failed in a fresh consumer. The failure was
  reproduced; the scaler now has a stable importable module and frozen source
  hash. Cross-process serialization and changed-source rejection pass.
