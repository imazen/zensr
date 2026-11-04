# zensr project notes

## Known Bugs

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
  block cell renames across all generator workers. Use storage-local counts and
  separate live PID/scope checks; reserve the full status scan for stopped runs.
  The observed scan was terminated and worker activity recovered.

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
