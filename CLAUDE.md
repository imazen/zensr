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
