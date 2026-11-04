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
