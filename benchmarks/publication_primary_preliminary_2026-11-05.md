# Preliminary 512-pixel routing frontier

Reserved confirmation, worst-loss assessment on the reserved cohort and deployment cost remain incomplete. These results cannot select a production replacement yet.

These are origin-balanced selection results from five origin-separated development folds: 235,224 cells, 484 references, 404 origins. Hyperparameters and thresholds were selected using these predictions, so they are not independent confirmation. The reserved cohort is still being generated.

Harm uses `0.5 - 0.04*clamp(identity_ssim2-90,0,10)` as the loss threshold. The taper above 90 is a proposal; no policy guarantees never-worse quality.

| Family | Selection harm budget | Mean SSIM2 gain | Empirical harm | Restored |
|---|---:|---:|---:|---:|
| threshold | 0.5% | +0.8798 | 0.1533% | 13.23% |
| threshold | 1% | +2.2149 | 0.9949% | 63.20% |
| threshold | 2% | +2.5746 | 1.8851% | 93.59% |
| coefficient | 0.5% | +2.4029 | 0.4997% | 69.99% |
| coefficient | 1% | +2.5655 | 0.9506% | 88.74% |
| coefficient | 2% | +2.5967 | 1.9343% | 96.08% |
| chooser | 0.5% | +2.3430 | 0.4974% | 67.17% |
| chooser | 1% | +2.4960 | 0.9997% | 77.41% |
| chooser | 2% | +2.5885 | 1.8910% | 95.66% |
| edge | 0.5% | +2.3143 | 0.4586% | 65.09% |
| edge | 1% | +2.5223 | 0.9970% | 83.20% |
| edge | 2% | +2.5893 | 1.9855% | 95.85% |
| decoded | 0.5% | +2.4918 | 0.4851% | 80.74% |
| decoded | 1% | +2.5893 | 0.9970% | 91.39% |
| decoded | 2% | +2.6041 | 1.7943% | 96.62% |
| decoded_legacy_chooser | 0.5% | +2.4909 | 0.4824% | 80.56% |
| decoded_legacy_chooser | 1% | +2.5804 | 0.9385% | 90.33% |
| decoded_legacy_chooser | 2% | +2.6034 | 1.7962% | 96.49% |
| post | 0.5% | +2.5555 | 0.4879% | 83.45% |
| post | 1% | +2.6247 | 0.9859% | 94.29% |
| post | 2% | +2.6268 | 1.2305% | 95.96% |

The coefficient family improves the development tradeoff over threshold routing at the 0.5% selection budget. Decoded features add gain at that budget. Post-restoration features require running restoration before deciding, so their quality frontier alone does not establish a useful compute-saving router. Chooser families retain unresolved historical training/feature-version provenance.

The coefficient model selected at the 0.5% and 1% budgets is `coefficient-hist-3` (255,788 serialized Python bytes). The decoded model at 0.5% is `decoded-linear-1` (5,660 serialized Python bytes). These are artifact sizes, not Rust deployment or resident-memory measurements. The image restoration weights remain 86,406 bytes.

Source: frozen `primary-fit-20261105-v1/selection.freeze.json`, verified on i265 on 2026-11-05.

Selection SHA256: `8acb37fe5a32fa251bf728b36ce9650dab7089821564d8135afecab5a1b8ebb8`.

Runtime routing remains the shipped policy. The reserved comparison will report the frozen candidates before deployment selection. Full-resolution behavior and the dense-size sweep are outside this initial 512-pixel result.
