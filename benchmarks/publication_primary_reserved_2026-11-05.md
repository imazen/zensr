# Reserved 512-pixel routing frontier — 2026-11-05

The score-table comparison is complete. Pixel-artifact auditing was cancelled by the owner and remains incomplete. No production policy was changed.

## Evidence and scope

140,940 completed cells, 290 references, 269 grouped origins, four encoders, 420/444, and the frozen 61-value quality grid. The owner-approved oversized-reference exclusion removed one 124,227,840-pixel PNG and its 486 required cells. The frozen development fit and operating points were not reselected on reserved outcomes.

Harm is loss greater than `0.5 - 0.04*clamp(identity_ssim2-90,0,10)`; the taper above 90 is a proposal. Gain and harm below are origin-balanced; restoration coverage is cell-weighted. Budget labels are development selection targets, not reserved-set guarantees.

| Policy | Mean SSIM2 gain | Observed harm | Restore coverage | Worst SSIM2 change |
|---|---:|---:|---:|---:|
| current_fidelity | +2.4949 | 1.9615% | 93.51% | -18.5344 |
| current_conservative | +2.1429 | 1.0571% | 61.17% | -18.5344 |
| threshold/budget-0.005 | +0.8774 | 0.1165% | 13.40% | -18.5344 |
| coefficient/budget-0.005 | +2.3649 | 0.6628% | 70.21% | -18.5344 |
| decoded/budget-0.005 | +2.4260 | 0.5490% | 80.43% | -18.5344 |
| decoded/budget-0 | +0.1488 | 0.0000% | 1.06% | -0.0000 |

## Interpretation

At the 0.5% development budget, decoded-feature routing improves mean gain over current Conservative by 0.2831 SSIM2 (pointwise paired 95% origin-bootstrap interval: 0.2503–0.3184), and reduces empirical harm by 0.5082 percentage points (interval: 0.0807–0.9625 points lower). Coefficient routing also improves that tradeoff: gain difference 0.2220 (0.1867–0.2614), harm 0.3943 percentage points lower (0.0152–0.7914 lower). These are paired comparisons on the designed reserved cohort, not population guarantees.

Neither 0.5%-budget candidate met a 0.5% observed-harm target on this cohort. Both retain the same worst loss as current routing: 18.5344 SSIM2, on a line plot at turbo 420 q6 (identity 8.1634, restored -10.3710). The lower harm frequency does not solve this tail. The zero-budget decoded point has no observed harmed cells but retains much less gain; it does not establish a never-harm guarantee.

The decoded 0.5%-budget router is a 5,660-byte serialized Python model; the coefficient equivalent is 255,788 bytes. These are not Rust deployment sizes or resident-memory measurements. Restoration weights are separate: 86,406 bytes. Feature extraction and deployment cost have not been confirmed by a valid runtime comparison.

Full-resolution behavior, size-dense generalization and encoder-mode generalization remain unestablished. Corpus composition and incomplete historical training-pair provenance limit broader claims. Legacy-chooser policies are included in the full report with their unresolved provenance stated.

## Artifacts and verification

Verified on i265. Full score report: `/home/lilith/data/zensr-array-archive/primary-reserved-frontier-scores-only-20261105-v1.json`. Rendered CSV, Markdown, SVG, PDF, PNG, operating points and worst-case records: `/home/lilith/data/zensr-array-archive/primary-reserved-report-scores-only-20261105-v1/`.

Result SHA256: `34ce54cfdf640594861b164d9dd33841fe31cc513b181e401dc154dfdb4108b3`.
Selection SHA256: `8acb37fe5a32fa251bf728b36ce9650dab7089821564d8135afecab5a1b8ebb8`.
Score-table digest: `d4f41113a676917cacd3af328785203801593229ac56d820a4c88bebbc52288f`.

The explicit score-table loader checks completion, roles, unique measurements and exact amended coverage. The original frozen evaluator checks model bytes, scientific inputs and numerical environment. No completed pixel-audit marker was fabricated. The stopped audit retained 116,352 checked cells and 578,792 verified artifacts. The report and figure disclose incomplete artifact verification.

Evaluation command: `just publication-routing-scores-only <reserved-primary> <frozen-selection> <result.json> <AUDIT_CANCELLED.json> --allow-unaudited-artifacts`, pinned Python under `run-heavy --mem 16G --jobs 4`. Its wrapper recorded `rc=0 165s | peak-RSS 2.27GiB | min-avail 19405MiB | peak-load 2.12`. This is offline analysis cost, not restoration latency.
