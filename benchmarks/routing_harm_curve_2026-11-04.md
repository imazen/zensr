# Coefficient routing under quality-dependent harm limits — 2026-11-04

## Harm definition and selection

The evaluation uses loss greater than 0.5 SSIM2 when the plain decoded reference score is at or below 90. As an initial curve above 90, the tolerated loss falls linearly to 0.1 at 100: `0.5 - 0.04 * clamp(identity_ssim2 - 90, 0, 10)`. Thus 95 tolerates 0.3 and 98 tolerates 0.18. The above-90 taper is an explicit design assumption, not a fitted perceptual law. The pristine reference is used only for evaluation labels; inference still uses coefficient content class, corrected probe quality/family/subsampling and record AQ metadata.

The prior report counted losses greater than 0.1 everywhere. Decimal subtraction now preserves the recorded TSV precision: exactly 0.100 is not greater than 0.1. This fixes three falsely counted cases in each of the original full-grid Fidelity/Conservative tallies. Mean gains do not change.

Gain curves, coefficient threshold 0.20, AQ deltas, model, guards, projection and base high-quality gates are frozen. This experiment selects among 420 combinations plus skip-all: minimum predicted gain in {0.25,0.5,0.75,1,1.5,2,3,4,6,10}, quality gate margin in {0,2,4,6,8,10}, distance gate margin in {0,0.1,0.25,0.5,1,2,6}. The two margin scales have separate units. The quality-only ablation uses frozen pooled curves instead of coefficient content classes.

For each budget, select the greatest mean calibration gain whose empirical harm fraction is within the budget. Ties prefer fewer harms, then less work. Only the 33 calibration images select policies. The 30 audit images were already inspected in the shipping work: they are held out from selection, but are not a fresh independent confirmation set. Each image has 192 encoder/subsampling/quality variants; these observations are correlated, not thousands of independent images. Grid coverage remains q15–100 at the original 12 sampled qualities, two subsamplings, eight encoder legs and 512-pixel crops; this is not full production-range validation.

## Audit results

Harm rates below use the new curve and count all 5,760 observations, including skipped observations with zero gain. Worst loss is shown regardless of whether it crosses the curve.

| Policy | Mean gain | Restored | Harmed | Worst loss | Images with harm / 30 |
|---|---:|---:|---:|---:|---:|
| Current Fidelity | +1.6261 | 68.65% | 160 (2.78%) | 2.664 | 9 |
| Current Conservative (margin 6) | +1.0409 | 25.50% | 27 (0.47%) | 1.396 | 6 |
| Coefficient, 0% calibration budget | +0.0000 | 0.00% | 0 (0.00%) | -0.000 | 0 |
| Coefficient, 0.5% calibration budget | +1.0357 | 17.93% | 15 (0.26%) | 1.009 | 4 |
| Coefficient, 1% calibration budget | +1.3121 | 29.08% | 39 (0.68%) | 2.143 | 5 |
| Coefficient, 2% calibration budget | +1.4681 | 48.65% | 79 (1.37%) | 2.143 | 7 |
| Coefficient, 5% calibration budget | +1.6261 | 68.65% | 160 (2.78%) | 2.664 | 9 |

No searched policy retained positive gain with zero calibration harms. The lowest-risk useful candidate (0.5% budget) nearly preserves current Conservative gain and reduces both harm count and worst loss on this audit. The 1% candidate retains more gain, but has a larger worst loss; it does not dominate Conservative on every risk measure.

| Calibration budget | Minimum gain | Quality margin | Distance margin | Calibration harm | Coefficient audit gain / harm | Pooled audit gain / harm |
|---|---:|---:|---:|---:|---|---|
| 0% | 10 | 0 | 0 | 0.00% | +0.0000 / 0.00% | +0.0000 / 0.00% |
| 0.5% | 2 | 0 | 0 | 0.44% | +1.0357 / 0.26% | +0.5589 / 0.05% |
| 1% | 0.75 | 4 | 1 | 0.98% | +1.3121 / 0.68% | +1.0598 / 0.61% |
| 2% | 0.25 | 0 | 1 | 1.72% | +1.4681 / 1.37% | +1.4514 / 1.46% |
| 5% | 0.25 | 0 | 0 | 3.33% | +1.6261 / 2.78% | +1.5983 / 2.76% |

The ablation is budget-matched on calibration, not matched to identical audit harm. The JSON includes per-encoder results and sensitivity fits under flat 0.1 and flat 0.5 cutoffs. Runtime policy defaults are unchanged by this experiment; choosing a deployment budget is a separate decision. The 0.5% curve-budget candidate is already expressible as Fidelity plus `Routing::Auto { min_gain: 2.0 }`.

## Reproduction and evidence

Inputs and generated artifacts are retained on i265. No original TSV, weights or saved results were removed. Source baseline: `904c175fb236803b6f935bc8bfcc123e4afd7ef5`.

```sh
just canon4-frontier \
  --scores /home/lilith/data/zensr/benchmarks/prod_dejpeg_rt24i_canon4_nogate_2026-11-04.tsv /home/lilith/tmp/handoff/canon4-ship/canon4-tail.tsv \
  --metadata /home/lilith/tmp/handoff/canon4-ship/frontier-runtime.tsv \
  --fit /home/lilith/tmp/handoff/canon4-ship/fit-final.json \
  --out /home/lilith/tmp/handoff/canon4-ship/routing-frontier-curve.json
just canon4-routing-tests
```

Run longer commands under the host resource wrapper. The metadata audit regenerated all 12,096 identity/probe rows and agreed on 1,451,520 threshold/gate decisions (ten gain cutoffs, ten candidate margins plus the two original gate columns). Maximum predicted-gain difference was 0.00005965. These checks compare actual Rust estimates and gates to the Python analysis; they do not claim independent ground truth for quality.

Metadata run: 193 s; run-heavy peak RSS 0.46 GiB, minimum available 22,591 MiB, peak load 8.88. Full log: `/home/lilith/tmp/handoff/canon4-ship/frontier-runtime.log`.

Artifact SHA-256:

- `fit-final.json`: `bd6a3f87288dc9e2e5e99e2ec2b4dffb6ab7a72caaf12557b63e7b620d1f2bb6`
- `frontier-runtime.tsv`: `b1492ed13be2c7c89bd8e75bbc77d0da31273661bf4044a88d0b34b0dafcba92`
- `routing-frontier-curve.json`: `26852add073b821b04b70674972f3c169c40d85614e6e2733344a7fbdddb34b3`

## Regression checks

`just canon4-routing-tests` passes ten Python tests covering calibration/audit
separation, constrained selection, denominator definitions, decimal loss boundaries,
the harm curve, gate direction/units, and exclusion of the pristine score from
inference. `just canon4-test` passes the full micro/integration suites, including
a new actual-model test that restores at the exact predicted-gain threshold and
returns pixel-identical decode above it. No existing test was removed or weakened.
`just canon4-lint` passes formatting and clippy. Test log: `routing-tests.log`
under the artifact directory; run-heavy: 79 s, peak RSS 0.21 GiB, minimum
available 23,086 MiB, peak load 7.25.
