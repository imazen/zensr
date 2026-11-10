# JPEG repair readiness — 2026-11-07

The agreed ×1 implementation is committed and its production code passed CI.
The project is a locked-source integration with explicit measurement limits;
it is not a packaged release or a fully validated general-resolution deployment.

November 9 update: `a01962d6` corrects binary16 subnormal weight expansion.
Exhaustive conversion and product tests pass. The quality/routing observations
below describe the earlier executable; a full corrected-weight routing replay
has not been performed. Existing numeric predictor goldens are not a substitute
for rescoring delivered pixels.

## What is complete

- Owner-approved 34-input routing default, excluding file bpp and dimensions.
  Source: `5a8fb79bc5f6`; approval documentation: `d3cb0725918f`.
- Actual model inference, native-size windows, caller routing overrides,
  coefficient projection, tiny/odd sampling geometry fixes and decode reuse.
- Independent sklearn prediction replays for all 140,940 vectors of both the
  original and active routers. Baseline expectations remain preserved.
- Real-JPEG feature replay, selected-output pixel parity, cached/fresh float-bit
  and report parity, model goldens, whole/tiled checks, metadata invariance and
  feature-off checks. External-data diagnostics are explicitly caller-selected.
- Source CI passed every job: Linux x86 and ARM, Windows x86 and ARM, macOS ARM,
  i686 via cross, formatting and Clippy. Verified directly in
  [run 42205004428](https://github.com/imazen/zensr/actions/runs/42205004428)
  for implementation commit `5a8fb79bc5f6`.
- [README](../README.md), [default routing contract](DEFAULT_ROUTING.md),
  [dependency chain](PRODUCTION_DEPENDENCIES.md),
  [feature comparison](../benchmarks/routing_feature_ablation_2026-11-06.md)
  and [delivered-pixel projection diagnostic](../benchmarks/projection_diagnostic_2026-11-06.md).

## Production limits

The source can be integrated using the committed lockfile and caller-loaded
canon4 weights. Distribution still needs a dependency/publication plan:
`zensr-micro` is `publish = false`, the integration uses git dependencies/root
patches, and fresh consumer lockfiles can resolve different analysis revisions.
No crate release was published by this work.

Arbitrary-size execution does not establish routing quality across sizes.
Native feature checks extend through 4096×2160, but the reported quality frontier
is 512-pixel only. Deployment latency, representative request cost and resource
behavior at production image sizes have not been established by a controlled
end-to-end measurement. Those are the remaining production validation tasks.

Projection is a family-calibrated coefficient constraint. It does not certify
exact JPEG re-encoding, final RGB8 coefficient bounds or perceptual improvement.
`Consistency::Certified` records that projection was applied to all three
components; `require_consistency` requires that availability, not a final-pixel
bounds check. The worst observed reserved loss remains 18.5344 SSIM2.

## Publication scope

The current 34-input policy was selected on development inputs. Its secondary
comparison reuses the already-inspected reserved cohort: 140,940 cells,
290 references and 269 origins, yielding +2.4173 origin-balanced mean SSIM2,
0.5662% observed harm and 79.75% cell restoration coverage. Harm follows the
stated quality-dependent curve. These are cohort observations, not guarantees.

The reports support a scoped technical account with those disclosures. A claim
of fresh held-out validation for the final policy requires fresh independent
evaluation; a claim of general-resolution quality also requires corresponding
size-generalization evidence. Corpus composition and historical training-pair
provenance limits remain part of the methods discussion.

Pixel-artifact auditing was explicitly cancelled by the owner. Its incomplete
status must remain disclosed; this readiness review does not restart it or
pretend it passed. Spatial-window alternatives and local-guard proposals remain
experimental and are not included in the active policy's quality claims.

The 2026-11-10 edge investigation confirms severe amplification from partial-
block padding in coefficient constraints on two tiny 420 witnesses. A bounded
encoded-padding control removes that amplification, but runtime projection is
unchanged and requires mixed partial-block validation before replacement.
The separate JPEG input-generation padding fix is on zenjpeg PR 261. See
[the mechanism and controls](../benchmarks/chroma_edges_2026-11-10.md).
