# Documentation

## Using JPEG repair

- [Routing contract](DEFAULT_ROUTING.md): defaults, overrides, supported inputs
  and native-resolution analysis.
- [Integration and dependencies](PRODUCTION_DEPENDENCIES.md): crate graph,
  model loading and reproducible source builds.
- [Readiness](REPAIR_READINESS.md): verified behavior and remaining validation.
- [Single-file runtime models](MODEL_CONTAINER.md): packing, loading and format.
- [Models and training](MODELS.md): the model family and canon4's training recipe.
- [Pair provenance](PAIR_PROVENANCE.md): reconstruct and verify saved crop identities.
- [Train from the preview](TRAINING_PREVIEW.md): assemble pairs from published imazen-26 training IDs.
- [Split authorities](SPLIT_DEFINITIONS.md): imazen-26 membership, zensr reassignment and internal validation.
- [Repair APIs](API.md): application containers, codec-supplied samples and result semantics.

## Evidence

- [Corrected edge projection](../benchmarks/projection_fix_2026-11-10.md): full small-bank replay, quality tails and native/browser checks.

- [Canon4 followed by ×4 SR](../benchmarks/sr_cascade_2026-11-09.md): matched
  direct/repair-first SPANF and DSCF quality, routing coverage and retained PNGs.

- [Public external-model comparison](research/EXTERNAL_MODELS_2026-11-09.md):
  runnable releases, JPEG repair and separate ×4 SR results by quality/category.
- [Current routing comparison](../benchmarks/routing_feature_ablation_2026-11-06.md).
- [Original reserved frontier](../benchmarks/publication_primary_reserved_2026-11-05.md).
- [Projection diagnostic](../benchmarks/projection_diagnostic_2026-11-06.md).
- [Benchmark index](../benchmarks/INDEX.md).
- [Model provenance](MODEL-PROVENANCE-AUDIT.md).

## Research and engineering

- [Small-image candidate bank](../tools/sr-bank/README.md): split-preserving
  CUDA repair/SR generation and lossless IMIJ retention for offline tuning.

- [Browser workbench](BROWSER_DEMO.md): concurrent WASM/WebGPU SR, optional JPEG
  repair, native-detail viewer and local serving. [Measured runs](../benchmarks/browser_sr_2026-11-09.md).
- [Research record](research/README.md): plans, experiment history and previous results.
- [Canon4 INT8 QAT](CANON4_QAT.md): controlled weight-compression experiment and measured limits.
- [Quality model integer inference](QUALITY_INT8.md): frozen float/INT8 speed and RGB-precision comparison.
- [VSR source review](VSR_REVIEW.md).
- [ARM kernel audit](engineering/ARM_AUDIT_2026-08-01.md).
