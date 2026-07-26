# zensr public API contract (2026-07-26)

**Snapshots:** `zensr-micro.public-api.txt`, `zensr-zenjpeg.public-api.txt`
(regenerate: `cargo public-api -p <crate> --simplified`). Diff against these
in review; growth of the stable surface requires deliberate sign-off.

## Stable surface (semver-honored once published)

- `zensr-zenjpeg`: `restore_jpeg`, `RestoreConfig` (non_exhaustive; Default +
  `with_*`), `Restored`, `RestoreReport` (non_exhaustive), `RestoreError`
  (non_exhaustive), `Projection` (non_exhaustive), `policy_wants_auto`,
  `slack_for`, and the narrow re-exports (`AdoptedModel`, `ModelSpace`,
  `GuardConfig`, `GuardReport`, `ProjectionConfig`, `ProjectionReport`).
- `zensr-micro`: `adopted` (model loading/inference), `guards`, `consist`,
  `decode_all_f16`. Config structs are non_exhaustive — construct via
  `Default` and mutate/builder.

## Explicitly NOT contract

- Anything behind the `internals` feature: the `simd`/`tiled` modules, their
  root re-exports, and `px` (which now requires `internals` too). Without the
  feature these modules are `pub(crate)` — absent from the public surface.
- Root items marked `#[doc(hidden)]` (SPANF research types: `SpanfWeights`,
  `SpanfModel`, `Scratch`, layout consts, `decode_f16_weights`,
  `decode_int8pc_weights`, scalar `spanf_x4`). They stay compiled because the
  kernel unit references them, but they are excluded from docs and from the
  cargo-public-api snapshot, and carry no stability promise.
- The `zensr-verify` bin (requires `internals`) and `zensr-bench` entirely
  (dev tooling).

Verified 2026-07-26: the default-features snapshot contains zero SPANF
entries; default features are now `["avx512"]` (px dropped — it was
SPANF-only and unused by any consumer).

## Deferred (recorded 2026-07-26): zensr-kernels crate extraction

Product cold build is already 11.9 s and core-edit iterate 2.0 s; the
extraction's remaining win is marginal, and moving kernel code is exactly
the 26x-incident hazard class (retiming required either way). Do it as its
own change with a dedicated retiming budget, not inside a feature milestone.
