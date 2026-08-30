# Changelog

All notable changes to zensr are documented here. (Started 2026-08-28; earlier
history lives in `git log`, `PLAN.md`, and `benchmarks/`.)

## [Unreleased]

### Changed

- **The chooser takes an `Offer`; the `&dyn FeatureProvider` path is gone.**
  `zenanalyze-api` cut `FeatureProvider` / `ProviderError` / `OwnedCatalog` before
  its 0.1.1 shipped — the contract is data, not behaviour, and the model is push
  (the host hands over an offer, the codec answers yes/no, and scans itself on
  "no"). Accordingly:
  - `classify_with_provider(&dyn FeatureProvider, …) -> Result<_, ProviderError>`
    → removed. Use `classify_from_offer` / `classify_from_owned_offer` when a host
    already ran a pass over `center_crop_rgb8`, or the new
    `classify_rgb8_scanning(rgb, w, h) -> Result<ChooserReport, AnalyzeError>`.
  - `bundled_provider()` → removed.
  - `classify_rgb8` keeps its signature and now scans via
    `zenanalyze::offer_for_request`.

  `zensr-zenjpeg` is not published, so nothing downstream breaks and no version
  bump is owed.

- **`chooser-bundled` removed; `chooser` now pulls `zenanalyze` too.** The split
  meant CI's first chooser step (`--features chooser`) was a green step that
  tested nothing — every behavioural test was gated behind the second flag — and
  the configuration it protected, "chooser without zenanalyze", was one nobody
  shipped. Now one flag, one configuration, one gate that both compiles the
  boundary and runs the rule's tests.

  The rule that actually matters is unchanged: no `zenanalyze` type appears in a
  public signature, so a host on a different `zenanalyze` version can still drive
  `classify_from_offer` with its own pass.

### Fixed

- `classify_rgb8_scanning` returns the analysis error instead of swallowing it.
  `classify_rgb8` keeps the infallible `Photo`-on-failure fallback, which is the
  precision-biased safe direction the rule was fit for.

### Fixed

- **CI is green again.** The `fmt + clippy` job had been failing since
  2026-08-25 (three consecutive runs) on `clippy::chunks_exact_to_as_chunks` —
  a lint that landed in a newer stable clippy than the code was written against.
  Six sites in `zensr-micro` (`decode_all_f16`, the f16 and int8 weight
  readers, `zensr-verify`, `zensr-adopted-verify`) and one in `zensr-zenjpeg`
  (`api.rs`'s zero-AC block scan) now use `as_chunks::<N>().0`, which has
  identical semantics — the remainder is dropped either way — and additionally
  yields `&[u8; N]`, so `from_le_bytes(*c)` replaces the element-by-element
  array rebuild and LLVM can prove the indexing safe. Behaviour unchanged;
  16 + 29 tests pass as before.

  (Not a drive-by: a red gate can't tell you whether the change you just pushed
  passed. This was fixed at the root rather than pushed past.)

### Changed

- **`zenpixels` is now a two-minor range, not a caret pin.** Both requirement
  lines — `zensr-bench` and `zensr-micro`'s optional `px` dep — move from
  `"0.2.14"` to `">=0.2.14, <0.4.0"`. The floor stays at 0.2.14 (this graph's
  own floor, below the published 0.2.16); only the ceiling moves.

  Why: a caret requirement on a `0.x` crate caps at the *next* minor, so a
  consumer on `0.2.x` and a consumer on `0.3.x` are semver-incompatible and
  Cargo resolves **two copies**. Two copies means `PixelSlice` from one is not
  `PixelSlice` from the other and types stop unifying across the crate
  boundary. Widening every consumer in the workspace family to span the
  published minor *and* the next keeps one copy in the graph through the next
  `zenpixels` minor bump. Verified here: `cargo metadata` resolves one
  `zenpixels` (0.2.14), one `zenpixels-convert` (0.2.14), one `zencodec`
  (0.1.26), and `Cargo.lock` is byte-identical before and after the edit.

- **The `chooser` feature now speaks the `zenanalyze-api` contract** — it takes
  an `Offer` or a `&dyn FeatureProvider`, so a host on any `zenanalyze` version
  can drive the rule, and `chooser` alone pulls in no `zenanalyze`. Owner
  directive 2026-08-28: "zenanalyze-api should be the sole contract and
  intermediary so different zenanalyze versions can compile together", corrected
  the same day to "a direct dep is okay though, a reanalysis might be needed
  anyway if the upstream provided features are insufficient" — so
  `chooser-bundled`'s direct dep is permitted and normal
  (`docs/sole-contract.md` in imazen/zenanalyze).

  `chooser::classify_rgb8` used to call `zenanalyze::analyze_features_rgb8`
  directly against a git-rev pin (`a7d8224`), which put a concrete zenanalyze
  version in zensr's library graph — so zensr could not link beside a codec that
  pinned a different one. The rule now reads its 21 feature values out of a
  `zenanalyze_api::Offer`, or extracts them through a `&dyn FeatureProvider` the
  caller supplies.

  New API (contract-only, `chooser`): `chooser_request()`,
  `center_crop_rgb8()`, `classify_from_offer()`, `classify_from_owned_offer()`,
  `classify_with_provider()`. The first two exist so an orchestrator can produce
  an offer on the right geometry — the rule was calibrated on the center 512×512
  crop and its features are not scale-invariant.

  New feature `chooser-bundled = ["chooser", "dep:zenanalyze"]` supplies
  `zenanalyze::Analyzer` as a default provider and keeps `classify_rgb8` working
  unchanged for callers that don't want to plumb one. That is zensr picking an analyzer
  version on the caller's behalf.

  Behaviour is preserved: the request is still `FeatureSet::SUPPORTED`
  (`Select::All`) rather than the 21 columns the rule reads, because narrowing it
  changes which analysis tiers run and would need re-validation against the
  pinned eval split. All 21 features carry golden version rows, so none is
  dropped by the offer's version-row filter (verified against
  `zenanalyze/benchmarks/feature_qualified_names.tsv`). `classify_rgb8` now
  falls back to `Photo` at `p = 0` if extraction fails — the safe direction, per
  the rule's deliberate precision bias.

  **Known gap:** the rule is *fitted*, so it should pin each column's code
  version (`Select::Features` over qualified `name@hex8`) and decline on a drift.
  It can't yet — the 2026-07-26 fit did not record the feature versions it
  trained against, and synthesising them from whatever the current build
  produces would be a provenance claim with nothing behind it. This is no more
  version-blind than the pre-contract code was; pinning lands with the next
  re-fit, which should stamp
  `zenanalyze::versioning::feature_qualified_names()` alongside the weights.

- `zenanalyze` / `zenanalyze-api` are declared as crates.io versions resolved
  through one workspace-root `[patch.crates-io]`, replacing per-manifest git-rev
  pins. Cargo unifies by source, so a rev pin is its own source: two consumers on
  different revs get two incompatible `Offer` types. A root patch rewrites every
  edge at once — including `zenanalyze`'s own internal `{ version, path }` dep on
  the contract, which a rev pin cannot reach. Drop the patch entries once
  zenanalyze 0.2.x / zenanalyze-api 0.1.1 publish.

- CI runs the chooser twice: `--features chooser` (the contract-only build must
  compile everywhere) and `--features chooser-bundled` (where the behavioural
  tests live). Running only the first would have been a green step that tested
  nothing.
