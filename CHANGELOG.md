# Changelog

All notable changes to zensr are documented here. (Started 2026-08-28; earlier
history lives in `git log`, `docs/research/PLAN.md`, and `benchmarks/`.)

## [Unreleased]

### QUEUED BREAKING CHANGES

None queued by the canon4 update.

### Fixed

- Browser SR overviews use output dimensions within the 4 MP budget, preserving
  full overview resolution for smaller upscales. Independent author-PyTorch
  comparisons cover both models, ×2/×4 and WASM/SwiftShader WebGPU.
  [Results](benchmarks/browser_torch_2026-11-09.md).

- Binary16 subnormal model weights now expand to their exact IEEE f32 values.
  An exhaustive independent regression covers all binary16 bit patterns (`a01962d6`).

- Routine routing and QAT scoring no longer persists every decoded/repaired RGB
  buffer or requests Butteraugli maps (`930476d7`). Explicit score-only identities
  preserve legacy artifact requirements; completed-run pruning protects source
  references and encoded JPEGs (`80748c5a`). [Retention](docs/SCORING_RETENTION.md).

- Tiny/odd JPEG chroma projection uses component sampling factors rather than
  padded block dimensions. Preserves 420 half-resolution projection and leaves
  422/440 chroma unprojected even when MCU padding masks the subsampling.

### Added

- Offline aspect-preserving CUDA repair/SR bank with official split intersection,
  existing feature-offer negotiation and lossless IMIJ storage.
  [Tool contract](tools/sr-bank/README.md).

- Resident SR blend evaluator using retained SPANF/DSCF PNGs, zenresize filters,
  zenmetrics PSNR and pinned adopted zensim-A. Scalar category/quality frontiers
  remain exploratory; no runtime default changes.
  [Results](benchmarks/sr_blends_2026-11-10.md).

- Browser workbench with bounded concurrent WASM tiles, WebGPU selection,
  streaming PNG output, native-detail zoom/pan/flicker, zenjpeg JPEG decoding,
  optional canon4 repair with a retained 1× checkpoint, and an isolated
  experimental relaxed-SIMD repair build (`29dd340d`, `1b58fad3`, `a2479fae`).
  [Behavior and build](docs/BROWSER_DEMO.md) ·
  [Measured browser runs](benchmarks/browser_sr_2026-11-09.md).

- Private SR memory experiments: native completed-tile RGB8/RGB16 storage and
  earlier DSCF feature release, with exact paired output and frozen float replay
  (`86682240`). [Measurements](benchmarks/sr_memory_2026-11-09.md).

- Two-call FP32 PyTorch CUDA 2048→8192 SPANF/DSCF measurements, including
  allocated/reserved VRAM and separate pinned transfers (`065bbe06`).
  [Results](benchmarks/sr_8k_gpu_2026-11-09.md).

- Two-call 2048→8192 CPU measurements for native SPANF and tract/PyTorch DSCF,
  including peak process RSS and retained run identities (`ccbaeeab`).
  [Results](benchmarks/sr_8k_cpu_2026-11-09.md).

- Public external-model evaluation with scalar-only streaming, pinned author
  checkpoints, author inference parity and quality × category reports
  (`788de6a8`, `6ff0d347`, `922326af`).
  [JPEG repair](benchmarks/external_jpeg_models_2026-11-09.md) ·
  [Super-resolution](benchmarks/external_sr_models_2026-11-09.md).

- JPEG-quality × category reporting for the larger repair model, including harm
  rates, worst losses and encoder/sampling breakdowns (`2092319a`). Each category
  currently contains one source image. [Results](benchmarks/quality_by_category_2026-11-08.md).

- Larger-model PTQ comparison with a complete 2,960-input cohort, RGB precision
  measurements, original float-head controls and independent ONNX reference
  execution (`aea36936`, `9b2b7d3f`). The tested activation-quantized formats lose
  quality; native float remains the serving implementation.
  [Results](benchmarks/quality_int8_2026-11-08.md).

- Controlled canon4 W8A32 QAT study with three paired seeds, native packed-weight
  replay, float/PTQ controls, source-excluded evaluation and model-only draft
  packaging. The f16 default is unchanged (`be26a373`, `369b5f74`).
  [Results](benchmarks/canon4_qat_2026-11-08.md).

- Draft artifact packaging for pair subsets identified against imazen-26's
  published ID split and the retained
  model collection, with source-ID manifests, archive hashes and bounded
  inventory of material not uploaded (`4fd49461`, `04617a3c`, `0177d775`).
  The trainer honors dataset validation-tail metadata while preserving its
  original default. [Preview records](benchmarks/preview_assets_2026-11-07.md).

- Reconstruction of retained v4 pair source IDs and crop coordinates, with
  exact saved-target/recipe checks and an independent coordinate-based verifier
  (`f11177a3`, `5f2a500b`). All 48,000 canon4 pairs matched; the manifest records
  exposure to imazen-26's published validate/test IDs. [Results](benchmarks/canon4_pair_sources_2026-11-07.md).

- Development-only routing feature ablation and secondary reused-cohort check;
  native 34-input candidate replay, APP15 invariance controls and experimental
  structure-selected analysis windows. Spatial selection remains experimental.

- Retained-input projection on/off diagnostic measures SSIM2 and coefficient
  residuals reconstructed from delivered RGB8; independent DCT and odd-edge
  box-sampling checks. See the four-case diagnostic in benchmarks.

- Arbitrary-resolution decoded routing through at most five native windows,
  aligned to luma DCT blocks and 420 MCUs, with restoration at original dimensions. Includes
  strided crop, coefficient-padding, flat-image, wide/tall and selected-output
  checks. Quality estimates remain scoped to the 512-pixel cohort (0e88fa9d).

- Native decoded-feature routing selected from the reserved 512-pixel frontier:
  default Fidelity requests use the frozen linear gain/risk policy, with explicit
  routing overrides retaining their existing rules.
  Pixel features enable `chooser` by default. Includes sklearn goldens, complete
  numeric replay and retained-JPEG pixel parity (9f9dbe95, 79822605).

- Publication storage observation timeouts freeze generation before one retry;
  recovery preserves pre-existing freezes and unchanged capacity limits
  (65d8d8ab, 5a962cf7).
- Read-only publication-run status checks verify writer and guard process
  identities without scanning active cell files (de06aa35).

- Publication model integrity: matched origin-balanced coverage and severity
  (d00746e7), rejection of unconverged estimators (7bcbb23e), numerical-runtime
  identity checks (d3bac492), and portable scaler serialization (0ec26cd7).

- Publication-study storage safeguards: hash-verified bounded staging and
  ended-writer final draining (3379bc05, 944a33b9), plus strict validation of
  resumed routing predictions (635771c9). The study remains incomplete;
  these changes do not select or deploy a new routing policy.

- Coefficient-routing comparisons under quality-dependent harm budgets,
  including quality-only ablation and calibration/audit separation (112c2a92).
  [Results and curve definition](benchmarks/routing_harm_curve_2026-11-04.md).
- Actual-model routing boundary regression and 1,451,520 checked runtime
  threshold/gate decisions; Python calibration integrity tests run in CI
  (73af230b).
- Exact decimal score differences prevent losses equal to the harm threshold
  being counted as strictly greater (3d1bedfc).

- `tools/exactness/`: a real-model matrix (`just exact-matrix`) requiring the
  same raw output from whole-image and tiled runs at threads 1/4/8 on tiers
  auto/v3/scalar, and a `restore_jpeg` probe (`just exact-product`) requiring
  the same final RGB8 at threads 1..16 on every tier, with the counterexample
  JPEGs recorded in
  [the pointer](benchmarks/exactness_counterexamples_2026-10-18.pointer.md).
  Results: [the repair note](benchmarks/exactness_repair_2026-10-18.md).
  `--compare-bin` checks a second worker on every configuration.
- Run existing core runtime unit tests from the isolated profiling workspace,
  including default, no-default-features, and tier_v4 configurations; test
  source and assertions unchanged (14435363).
- Standalone compact-model Coz profiler with exact raw-output repeatability
  checks; production runtime unchanged (4abe56e7). See
  [the probe record](benchmarks/coz_runtime_2026-10-18.pointer.md).

### Changed

- Owner-approved metadata-independent default: 34 numeric inputs exclude file
  bpp and dimensions. Reused reserved-cohort gain +2.4173 SSIM2, observed harm
  0.5662%; original model/replays retained. APP15 padding preserves predictions.

- Reuse per-request JPEG coefficient data and compatible plain decoded pixels
  across routing and restoration. Auto deblocking retains its distinct decode.
  Cached/fresh parity checks preserve output float bits, RGB8 and reports.

- Realtime tool defaults select the four-encoder `dejpeg_rt24i_canon4` model
  (86,406 f16 bytes). Refit routing curves, AQ correction, content threshold
  and family-aware identity gates using an image-separated calibration.
  Default routing restores more images and gains more mean SSIM2, with a
  higher harm fraction; see [the measured tradeoff](benchmarks/ship_rt24i_canon4_2026-11-04.md).
  The library continues to accept caller-loaded models; no public signatures
  changed. The old artifact remains available.

- **zenjpeg git pin e277e9c9 → b88f02e3 (2026-10-12), so the graph resolves one
  zenanalyze.**
  - e277e9c9 pinned zenanalyze by git rev 13d40c3b. The root
    `[patch.crates-io]` rewrites only registry requirements, so two zenanalyze
    0.2.0 sources compiled, against the root manifest's stated invariant.
  - b88f02e3 is the newest zenjpeg `main` rev whose zenanalyze and zencodec
    edges are registry requirements.
  - Later revs pin zencodec and zenmeta by git. Head 64a92f78 would add a
    second zencodec (0.1.27 git beside 0.1.26) and a second ultrahdr-core
    (0.6.0 beside 0.5.0).
  - Lock changes: the 13d40c3b zenanalyze is gone; zenjpeg and zenyuv change
    source; zenyuv gains an `imgref` edge. Nothing else.
  - `restore_jpeg` gives the same recorded digests, and `zensr-adopted-verify`
    prints identical lines.
- **Released archmage/magetypes 0.9.30 instead of the f46cf941 git pin.** The
  pin existed for magetypes' `concat_shift`, which no release has; conv3x3 now
  reads its three taps as three unaligned loads (same lane values, same
  arithmetic; speed effect not measured) and uses the `_t` constructors 0.9.30
  asks for. Both lockfiles were regenerated from a clean tree, so the workspace
  lock records the committed git zenjpeg rather than a local path override.
  Raw outputs are bit-identical to the pinned build on a 378-case model matrix.
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

- Partial luma-only projection on 4:2:2/4:4:0 no longer reports full
  consistency. Requests requiring consistency return decoded pixels; other
  requests explicitly report unprojected chroma. Clarified that calibrated
  projection slack does not guarantee bit-exact coefficient re-encoding.

- Routing skips now return the deployment decode before model inference.
  Previously `Routing::Never` and an Auto rejection still ran restoration
  while labeling its output unchanged. A real-model regression test compares
  both skip modes against the decoder byte for byte.

- **Threads, tiles and CPU tier no longer change the output.** The same JPEG
  restored with 2 threads differed from 1 thread, and AVX2 from AVX-512, by one
  RGB8 level (2026-10-18 counterexamples, dejpeg7_graphics on gb82 `girl` and
  `city` at q40). Raw model output differed in the last bit between whole and
  tiled runs and between tiers on every shipped model. Causes: conv3x3's
  overlapped final tile summed all taps in one chain while the main loop sums
  16-channel blocks, so any column it covered depended on the row width and
  vector width when cin > 16; rows narrower than one vector used an unfused
  scalar loop; conv1x1, SiLU, gate and PReLU computed their last
  `len % W` elements with different scalar code (libm `exp` instead of
  `exp_midp`, unfused sums); and the scalar backend's `mul_add` rounds twice.
  Every path now runs the main loop's per-element sequence (tail and narrow
  tiles in registers, remainders as zero-padded vectors), and every FMA on the
  default path is `mul_add_portable`, which is the same FMA on x86 v3/v4 and NEON and a
  one-rounding software FMA on scalar and WASM. PReLU and the exp clamp use
  compare-and-select with x86 `maxps`/`minps` semantics, because the wasm and
  NEON `max`/`min` return NaN for a NaN operand where x86 returns the other.
  Outputs of the main loop on the FMA tiers are unchanged. New exact
  regressions: `simd::exactness`.
- **Opt-in Winograd (`ZENSR_WINOGRAD=1`) no longer depends on threads, tiles
  or tier.** A Winograd output depends on which 2×2 tile it falls in. Tiles
  started at buffer row and column 1, so a tiled run grouped image pixels into
  different tiles than the whole-image run. Beyond that:
  - an odd interior width computed its last column with the direct formula;
  - an odd interior height re-ran an overlapping last tile row;
  - buffers under 4 px, or under one vector of tiles, took the scalar
    `wino.rs` path with unfused multiply-adds;
  - the tile halo did not count a Winograd layer's 2-pixel reach.

  On the previous build, 14 of 68 real-model matrix cases disagreed, 5 of them
  in RGB8. The largest was animevideo_x4v3 150×140, tiled vs whole: 4 RGB8
  values.

  The vector kernel now runs every shape:
  - tiles are anchored to the image's row and column parity, with
    zero-padded lanes;
  - edge tiles store only interior pixels;
  - the one-pixel border uses `direct_px`;
  - the GEMM uses `mul_add_portable`;
  - `halo()` adds one pixel per Winograd layer.

  All 68 cases now agree, and the default path is byte-identical to the
  previous build. New regressions:
  `simd::exactness::winograd_windows_and_tiers_match_the_whole_image` and
  `adopted::tests::winograd_models_tiled_equals_whole_bitwise`.
- `tools/coz-profile`: forcing `ZENSR_PROFILE_TIER=v3` or `scalar` now also
  disables the v4 token, and it panics if a token cannot be disabled instead of
  ignoring the result. Off x86-64, a forced tier is an error. The tool enables
  archmage's `testable_dispatch`, under which the disable cannot fail, so
  earlier matrix rows are unaffected.
- `tools/coz-profile` under Coz: a positive `threads` now exits with an
  error before the model loads. coz-rs asks for `coz::thread_init()` on every
  thread Coz samples, and `upscale_tiled`'s workers never call it, so profile
  with `threads=0`. Positive `threads` remain a workload mode outside Coz.
  - `just coz-build` now builds with `--locked`.
  - `just coz-run` writes the profiled binary's sha256 next to the profile.
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
