# Integration status — 2026-11-12

The engine, browser workbench and corrected coefficient projection are on
zensr main. This is an unreleased source integration, not a claim of production
or publication readiness. The native zenjpeg adapter is still a draft.

## Delivery and PR disposition

| Work | Status | Revision / review |
|---|---|---|
| Codec-independent engine, browser fixes, retained SR studies, partial-block projection | Merged to zensr main | `5042bccd10027964c25d86a13cdc46ef926e6dbe`, [PR 4](https://github.com/imazen/zensr/pull/4) |
| Single-file models, CLI loading and documentation cleanup | Merged to zensr main | `3d964681a1b5469c8e35f65e29513ee94384ec9e`, [PR 1](https://github.com/imazen/zensr/pull/1) |
| JPEG encoder strip-edge padding | Merged to zenjpeg main | `a9c34a865424c0b33b5b313ce0eca49869c82fd6`, [PR 262](https://github.com/imazen/zenjpeg/pull/262) |
| Small-bank corpus registry | Merged metadata; original artifacts remain frozen | `21009633b62610415b7dcaccdec7806726d9e10c`, [imazen-26 PR 4](https://github.com/imazen/imazen-26/pull/4) |
| Native decoder repair adapter | Keep draft; conflicts with current main | [zenjpeg PR 261](https://github.com/imazen/zenjpeg/pull/261) |

The obsolete native restoration proposal [zenjpeg PR 191](https://github.com/imazen/zenjpeg/pull/191)
is superseded by PR 261's codec-owned integration. Existing commits remain
available; no implementation was discarded. Other zenjpeg API/dependency PRs
are separate work and are not merged as part of this cleanup.

## What is verified

The engine/browser head `921d4878` passed all 12 jobs in
[CI run 42757405497](https://github.com/imazen/zensr/actions/runs/42757405497),
including Linux x64/ARM64, macOS ARM64, Windows x64/ARM64 and i686 via cross.
The combined loader/engine head `5da3eb238debcf3c9c4056eee5004b188355a45b`
also passed all 12 jobs in [CI run 43163821130](https://github.com/imazen/zensr/actions/runs/43163821130).
The main merge tree was verified byte-identical to that tested head.

Two standalone evaluation locks initially missed the loader dependency. All
five standalone consumers now resolve with `--locked`; the exactness product
root also mirrors the required codec patches. Existing QAT/storage and integer
regression checks plus strict lint pass (wrapper rc=0, 60 seconds, peak RSS
2.67 GiB, minimum available 19,202 MiB). No pixel expectations changed.

macOS dependency fetching initially failed in Cargo libgit2 while cloning the
nested `skcms` submodule, before tests. CI now uses system Git transport; the
retry passed on `aarch64-apple-darwin`. No assertion or threshold was relaxed.
The model-container merge retained both CI contracts. Its local default and
no-chooser suites pass (61/44 cases, five existing explicit publication tests
unselected in the default suite), with six packer tests, both strict Clippy
configurations, doctests and scoped formatting. The wrapper recorded rc=0,
52 seconds, peak RSS 0.67 GiB, minimum available 21,241 MiB on i265.

The corrected projection replays 16,104 JPEG cases from 244 origins; all 4,928
fully visible cases remain RGB8-identical. Both old and new image/scalar
identities are preserved. The 136 native preview comparisons match corrected
compatibility pixels and routing; strict native/WASM parity, EXIF permutations,
identity/malformed ICC and relaxed-first/unfused fallback browser checks pass.
See [the qualification and quality tails](../benchmarks/projection_fix_2026-11-10.md).

The focused encoder PR has independent plane/coefficient references, native
and cross-platform hash checks and libjpeg-turbo decoding. Its coverage and
remaining oracle limits belong to
[zenjpeg's report](https://github.com/imazen/zenjpeg/blob/a9c34a865424c0b33b5b313ce0eca49869c82fd6/benchmarks/encode_strip_padding_2026-11-10.md).

## Remaining native integration work

PR 261 is the concrete next integration task. Reconcile its adapter onto
current zenjpeg main, preserving the already-merged encoder fixes. Review the
new public API, then cover admitted pixel layouts, borrowed/owned rows,
`decode_into`, crop, arbitrary ICC profiles, cancellation/error propagation and
memory limits. Review repeated parsing/entropy decoding and allocations (`zensr`
now provides strided/`zenpixels` pixel writers, stack `CoefficientsView`,
`JpegDctBlocks` zigzag views and zero-allocation `RepairWorkspace` /
`ModelWorkspace` scratch reuse for `decode_into` integration).
Required-platform integration CI must pass before that draft is merge-ready.
The 136-case RGB8 comparison and identity ICC fixtures do not establish those
missing paths. Do not merge the conflicting historical branch merely to close it.

The preview intentionally pins zenjpeg `06d0e6f4` and engine `0bb29cf6`.
The compatibility/model-container path still pins historical zenjpeg `24b6d3fb`.
Neither pin tracks current zenjpeg main. See [the dependency graph](PRODUCTION_DEPENDENCIES.md)
and [API boundaries](API.md). Container loading has not migrated into the new
codec-independent engine. The independently pinned local/Git micro crates
also have distinct Cargo type identities; align sources or provide an explicit
bridge before sharing loaded models across these APIs. No release is implied
by the documentation.

## Quality and publication limits

The approved metadata-independent router's reported +2.4173 mean SSIM2,
0.5662% observed harm and 79.75% restoration coverage describe the previously
inspected 512-pixel cohort. Those scores predate both the binary16 weight-loader
and partial-block projection corrections. The full final router has not been
rescored with the corrected pixels, and this is not fresh held-out evidence.
The historical worst reserved loss is 18.5344 SSIM2.

The new projection study is forced repair, not the router's operating point:
33 old-to-new losses exceed 0.5 SSIM2; worst loss 2.41581. Whole-frame execution
supports arbitrary dimensions; resolution-generalized quality and deployment
latency remain unestablished. `Consistency::Certified` records projection
application, not exact re-encoding, final RGB8 bounds or perceptual safety.
The owner cancelled pixel-artifact auditing; this cleanup does not restart it.

The original small-bank JPEGs contain the old encoder tint, and its old repair
outputs contain the old projection behavior. They remain historical controls.
The fresh corrected SR witnesses isolate projection while retaining those same
JPEG inputs. SPANF/DSCF retain their measured small grayscale color bias; no
model-padding change or learned blend selector was adopted. Historical canon4
training exposure and [split provenance](SPLIT_DEFINITIONS.md) remain disclosed.

## Retained artifacts and reproduction

Use [model loading](MODEL_CONTAINER.md), [browser build/test recipes](BROWSER_DEMO.md),
[the small-bank tool contract](../tools/sr-bank/README.md) and the owning
benchmark reports. Model/data release previews remain unpublished drafts;
no crate or release was published by this cleanup.

The small bank is under `sr-bank-20261110/` in the retained zensr output tree.
Original bank/scalar archive: 7,749,263,360 bytes, SHA-256
`35689212375377ed1bd479c952db1248d52f769158872faa9f971b7e45b6f56d`.
Corrected projection/browser supplement: 188,467,200 bytes, SHA-256
`11e7b9fa581e47419ec22aded9eb66d1ac3b68fc2b951c2e9b6a9ddc41095c98`.
Transfer hashes, server synchronization and seeded member readbacks were
verified at generation time; local receipts and reports retain the identities.
No source images, candidates, scores, model files or failed probes were deleted.
Only superseded prose/API dumps were removed; Git history retains them.

### Corpus and dataset repository convention

Training, calibration and evaluation corpora follow the workspace `*-datasets`
separation pattern alongside `imazen-26` (and sibling `zensally-datasets`):
inference crates (`zensr`, `zensr-micro`) stay free of corpus dependencies, Git
tracks deterministic manifests, SHA-256 content identities and image-separated
split provenance ([split definitions](SPLIT_DEFINITIONS.md)), and bulk image or
candidate bytes live in dedicated dataset repositories (`zensr-datasets` / R2 /
retained `/mnt/v` mirrors) rather than inside library source trees.
