# Restoration dependency chain

Resolved from `Cargo.lock`, with default features checked on 2026-11-05. Reproduce with:

```sh
cargo tree --locked -p zensr-zenjpeg -e normal --depth 3
cargo tree --locked -p zensr-zenjpeg --features chooser -e normal --depth 1
cargo tree --locked -p zensr-micro-abi -e normal --depth 2
```

```text
application
└── zensr-zenjpeg 0.1.0
    ├── sha2 0.11.1 (single-file model checksums)
    ├── enough 0.4.5
    ├── zenanalyze 0.2.0 @ e676651e (default chooser feature)
    ├── zenanalyze-api 0.1.1 @ e676651e (default chooser feature)
    ├── zensr-micro 0.0.1
    │   ├── archmage 0.9.30
    │   └── magetypes 0.9.30
    └── zenjpeg 0.9.0 @ 24b6d3fb
        ├── zenyuv 0.1.3 @ 24b6d3fb
        ├── zencodec 0.2.0 @ 61fb96f1
        ├── zenmeta 0.1.0 @ 61fb96f1
        ├── zenpixels 0.3.1 @ f23ac753
        ├── ultrahdr-core 0.6.0
        │   ├── zencodec 0.1.26
        │   └── zenpixels 0.2.16
        └── pixel/SIMD support crates
```

`zensr-micro-abi` is a separate entry point into `zensr-micro`; it does not
provide the JPEG restoration pipeline. `zensr-micro` has no JPEG decoder,
PyTorch, ONNX, or GPU dependency. The f16 model weights are a separate artifact,
loaded into the runtime or [bundled with their policy](MODEL_CONTAINER.md); committing a model directory does not embed a default
model into the public library.

With `chooser` (now enabled by default), `zensr-zenjpeg` additionally uses `zenanalyze 0.2.0` and
`zenanalyze-api 0.1.1`, both resolved at `e676651e`. This feature is optional;
explicit coefficient-based Auto routing does not require it. The decoded-feature
default uses it for feature extraction, with frozen feature-definition identities
checked before inference. `--no-default-features` retains coefficient routing.
The root patches for
these two crates specify a git repository without a revision: the checked-in
lockfile fixes this checkout, but consumers resolving a fresh lockfile can
select a different commit. Use `--locked` for reproducible application builds.

Two zencodec and two zenpixels versions coexist in the default integration
graph because the pinned JPEG graph and ultrahdr-core use different versions.
This is an observed dependency duplication, not evidence that their types
interchange. No dependency versions were changed for the model swap.

`zensr-bench` is tooling, with a larger graph: tract-onnx, metric crates,
zenanalyze, additional image codecs, and command-line reference encoders. Those
are not dependencies of `zensr-micro` or the default `zensr-zenjpeg` entry point.

## Quality tier and inference engine

`Budget::Quality` selects the caller-loaded `AdoptedModel` in `Restorer`.
Both Quality and Realtime use `zensr-micro` and its tiled SIMD execution;
the quality tier does not switch to Tract. The adopted graph implementations
are Compact and SPAN-48. Tract serves the separate ONNX benchmark executable.
SSIMULACRA2 and Butteraugli evaluation are offline measurements, not part of
serving inference.

## Distribution constraints

The tree is not a crates.io release: `zensr-micro` is `publish = false`, and
JPEG integration uses git dependencies and root patches. The compatibility integration still pins the historical encode-record revision
`24b6d3fb`; it does not acquire later encoder fixes just because zenjpeg main
advances. In particular, that pin predates the padding correction merged in
zenjpeg PR 262. Application builds can use this locked source tree. A crate
release requires a separate dependency/publication plan and release approval.
The model artifact is 86,406 bytes of f16 weights; runtime code and dependencies
are additional and have not been size-measured in this task.

## Native decoder and browser integration — 2026-11-10

The review integration resolves separately from the compatibility tree above:

```text
browser repair binding / native codec caller
└── zenjpeg @ 06d0e6f4 (restoration feature; decoder/color/orientation)
    └── zensr @ 0bb29cf6 (routing, guards and repair policy)
        └── zensr-micro @ 0bb29cf6 (inference and corrected projection)
```

Pins are in `tools/browser-demo/repair/Cargo.toml` and
`tools/codec-integration/Cargo.toml`; resolve them with
`cargo tree --locked --manifest-path tools/browser-demo/repair/Cargo.toml -e normal`.
The shared projection correction adds no public signatures. The engine and
browser work merged in zensr PR 4 (`5042bccd`). The pinned zenjpeg adapter
remains draft PR 261 and conflicts with current main; it is a review preview,
not the current zenjpeg release. [Correction and validation](../benchmarks/projection_fix_2026-11-10.md).
