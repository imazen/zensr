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
loaded into the runtime; committing a model directory does not embed a default
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
JPEG integration uses git dependencies and root patches. The zenjpeg revision
is the encode-record work associated with its PR 253; this task does not alter
or merge that PR. Application builds can use this locked source tree. A crate
release requires a separate dependency/publication plan and release approval.
The model artifact is 86,406 bytes of f16 weights; runtime code and dependencies
are additional and have not been size-measured in this task.
