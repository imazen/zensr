# Compact-model causal profiling

This standalone development crate profiles the existing `zensr-micro` runtime.
It is outside the benchmark workspace's codec dependency graph and uses the
same pinned archmage implementation as the runtime workspace. It adds no
instrumentation or dependencies to production crates.

Build with `just coz-build`. The release profile includes DWARF line tables
and thin LTO. Install [Coz](https://github.com/plasma-umass/coz) separately;
the Rust counters use [coz-rs](https://github.com/plasma-umass/coz/tree/master/rust).

The executable accepts:

```text
weights_f16.raw nf nc scale height width iterations threads [raw-output-path]
```

Use the exact architecture values in the model's `meta.json`. Only compact
models are supported here. `threads=0` measures whole-image `forward`;
positive values measure `upscale_tiled` with a 128-pixel tile size. The input
is deterministic synthetic CHW RGB in [0,1], with seed `7b19e493`, noise,
and alternating blocks. This is a runtime-cost probe, not a quality dataset.
Weights are read without modification. Model parsing and input generation
happen before the progress loop; the called inference operation retains
its normal allocations. Output is consumed through `black_box` each iteration
and checked for nonfinite values after the loop.

For example, using the existing realtime model:

```sh
just coz-run profile.jsonl models/adopted/dejpeg_rt24g/weights_f16.raw 24 8 1 64 64 500 0
```

`COZ_BIN` can select an installed profiler executable. Wrap builds and runs
in the resource limits required by your environment. Start with a cheap probe;
confirm source samples, experiments with selected samples, and completed
progress before scaling the run. Coz predictions need actual interleaved A/B
measurements before an optimization is accepted.

The printed fingerprint is a quick diagnostic. For exact pre/post validation,
pass a new raw output path, then compare the complete little-endian f32 files
with `cmp` and record their SHA256 hashes. The writer refuses to replace an
existing file. Validate each applicable model, size, thread mode, and SIMD
tier separately. This tool does not establish cross-tier numerical parity.
