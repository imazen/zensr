# Exactness tools

Every output pixel must have one value whatever the thread count, tile size or
CPU tier. These tools check that on real models and on the shipped pipeline;
the kernel-level oracles live in `crates/zensr-micro/src/simd/exactness.rs` and
run with `cargo test`.

## Real-model matrix — `matrix.py`

Runs each adopted model through the `tools/coz-profile` worker (deterministic
synthetic input, raw f32 dump) as a whole-image forward and as the tiled runner
at threads 1/4/8, on tiers `auto`, `v3` and `scalar`, and requires the same
bytes from every configuration of a (model, shape). Disagreements are reported
with the first differing element and with the shipping RGB8 quantizer applied.

```sh
cargo build --release --manifest-path tools/coz-profile/Cargo.toml
python3 tools/exactness/matrix.py --bin tools/coz-profile/target/release/zensr-coz-profile \
    --out ~/tmp/exact-matrix-1
```

- The scalar tier computes each multiply-add with a one-rounding software FMA,
  so it runs only on small inputs (up to 64×64 for nf ≤ 24, 17×13 otherwise).
- `--winograd` sets `ZENSR_WINOGRAD=1` (the opt-in algorithm); its results are
  compared within that mode only.
- `--compare-bin` also runs every configuration on a second worker (for example
  a baseline build) and reports, per configuration, whether it gives the same
  bytes. This is informative and does not change the exit status.

Models are read from `models/adopted/<dir>/weights_f16.raw`. Only
`dejpeg_rt24g` is tracked in git; the others are local adoption dumps, and a
missing one is an error, not a skip (choose with `--models`).

## Shipped pipeline — `product.py` and `product/`

`product/` is a standalone crate (the product dependency graph is unchanged).
It restores JPEGs with `zensr_zenjpeg::restore_jpeg` (guarded model,
quantisation-consistency projection, RGB8) at several thread counts, with the
default automatic tile. `product.py` runs it on each tier and requires every
RGB8 output of a (JPEG, model) pair to be byte-identical.

```sh
cargo build --release --manifest-path tools/exactness/product/Cargo.toml
python3 tools/exactness/product.py --bin tools/exactness/product/target/release/zensr-exactness-product \
    --out ~/tmp/exact-product-1 \
    --jpeg /mnt/v/output/zensr/exactness-2026-10-18/inputs/girl-576x576-q40-420.jpg \
    --jpeg /mnt/v/output/zensr/exactness-2026-10-18/inputs/city-576x576-q40-420.jpg \
    --model q7=models/adopted/dejpeg7_graphics/weights_f16.raw:64:16 \
    --model rt24g=models/adopted/dejpeg_rt24g/weights_f16.raw:24:8 \
    --expect tools/exactness/product-expect-2026-10-18.json
```

The two JPEGs are the counterexamples that exposed the defect; their
provenance and hashes are in
[the pointer](../../benchmarks/exactness_counterexamples_2026-10-18.pointer.md).
`ZENSR_EXACT_TIER` forcing is implemented for x86_64; elsewhere the probe runs
the native tier only.
