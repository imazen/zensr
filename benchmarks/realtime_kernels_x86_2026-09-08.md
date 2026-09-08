# Realtime tier on AVX-512: where the time is, and why Winograd is still off

First x86 kernel measurement for `zensr-micro` (the 2026-08-01 ARM audit was
aarch64). Host: this workstation — WSL2, **AMD Ryzen 9 7950X**, 32T, avx512
f/bw/cd/dq/ifma/vbmi/vl. Commit `ddb8901d977e`. zenbench, interleaved, paired.

**Read the tier label.** This bench previously hardcoded its arm as
`v3(avx2)` on every x86_64 host while the dispatch ladder
(`[v4x(cfg(avx512)), v3, ...]`) picked **v4x**. It is now probed at runtime and
prints `comparing v4x(avx512) vs forced scalar`. Every x86 reading of this bench
before 2026-09-08 named the wrong tier.

## 1. Kernels — conv3x3 is the one with headroom

`cargo bench -p zensr-micro --features internals --bench kernel_tiers`

| kernel | v4x(avx512) | forced scalar | ratio |
|---|---|---|---|
| `silu_dispatch` / 1M f32 | **222.7 ±4.6 µs** (17.5 GiB/s) | 2297.7 ±8.5 µs (1.70 GiB/s) | **10.3×** |
| `conv3x3_dispatch` / 32→32ch, 128×128 | **3.8 ±0.1 ms** (137 Melem/s) | 9.5 ±0.5 ms (55.0 Melem/s) | **2.5×** |

conv3x3 is 32·32·9·128·128 = 151 MMAC = **302 MFLOP in 3.8 ms → 79.5 GFLOP/s**,
roughly half of single-core AVX-512 FMA peak on this part. The 2.5× ratio
understates the SIMD win — the "scalar" arm is still autovectorized by LLVM at
baseline SSE2 — but the absolute figure is the one that matters, and it says the
convolution is where the remaining time lives. SiLU does not need work.

## 2. Winograd F(2×2,3×3) is still a loss, now confirmed on AVX-512

`wino.rs` cuts multiplies **2.25×** and is opt-in behind `ZENSR_WINOGRAD=1`
because v1 measured "1.8× slower ... scalar transforms + untier'd GEMM lose more
than the 2.25× multiply cut saves" (`adopted.rs:99`). That was recorded on a
different box. Re-tested end-to-end here with `prod_bench`, model
`dejpeg_rt24g`, 3 reps, `total = α + β·MP`:

| threads | direct conv3x3 | Winograd v1 | |
|---|---|---|---|
| 1 | `−13.2 + **1346.4**·MP` | `+26.9 + **2390.2**·MP` | **1.78× slower** |
| 12 | `+6.1 + **192.0**·MP` | `−35.0 + **333.1**·MP` | **1.73× slower** |

Per-size, 12 threads: 202.2 → 276.1 ms at 1024px, 816.7 → 1273.5 ms at 2048px,
3226.5 → 5577.4 ms at 4096px.

**The 1.8× verdict is robust** — it survives a different machine, a different
microarchitecture and the AVX-512 tier, landing within 4% of the original.

### What that implies for the fix

Winograd does 2.25× fewer multiplies and still comes out 1.75× behind, so its
non-multiply overhead currently costs about **3.9× what the multiply saving
returns**. That overhead is the transforms and the GEMM, and neither is tiered:
`wino.rs` is written as fixed `[f32; T]` blocks "so LLVM vectorizes the inner
loops" rather than through magetypes, and its `T = 16` was chosen as "2 AVX2
regs per row" — which on AVX-512 is exactly one zmm register, a shape it was
never tuned for.

So the optimization target is **tiering the Winograd transforms and GEMM with
magetypes**, not new intrinsics and not an archmage upgrade. Closing a 3.9× gap
is not guaranteed — but the transforms are currently scalar, where the measured
scalar→v4x factor on this box is 10.3× (silu) and the transform entries are
exact-in-f32 (0, ±1, ±0.5), so there is real room. Re-measure before believing.

## 3. archmage 0.9.29 is a correctness upgrade, not a speed one

0.9.29 adds `silu_midp()`/`sigmoid_midp()` and restores the fast `recip()`
lowerings. zensr's SiLU already computes `one / ((-v).exp_midp() + one)` —
exact division — and 0.9.29's `silu_midp` "keeps exact division internally so
saturated lanes stay exactly 0/1". Same arithmetic. Taking it replaces a
hand-rolled kernel with one differentially tested on every backend, and
obsoletes the `recip(inf) = NaN` workaround note, but no speed claim should be
attached to it without measuring.

## Reproduce

```
cargo bench -p zensr-micro --features internals --bench kernel_tiers
cargo build --release -p zensr-bench --bin prod_bench
ZENSR_PB_MODEL=dejpeg_rt24g ./target/release/prod_bench 3
ZENSR_WINOGRAD=1 ZENSR_PB_MODEL=dejpeg_rt24g ./target/release/prod_bench 3
```
