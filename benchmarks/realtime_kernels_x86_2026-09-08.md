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

### What that implies — and a correction

**First reading of this was wrong and is retracted.** I read `wino.rs`, saw
fixed `[f32; T]` blocks relying on LLVM autovectorization, read
`adopted.rs`'s comment ("kept for a future vectorized v2"), and concluded the
remaining win was to tier the transforms and GEMM with magetypes.

That work already exists. Commit `b714829` — *"Winograd v2/v3 —
magetypes-tiered kernel (deinterleaved vector transforms, quad GEMM, row-wide U
amortization)"* — lives in `simd.rs::conv3x3_wino_dispatch`, and it is the path
`adopted.rs` calls under `ZENSR_WINOGRAD=1`. `wino.rs` is only the scalar
fallback, taken when `h < 4 || wd < 4` or when the tile count `nt` is under one
vector width. At the sizes measured here (wd ≥ 1024 → nt = 511, W = 16) the
**tiered** kernel is what ran.

So the honest result is stronger than "needs tiering": **vectorizing the
transforms did not close the gap.** Winograd does 2.25× fewer multiplies and is
still 1.75× slower with a fully tiered implementation, which means the transform
plus scatter/gather overhead — not the multiply count, and not the ISA — is what
decides this at nf=24. That is a negative result to keep, not a TODO.

The remaining conv3x3 headroom (79.5 GFLOP/s against ~half of single-core FMA
peak) is therefore a blocking/scheduling question inside the direct kernel, not
a tiering question. The hot path is already tiered end to end.

## 3. Where conv3x3's time actually goes, and a bit-exact +6%

Profiled with `perf` on an isolated loop (`examples/conv_probe`, 32→32ch
128×128, v4x tier) rather than reasoned about. Two hypotheses died immediately:
the weight splats **are** folded into AVX-512 embedded broadcasts
(`vfmadd231ps (%rbx,%rdi,4){1to16},%zmm9,%zmm8`), and the kernel is nowhere near
memory-bound — arithmetic intensity is ~75 FLOP/byte. IPC is 3.96, but only
~12.8% of retired instructions were FMAs.

| instruction class | before | after |
|---|---|---|
| vector (packed) | 53% | **62%** |
| **branch/compare** | **28%** | **2%** |
| integer/addressing | 12% | 27% |
| scalar float (border path) | 7% | 2% |

The branches were per-(ic,ky)-per-tile indexing — `rowtab[ic*3+ky]`, re-slicing
`wts[o..o+12]`, and three `from_slice` calls each doing
`slice[..W].try_into().unwrap()` — about five checkable operations per twelve
FMAs. Fixed by building the (row, weights) pairs once per output row and walking
them by iterator, plus one checked window per tap from which l/m/r are constant
sub-ranges of a fixed-size array.

**Paired, interleaved, 5 pairs of 1500 iterations each** (single runs are not
enough — the run-to-run spread on this box is ~3.4%, comparable to the effect):

| pair | old GFLOP/s | new GFLOP/s |
|---|---|---|
| 1 | 84.3 | 89.7 |
| 2 | 86.5 | 89.1 |
| 3 | 84.9 | 89.4 |
| 4 | 83.2 | 88.6 |
| 5 | 82.7 | 91.4 |

Median **84.3 → 89.4 GFLOP/s, +6.0%**, 5/5 pairs positive.

**Bit-exact**, which was the gate rather than the speed: iteration order is
unchanged, so FP accumulation order is untouched. `conv_probe`'s checksum is
identical (−82.456), all 16 zensr-micro tests pass, and `zensr-verify` PASSes
with golden deltas unchanged (5.364e-7 / 1.192e-7 / 2.027e-6). This kernel ships
in a binary; a reordering would have silently changed users' pixels and
invalidated every cached derivative.

### Accumulator widening: 8 chains, and why not 12

Authorised to reorder FP addition, so the four-accumulator dependency chain was
widened. **The chain count is a cross-tier decision, not a throughput knob.**

Measured paired and interleaved, per tier:

| chains | AVX-512 (v4x) | AVX2 (v3) |
|---|---|---|
| 4 (original) | ~88.7 | ~63.4 |
| **8** (shipped) | **~93.5** (+5.4%) | **~66.2** (+4.4%) |
| 12 (one per tap) | ~94.2 (+6.2%) | **~55.3 (−12%)** |

Twelve chains — one per tap — is fastest on AVX-512 and a **12% regression on
AVX2**: 12 accumulators plus 3 loads plus a broadcast temp exceed the 16 ymm
registers, and AVX2 has no embedded broadcast to fold the weight operand the way
AVX-512's `{1to16}` does. Eight chains ((l,m) share one per output, r gets its
own, longest run 2 instead of 3) fits both register files and captures nearly all
of the AVX-512 gain.

We ship one binary and the **user's CPU picks the tier**, so a change that helps
the newest hardware and penalises everything older is not an optimization. This
is the concrete case the per-tier gates in `docs/SHIPPING-METHODOLOGY.md` exist
to catch, and it would have been invisible measuring only the fastest tier —
which, before the label fix above, is exactly what this bench appeared to do.

Cumulative on AVX-512, from the kernel as it stood before this session:
**84.3 → ~93.5 GFLOP/s, +10.9%.**

No golden regeneration was needed. The reorder moves `golden 17x18` from
2.027e-6 to 1.907e-6 against a 1e-3 gate, all 16 tests pass, and `zensr-verify`
PASSes — the goldens are tolerance-based, not bit-exact. Cross-tier determinism
is preserved because the per-pixel accumulation sequence does not depend on the
vector width; only the lane count does.

## 4. The real bottleneck at production sizes was the TLB

Everything above was measured at 128x128, which is the size the kernel bench
uses — and it is the one size where the kernel is compute-bound. Sweeping sizes
told a different story:

| size | AVX-512 GFLOP/s (as of §3) |
|---|---|
| 128px | 94.5 |
| 256px | 104.6 |
| 512px | 73.6 |
| 1024px | **29.7** |

Throughput collapses by 3.5x. Not DRAM bandwidth — the kernel needs well under
1 GB/s at these sizes. **It is the TLB.** In the planar layout the `cin*3` input
rows a tile needs are `cs` floats apart, so at 1024px they sit on `cin*3 = 96`
distinct pages, against Zen 4's **64-entry L1 dTLB**. Every tile evicts the whole
TLB.

Measured page walks per MFLOP: **0.019 at 256px, 13.3 at 1024px** — a 700x
increase. And varying only the channel count at 1024px puts the cliff exactly at
the TLB boundary:

| cin | pages per tile | GFLOP/s |
|---|---|---|
| 8 | 24 | 99.6 |
| 16 | 48 | 100.8 |
| 32 | **96** | **48.9** |

### Two fixes, both bit-exact in effect and large

**Loop order** (`oy` outer, output quad inner). The old nesting walked every row
for one quad before returning to row 0 for the next. Bit-exact — output writes
are disjoint per `(oc0, oy)`.

**Channel blocking** (`IC_BLOCK = 16`, so 48 pages per tile). Partial sums cross
blocks through `out`, which stays in L1 for the row. One trap, caught by
`arbitrary_dims_simd_vs_scalar` at 8x19: the overlapped final tile recomputes
columns the main loop already wrote, which is idempotent when it *overwrites* but
**double-counts every block after the first** when it accumulates. It now runs
once, outside the block loop, over the full tap set — one tile per row, so its
TLB cost is nil.

| size | original | + loop order | + channel blocking |
|---|---|---|---|
| **AVX-512** | | | |
| 128px | 94.5 | 95.5 | 94.7 |
| 256px | 104.6 | 106.7 | 106.5 |
| 512px | 73.6 | 95.6 | **108.0** |
| 1024px | 29.7 | 46.3 | **100.1** |
| **AVX2** | | | |
| 128px | 68.9 | 69.0 | 69.2 |
| 512px | 67.2 | 65.8 | **72.3** |
| 1024px | 25.4 | 35.4 | **68.4** |

At 1024px: **AVX-512 +237%, AVX2 +169%.** Throughput is now flat across sizes
instead of collapsing.

### The bench was measuring the right size after all — correction

I first wrote that the 128px bench "was hiding" this and would "approve the wrong
kernel". **That overstated it, and the correction matters more than the original
claim.** `restore_jpeg` runs the model through `upscale_tiled`, whose default
`tile` is **128** — so production convolves 128×128 tiles (plus halo), never a
whole 1024px plane. 128px is the *representative* size, not the wrong one.

So the honest accounting of these two cache fixes:

- On a **whole plane** (the shape `conv_probe` measures, and what a caller using
  the untiled path gets) they are worth up to **+237%**.
- On the **tiled production path** they are worth ≈0%, because 128px tiles never
  reach the TLB cliff.
- The measured end-to-end gain — `restore` at 1024px/12T, **202.2 → 168.3 ms,
  −16.8%** — comes from the *compute-bound* work instead: bounds-check hoisting
  (+6%) and 8 accumulator chains (+5.4%), which together are +12.3% at 128px.

The size sweep is still worth having, for two reasons that survive the
correction. It documents a real cliff that anyone raising the tile size or using
the untiled path would fall into; and removing that cliff is what makes a larger
tile *viable*, which is the actual production lever — at tile=128 with halo=10
each tile computes (128+20)²/128² = **1.34× the pixels it keeps**, so 34% of the
work is discarded halo. 256px would cut that to 16%.

`examples/conv_probe` takes size and channel-count arguments and `kernel_tiers`
now sweeps 64/128/512, so both regimes stay visible.

### What is left

At 89.4 GFLOP/s the kernel is at ~57% of single-core AVX-512 FMA peak. Accumulator widening is done (above). At ~93.5 GFLOP/s the kernel sits at
**~55% of Zen 4's 32 FLOP/cycle** AVX-512 peak, up from 51%. Instructions per
iteration halved across the two changes (73.6M → 35.6M) and FMAs went from 12.8%
to 26.5% of retired instructions; IPC fell 3.96 → 2.07, which is the expected
shape when cheap integer work is removed and what remains carries real latency.

The next bottleneck is **loads**, not arithmetic: the hottest single instruction
is now `vmovups (%rdx,%rdi,4),%zmm16` at 8.3%. Two of the three taps (x−1 and
x+1) are 4-byte-misaligned by construction, so a 64-byte load crosses a cache
line on every one. The standard fix is to load aligned and synthesise the
shifted vectors with `valignd`/shuffles, trading two misaligned loads for two
permutes — untested here.

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
