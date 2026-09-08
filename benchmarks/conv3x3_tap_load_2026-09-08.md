# conv3x3 tap load: two loads plus a funnel shift, and one cache line of skew

2026-09-08. Host `dev` (WSL2 on a **7950X**, 16C/32T — not the 9950X3D; the
hostname is shared). `cargo build --release`, no `-C target-cpu=native`.
Measured with `examples/conv_probe`, which runs the dispatching kernel only,
and `perf stat` for the counters. Raw data: `conv3x3_tap_load_2026-09-08.tsv`
(the shipping A/B) and `conv3x3_tap_load_arms_2026-09-08.tsv` (the three-arm
run that separated the two effects).

Every comparison is **paired and interleaved**: within one repetition each arm
runs back to back, and the reported figure is the median of the per-repetition
differences with the win fraction beside it. A difference of medians across
separate runs would not survive this box's drift.

## What changed

The 3x3 kernel needs three W-lane vectors at `x-1`, `x`, `x+1`. It used to load
all three. It now loads two vectors a full W apart and derives the middle and
right taps with `magetypes`' `concat_shift`, the cross-vector funnel shift
(imazen/archmage#111).

**Which instruction that becomes is LLVM's choice, and it is not the obvious
one.** Disassembling this binary: the AVX-512 path emits **`vpermt2ps` +
`vpermt2pd`** with both mask registers hoisted out of the loop — not the
`valignd` the intrinsic names, and `vpermt2pd` because LLVM noticed the N=2
dword shift is a qword shift. The AVX2 path emits one **`vperm2f128`** CSE'd
across both shifts, then `vshufps` x2 and `vshufpd` — not the `vpalignr` the
source asks for. Both are one instruction per shift inside the loop, which is
the property that matters; the mnemonic is not a contract, and archmage's
`scripts/verify-asm.sh` gates the absence of a scalar lane gather instead.

That change forces a second change: the second load runs off the end of the row
at the last tile, so the padded row needs trailing slack. `pwd` went from
`wd + 2` to `wd + 18` — the two zero columns plus exactly one 64-byte cache
line.

## Result (10 paired reps, c=32)

| tier | size | before | after | median Δ | wins |
|---|---|---|---|---|---|
| v4x (AVX-512) | 128 | 126.0 | 127.5 | +1.5% | 6/10 |
| v4x | 256 | 124.8 | 128.1 | +2.3% | 9/10 |
| v4x | 512 | 122.6 | 128.6 | **+4.7%** | 10/10 |
| v3 (AVX2) | 128 | 82.7 | 86.5 | +4.9% | 8/10 |
| v3 | 256 | 81.8 | 92.1 | **+11.2%** | 10/10 |
| v3 | 512 | 76.3 | 90.7 | **+19.5%** | 10/10 |

GFLOP/s. Checksums are identical to the previous kernel in every cell, and the
tiers agree with each other and with scalar (`arbitrary_dims_simd_vs_scalar`,
plus a new `arbitrary_dims_v3_matches_scalar` that disables the AVX-512 tokens
so the AVX2 arm is exercised rather than shadowed).

## End to end, on the full production pipeline

`prod_bench` (restore_jpeg -> guarded x1 model -> S10 projection -> RGB, then
the chained x2 SR step) on turbo q75 4:2:0, 3 interleaved paired reps, raw log
`conv3x3_tap_load_e2e_2026-09-08.log`. **Every one of the 27 cells is a win.**

| stage | 64px | 256px | 1024px | 2048px | 4096px |
|---|---|---|---|---|---|
| restore, 1 thread | +4.6% | +9.1% | +11.0% | +12.6% | — |
| restore, 12 threads | +5.5% | +6.4% | +6.3% | +9.2% | +10.2% |
| sr_x2, 1 thread | +3.5% | +7.6% | +8.7% | +10.1% | — |
| sr_x2, 12 threads | +7.8% | +9.7% | +5.0% | +9.4% | +7.9% |
| chain, 1 thread | +4.1% | +10.5% | +9.9% | +11.5% | — |
| chain, 12 threads | +5.6% | +7.4% | +5.7% | +8.3% | +8.3% |

3/3 paired wins in every cell. (4096px at 1 thread is skipped by the harness on
cost/benefit.) The gain rises with size and is smaller at 12 threads, which is
what a kernel-level change looks like once thread scaling absorbs part of it.
conv3x3 is 91.3% of this pipeline, so a kernel gain of +5..+20% arriving as
+4..+13% end to end is the expected pass-through.

## The isolated harness predicted the wrong tier

`examples/tap_load_probe` prices the two formulations in a bare tap loop and
said **+13% on AVX-512** (125.1 -> 141.9 GFLOP/s). In the real kernel the
AVX-512 gain from the load change alone is **+0.8%, 3/5** — noise. The tier
that actually gains is AVX2, which the probe never measured.

The probe is not wrong about what it measured; it measured a loop with no
channel blocking, no output stores and everything resident. Read it as an upper
bound on one effect in isolation, never as a prediction for the kernel.

## The two changes work at opposite ends, and the skew does the heavy lifting

Separating them (three-arm run, 7 paired reps) against the committed kernel:

| tier | size | stride only | load count only | both |
|---|---|---|---|---|
| v4x | 128 | −3.7% 0/5 | +0.9% 3/5 | −0.5% |
| v4x | 512 | +1.0% 3/5 | +0.8% 3/5 | +1.8% |
| v3 | 128 | −4.3% 1/5 | **+16.0% 5/5** | +11.1% |
| v3 | 512 | +5.7% 5/5 | **+11.6% 5/5** | +18.0% |

and `perf stat` says why (c=32, 120 iters):

| cell | arm | GFLOP/s | L1 loads | L1 misses | miss rate |
|---|---|---|---|---|---|
| v3 512 | 3 loads, `wd+2` | 76.7 | 71.0e9 | 5.25e9 | 7.39% |
| v3 512 | 2 loads, `wd+2` | 76.6 | 65.6e9 | 6.81e9 | **10.37%** |
| v3 512 | 2 loads, `wd+18` | 92.4 | 65.6e9 | 2.63e9 | 4.01% |
| v4x 512 | 3 loads, `wd+2` | 126.2 | 39.5e9 | 4.33e9 | 10.97% |
| v4x 512 | 2 loads, `wd+2` | 124.7 | 35.9e9 | 6.25e9 | **17.39%** |
| v4x 512 | 2 loads, `wd+18` | 129.0 | 36.0e9 | 3.04e9 | 8.43% |
| v3 128 | 3 loads, `wd+2` | 84.7 | 4.6e9 | 0.18e9 | 3.95% |
| v3 128 | 2 loads, `wd+18` | 91.3 | 4.2e9 | 0.23e9 | 5.38% |

Three findings, none of which were the expected one:

1. **The load-count change issues fewer loads but touches MORE cache lines.**
   Three loads at `x`, `x+1`, `x+2` live inside one W-window and hit at most two
   lines; two loads a full W apart span two windows. At 512px that raises the L1
   miss rate by half (7.4% -> 10.4% on v3, 11.0% -> 17.4% on v4x) and eats the
   entire instruction-count saving. **On its own it is a wash at large sizes.**
2. **The one-line skew halves the miss rate** — below the original, on both
   tiers — and that is where the large-size win comes from. `wd = 512` floats is
   exactly half a 4 KiB page, so a `wd + 2` stride puts rows 0 and 2 of every
   tap group in the same L1 set; one line of skew separates them. The mechanism
   is a hypothesis; the miss counts are not.
3. **At 128px the miss rate goes UP and the kernel still gets faster** (v3 84.7
   -> 91.3). The working set is small enough that L2 absorbs it, so the load
   count is what is left to win.

Instruction count moved the wrong way throughout: 132.5e9 -> 139.7e9 on the v3
512 cell, while cycles went 47.7e9 -> 37.4e9 and IPC 2.78 -> 3.73. Counting
instructions would have rejected this change.

## What was rejected

- **Slack on the allocation instead of the stride** (`pwd` stays `wd + 2`, the
  buffer carries the extra floats, rows read a little into the next slot). It
  keeps the small-size behaviour — v4x 128px +0.7% against −0.5% — but throws
  away the whole large-size win: v3 512px **−1.1%, 3/7**. Cheaper in memory,
  and the memory was never the problem.
- **`pwd = wd + W`** (tier-dependent slack, the minimum the second load needs).
  Correct but worse than one flat cache line at every size except v3 128px:
  v4x 512px +0.8% against +4.5%, v3 512px +16.2% against +20.3%.
