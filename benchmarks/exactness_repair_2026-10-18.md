# Exactness repair — 2026-10-18

Correctness record, no performance claim: nothing here was timed.

## The defect

The same input gave different output depending on thread count, tile size and
CPU tier. Final pixels from `restore_jpeg` (dejpeg7_graphics, q40 4:2:0 JPEGs
of gb82 `girl` and `city`; see
[the counterexample pointer](exactness_counterexamples_2026-10-18.pointer.md)):

- `girl`, AVX-512: threads 2 and 4 gave one RGB8 byte different from threads 1
  (x=296 y=525 R 202→203).
- `city`: AVX2 (v3) gave one byte different from AVX-512 (x=572 y=280 R 107→108)
  at every thread count.

At the model level, 135 of 311 tiled-vs-whole and tier-vs-auto comparisons on
the bundled models differed in raw f32 output, 22 of them in RGB8. That
includes span48, which has no PReLU.

## Causes, and what changed (`crates/zensr-micro/src/simd.rs`)

| path | before | after |
|---|---|---|
| conv3x3 overlapped final tile (`wd % W != 0`) | one FMA chain over every tap, while the main loop sums 16-channel blocks and adds them through `out`; for cin > 16 the tail's columns depended on the row width and on W | `tile_complete`: the main loop's per-block sequence, carried in registers |
| conv3x3 rows narrower than W | unfused scalar loop, different order | one zero-padded tile through `tile_complete` (`pwd >= 2W`) |
| conv1x1 last `plane % W` elements | scalar sum, unfused | one zero-padded vector tile, same FMA chain |
| SiLU / gate last `len % W` | libm `exp` instead of `exp_midp` | zero-padded vector, same formula |
| PReLU last `plane % W` | `f32::max`/`min` | zero-padded vector, same formula |
| PReLU and the exp clamp, NaN operand | magetypes `max`/`min`: x86 returns the other operand, wasm128 and NEON return NaN (wasm128 PReLU(NaN) = NaN, x86 and scalar 0) | `max_x86`/`min_x86`: compare-and-select with x86 `maxps`/`minps` semantics on every backend |
| every FMA | `mul_add`, which rounds twice on the scalar and WASM backends | `mul_add_portable`: the same FMA on x86 v3/v4 and NEON, a one-rounding software FMA on scalar and WASM |
| SiLU / gate exp | magetypes `exp_midp` (unfused polynomial on scalar) | `exp_midp_portable`: the same algorithm with one-rounding steps |

What the repair leaves unchanged:
- The main tile loop on the FMA tiers still produces the same bits. Every case
  in the real-model matrix whose width is a multiple of 16 (so every layer runs
  only the main loop) is byte-identical to the build before the repair: 20 of
  72 cases, all seven models, span48 included.
- The cases that changed are exactly those that used a tail, a narrow row or a
  remainder.

## Oracles and regressions

`crates/zensr-micro/src/simd/exactness.rs` calls the generated per-tier
functions with summoned tokens (no process-wide token flags):

- **conv3x3:**
  - Every output must equal the same kernel run on the input zero-extended to a
    multiple of 16 columns (so only the main loop runs on any tier), cropped
    back. Zero extension is the conv's own border padding, so each kept column
    sees the same receptive field.
  - Each tier's oracle must equal the scalar tier's.
  - Shapes: cin 3..64, cout 4/8/24, h 1/3/5, widths 1..49 including every tail
    and narrow case.
- **Receptive-field counterexample:** columns 0..=30 of 32- and 36-wide rows,
  and 0..=10 of a 12-wide row, must be identical for cin 8/16/17/24/64.
- **conv1x1:** remainders against full tiles and across tiers.
- **SiLU, gate, PReLU:**
  - remainders against full vectors and across tiers
  - inputs include ±0, subnormals, ±MIN_POSITIVE, ±MAX, ±inf and NaN, plus
    slopes ±0
  - NaN is compared as NaN; Rust and the backends do not specify its payload
- **`exp_midp_portable`:**
  - bit-identical to magetypes `exp_midp` on the FMA tiers
  - equal on every tier
  - checked on the special values plus 200,000 inputs over [-100, 100]
- **Adopted models:**
  - procedural compact nf 4/12/24/64, scales 1–4
  - procedural span48 ×2/×4
  - the tiled runner at (threads, tile) (1,32), (3,32), (2,40), (1,64), plus
    auto tiles (4,0) and (16,0) at 37×53, must equal the whole-image forward
    bit for bit

All six fail on the unrepaired kernels (scratch copy of the dependency commit
plus this file, with `exp_midp_portable` shimmed to `exp_midp`). Examples:
- `Scalar cin=3 cout=4 h=1 wd=1: element 2 0xbc1b7cfc != main-loop 0xbc1b7cf8`
- `compact nf24 nc2 s1 37x53 threads=1 tile=32: element 78 2.2897841e-1 != whole 2.2897843e-1`

All six pass on scalar, v3 and v4x, and on v4 with `--features tier_v4`;
under qemu user mode on aarch64 (NEON tier, emulated) all six pass; under
wasmtime with `+simd128` (wasm128 tier) the five kernel tests pass (the model
test needs threads, which wasip1 lacks). wasmtime is what found the NaN case:
wasm128 PReLU(NaN) gave NaN where x86 and scalar give 0, now `max_x86`/`min_x86`.

## Real-model matrix (`tools/exactness/matrix.py`)

Built from the repaired tree; the worker is `tools/coz-profile`.
- Models: all seven adopted models (rt24g, dejpeg7_graphics, people_rtc_2x,
  nomosuni_compact_2x, general_x4v3, animevideo_x4v3, nomosuni_span_4x).
- Shapes: 1×1 up to 1024×1024, including 997×17, 17×997, 48×4096 and 48×4093.
- Configurations: whole forward plus tiled at threads 1/4/8, on tiers auto/v3,
  and scalar on small inputs.

Result: **72 cases, 656 runs, 0 inconsistent.**

## Shipped pipeline (`tools/exactness/product.py`)

`restore_jpeg` on the two counterexample JPEGs, with dejpeg7_graphics and
dejpeg_rt24g:
- tiers auto and v3 at threads 1/2/4/8/12/16
- the scalar tier at threads 16, 12, 8, 4, 2 and 1 (threads 1–12 on the
  probe build before the NaN-semantics change, which changes no output in the
  real-model matrix: 0 of 72 cases differ)

Every output of each (JPEG, model) pair has one SHA-256, recorded in
`tools/exactness/product-expect-2026-10-18.json`:

| input | model | RGB8 sha256 |
|---|---|---|
| girl | dejpeg7_graphics | `85b6201c467522851d6929c7768c36f40ec30376e85325e783e749bb9b42b973` |
| city | dejpeg7_graphics | `b293bdd06245d3b7dbd99c1366a8d088fb34fcf7b0a5e853e2a1c7f68c5720e8` |
| girl | dejpeg_rt24g | `6ca613b8445d631a15cbd0cef15e6190f02588b050c2f497261a71b6225c3808` |
| city | dejpeg_rt24g | `3bc033a92611e96ffdec65e768b03e2485aba802606a38de394ca785cc4c40a4` |

`zensr-adopted-verify` (torch goldens, 5e-4) reports the same result before
and after: the same four checks fail (dejpeg11_teacher 40x36/17x13,
dejpeg12_teacher_big 40x36/17x13, max 8.6e-4 to 1.46e-3, identical values in
both trees); every other check passes, its max difference moving in the third
significant digit (for example animevideo_x4v3 40x36 2.66e-4 → 2.67e-4,
dejpeg2_auto 17x13 3.63e-5 → 3.68e-5), because the 40x36 and 17x13 goldens use
widths that are not multiples of 16.

## Not covered, or still inconsistent

- **Winograd (`ZENSR_WINOGRAD=1`, opt-in, off by default) is still not
  consistent within its own mode.** On the repaired build:
  - animevideo_x4v3 150×140, tiled vs whole: 77,305 f32 values and 4 RGB8
    values differ
  - nomosuni_compact_2x 33×19, v3 vs auto: 2 RGB8 values differ
  - tier disagreements in 14 of 68 cases
  
  Its borders and buffers under one vector fall back to the scalar `wino.rs`
  path, so a pixel's sequence depends on where tile borders fall. Aligning it
  needs that fallback redesigned; not done here.
- NEON and WASM ran only under emulation (qemu user mode, wasmtime); no ARM
  hardware or browser run. Windows (x64, ARM64) was compiled, not run.
- The scalar tier and WASM now compute every multiply-add with a software FMA,
  so they are slower than before; not measured.
- The per-tier golden digest that `docs/SHIPPING-METHODOLOGY.md` asks for is
  still missing.
