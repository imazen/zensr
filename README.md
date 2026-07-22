# zensr — SR-on-CPU experiments (bench harness + micro-runtime)

Feasibility work from 2026-07-22 (see `~/work/zen/SR-MODELS-RUNTIMES-2026-07-21.md` for the
full model/runtime survey). Two crates:

- **`zensr-bench`** — tract-onnx CPU bench for fixed-shape SR ONNX exports
  (`tools/export_ntire25.py` exports NTIRE2025_ESR zoo checkpoints from
  `~/work/superrez/NTIRE2025_ESR`; ONNX files land in gitignored `models/`).
- **`zensr-micro`** (+ `zensr-micro-abi` cdylib) — hand-rolled SPANF (fc=32, x4) inference in
  safe Rust; FFI unsafe isolated in the -abi crate (zentract-abi pattern). Op set: grouped/plain
  conv3x3, conv1x1, SiLU, sigmoid gate, concat, PixelShuffle(4).

## Measured results (2026-07-22, this box, single-thread)

| Thing | Number |
|---|---|
| SPANF x4 via tract 0.23.4 | 3.27 MP-out/s (128², 80 ms) · 3.33 (256², 315 ms) |
| TSR / EFDN / NanoSR via tract | 2.0 / 1.1 / 1.25 MP-out/s (128²) |
| zensr-micro-abi cdylib, scalar-only | 271,432 B (265 KB) |
| **zensr-micro-abi cdylib, full SIMD dispatch** | **381,816 B (373 KB)** (v4x AVX-512 + v3 AVX2 + mandatory scalar fallback; `tier_v4` feature off by default; `incant!` always links scalar) |
| correctness vs PyTorch golden (every tier v3/v4/v4x) | max_abs 8.6e-6 (PASS) |
| **f16 weights (297,696 B)** | **59.2 dB ramp / 75–76 dB photo PSNR vs fp32 — TRANSPARENT, ship it** |
| int8-pc weights (152,400 B) | 16.8 dB ramp / 35–36 dB photo — **NOT viable** (see below) |
| SPANF weights fp32 | 593,152 B |

Full grids: `benchmarks/tract_cpu_2026-07-22.tsv`, `benchmarks/quant_accuracy_2026-07-22.tsv`.

## Weight-compression verdicts (tools/quant_accuracy.py, torch sim + Rust decoders agree exactly)

- **f16: production-safe.** Worst case 0.3% max-error on photos, 2.5% of pixels shift ±1 u8 step
  (75-76 dB). Rust decoder (safe bit-twiddle, subnormal-correct) in `decode_f16_weights`.
- **int8 post-training quantization: broken for SPANF** (35 dB photo, 90%+ u8 mismatch on ramp).
  The gate-exemption probe (quantize all but the σ-gate-feeding convs) changed nothing —
  sensitivity is distributed across the conv chain, so mixed-precision doesn't rescue it.
  int8 for SPAN-class requires QAT (MAI recipe) or an int8-by-design arch (ABPN/ECBSR/ETDS).
- int6/int4/k-means-256 codebook: 5-21 dB, dead on arrival.

## SIMD implementation notes

- One `#[magetypes(v3, neon, wasm128, scalar)]` family (f32x8) + hand `#[arcane]` `_v4x` (and
  feature-gated `_v4`) on native 512-bit f32x16; kernels macro-instantiated per width; whole
  forward inlines into one target_feature region per tier; `incant!` dispatches once per call.
- **magetypes gotcha (root-caused here): `recip()` is a Newton-refined approximate reciprocal —
  `recip(inf) = NaN` (inf·0 in the refinement step). Sigmoid after `exp_midp` saturation MUST use
  exact division.** Real SPANF pre-activations reach ~±100, so exp(-v) overflowing to inf is by
  design, not an anomaly. Symptom was 97% NaN output while `f32::max`-based diff stats looked
  clean — always NaN-check in diff stats.
- Archmage tier map (current, 0.9.28): v3 = AVX2+FMA (native 256), v4 = AVX-512 base,
  v4x = AVX-512 extended; `F32x8Convert` (transcendentals bound) is NOT implemented for v4/v4x
  tokens — use f32x16 there.

## Reproduce

```sh
just export      # NTIRE25 ckpts -> models/*.onnx (needs torch; box GPU used by SPANF init)
just dump        # SPANF weights + golden -> models/*.raw
just bench       # tract grid -> benchmarks/
just verify      # golden-gate zensr-micro + timing
just size        # build + report micro-abi cdylib size
```

## Next steps (QUIET-BOX perf pass — blocked on machine contention 2026-07-22 night)

The box ran at load 23-38 all evening (backup rsync + fleet jobexec + 2 other sessions);
128² timings came out "slower" than 256² — incoherent, so per benchmarking discipline NO
speed conclusions were drawn for the SIMD kernels. Three kernel structures are committed
(scratch-row, register-acc, packed+register-acc — all golden-clean); on a quiet box:

1. zenbench shootout: micro (3 structures × v3/v4x) vs tract vs scalar, sizes 64-512,
   α+β·pixels fit. cargo-bloat attribution of the 373 KB.
2. **Plane-stride padding**: at 128² the 64 KB power-of-2 plane stride maps all 112 conv1x1
   input streams to the same L1 sets — pad plane stride by +16 floats to break aliasing.
3. Fuse SiLU/gate into conv stores; skip materializing `pre` (pixelshuffle straight from the
   conv2 tile); tile-parallel scaling across cores.
4. Single-tier builds (e.g. v4x-only for known fleets) to push full-SIMD below 300 KB.
5. If int8 compute is ever wanted: QAT via MAI recipe, or switch arch to ECBSR/ETDS class.

Status honestly stated: correctness (all tiers) + size menu + f16 path are proven; SIMD
speed vs tract is UNMEASURED (load noise); nothing is product-wired (no zenpipe/imageflow
integration, no zencodec surface). Weight licensing: NTIRE2025_ESR repo is MIT but per-team
weight terms are unaffirmed — fine for experiments, diligence (or retrain on our corpora)
before shipping.
