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
| **zensr-micro-abi cdylib (release-min)** | **271,432 B (265 KB) — sub-300 KB target MET** |
| zensr-micro correctness vs PyTorch golden | max_abs_diff 5.66e-6, rmse 5.9e-7 (PASS) |
| zensr-micro speed (naive kernels) | 1.22 MP-out/s @128² — **2.7× slower than tract** |
| SPANF weights | 593,152 B fp32 (separate; 296 KB fp16 / 148 KB int8) |

Full grid: `benchmarks/tract_cpu_2026-07-22.tsv`.

## Reproduce

```sh
just export      # NTIRE25 ckpts -> models/*.onnx (needs torch; box GPU used by SPANF init)
just dump        # SPANF weights + golden -> models/*.raw
just bench       # tract grid -> benchmarks/
just verify      # golden-gate zensr-micro + timing
just size        # build + report micro-abi cdylib size
```

## Next steps (speed roadmap for zensr-micro)

1. conv3x3: single pass over the 3 taps (currently 3 row sweeps) + output-channel register
   tiling (4-8 oc per input-row read) — the standard direct-conv blocking.
2. Fuse SiLU into the conv store; fuse the gate; run conv1x1 straight off `near`/`b5`/`b1`
   without materializing the 112-ch concat.
3. magetypes `f32x8` FMA kernels via archmage dispatch once the scalar blocking is right.
4. Then: int8/VNNI experiment, tile-parallel scaling, and a 2x-scale SPAN variant.

Status honestly stated: correctness + size are proven; speed is not yet competitive with
tract's assembly GEMMs (2.7× gap at 128²); nothing here is product-wired (no zenpipe/imageflow
integration, no zencodec surface). Weight licensing: NTIRE2025_ESR repo is MIT but per-team
weight terms are unaffirmed — fine for experiments, diligence (or retrain on our corpora)
before shipping.
