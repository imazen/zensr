# zensr systems — five deployable SR/artifact-removal systems

Assembled 2026-07-23 (10-hour director session). Every system = **model(s) +
guard layer + policy**, runs in zensr-micro (CPU, AVX-512/AVX2 dispatch,
seam-exact tiling, arbitrary dims), and is designed for **blind en-masse usage
on collections**: the guard layer bounds the worst case by construction.

## The guard layer (common to all systems)

`zensr_micro::guards` — three independent mechanisms anchored to a bilinear
upsample of the input:
1. **Residual clamp** (default τ=0.25): `out = base + clamp(SR−base, ±τ)` —
   no model failure can deviate from the linear baseline by more than τ.
   Property-tested against adversarial model output.
2. **Texture gate**: per-16px-cell high-pass energy shrinks the SR residual
   (α∈[0.35,1], bilinearly smoothed) where content is stochastic texture —
   the one content class where SR measurably loses to linear filters.
3. **Round-trip fallback**: box-downscale of the output is compared to the
   input; MAE beyond threshold blends the whole image back toward baseline
   (catches off-distribution inputs: dithers, noise fields, exotic content).

`GuardReport` (clamped fraction, mean α, round-trip MAE, fallback blend) is
returned per image — en-masse runs can log it and flag outliers for review.

## The systems

| # | name | scale | models (license) | weights | intended use |
|---|---|---|---|---|---|
| **S-A** | Guarded Fast Photo | ×2 / ×4 | 2xNomosUni_span_multijpg / 4xNomosUni_span_multijpg (CC-BY-4.0, JPEG-trained) | 1.6 MB f32 ×2 | blind default for photo/mixed collections |
| **S-B** | Quality Restore | ×4 | realesr-general-x4v3 + wdn (BSD-3), **severity-blended**: wdn weight α from JPEG quality (product: from decoded quant tables) | 9.8 MB pair | opt-in max quality, heavier compute |
| **S-C** | Pure Artifact Removal | ×1 | 2xNomosUni_compact_multijpg (CC-BY-4.0) upscale→box-down | 2.4 MB | dejpeg/deblock at original size |
| **S-D** | Anime/Graphics | ×4 (+×2 via HFA2k) | realesr-animevideov3 (BSD-3), 2xHFA2k_compact (CC-BY-4.0) | 2.5 MB | illustration/anime collections |
| **S-E** | Realtime | ×2 | rt_distill_2x — Compact nf24/nc8 (~47 K params) distilled this session from the S-A ×2 teacher on turbo-JPEG-degraded imazen-26 pairs | ~190 KB | previews, thumbnails, latency-critical paths |

Notes:
- S-B's severity blend is the wdn weight-interpolation mechanism (PLAN S2d);
  in the eval it is keyed by known q, in the product by zenjpeg quant tables.
- S-C's 1× repair uses the 2×-restore-then-box-down identity; a native 1×
  RealPLKSR (1xDeJPG) upgrade is queued behind the large-kernel port.
- S-E is the S9 distillation-ladder pilot; if its eval misses the bar the
  documented fallback is S-C's compact at ×2 (3× the compute, still fast).
- All CC-BY-4.0 components require attribution (Philip Hofmann / Phhofm) in
  product credits; BSD-3 components retain the Real-ESRGAN copyright notice.

## Evaluation protocol

`systems_eval` bin: 8 imazen-26 subcorpora × first-8 frozen split × degradation
{clean, turbo-q75, q50, q35 — REAL system cjpeg, 4:2:0, -optimize} × tracks
{×2, ×4, ×1}. Baselines: Lanczos/CatmullRom (identity for ×1). Metrics:
PSNR, SSIMULACRA2, butteraugli-3norm; aggregates report median, **p10
(worst-decile)** and **worse-than-baseline rate** — the bounded-downside
numbers. Guard ablation: S-A ×2 runs guarded AND raw.

## Results

See `benchmarks/systems_eval_2026-07-23.tsv` + the summary table committed
alongside. (Filled by the session's eval run; the distilled S-E row lands when
training completes.)
