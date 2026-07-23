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

### Eval run 1 (flawed SPAN, kept as `*_flawed-span.tsv`) — what it taught

The first run shipped a **miswired SPAN-48 graph** (missing `(x−mean)·255`
input norm + official's `SiLU(inplace=True)` concat semantics) that produced
near-constant gray — caught because eval numbers, not goldens, went sideways;
my consistency goldens agreed with my own broken torch reimplementation. Fix
verified against spandrel (all 7 models ≤6e-6); the dump now hard-gates every
model against the reference implementation. Compact-arch rows (A2c, B, D, C)
in run 1 pass that same cross-check, so their numbers stand:

1. **The guard did its job under total model failure** — the strongest
   bounded-downside evidence we could have asked for, obtained by accident:
   raw broken SPAN scored **9.4 dB PSNR / SSIM2 −357** (garbage), while the
   guarded path held it at **25.7 dB / SSIM2 54** on clean ×2 (baseline
   Lanczos 28.8 / 67). A catastrophically wrong model degraded blind output
   to "slightly worse than bilinear", not to noise. (`A2_span` vs
   `A2_span_raw`, run 1.)
2. **Severity gating is mandatory policy, not a nice-to-have.** On clean
   input every model loses to Lanczos (worse-than-baseline 77–98 %); at q50
   and below the compact restore models win decisively (A2c 20–38 % worse
   rate, i.e. better on 62–80 % of images; SSIM2 median +2–4 over Lanczos;
   butteraugli better). Blind policy: detect JPEG quality from quant tables
   → model only when degraded, resample when clean.
3. **S-C ×1 repair via 2×-up→box-down loses to identity** at q35–q75 on all
   three metrics (worse rate 44–89 %). Suspect the box blur; run 2 adds a
   CatmullRom-down variant (`C_repair_cr`). If that still loses, the honest
   verdict is that ×1 JPEG repair needs a native ×1 model (1xDeJPG /
   RealPLKSR port), not a scale round-trip.
4. **×4 blind upscaling is brutal**: SSIM2 medians are negative even for
   Lanczos on degraded inputs. ×4 claims should be framed against that
   reality — nothing "restores" q35 at ×4; the models only lose less.
