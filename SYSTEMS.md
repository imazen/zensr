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

Data: `benchmarks/systems_eval_2026-07-23.tsv` (+ `.summary.txt`; regenerate
with `systems_eval summarize <tsv>`). n=64 images (8 subcorpora × 8), real
system-cjpeg degradation, medians unless noted. E_rt rows appended from
`ert_eval` after the distillation finished.

### Run 2 (fixed SPAN graph) — the five-system verdict

**×2 (S-A / S-C-class models).** On clean input every model loses to Lanczos
(SSIM2 62–63 vs 67; worse-rate 83–97 %) — resample clean images, full stop.
On JPEG input the models win and the ranking is quality-dependent:

| deg | best SSIM2 | best butteraugli | best worse-rate |
|---|---|---|---|
| q75 | A2c 50.3 (Lanczos 48.9) | A2c 2.82 (L 3.01) | A2c 56 % |
| q50 | A2_span 39.1 (L 36.5) | A2c 3.34 (L 3.72) | A2c 38 % |
| q35 | A2_span 32.9 (L 27.1) | A2c 3.67 (L 4.22) | A2c 20 % |

The span model restores the most structure at heavy degradation (+5.8 SSIM2
over Lanczos at q35) but its sharper hallucinations cost butteraugli; the
compact model is the safer blind default (lowest worse-rate at every q,
best butteraugli everywhere). Policy: **A2c for blind en-masse, A2_span for
"restore harder" opt-in at q≤50.**

**Guard ablation (A2_span vs A2_span_raw), now on a working model:** the
guard costs ~1.2 SSIM2 median at q35 (32.9 vs 34.1) but improves worst-decile
(p10 8.5 vs 7.5), worse-rate (27 % vs 41 %), and butteraugli (4.34 vs 4.89).
On clean input the guard *raises* the median (62.1 vs 57.2): multijpg-trained
models "fix" texture that isn't broken, and the bilinear anchor tempers it.
Insurance that pays for itself everywhere except a small median cost at the
degradation level where you'd opt into raw anyway.

**×4.** Severity split is stark. Clean: **F_spanf (NTIRE clean-specialist)
dominates** — SSIM2 32.9 vs Lanczos 29.6, worse-rate only 19 %, best
butteraugli — while every restore-trained model loses to Lanczos. Degraded:
F_spanf collapses (worse-rate 61–69 %) and **D_anime is the surprise blind
winner** (worse-rate 23 % at q35/q50, best/near-best SSIM2 + butteraugli),
with B_quality close behind; A4_span mid-pack. ×4 remains "lose less":
SSIM2 medians are negative for everything on degraded input. Policy:
**F_spanf on clean, D_anime (or B at q75) on JPEG input.**

**×1 repair (S-C): honestly negative.** CatmullRom-down beats box-down
slightly but still loses to identity at q50/q75 on all metrics (butteraugli
identity 1.18–1.91 vs repair 1.60–2.07); at q35 it's a median tie (SSIM2
59.8 vs 59.6, wins 58 % of images) — a weak positive only at the heaviest
degradation. The 2×-up→down round trip is falsified as a ×1 strategy for
q≥50; a native ×1 artifact model (1xDeJPG-class, needs the RealPLKSR port)
is the queued replacement.

**Cross-metric note:** butteraugli consistently prefers conservative output
(A2c, identity); SSIM2 rewards restored structure (span). Both agree on
severity gating and on every policy call above; report both, never one.

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

### S-E (rt_distill_2x) — pilot verdict

45,156 params (180 KB f32 / 90 KB f16), distilled in 20 min (60k steps,
GPU-resident set) from the S-A ×2 teacher on 13.5k imazen-26 turbo-JPEG
pairs. Golden-verified in the runtime (≤4.8e-7, tiled exact).

Quality (guarded, vs the same eval): at q35 it keeps **68 % of A2c's SSIM2
gain over Lanczos** (+2.5 vs +3.7) at 13× fewer params; q50 marginal (+0.5);
**q75 and clean lose to Lanczos** — the severity gate must route light/no
degradation to A2c or a resampler. Worst-decile slightly below A2c
(p10 6.9 vs 8.1 at q35). A real result for a first distillation pilot; the
S9 ladder (more pairs, longer schedule, maybe nf32) is the obvious next rung.

### Speed (quiet box, 5-rep min, benchmarks/systems_bench_2026-07-23.tsv)

| system | 1T MP-out/s | 12T MP-out/s | note |
|---|---|---|---|
| S-A ×2 span | 0.24 | 1.31 | |
| S-A ×4 span | 1.09 | 3.70 | |
| S-B ×4 general | 0.35 | 0.98 | heaviest |
| S-C ×1 (compact2x) | 0.19 | 0.99 | = 0.25 input-MP/s; expensive for a losing repair |
| S-D ×4 anime | 0.84 | 2.75 | |
| **S-E ×2 rt** | **3.07** | **21.93** | realtime-tier gate ✓ (≤0.05M params, ≥15 MP/s @12T) |

Guard overhead is a flat 21–26 ms per output MP — negligible for S-A/B/C/D,
but ~55 % of S-E's model time; SIMD-ifying `guards.rs` is the queued fix.
