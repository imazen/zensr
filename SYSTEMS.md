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

### S9 rung 2 (capacity): falsified — teacher choice is the next lever

rt32_distill_2x (nf32/nc12, 115,756 params, same data/recipe/steps) closed
+0.85 dB of the val-vs-teacher gap but translated to only **+0.5 SSIM2**
over the 45K student on ground truth (q35 30.2 vs 29.7; q75 46.3 vs 45.8)
at **2.7× the compute** (8.5 vs 23.0 MP-out/s @12T). Verdict: capacity is
NOT E_rt's bottleneck; the 45K shape stays the realtime pick. The gap to
A2c at q75 tracks the *teacher's* q75 weakness (span 48.3 < A2c 50.3) —
rung 3 tests an A2c-teacher distillate (E_rtc) on identical data settings.
Guard after the separable rewrite: 9 ms/MP (was 26).

### S9 rung 3 (teacher choice): CONFIRMED dominant — E_rtc ships as S-E

rtc_distill_2x: identical 45K shape/data/recipe, teacher switched from
span-2x to **A2c (2xNomosUni_compact)**. Ground-truth results vs E_rt:
q75 SSIM2 **48.4 vs 45.8** (teacher 50.3 — the 45K student lands within
1.9 of its 600K teacher); butteraugli **matches the teacher** at every
quality (2.83/3.42/3.80 vs 2.82/3.34/3.67) and beats Lanczos at q35;
worse-rate q35 **28 % vs 50 %**; clean 61.4 vs 58.8. Only q35 SSIM2 keeps
a sliver of span-teacher advantage (29.5 vs 29.7 — noise-level).
Speed identical to E_rt (same shape): 23 MP-out/s @12T, ~23× A2c.

**S9 ladder conclusions (3 rungs, one session):** distillation transfers
restoration behavior at 13× compression; capacity is not binding at 45K
(rung 2 falsified); **teacher choice is the dominant lever — distill from
the policy winner (worse-rate/butteraugli king), not the SSIM2-median
king.** S-E ships rtc_distill_2x; rt/rt32 kept on Tower as ablation record.

### Per-subcorpus breakdown (n=8 each — medians noisy, wins/n is the honest read)

Δ SSIM2-median vs baseline | per-image wins. Pattern across every track:
**restoration wins on graphic/text/synthetic content (documents, maps,
renders, screen) and loses on natural-texture content (people, textures,
art-scans/halftone) except at heavy degradation.**

x2 q35 (vs Lanczos): documents +5.7 (8/8), renders +10.1 (7/8), maps +4.7
(8/8), screen +4.6 (7/8), photos +4.0 (6/8) — but art-scans −2.7, people
−1.8, textures −0.9 (A2c). E_rtc mirrors it with a weaker tail (textures
−5.3): the 45K student is weakest exactly on the stochastic class.

x2 q75 (vs Lanczos): wins survive ONLY on screen +1.0 (7/8), maps +1.0
(6/8), documents +0.9; photos/people/art-scans/textures all negative →
at q75 the gate should be severity×content, not severity alone.

x4 q50: D/B are huge on documents (+13.5/+15.6, 8/8), maps, renders;
negative on people/screen/textures. x4 clean F_spanf: positive on ALL 8
subcorpora (art-scans thinnest +0.8, 5/8) — the most robust single claim
in the eval. x1 q35 repair: wins only maps +4.0 (8/8) and renders +5.4
(7/8) — graphics deblocking; everything else ≤ identity.

Policy consequence: the blind router needs a cheap content signal next to
the quant-table severity signal — the guard's own texture-energy map is
already computed and is the natural first feature (S7 ties in). Recorded
as the next eval axis; n=8/subcorpus must grow before shipping per-class
thresholds (sweep-discipline: ≥50/class).

### S9 rung 4 (teacher adoption for people): FALSIFIED — no off-the-shelf heavyweight qualifies

Audition (`tools/teacher_audition.py` + `audition_score`, x2-target protocol,
frozen eval files, system-cjpeg degradations; TSV + summary in benchmarks/):
RealESRNet_x4plus, RealESRGAN_x4plus (BSD-3, 17M RRDB), 4xNomosWebPhoto_
RealPLKSR, 4xFaceUpDAT (CC-BY-4.0) vs the incumbent A2c teacher and Lanczos
on people / textures / art-scans / photos.

**Every candidate loses to BOTH Lanczos and A2c on people, textures, and
art-scans at every quality.** FaceUpDAT — the face specialist — goes 0/8 on
people at all four degradations (q35 SSIM2 11.2 vs Lanczos 21.6, A2c 20.1).
RealESRGAN_x4plus is catastrophic (people q35 −1.4; art-scans −23.2).
The one candidate bright spot: RealESRNet (fidelity RRDB) on photos q35
(33.2, 6/8 wins — above A2c's 30.8) — but with much worse butteraugli
(4.96 vs 3.86) and a total art-scans collapse (−4.5). n=8/slice.

Interpretation: the community heavyweight zoo optimizes ×4 perceptual
restoration (invented texture, no-GT aesthetics). Under fidelity metrics at
×2 web-JPEG with ground truth, invention is penalized — the 600K JPEG-
trained compact beats every 17M+ model. Consequences: (a) "adopt a bigger
teacher" is closed as the people path; (b) people improvement = region-wise
conservatism (skin/face-aware guard alpha) now + ground-truth fidelity
fine-tune under our exact degradation model at P2, shipped as an S2 band;
(c) the audition harness is reusable for any future candidate in minutes
(and its A2c rows cross-validate the systems_eval per-subcorpus numbers).

### People fine-tune (P2-mini, 2026-07-24): the people gap is CLOSED by GT training

Corpus: **zensr-people-v1** — 2,500 CC0 people photos (pxhere via HF dump,
per-image URL provenance), frozen 64-image eval slice (shard-spread,
`eval_ids.txt`), 24k GT crop pairs (image-level val split, q40–95 turbo).
Two warm-started fine-tunes (20k steps each, ~25 min total GPU):

| model | params | init | q35 SSIM2/butter/worse% | q75 SSIM2/butter/worse% |
|---|---|---|---|---|
| Lanczos | — | — | 30.3 / 3.19 / — | 57.9 / 2.19 / — |
| A2c (incumbent) | 600K | — | 33.8 / 3.01 / 20 % | 56.4 / 2.23 / 70 % |
| **P_rtc** | **45K** | rtc student | 33.5 / 2.92 / **8 %** | 59.8 / 1.97 / 25 % |
| **P_a2c** | 600K | A2c | **39.8 / 2.67 / 5 %** | **62.6 / 1.88 / 14 %** |

(frozen pxhere slice, n=64.) P_a2c is the first model to decisively win
people at q75. P_rtc at 45K beats the 600K generalist at q50/q75 and fixes
E_rt's people weakness (q75 59.8 vs 51.8). Clean stays route-to-resample
on SSIM2 (P_a2c butteraugli actually ties Lanczos there).

**Cross-source control** (imazen-26 unsplash-people, n=8, zero training
exposure): P_a2c +3.3 over Lanczos at q35, +2.0 at q50, q75 tie (44.3 vs
44.9; A2c was 40.6) — transfer confirmed, home-field inflates magnitude
but not sign. n=8 caveat applies to the control, not the n=64 primary.

Consequences: (a) the S2 external-band mechanism now has its first proven
band — people — with P_a2c as the quality band and P_rtc as the realtime
band; (b) GT fine-tuning on 2.5k targeted CC0 photos + 25 GPU-minutes
moved a content class from "loses to Lanczos" to "wins at every quality"
— this is the per-class recipe (textures and art-scans are next); (c) the
n=8 people readout that motivated rung 4 was noise-pessimistic: at n=64
A2c already won q35 people; per-class evals need n≥50 (sweep discipline
held).

### Split audit + TRUE-TEST confirmation (2026-07-24, user-prompted)

Audit of train/val/test hygiene found: (1) people GT training clean
(eval excluded + image-level val, verified in logs); (2) ONE leaked image
in the distill data (textures/teresa — the Rust eval's "first 8 usable"
slid past a 101MP decode-skip while the exclusion cut "first 8 sorted");
(3) distill val tail was crop-level (monitoring only); (4) the frozen
slices had been reused for rung selection — they are DEV sets.

Fixes: the imazen-26 eval split is now a PINNED COMMITTED FILE LIST
(`eval_split/imazen26_eval_files.tsv`) and training excludes first-8 ∪
pinned; distill gen val split is image-level; rtc (S-E) retrained on the
clean regen. Policy: frozen-by-file-list, dev slices for selection, test
slices touched once per milestone.

**True held-out TEST — people-test-v1** (64 images, 6 virgin pxhere
shards, zero id overlap with any pool/dev slice, scored exactly once):

| deg | Lanczos | A2c | P_rtc 45K | **P_a2c 600K** |
|---|---|---|---|---|
| q35 | 33.5 / 3.14 | 38.1 / 2.98 / 16 % | 36.9 / 2.93 / **3 %** | **41.4 / 2.74 / 2 %** |
| q50 | 44.7 / 2.70 | 46.2 / 2.67 / 30 % | 46.9 / 2.53 / 6 % | **50.3 / 2.40 / 2 %** |
| q75 | 59.0 / 2.11 | 57.2 / 2.18 / 69 % | 59.9 / 1.93 / 22 % | **61.6 / 1.85 / 8 %** |
| clean | 82.2 / 1.18 | 77.6 / 89 % | 76.1 / 97 % | 80.2 / 89 % |

(SSIM2 med / butteraugli med / worse-than-Lanczos; n=64; lanczos clean
n=63, one row filtered.) Test ≥ dev on every degraded metric — the dev
claims transfer: P_a2c wins every degraded quality on all three metrics
(q35 worse-rate 2 %); P_rtc at 45K beats Lanczos q35–q75 and the 600K
generalist at q50/q75; clean stays route-to-resample. These are the
citable people-band numbers.

**Audit closure:** rtc retrained on the leak-free regen (`E_rtc2` rows in
the day TSV): within ±0.25 SSIM2 of v1 at every quality including textures
(clean marginally better than leaked, 13.9 vs 13.2 at q35 — both still lose
the subcorpus) — the teresa leak was immaterial; verdicts unchanged. S-E now
ships the clean weights; goldens ≤3e-7. Distill val is image-level (clean
val 32.42 ≈ v1's crop-level 32.35 — old monitoring numbers were honest).
