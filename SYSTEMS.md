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

### S6 native ×1 dejpeg (2026-07-24): round-trip retired, direct inversion wins everything

User called it ("just train inversions of jpeg artifacts, right?") — the
scale round-trip was a day-1 adoption-only expedient. dejpeg_1x: Compact
nf64/nc16 at scale 1 (595,459 params; runtime gained s=1 via zero-channel
head padding), warm-started from A2c's body (51/53 tensors), 23.5k
same-size turbo-JPEG pairs (q35–95, pinned-exclusion imazen-26), 20k
steps ≈ 25 GPU-min. n=64/cell vs identity:

| deg | identity | C_repair (round-trip) | **S6_dejpeg** |
|---|---|---|---|
| q35 | 59.6 / 1.91 | 58.6 / 2.08 / 44 % | **65.4 / 1.68 / 2 %** |
| q50 | 66.8 / 1.64 | 64.6 / 1.87 / 62 % | **71.6 / 1.39 / 2 %** |
| q75 | 77.2 / 1.18 | 72.9 / 1.59 / 89 % | **79.6 / 0.99 / 14 %** |

(SSIM2 med / butteraugli med / worse-than-identity.) Wins every quality
on every metric incl worst-decile (p10 72.7 vs 69.8 at q75); +1.7–2.0 dB
PSNR. S-C is replaced by S6; the RealPLKSR port is no longer on the
critical path (still interesting for a quality-tier ×1 later).

### S6 v2 — zenjpeg-native, 4-encoder, deblock 2×2 (2026-07-24)

Directive: cooperate with zenjpeg deblocking or determine it should be
disabled; cover {turbo, mozjpeg, jpegli, zenjpeg} × {420,444} × any q;
avoid encode-space gating if possible. Training pairs decoded by the
LOCAL zenjpeg 0.9 (deployment decoder; 0.8.4 pub(crate)-locks
DeblockMode), both arms from the same encoded bytes.

**Deblock verdict: DISABLE (which is zenjpeg's default).** 2×2 mean of
per-cell SSIM2 medians (4 enc × 2 ss × q∈{15,35,55,75,90}, n=24):
model_off **69.45** > model_auto 68.64 > identity_auto 66.70 >
identity_off 66.22. Best-arm-by-butteraugli: model_off 26/40 cells,
model_auto 10, identity_off 3 (raw q90 cells), identity_auto 0.
Deblocking helps when NO model runs (+0.5 alone) but destroys block-
boundary structure the trained inversion uses — train on pixel-exact
decode. Val agrees: off-arm 33.42 vs auto-arm 33.24 dB.

**Coverage:** model_off beats identity at q15–q75 on every encoder and
both subsamplings (worse-rate 0–38 %; biggest wins at low q: mozjpeg
420 q15 23.7→31.0). Largest gains on mozjpeg/turbo (Annex-K-scaling
artifacts); jpegli/zenjpeg gain less (their AQ encodes are cleaner).
**Sole gap: q90** — 5/8 enc×ss combos show −0.01…−0.61 SSIM2 medians
(visually nil, guard-bounded). Fix in flight WITHOUT gating: qboost
rung (3× oversampling of q≥85 + clean pairs, fine-tune from the off
arm) targeting identity behavior at transparent qualities.

**Fingerprint validation (n=960 encodes):** mozjpeg→Mozjpeg 100 %,
jpegli→CjpegliYcbcr 100 %, zenjpeg→CjpegliYcbcr 100 % (jpegli lineage —
indistinguishable; zenjpeg#189 filed for an encoder-embedded parameter
record), turbo -optimize→ImageMagick 80 % / Unknown 20 %. Family-level
routing is reliable today; #189 upgrades zenjpeg files to ground truth.

### S6-v2 qboost rung + the "any quality" verdict

qboost (3× oversampling of q≥85+clean, 12k-step fine-tune) closes q85–90:
q85 positive on all 8 enc×ss (+0.15…+2.72), q90 majority positive. q93–96
remains metric-negative in 17/32 cells (worst −1.2 SSIM2) — but in
butteraugli the entire q93+ downside is **+0.013…+0.038 on a 0.33–0.52
baseline**, an order of magnitude under JND. At those tables the file is
transparent; there is nothing to recover (the S10 view: per-band Q/2
boxes are tiny, so any model perturbation is pure downside).

**Deployment call: blind-always activation is defensible today** —
guard-bounded, perceptual worst case nil. The principled closure of the
q93+ tail is NOT more training and NOT encode-space gating: it is the
**S10 quantization-consistency projection** (clamp model output into the
per-coefficient DCT box), which scales the allowed correction with the
tables themselves — continuous, exact, and zero-configuration. Queued
next together with S5b YCbCr-native models (projection is tight in the
decoder's YCbCr domain).

### Sub-q15 probe (2026-07-25, user-prompted): the deblock verdict FLIPS at q≤8

Training had been q≥10 and eval q≥15 — convention, not measurement, and
counter to the workspace low-q mandate. A q∈{5,8,12} probe (n=24, full
2×2) shows structure the q15+ survey missed:

- **2×2 mean flips: model_auto 20.16 > model_off 19.49** > identity_auto
  15.98 > identity_off 12.70. At q5–8 on the IJG-family encoders
  (mozjpeg/turbo — brutal Annex-K steps), Knusperli + cooperating model
  wins (e.g. turbo 420 q8: auto-model 10.3 vs off-model 5.7 vs identity
  −2.4); on the AQ encoders (jpegli/zenjpeg) model_off still leads.
  q12 returns to model_off majority. Worse-rates stay 0–25 % everywhere;
  both models beat identity by huge margins (jpegli 420 q5: +13.3).
- **Why this makes sense (S10 lens):** Knusperli is a DCT-coefficient-
  domain consistency method. At extreme q the per-band boxes are huge and
  pixel-domain inversion is underdetermined — coefficient-domain
  information carries real signal a pixel-space model never sees. The
  sub-q9 flip is more evidence that the coefficient-aware S10 direction
  is the unifying mechanism, not per-range gating.
- **Caveats:** current models trained with q floor 10 (q5–8 is out of
  distribution — impressive that worse-rates stayed tiny), and the auto
  arm lacks the qboost treatment. Floor-5 data (v3) is regenerating;
  both arms retrain before any deployment change. Until then the
  no-gating call stands: Off+model everywhere is still hugely positive
  at every q (just leaves ~1–4 SSIM2 on the table at q5–8 IJG vs the
  cooperating path).

### Floor-5 matched re-verdict (2026-07-25) — final S6-v2 quality-axis picture

v3 arms (q∈U(5,96), qboost, warm from v2b/v2-auto) both improve at low q
(off 19.49→20.77, auto 20.16→21.26 mean cell-median) and the standard
grid HOLDS (off 70.89 > auto 70.17 > id_auto 67.49 > id_off 66.92 —
no regression; v3_off ships as S6).

**The q≤8 Knusperli advantage is STRUCTURAL, not a distribution
artifact**: in-distribution, auto still wins q5–8 on IJG-family
(turbo q5 −18.9 vs −23.2; mozjpeg q8 +0.4 vs −1.4) while AQ-family
(jpegli/zenjpeg) stays off-favored at every q. Coefficient-domain
correction carries information pixel-space models cannot see, exactly
where quantization boxes are huge — third independent confirmation of
the S10 thesis.

**Deployment (final):** ship ONE model (dejpeg3_off) + deblock Off at
all qualities — blind, no gating, wins everywhere except the q≤8
IJG corner where it still beats identity by wide margins, just not by
as much as the cooperating path (~2–4 SSIM2 left on the table in a
regime where content is already destroyed, SSIM2 < 0). That corner is
fingerprint-detectable (family + est-q from probe) if it ever matters
commercially before S10; the principled closure is the S10
coefficient-aware model, which subsumes Knusperli's advantage into the
single-model story.

### Final S6 deployment: dejpeg4_policy + the two-line decoder rule (2026-07-25)

User challenge ("why not enable Knusperli at low q?") accepted — the
constraint was never about a decoder flag, only about not bounding model
activation. Shipping configuration:

- **One model** (`dejpeg4_policy`, 595 K), fine-tuned on the exact decode
  mix the runtime produces; **runs on every image, every quality**.
- **Two-line decoder rule** (measured-exact inputs): probe family not
  Cjpegli* AND quality scale Ijg/Mozjpeg AND estimate ≤ 9.5 →
  `DeblockMode::Auto` (Knusperli), else `Off`. Probe q is exact at 5/8/12
  for turbo (as Unknown-family) and mozjpeg; AQ files never trigger.
- **End-to-end (probe→decode→model) validation**: low-q mean 21.23 —
  captures the cooperating specialist (21.26) vs pure-off 20.77;
  standard grid 70.85 ≈ off-specialist 70.89, with **0/40 cells** more
  than 0.3 below the per-cell best-of-both-specialists oracle.
- "Disable ML at low q" was refuted outright: the model's largest wins
  on the whole axis are at q5–15 (up to +13 SSIM2 over identity).
S10 remains the roadmap item that folds Knusperli's coefficient-domain
advantage into the model itself, retiring even the decoder flag.

### Conditioning ablation, part 1 (2026-07-25, LAN fleet): precision confound dead, S2a falsified

Distributed over the household fleet (lianli 2080 / jason 3070 / ian
1660Ti; ian lost mid-run to a power-off — kids' box). Matched arms
(24k v4 pairs, policy-mix decode, 14k steps, batch 48, warm from
dejpeg4): **none-Turing 32.59 ≡ none-Ampere 32.61** (every stratum
within 0.06 dB) — bf16-vs-fp32-fallback compile paths do NOT matter;
cross-box arms are comparable. Therefore the scalar arm's result is
real: **+13.7 dB on clean, −2.7…−3.5 dB on every degraded band** — a
global severity scalar is an over-strong prior that fights local
evidence. S2a input-conditioning falsified in scalar form (and the
clean win is redundant with qboost). S10 dmap (local, exact,
per-block) still in training — the surviving conditioning hypothesis.

### Conditioning ablation, part 2 — VERDICT: pixels-only wins; input-channel conditioning is a shortcut trap

dmap (S10-as-input, local + exact) landed **identical to the global
scalar** (31.05 vs 31.07 val; every stratum within 0.15 dB; clean 75.9):
the network extracts only the "overall severity" bit from the rich map
and uses it the same shortcut way — conditioning substitutes for pixel
analysis instead of augmenting it, costing ~3 dB on every degraded band
for a clean-identity win qboost provides anyway. **S2a and S7-as-input
falsified; S10's value is the OUTPUT-side projection** (DCT-box clamp —
no training, hard re-encode-consistency guarantee), not input channels.
The shipping architecture stays pixels-in/pixels-out.

Fleet postscript: 6 training runs + 1 confound probe across lianli/
jason/ian in one afternoon; ian was lost mid-run to a physical power-off
(kids' box — WoL unarmed after hard cut; needs the button) and its arm
completed on lianli. Kids' boxes flipped back to Windows at wrap-up.

### S10 projection — shipped and measured (2026-07-25/26)

Implementation: `zensr_micro::consist` (decoder-agnostic DCT-box clamp,
property-tested: decode-in-box no-op, adversarial re-encode consistency)
+ `zensr-zenjpeg::restore_jpeg` (probe → deblock policy → guarded model →
family-slack projection; the eval's `model_proj` arm IS this call).
Slack calibrated on 1M coefficients/cell: turbo p99≤0.07Q; mozjpeg
trellis p99≤0.23Q (max ~15Q zeroed runs); jpegli/zenjpeg AQ p99≤0.41Q
(stored DQT understates per-block quant) → family slack 0.15/0.35/0.45.

Grid results (Δ SSIM2 median vs identity, mean over cells):

| grid | policy (no proj) | **+ projection** | cells < identity |
|---|---|---|---|
| high-q 85–96 | +0.18 | **+0.37** | 14→12 /32 |
| standard 15–90 | +3.93 | **+3.96** | 5→3 /40 |
| low-q 5–12 | +8.54 | +8.55 | 0→0 /24 |

Projection doubles the high-q margin, never costs anything at q≤90, and
carries the hard guarantee (output re-encodes to the file's own
coefficients, per-family slack caveats aside). Honest residuals at q96:
jpegli −0.60→−0.25 and zenjpeg −0.74→−0.54 (big improvements), but
turbo −0.75→−0.87 / mozjpeg −0.32→−0.40 — traced to calibration tails
(turbo q92 shows 2.2 % violations up to 1.2Q from integer-DCT skew;
slack 0.15 clips them). Full q96 erasure additionally needs 420-chroma
back-projection + the YCbCr-native pipeline (S5b) — the projection
currently covers luma always and chroma only at 4:4:4.

Build/deps milestone: product cold build (zensr-zenjpeg+micro+zenjpeg-
from-source) **11.9 s**; core-edit loop 2.0 s; zenjpeg-edit loop 7.5 s.
zenjpeg#190 filed: impl-Stop monomorphizes the decode pipeline into
every consumer (46k LLVM lines in our 300-line glue; 11.6 s glue edits)
— dyn-inner fix proposed.

### Production hardening pass (2026-07-26): API contract + geometry review

The `code review / refactor / minimize public API` milestone (user-directed):

- **Public surface minimized and snapshotted.** `zensr-micro` default API went
  260 → 169 lines (`apidoc/zensr-micro.public-api.txt`); the SPANF research
  surface (simd/tiled/px modules, root re-exports, `SpanfWeights`/`SpanfModel`/
  `Scratch`/layout consts/f16+int8 decoders, scalar `spanf_x4`) is gone from
  the default surface — modules are `pub(crate)` unless the `internals`
  feature opts in; root items that must stay compiled are `#[doc(hidden)]`.
  `px` was dropped from default features (SPANF-only, zero consumers) and now
  also requires `internals`; the `zensr-verify` bin is
  `required-features = ["internals"]`. Contract doc: `apidoc/PUBLIC_API.md`.
  Product-crates rebuild: 1.56 s (was 2.0 s with zenpixels in the default
  graph).
- **Review caught a real 4:2:2/4:4:0 bug** in `restore_jpeg`: the old
  `half_res` test (`blocks*16 >= dim` on both axes) classified half-ONE-axis
  chroma (4:2:2 horizontal, 4:4:0 vertical — both encodable by zenjpeg) as
  4:2:0 and would have run the 2×2 box back-projection on them, corrupting
  chroma. Fixed with an explicit `PlaneGeometry {Full, HalfBoth, Other}`
  classifier; 422/440 now classify `Other` and stay unprojected. Regression
  test `geometry_classification_422_440_never_uses_420_backprojection` runs
  against real zenjpeg encodes of all four subsampling modes.
- **4-component (Adobe CMYK/YCCK) files skip projection entirely** — their
  coefficient planes are not the YCbCr space the pipeline reconstructs.
  Grayscale (1-component) keeps luma projection.
- `RestoreError` and `Restored` are now `#[non_exhaustive]`; the x1-model
  precondition returns `RestoreError::UnsupportedModel` instead of panicking.

### S10 high-q slack: the q96 residual is SOLVED — absolute sample-quantization noise (2026-07-26)

Per-quantizer-value calibration (slack_probe extended with per-Q breakdown +
nonzero split; 1M coeffs/cell, q88–q98; benchmarks/slack_calibration_highq_*
+ _nzsplit_2026-07-26.tsv):

- **Root cause of the q96 projection regression**: encoders with integer
  sample pipelines (libjpeg-turbo, mozjpeg) round YCbCr samples to u8 BEFORE
  their FDCT. That injects a bounded ABSOLUTE DCT-domain noise (turbo Q=1:
  p99 1.32, p99.9 2.15, max 3.70 ≈ 8·0.5 — exactly the ±0.5/255-per-sample
  worst case) which a relative slack can never cover as Q→1. At Q≥4 it hides
  under Q/2; at q≥94 the DQT is mostly Q=1..3 and 17% of Q=1 bands violate
  the box → the projection clamps CORRECT detail.
- **Fix shipped**: `ProjectionConfig.slack_abs` (additive, coefficient
  units); interval is now Q·(0.5+slack_q) + slack_abs. Family calibration in
  `slack_abs_for`: turbo/mozjpeg 1.5 (covers p99, most of p99.9), Cjpegli
  family 0.5 (float sample pipeline measured far cleaner: Q=1 p99 ≤ 0.24 —
  don't weaken valid high-q projection). Negligible at low q by construction
  (1.5/Q vanishes as Q grows).
- **Falsified: skip-zeroed-bands rescue for trellis/AQ.** Hypothesis was
  that mozjpeg/zenjpeg 15Q violations sit only on zeroed runs; the nonzero
  split disproves it — coded coefficients also violate (mozjpeg-nz p99 up to
  1.7, max 15Q, 8–17% violating). Trellis moves coded coefficients multiple
  steps for rate. No band-conditional skip can restore the truth-in-box
  guarantee for those families; their tail remains the documented
  approximation, arbitrated by eval only.
- jpegli at high q is nearly round-to-nearest (float pipeline) — the 0.45
  relative family slack is about AQ at LOW q, not high.
- Validation eval (high grid, per_sub=3, slack_abs active in Auto) launched;
  verdict lands with the next benchmarks commit.

### Generation loss (gen2/gen3) — measured 2026-07-26 (tower; benchmarks/gen_eval_2026-07-26.tsv)

User-directed question: how does the pipeline behave on 2nd/3rd-generation JPEGs?

- **The model's gain does NOT collapse on multi-gen input** — it grows: at
  matched final q, g2-social +3.22 ssim2 (vs +2.13 single t75), g2-cdn +3.02
  (vs +2.30 single m70), g3-meme +5.13 (vs +4.25 single t50), g3-deep +6.09.
  Multi-gen inputs carry more total artifact energy and the blind
  pixels-in/pixels-out model removes proportionally more of it.
- **But generation damage is mostly permanent**: restored g2-social lands at
  75.06 vs 78.92 for restored single-t75 — the model recovers only ~20% of
  the generation-loss delta (identity gap 4.96 → restored gap 3.86). That
  ~3-4 ssim2 gap is the headroom for gen-aware training (augmentation is
  implemented: ZENSR_GEN2/ZENSR_GEN3 in make_dejpeg_data4, default off).
- **S10 projection stays safe on multi-gen** (it certifies the FINAL
  generation): proj−noproj ≥ 0 on 13/14 chains; the one negative is −0.05 on
  g2-upq (60→90 re-encode: tight q90 boxes lock in gen1 damage) — neutral,
  not harmful. At low-q finals projection is ~0 as expected (huge boxes).
- **Grid-misaligned chains (crop+re-encode) are harder but not pathological**:
  g2-shift2 +2.60 vs g2-social +3.22.
- **g2-upq retro-validates pixels-only**: probe says "q90, mild" but the
  model corrects what it SEES (+3.37) — a severity-conditioned model would
  have under-corrected exactly here (the conditioning-ablation verdict,
  independently confirmed on generation loss).
- Approach going forward: enable ZENSR_GEN2≈0.25 / ZENSR_GEN3≈0.08 on the
  next dataset build and A/B against dejpeg4_policy on the gen chains; a
  double-quantization detector (DCT-histogram periodicity) stays a
  rung-if-needed, not a default.

### slack_abs validated + high-q identity gate (2026-07-26, benchmarks/dejpeg_proj_highq_slackabs_*.tsv)

High-grid A/B vs the 07-25 baseline (same files/grid/models; the only delta
is slack_abs in Auto projection):

- **slack_abs works where the mechanism said it would**: q93 residual erased
  (turbo −0.24→−0.02; moz +0.06→+0.38; jpegli +0.12→+0.26; zen +0.04→+0.13),
  q90 improved for all four families. Mean high-grid gain +0.160→+0.225.
  Cost: moz q85 −0.14 (wider boxes, slightly weaker corrections).
- **q96 reframed — it was never a projection problem.** proj−policy is
  POSITIVE at every q96 cell (+0.15..+0.81): the projection adds value; the
  MODEL loses to identity on near-pristine input (policy arm −0.5..−1.1).
  My earlier "q96 residual traced to calibration tails" was only part of
  the story; "solved by slack_abs" (113722d7 commit message) was premature.
- **Fix: measured high-q identity gate** (`policy_high_q_identity`, default
  on): skip the model at probe q>=94.5 (IJG/Moz scale, exact at these q) or
  d<=0.6 (Cjpegli family; q96 reads 0.3–0.5, q93 reads 0.7–1.0 and stays
  modeled). Top-end analog of the user-endorsed low-q deblock policy; the
  decode is already consistent so projection is skipped too (no-op).
  Analytic effect on the grid: negative cells 5/16 → 1/16 (turbo q93 −0.02
  marginal); mean high-grid gain +0.35. Report flag `skipped_model_high_q`.

### Boundary4Tap on high-q content — no slot (2026-07-26, user-prompted; from committed grids)

zenjpeg's second deblocker (H.264-style 4-tap pixel filter; zenjpeg Auto uses
it above ~q50) was already measured by the high-grid identity_auto arm:
- q96: BYTE-IDENTICAL to pixel-exact decode — strength derives from DC quant
  (0.25*dc_quant, threshold 0.4x; deblock/boundary.rs), so at DC quant 1 it
  is a structural no-op. Cannot help where the model lost to identity;
  independently confirms the high-q identity gate.
- q85–93: small net LOSS (turbo −0.19..−0.29, moz −0.14..−0.32 ssim2; butter
  also worse) — it smooths real detail where block edges barely exist. The
  model+projection arm (+0.35 mean) dominates.
- mid-q 55–75: alone +0.1..+0.35, but the model is +2..+4 and prefers
  pixel-exact input (model_off >= model_auto, deblock-mix verdict).
Policy unchanged: Knusperli (Annex-K, q<=9.5) -> model+projection -> identity
(q>=~95); Boundary4Tap dominated everywhere in between.

### Chained dejpeg->SR vs jpeg-specialized direct SR — CHAIN WINS (2026-07-26, user question; benchmarks/chain_eval_2026-07-26.tsv)

Protocol: systems_eval LR (catmullrom half + turbo 420) x q35/50/75, 24 eval
files; SR = nomosuni span/compact x2 (both degradation-trained); chain = full
restore_jpeg prod call (S10 projection incl.) then SR on restored planes.

Paired per-file ssim2 (chain - direct):
  q35: span +1.85 median (22/24 wins), compact +1.09 (19/24)
  q50: span +0.85 (18/24), compact +0.67 (16/24)
  q75: ~+0.4 median but ~0 mean, wins 13-15/24 — neutral
Butteraugli agrees (chain <= direct at q35/50).

VERDICT: separate the steps. SR does NOT need to jpeg-specialize — the
S10-projected x1 stage removes artifacts better than the SR models absorb
them internally (it works at input resolution, in the quantization-native
space, with coefficient information SR never sees), and hands SR an input
closer to its training distribution. Chaining INCREASED the gain at q<=50
and is neutral at q75. Runtime cost: ~+80% over SR alone per prod_bench
(restore 5.3 s/MP + span-x2 ~4.2 s/MP-input at 12T). Per-file tail at q75
is symmetric (+3.8/-4.6 extremes) — optional future refinement: skip the
x1 stage above ~q75 if tail-risk matters more than the median.

### Attribution: control separates training-time from specialization (2026-07-27)

dejpeg4b_control (+16k steps, ORIGINAL mix — benchmarks/dejpeg4b_control_std_*.tsv)
vs dejpeg4 baseline, paired std-grid ssim2 (photo rows / graphics rows):

- **control: FLAT** (photo −0.011 median, graphics +0.031) — the +16k-steps
  confound was a mirage; dejpeg4 was converged. All specialist deltas below
  are REAL effects (vs control):
- **graphics specialist (dejpeg7): wins BOTH classes** — photo +0.085 median
  (207/320), graphics +0.086 (209/320). Training on harder text/edge content
  improved general artifact discrimination. Ship-candidate as DEFAULT
  pending its high-grid safety eval (running).
- **YCbCr (dejpeg6): split** — photo −0.059 median (141/320, loses), graphics
  +0.296 median / +0.405 mean (wins clearly). S5b stays falsified as the
  general pipeline (also lost the high grid outright) but is a real
  GRAPHICS-specialist trait → compounding rung dejpeg9_gfxycc (graphics data
  x YCbCr space) training on jason.
- gen-aware (dejpeg8): deep chains +0.3..+0.45, light chains −0.1..−0.2,
  singles −0.05 — a wash for typical traffic, only deep-gen corpora want it.
  Not default; available as a routing option if gen-heavy traffic emerges.
- realtime tier: rt32 direct-train keeps 15–45% of quality-tier gain with
  ZERO negative cells (safe blind); rt32-DISTILL (teacher targets, S9
  recipe) exported — eval queued. rt24 rung on mac.

### Default-swap + realtime tier verdicts (2026-07-27)

- **dejpeg7 clears the default bar** (benchmarks/dejpeg7_high_2026-07-27.tsv):
  high grid +0.97/+0.69/+0.35 at q85/90/93 (vs dejpeg4's +0.80/+0.42/+0.19),
  q96 = 0.00 (identity gate), NO negative cells. With its std-grid wins on
  both classes (vs control): strictly better default. "graphics" in the name
  is a training-data label — it ships as the general model.
- **S9 distillation works at x1** (rt32d vs rt32, benchmarks/rt32d_std_*):
  +0.41 mean paired (745/960), q15 1.10->1.83, q35 0.98->1.64, q55 ->1.43,
  q75 ->0.84 — 26-69% of teacher quality at 0.68 s/MP (7.9x faster).
  q90 -0.00 neutral (near-clean edge traded away; the >=q95 gate covers the
  top). Ship shape: quality tier = dejpeg7-class; realtime = rt32d-class.
- Pending: dejpeg9_gfxycc compound rung (graphics x YCbCr, jason) — decides
  whether the chooser routes graphics to a compound specialist or retires;
  rt24 rung (mac) for the speed floor.

### FINAL production shape (2026-07-27, closes the 07-26/27 autonomous wave)

Model ladder (all golden-verified; std-grid mean ssim2 gain / full-pipeline speed @12T):

| tier | model | params | q15 / q35 / q55 / q75 / q90 | s/MP |
|---|---|---|---|---|
| quality (DEFAULT) | dejpeg7_graphics | 595k | +7.1*/+3.5*/+2.6*/+1.2*/+0.07* (dejpeg4 cols; dejpeg7 beats them per attribution & high grid) | 5.3 |
| realtime | dejpeg_rt24d (distilled) | 43k | +1.63/+1.58/+1.42/+0.86/+0.04 | 0.21 |
| low-q graphics route | dejpeg9_gfxycc | 595k | graphics rows +1.58/+0.65/+0.39 OVER dejpeg7 at q15/35/55; negative q75+ | 5.3 |

- **rt24d == rt32d quality at 3.2x speed** (43k saturates the S9 recipe;
  rt32d retired). Zero negative cells on both -> safe for blind mass use.
- **Routing rule (measured)**: default dejpeg7; IF chooser p_graphics>0.85
  AND probe est-q<=60 -> dejpeg9_gfxycc (the "more aggressive correction"
  slot from the user directive; bounded downside: photo-misroute at low q
  costs -0.14 median vs +0.4..+1.6 upside on target class). High-q identity
  gate (>=q95) and low-q Knusperli policy (Annex-K <=9.5) unchanged.
- Every number above is a committed TSV in benchmarks/ with box+commit
  provenance; models mirrored to Tower.

### Chain verdict, 4:4:4 arm (2026-07-27, mac; closes the 420-only scope flag)

Same protocol at 4:4:4 (benchmarks/chain_eval_444_2026-07-27.tsv; ZENSR_CHAIN_SS=444):
  q35: span +1.04 median (17/24), compact +0.38 (15/24) — chain still wins clearly
  q50: ~neutral (span +0.09, compact −0.12)
  q75: chain LOSES slightly (span −0.31, compact −0.40, wins 8-9/24)
vs 420 where chain won q35/q50 and was neutral at q75. Mechanistically consistent:
the x1 stage's biggest lever over direct SR is chroma repair on the subsampled
lattice (back-projection + native-space work); at 4:4:4 that lever is absent, so
the crossover point drops from ~q75 to ~q50. REFINED ROUTING: chain the x1 stage
when (ss==420) OR (est-q < ~50); skip it for 444 high-q before SR. Verdict
"separate the steps" unchanged — the switch is per-image, cheap, and probe-driven.
(Cross-box note: 444 arm ran on the mac with brew cjpeg vs dev-box turbo — the
within-run chain-vs-direct comparison is box-consistent; only cross-grid absolute
levels carry the box difference.)

### YCbCr de-confound: from-scratch pair (2026-07-27, user-prompted; benchmarks/scr_{rgb,ycc}_std_*.tsv)

User challenge: the warm-start comparisons inherited RGB-tuned convs — was the
S5b falsification an artifact? From-scratch pair (identical seed/box/steps/LR,
jason 3070, 16k @2e-4), paired ycc−rgb:

- ALL −0.137 median / −0.444 mean (416/960); photo −0.360 median / −1.031 mean.
- graphics rows: **+0.131 median** — same direction/size class as warm-start
  (+0.30) and as gfxycc's low-q graphics gains. The graphics trait is robust
  across BOTH bias regimes.
- The warm-start run's q15 advantage (+0.45 median) did NOT survive
  de-confounding (scratch q15 −0.44) — it was largely extra effective
  training, not the color space.
- Both experiments carry opposite convergence biases (warm favors RGB init;
  scratch tests equal-step convergence where YCbCr's low-variance chroma
  channels may learn slower) and AGREE qualitatively → S5b-as-general-model
  falsified robustly; YCbCr survives only as the graphics-specialist trait
  (dejpeg9_gfxycc). Residual caveat: fixed 16k budget tests equal-compute,
  not asymptote; not worth another rung given three concordant comparisons.
- context: scratch-rgb sits −0.92 median under production dejpeg4 (16k from
  scratch is undertrained vs the warm lineage — as expected; pairing is
  internal so this doesn't affect the ycc−rgb comparison).

### SCOPE CORRECTION on the YCbCr falsification (2026-07-27, user-prompted)

What the three concordant comparisons falsified is the COLOR BASIS: feeding the
same full-res pixel CNN YCbCr instead of RGB. That is theoretically ~neutral
anyway (RGB<->YCbCr is a fixed 3x3 linear map the first/last convs absorb; only
optimization dynamics differ — Cb/Cr low variance => smaller gradients, matching
the scratch result). They did NOT test a LATTICE-AWARE architecture: both arms
received chroma already upsampled to full res (zenjpeg DecodedYCbCr also returns
upsampled planes), so neither model ever saw the 4:2:0 grid. The S10 projection
is native-space/native-lattice in ALL arms (Y full-res box, chroma back-projected
on the half-res lattice) — the pipeline's frequency-domain component was never RGB.
OPEN (S5 two-trunk, the right next rung): Y trunk full-res + chroma trunk at
half-res fed PRE-UPSAMPLE chroma, reconstructible today via decode_coefficients
-> dequant+IDCT at native res (consist.rs has the exact basis) — no zenjpeg
change needed. A-priori ceiling on photos is modest (decoder upsampling + S10
back-projection already enforce the lattice constraints; chroma = 2 of 6
samples per 2x2), which the persistent graphics-only edge hints at — but that
is an argument, not a measurement.
Also recorded: strength tiering (direction robust across 3 designs; magnitudes
single-seed, ~24 effective units/grid, equal-compute-not-asymptote) and the x2
protocol caveat (synthetic catmullrom LR = valid for relative A/Bs incl. the
chain verdict, weaker for absolute real-web gains; x1 evals — ALL the ycbcr/
specialist/control verdicts — use the pristine original as GT with no downscale).

### Chroma ceiling + lattice decomposition (2026-07-27; benchmarks/chroma_ceiling_2026-07-27.tsv)

Oracle-chroma probe (swap GT Cb/Cr into pipeline output, turbo 420, dejpeg7):
restored->restored_oc gap = +10.1/+6.8/+6.2/+4.8/+3.1 ssim2 at q15/35/55/75/90 —
chroma is ~HALF the remaining error mass, and the current pixel CNN recovers
almost none of it (decode-side chroma gap 7.6 -> restored-side 6.8 at q35: the
model's +5.9 total gain is nearly all luma).

Lattice-floor decomposition (GT chroma, NO quantization, 2x2 box down +
bilinear up = 93.21 ssim2, loss 6.79): at q>=35 the ENTIRE oracle gap sits at
or under the lattice floor -> ~0% is unrecovered quantization damage. VERDICT
on S5-two-trunk (coefficient-faithful native-lattice arch): ~zero headroom
above q35 — decoder upsampling + S10 back-projection already saturate what the
coefficients contain. Only q15 keeps a quant-side component (+3.3 of +10.1),
matching where ycbcr-space training always showed its graphics edge.

REDIRECT: the recoverable pool is GUIDED CHROMA SR — luma-guided sharpening
past the lattice (the 6.8-point pool; the model already SEES full-res luma but
RGB charbonnier underweights chroma so it doesn't try). Cheapest lever first:
chroma-weighted loss (ZENSR_LOSS_SPACE=ycbcr + ZENSR_CHROMA_W, loss-only
change, model stays RGB drop-in). Paired experiment launched on jason:
dejpeg10_chw3 (w=3) vs dejpeg10_ctl (plain), both warm from dejpeg7_16000.
Caveats: decomposition is a crude subtraction in ssim2 space; bilinear
lattice floor slightly understates the best resampler; ssim2 (XYB-based)
weights chroma heavily — butteraugli columns in the TSV allow a cross-check.

### Guided-chroma rung 1: chroma-weighted loss — NULL (2026-07-27; benchmarks/dejpeg10_*_2026-07-27.tsv)

chw3 (YCbCr loss, Cb/Cr x3) vs paired plain-loss control, both warm from
dejpeg7_16000: ssim2 +0.009 median / 500 of 960 (420-only q-slices all within
noise), butteraugli flat. Loss reweighting does NOT unlock the chroma pool —
the model isn't ignoring chroma because of loss weighting; either the
luma-guided signal is weak in practice (the oracle bound includes truly
destroyed information) or a full-res conv stack can't express the guided
upsample. Also: both +16k continuations sit ~0.06 BELOW their dejpeg7 init —
re-confirms dejpeg7 is converged (third flat continuation).
Next (decisive, training-free): joint-bilateral-upsample arm on CLEAN
half-res chroma with GT luma guide vs the bilinear lattice floor (93.21).
If classic guided upsampling can't beat the floor materially, the practical
pool is far smaller than the 6.8 oracle bound and the chroma direction gets
CLOSED with measured bounds at every rung.

### Guided-chroma rung 2: JBU bound — NEGATIVE; chroma direction CLOSED (2026-07-27)

Joint-bilateral 2x chroma upsample with a GROUND-TRUTH luma guide on CLEAN
half-res chroma (best case for guided upsampling; benchmarks/
chroma_ceiling_v3_2026-07-27.tsv): 93.21 bilinear -> 90.95 JBU (-2.25 mean,
median -0.57, worst -13.2 documents, best +2.7 screen). Luma-guided range
weighting pulls chroma across mismatched edges — luma structure does not
reliably predict chroma structure on this corpus.

CLOSED with three concordant bounds: (1) oracle pool is ~all lattice loss
above q35 (destroyed information, not recoverable-by-fidelity); (2) chroma-
boosted loss on the end-to-end model: null; (3) classic guided upsampling
with a perfect guide: negative (untuned single sigma — caveat recorded).
Decoder upsampling + S10 back-projection are already at the practical
chroma frontier for 420. Re-open only via a dedicated learned chroma-head
rung with a sigma-swept/learned guide, expected value LOW. The honest
answer to "how can RGB be as good as YCbCr at 420": the chroma information
a native pipeline could read better simply isn't in the file — and what
could be hallucinated past the lattice is not reliably luma-predictable.

### f16 ship format + compression study (2026-07-27/28)

- f16 BAKED IN: trainer exports weights_f16.raw and generates goldens THROUGH
  the f16 roundtrip; verify + eval loader use f16 when meta carries
  f16_goldens (27 dirs pass, max golden diff 9.8e-6 vs f16-goldens; measured
  output delta vs f32 <=9e-4 = ~1/4 of an 8-bit step). Legacy dump dirs
  (span48, general_*) stay f32 pending their own pipeline re-dump — NOTE:
  general_* torch re-repro mismatches runtime at 1e-3; their goldens were
  restored from Tower after a near-poisoning; do NOT regenerate them with
  train_people's Student.
- Reproducibility: model meta.json now embeds a repro block (argv, ZENSR_*
  env, seeds, commit, host, torch version, init sha, full dataset meta,
  f16 sha) and every export writes an executable repro.sh.
- COMPRESSION (study in this section; script inline in jj history):
  f16+zstd19 = 1.07-1.09x; byte-plane split = 1.15x (dejpeg7) — conv weights
  are near-max-entropy. zenpredict zero_bias (tau*max per block) DOES NOT
  TRANSFER from picker MLPs to conv kernels: tau=0.002 already corrupts
  output 9e-2 (100x f16 noise) for +2% ratio; tau=0.02 = 0.52 corruption for
  1.26x. VERDICT: ship plain f16 (dejpeg tier 1.16MB, rt24d 86KB); skip
  container compression (<=15% for real complexity); zero-bias closed for
  conv SR. Reuse from zenanalyze stack: bake-format ideas only; zentrain is
  picker-specific — nothing to link for SR training.

### f16 metric-level A/B — TRANSPARENT (2026-07-28; benchmarks/f{16,32}_metric_ab_*.tsv)

320 paired cells (dejpeg7, full pipeline, loader f16 vs ZENSR_LOAD_F32):
ssim2 median +0.0010, butteraugli median -0.0001; tails symmetric on 3/320
knife-edge text cells (-0.24..+0.61, largest delta favors f16 = reshuffling).
f16 ship format fully cleared at golden, output, and metric levels.

### Cross-arch runtime benches (2026-07-28; benchmarks/neon_mac_benches_*, lianli logs)

Full restore_jpeg fits (total_ms = a + b*MP), quality tier / rt24d @12T:
  lianli (bare Linux x86): 2.71 s/MP (439 GFLOPS agg) / 0.153 (565)
  WSL dev box:             5.3        (225)           / 0.21  (410)
  mac M4 Pro (NEON):       4.45       (267)           / 0.253 (342); 1T rt =
  2.03 s/MP = 42.6 GFLOPS/thread — the FASTEST single-thread of the fleet
  (NEON path healthy; MT limited by 4 E-cores in the 12).
Implications: cache penalty at nf=64 is box-dependent (lianli 78% of rt-tier
rate vs WSL 55%) -> tile/strip-fusion headroom ~25% on good caches; Winograd
(2.25x flops) is the dominant portable lever. WSL numbers understate
production Linux by ~2x on the quality tier.

### Tile sweep (lianli, 1MP, 12T; benchmarks/tile_sweep_lianli_2026-07-28.tsv)

48:4575ms 64:3353 96:2620 128(default):2726 192:3163 256:3659.
Default 128 is within 4% of the optimum (96); curve shallow 96-128, steep at
extremes (tile 48 pays ~3x halo recompute; 256 blows cache). Tile lever
EXHAUSTED — Winograd F(2x2,3x3) is the next and dominant CPU rung (2.25x
arithmetic; golden-tolerance gate mandatory; same-box retime on lianli).

### Winograd F(2x2,3x3) v1 — correct but FALSIFIED on speed (2026-07-28)

Implemented (crates/zensr-micro/src/wino.rs): interior 2x2 tiles, 16-tile GEMM
batches, scalar borders, exact zero-pad semantics. Correctness clean first
try: kernel-equivalence test <1e-4 across odd sizes/strides/asym channels;
ALL 31 model goldens pass with it active (f16 checks at 9.95e-6).
Retime (lianli, 1MP, 12T, paired): quality 4834ms vs 2699 direct; rt24d 277
vs 154 — 1.8x SLOWER both tiers. The 2.25x multiply cut is swamped by scalar
strided transform gather/scatter + an untier'd GEMM vs the 439-565 GFLOPS
magetypes direct kernel. Default OFF (ZENSR_WINOGRAD=1 opt-in).
v2 bar (if ever re-attempted): vectorized tile transforms (contiguous
[T]-major layouts), f32x8/f32x16 GEMM microkernel via magetypes, and it must
beat 2.7 s/MP quality-tier on lianli — realistic ceiling ~1.3-1.5x over
direct per literature; EV modest. CPU opt status: direct kernel STANDS as
the frontier (tile default within 4%, cache levers ~25% box-dependent,
winograd-naive falsified).

### Winograd v2/v3 (magetypes-tiered) — still loses; root cause PROFILED (2026-07-28)

User challenge "insufficient intrinsics" was half right: proper tiering
(deinterleaved even/odd staging -> all-contiguous vector transform loads,
quad-blocked FMA GEMM, per-tier f32x8/f32x16 via the macro) cut the deficit
1.8x -> 1.17x (quality tier 3184 vs 2701ms; rt24d worse: 236 vs 152).
Row-wide U amortization (v3) changed nothing -> weight-traffic hypothesis
falsified. perf stat (lianli, quality tier, whole run):
  wino=1: 1825G instr / 364G cyc = IPC 5.0, 17.3G L1d misses
  wino=0:  864G instr / 323G cyc = IPC 2.7, 30.1G L1d misses
VERDICT: wino executes 2.1x MORE INSTRUCTIONS at near-peak IPC — the 2.25x
multiply cut is erased by transform bookkeeping, the stride-2 scalar
scatter (~hundreds of scalar instr/px across 64 oc), and bounds checks on
runtime-strided slice accesses. Not memory-bound, not stall-bound:
instruction-count-bound. A v4 would need ~3x fewer non-GEMM instructions
(vectorized interleave via shuffle stores, hoisted-bounds patterns,
fused transform+GEMM registers) for a best-case ~1.3x quality-tier win and
a certain rt-tier loss -> LOW EV; direct kernel remains the CPU frontier.
Wino stays opt-in (ZENSR_WINOGRAD=1), correctness-tested at both scalar and
vector tiers. Engineering note: a bad edit anchor amputated simd.rs mid-
iteration; rebuilt from committed base — commit kernel files BEFORE
iterating on them.
