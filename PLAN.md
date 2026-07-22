# zensr — long-term plan: CPU super-resolution for web JPEGs

**Mission:** be the best compact CPU engine at upscaling (and repairing) images encoded with
**libjpeg-turbo and mozjpeg** — the encoders that produce the overwhelming majority of
imageflow's real inputs. Fidelity-first, web-focused, shippable inside a .dll.

**Unfair advantages this plan is built around:**
1. **We own the decoder.** zenjpeg hands us exact quant tables, subsampling mode, progressive
   flag, and (optionally) native Y/CbCr planes before chroma upsampling. Everyone else does
   blind restoration; we can do *exact-conditioned* restoration.
2. **We own the target encoders.** mozjpeg-rs (ours) + pinned libjpeg-turbo mean training
   degradations are the *actual* deployment degradations, bit-for-bit reproducible.
3. **We own the runtime.** zensr-micro (400 KB-class, AVX-512, seam-exact tiling, f16 weights)
   defines a hard op vocabulary that doubles as an architecture-search constraint.
4. **We own the metrics.** zensim / fast-ssim2 / butteraugli for eval; imazen-26 for
   provenance-clean subcorpus evaluation.

**Non-goals:** GPU inference; diffusion/GAN hallucination as default (a light-GAN quality tier
is opt-in, fidelity tier is the default); attention/Mamba architectures (CPU-hostile); chasing
leaderboard PSNR on bicubic-DIV2K (not our distribution).

---

## Engine constraints (architecture search space)

Models must compile to the zensr-micro op vocabulary — this is a feature, not a limitation:

- conv3×3 (+bias), conv1×1, grouped conv3×3
- elementwise: SiLU, PReLU*, LeakyReLU*, sigmoid gate, add/mul, residual adds*
- PixelShuffle(s) for s ∈ {1, 2, 3, 4}* (s=1 ⇒ pure artifact removal)
- channel concat; nearest-upsample residual*
- (*) = small planned additions, Phase 0

**Three product tiers** (all within the op vocabulary; targets are single-thread on the
7950X-class reference box, MT scales ~4-6× via tiling):

| tier | params | compute (per MP-out, ×4-equiv) | speed target (1T) | role |
|---|---|---|---|---|
| **realtime** | ≤ 0.05 M | ~2–5 GFLOP (≈2–4× Lanczos cost) | ≥ 15 MP-out/s | previews, video stills, on-the-fly serving; ×2-first |
| **fast** | ≤ 0.20 M | ~9–12 GFLOP | ≥ 3.5 MP-out/s | default imageflow upscale (SPAN-class) |
| **quality** | ≤ 1.3 M | ~150 GFLOP | ≥ 0.25 MP-out/s | opt-in max quality (Compact-class) |

Weights ship f16 (proven transparent, 75–76 dB). Scale priority is **×2 first, then ×4, ×1, ×3**
— web traffic is dominated by ≤2× enlargement (thumbnail→retina); challenge culture over-indexes
on ×4. Runtime perf discipline: **retime after every kernel-adjacent commit** (26x-incident
lesson; see README post-mortem) — keep `#[arcane]`-adjacent bodies under the MIR inline cap.

## Degradation model (the science core)

Web JPEG reality we must match (sampled, not full cross-product; per-sample metadata recorded):

| axis | values | notes |
|---|---|---|
| encoder | libjpeg-turbo (pinned ver), mozjpeg (mozjpeg-rs, pinned) | per-encoder quant-table families differ (Annex-K scaling vs mozjpeg tables + trellis) |
| quality | **dense low end per workspace rule**: q5–70 step 5, q70–95 step 2 | web weight centered ~q30–85 |
| subsampling | 4:2:0 (dominant), 4:4:4, (4:2:2 minor) | 420 chroma artifacts are half the problem |
| progressive | on/off | mozjpeg default progressive |
| pipeline order | (i) HR→jpeg→upscale (repair+SR); (ii) HR→downscale→jpeg→upscale (CDN thumbnail); (iii) double-encode chains (re-saves, q₂<q₁ and q₂>q₁) | (ii) is imageflow's core case |
| pre-scale kernel | the kernels CDNs actually use: Lanczos3, CatmullRom, box/area | zenresize |
| minor extras | slight blur/noise before encode (camera/app pipelines), EXIF-stripped re-saves | Real-ESRGAN-style second-order, but with REAL encoders |

Ground truth = the pre-degradation image at target scale (for (ii), the HR itself).

## Science questions (each falsifiable, each an experiment with a decision it informs)

- **S1 Encoder-match:** does training on turbo+mozjpeg beat simulated-JPEG (DiffJPEG-style) and
  single-encoder training, evaluated per-encoder? → decides degradation generator scope.
  *Falsified if* generic-JPEG training is within noise on both encoder test sets.
- **S2 Conditioning:** blind vs (a) q-scalar input channel, (b) **q-banded weight swapping**
  (2–4 models selected by decoded quant table; zero runtime cost, f16 files are 300–600 KB so
  a 3-band zoo is ~1–2 MB), (c) FiLM scale/shift (within op vocab). Hypothesis: (b) captures
  most of the win. → decides product shape.
- **S3 Pipeline-order coverage:** does joint training on (i)+(ii)+(iii) degrade the pure-(ii)
  CDN case vs a (ii)-specialist? → decides whether we ship one model or per-flow models.
- **S4 Capacity/arch frontier:** SPANF-32 vs SPAN-48 vs Compact-16/32 at matched degradation
  training; loss ablation: Charbonnier vs +DISTS/LPIPS vs light-GAN (quality tier only).
  Eval on zensim/ssim2/butteraugli, not just PSNR. → locks the two shipped tiers.
- **S5 Native-plane input (Phase ≥3):** model consumes zenjpeg's Y(full) + CbCr(half) planes +
  a chroma trunk with PixelShuffle(2) merge, skipping decoder chroma upsampling entirely.
  Potentially our biggest quality differentiator on 4:2:0. → decides deep-decoder integration.
- **S6 1× repair mode:** same pipeline, scale=1 (deblock/dering only) as an imageflow filter.
- **S7 Quant-table features:** beyond q-banding, do cheap table-derived scalars (per-band quant
  energies) as broadcast input channels help at band boundaries?
- **S8 Realtime frontier:** at the ≤0.05 M budget, which shape wins on degraded inputs:
  (a) RVSR/RT4KSR-class reparam plain-conv stacks; (b) ZenoSR-class NTIRE param-champ
  distillates; (c) a process-at-low-res shape (learned downscale → convs at ½ res → larger
  PixelShuffle) — the Bicubic++ *pattern*, independently reimplemented and retrained (their
  code is CC-NC; the architectural idea is not copyrightable — zero code reuse, own weights).
  → locks the realtime tier arch.
- **S9 Distillation ladder:** quality→fast→realtime teacher-student (feature + output
  distillation, standard ESR-meta recipe). Hypothesis: distillation closes ≥30 % of the
  realtime↔fast quality gap for free at inference. → becomes the default training topology
  if confirmed.

Discipline: every experiment lands in `benchmarks/` with commit + config + n; conclusions that
inform shipped constants follow the workspace sweep/calibration rules (dense axes, held-out
validation, no hand-distilled N=1 anchors). Falsified branches get recorded in PLAN.md
(appendix) so they are never re-run blind.

## Data plan (workspace ML-data discipline applies in full)

- **HR corpus ("imazen-sr-hr-v1"):** commercially clean only — imazen-26 (existing provenance),
  Unsplash-license pulls, PD/CC0 (LoC, NASA, NPS, archive scans), own photography. Target
  5–20 k images ≥ 1200 px. **Licensing gate before any training run** (DIV2K/Flickr2K/LSDIR are
  research-grade — usable for *ablation pilots only*, never for shipped weights).
  Storage: `/mnt/v/zen/zensr-training/hr-v1/` + R2 (`s3://zentrain/zensr/hr-v1/`) + Tower
  mirror; `PROVENANCE.md` + `_MANIFEST.json` with `build_commit` per the rules.
- **Degraded variants:** generated by a `zensr-degrade` tool (Rust, in-repo `tools/`):
  content-addressed encoded bytes (**always persisted** per the no-exceptions rule), Parquet
  sidecar rows: `hr_sha, encoder, version, q, subsampling, progressive, chain, prescale_kernel,
  scale, encoded_sha, quant_tables_hash, bytes`. Regeneration is one command from HR + manifest.
- **Splits:** by origin (even/odd convention like the canonical picker datasets) — no rendition
  leakage; per-subcorpus held-out test sets frozen before first training run.
- **Canonical index:** this repo's `DATA.md` + an entry in `~/work/zen/DATA_PROVENANCE.md`.
- **Compute:** degradation gen on CPU via **zenfleet** (never hand-rolled fleets); training on
  local RTX 5070 (pilots) → vast.ai GPUs via zenfleet GPU executors (full runs); checkpoints to
  R2 with pointer files in-repo (nothing > 30 KB binary in git).

## Training stack

- Base: **neosr** (Apache-2.0, local clone, supports SPAN + Compact archs, OTF degradations) —
  adapted with a real-encoder degradation hook (subprocess/FFI to pinned cjpeg + mozjpeg-rs;
  pre-generated variant pools to keep GPUs fed). Decision checkpoint P2: if adaptation fights
  us, fall back to a minimal in-repo trainer (basicsr-style, ~1 kLoC).
- Export path stays: torch → fixed-order weight dump → zensr-micro golden verification (every
  trained model gets torch goldens at 64² + tiny shapes, same gates as SPANF today).
- Fidelity tier losses: Charbonnier + optional DISTS. Quality tier: + light UNet-GAN
  (Real-ESRGAN recipe) — clearly labeled, never the imageflow default.

## Evaluation protocol (the bar)

`zensr-bench/eval` extended with the JPEG axis: per subcorpus × encoder × q ∈ {30, 50, 70, 85}
× pipeline-order, methods = {ours-fast, ours-quality, Lanczos, realesr-general-x4v3 (baseline
to beat), SPANF-bicubic (regression guard)}, metrics = {psnr, ssim2, butteraugli-n3, zensim}
(+ ssim2/cvvdp-style baseline rows per workspace eval rules; medians AND p10 worst-case).
Reports committed to `benchmarks/` with commit hashes.

**Acceptance gates:**
- G1 (P0): baseline report exists; realesr-general-x4v3 runs in our engine bit-verified.
- G2 (P4 fast tier): beats Lanczos on degraded inputs on ≥7/8 subcorpora by ≥1 dB PSNR-equiv
  AND ≥5 ssim2 median at q∈{50,70}, ≤5 % clean-input regression vs SPANF-bicubic, ≥3 MP-out/s
  single-thread.
- G3 (P4 quality tier): beats realesr-general-x4v3 on turbo+mozjpeg test sets at ≤ its compute,
  on zensim AND ssim2 medians, no p10 collapse.
- G2rt (P4 realtime tier): ≥ 15 MP-out/s single-thread; beats Lanczos on degraded inputs by
  ≥ 0.7 dB PSNR-equiv and ≥ 4 ssim2 median at q∈{50,70} on ≥ 6/8 subcorpora; ≤ 4× Lanczos cost.
- G4: 4:2:0 chroma: measurable win from S5 or documented falsification.
- G5 (P5.5): at least one top-3 sub-track placement (runtime/params/overall) in a public
  challenge with shipped-identical models, or a documented near-miss analysis feeding P6.

## Path to SOTA — definition, evidence, and the drive

**The claim we are driving to** (falsifiable, scoped, honest): *"State of the art in CPU-budget
super-resolution of real-encoder (libjpeg-turbo/mozjpeg) compressed web images: at each of
three compute budgets, the best published quality on a public, reproducible benchmark — and
competitive results in the corresponding public challenges."* We do NOT claim global SR SOTA
(HAT/diffusion territory); we define and then dominate the niche that matches real web serving.

**Evidence pillars (all three required before the word "SOTA" appears in any README):**
1. **A public benchmark we release** — `webjpeg-sr-bench`: frozen public subcorpus (imazen-26 is
   PD/licensed — publishable), pinned encoder versions + degradation spec, the eval harness, and
   baseline numbers for every runnable competitor. A niche SOTA claim is only meaningful if
   others can run the bench and disagree; releasing it is what makes the claim load-bearing.
2. **The scoreboard, all green at matched compute** — competitors to beat at each tier on the
   bench (medians AND p10, zensim + ssim2 + butteraugli + PSNR):
   realtime: Lanczos/CatmullRom, RT4KSR, RVSR, bicubic++-pattern reimpl;
   fast: SPAN/SPANF (bicubic-trained), ECBSR/ETDS retrained, ArtCNN-class community compacts;
   quality: realesr-general-x4v3 + -wdn blend, animevideov3, best CC-BY community Compact/SPAN
   dejpeg models on OpenModelDB (re-scored per-model licenses permitting).
   The scoreboard lives in `benchmarks/SOTA-SCOREBOARD.md`, regenerated by the eval harness,
   each cell carrying commit + date. SOTA = every cell green at ≤ matched compute.
3. **External validation** — enter the public challenges with the shipping engine+models:
   NTIRE Efficient SR (CVPR, submissions ~Jan–Mar) and the real-world/real-time tracks
   (AIS/AIM RTSR; Mobile Real-World SR). Goal: top-3 in at least one sub-track (runtime or
   params or overall) with the *same* models we ship, plus the engine story (400 KB Rust DLL)
   in the factsheet. Challenge results are the third-party stamp no self-run bench provides.

**Compute budget (order-of-magnitude, revisited per phase):** pilots ≈ 50–100 GPU-h (local
RTX 5070, $0); S1–S4+S8–S9 matrix ≈ 20–30 runs × 8–24 GPU-h ≈ 300–600 GPU-h (vast.ai 4090-class
via zenfleet, ≈ $150–400); production + distillation ladder ≈ 200–300 GPU-h (≈ $100–200).
Total to first SOTA claim: **well under $1k of rented GPU** plus CPU fleet time for data
generation. This is deliberately cheap — sub-0.2 M models train in hours; the moat is the
degradation exactness + conditioning + engine, not brute compute.

**Drive mechanics (how this reaches the end instead of parking):**
- Every phase has a Definition-of-Done gate below; a phase without its gate met does not yield
  to the next except by a written kill/pivot note in the appendix.
- Every experiment run lands in `benchmarks/` same-day with commit + config; conclusions edit
  PLAN.md in the same change (workspace DOCS discipline).
- The scoreboard is regenerated at every P≥3 milestone; regressions block.
- Kill criteria are pre-declared per science question (see S1–S9 falsification clauses) — a
  falsified branch is recorded and never blocks the critical path (which is: data → q-banded
  fast tier → realtime distillate → scoreboard → bench release → challenge entry).
- Standing cadence once P2 lands: no idle GPU-week — there is always a queued experiment from
  the S-list, and always a scoreboard delta to publish.

## Phases

- **P0 — Foundation (now):** this plan; engine additions (PReLU, nearest-residual, scale-param
  PixelShuffle, parametric fc/blocks); Compact + SPAN-48 graph ports with goldens; eval JPEG
  axis + zensim column; **baseline report**. Gate G1.
- **P1 — Data:** HR corpus assembly + licensing gate; `zensr-degrade` + manifests + R2/Tower;
  frozen test splits; DATA.md + workspace provenance entry.
- **P2 — Training bring-up:** neosr adaptation; reproduce SPANF-class bicubic training (sanity
  vs published numbers); first degradation fine-tune pilot on local GPU; export+golden loop
  proven end-to-end.
- **P3 — Science:** S1–S4 (+S7) on fleet GPUs; decisions locked, falsifications recorded.
- **P4 — Production models:** realtime (S8 winner + S9 distillation) + fast (q-banded per S2)
  + quality tiers; ×2 first, then ×4/×1/×3; f16 ship files + goldens; full eval reports;
  first complete `SOTA-SCOREBOARD.md`. Gates G2rt, G2, G3.
- **P5 — Product wiring + bench release:** imageflow/zenpipe integration (decode-metadata →
  band pick → tiled upscale), size diet toward sub-300 KB; **publish `webjpeg-sr-bench`**
  (spec + harness + frozen subcorpus + all baseline numbers) — user-gated like any publish.
- **P5.5 — Challenge entries:** NTIRE ESR + real-world/real-time track submissions with
  shipping models (calendar-driven; prep starts when P4 models exist). Gate G5.
- **P6 — Frontier (ongoing):** S5 native-plane models, PLKSR-Rep-class large-kernel port,
  fusion-retry via per-tier `#[rite]` exp helpers, QAT-int8 if ever needed; scoreboard defense
  as competitors move.

## Standing engineering rules for this repo

1. Retime after every kernel-adjacent commit (paired/interleaved, min-of-N, load recorded).
2. Every model variant ships with torch goldens + arbitrary-dim + tiled seam tests.
3. Weights/datasets never in git — R2/Tower + pointer files; manifests carry `build_commit`.
4. Every experiment writes `benchmarks/<name>_<date>` with config + commit; falsified ideas get
   a line in the appendix below.
5. Licensing: shipped weights only from our-trained or Apache/BSD/MIT-verified sources.

## Appendix: falsified / rejected (do not re-attempt blind)

- +16 channel-stride padding at 4 KiB planes: measured 2× slower (2026-07-22).
- SiLU/gate store-fusion: 26× MIR-inline-cap regression; retry only via per-tier `#[rite]`.
- Row-band tiling: 50–377 MB/thread scratch on wide images.
- int8 weights-only PTQ on SPAN-class: 35 dB photo — broken; QAT or int8-first arch only.
- Bicubic++ / AnimeJaNai / eSR / RAISR / EchoSR / SAFMN / OmniSR / ABPN: license-blocked.
