# zensr — long-term plan: CPU super-resolution for web JPEGs

**Mission:** be the best compact CPU engine at upscaling (and repairing) images encoded with
**libjpeg-turbo primarily** (USER 2026-07-23: turbo-focus is fine), mozjpeg as a secondary
eval axis and later fine-tune — the encoders behind the overwhelming majority of imageflow's
real inputs. Fidelity-first, web-focused, shippable inside a .dll. **Severity-adaptive by
design:** the core ships one small built-in model; an optional external "severity pack"
provides degradation-strength adaptation without growing the core (see S2).

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
- **S2 Severity adaptation (USER 2026-07-23: adapt to severity WITHOUT model growth; optional
  external model set is acceptable).** Severity signal = computed EXACTLY from decoded quant
  tables (zenjpeg): map tables → equivalent IJG quality + subsampling flag → canonical severity
  scalar (mozjpeg tables map too; recompressed files expose only the last encoder's tables —
  a content-based correction is S7's job; **zenjpeg#189 filed 2026-07-24: encoder-embedded
  APPn parameter record would upgrade the severity signal from table-inversion estimation to
  ground truth for zenjpeg-encoded files** — fingerprint fallback stays for everything else).
  Candidate mechanisms, all keeping the core small:
  - **S2a input-conditioning:** severity as extra input channel(s) — single model, +O(100)
    params; risk: soft compromise at extremes.
  - **S2b banded external set:** N specialist fine-tunes selected by severity band; fast-tier
    f16 = ~300 KB/model → 4-band pack ≈ 1.2 MB external, core ships the mid band. Zero runtime
    cost; discrete boundaries.
  - **S2d endpoint interpolation (production-proven: Real-ESRGAN wdn):** TWO endpoint
    fine-tunes (light/severe) from a shared parent; runtime lerps weights by continuous
    severity α, then packs — µs-scale, cacheable per bucket. Continuous control, 2 files
    (~600 KB f16). Requires shared-basin fine-tunes — our warm-start topology gives this free.
  - **S2e LoRA-style severity deltas:** base + per-severity low-rank conv deltas (rank 4–8 ≈
    30–50 KB each, merged into weights at load, zero inference cost). Smallest pack; most
    granular; slight training complexity.
  Decision metric: quality-vs-severity curve (smoothness + endpoint quality) per pack size.
  Hypothesis: S2d wins simplicity/quality; S2e wins if we want ≥6 severity points.
  → locks the product shape: **built-in mid model + optional external severity pack**.
- **S3 Pipeline-order coverage:** does joint training on (i)+(ii)+(iii) degrade the pure-(ii)
  CDN case vs a (ii)-specialist? → decides whether we ship one model or per-flow models.
- **S4 Capacity/arch frontier:** SPANF-32 vs SPAN-48 vs Compact-16/32 at matched degradation
  training; loss ablation: Charbonnier vs +DISTS/LPIPS vs light-GAN (quality tier only).
  Eval on zensim/ssim2/butteraugli, not just PSNR. → locks the two shipped tiers.
- **S5 Native-plane input (Phase ≥3):** model consumes zenjpeg's Y(full) + CbCr(half) planes +
  a chroma trunk with PixelShuffle(2) merge, skipping decoder chroma upsampling entirely.
  Potentially our biggest quality differentiator on 4:2:0. → decides deep-decoder integration.
- **S5b YCbCr-native models (USER 2026-07-24):** run restoration in YCbCr (quantization's
  own space — artifacts axis-aligned; RGB models waste capacity learning the color rotation).
  zenjpeg exposes `DecodedYCbCr`; first rung = same Compact shape, planar YCbCr in/out, x1.
  Prereq for S10 (projection is tight only in the decoder's YCbCr domain). 444 first; 420
  chroma joins via S5's two-trunk shape — **420 chroma reconstruction ≡ guided ×2 SR of the
  chroma planes** (user insight: same op vocabulary, same math; decoder's fixed upsampling
  filter is where a learned PixelShuffle(2) trunk goes).
- **S10 Quantization-consistency guard (USER 2026-07-24, the DCT-table math):** per band
  |c_true − ĉ| ≤ Q[u,v]/2 ⇒ the file defines a convex DCT-space box (POCS constraint set).
  (a) per-block severity map Σ Q²/12 over zeroed/active bands = LOCAL damage conditioning
  (upgrades S7); (b) the box bounds allowed "invention" per band; (c) project model output
  into the box (DCT→clamp→IDCT, zenjpeg owns fast DCTs) ⇒ output PROVABLY re-encodes to the
  same coefficients — bitstream-consistency guarantee, strictly stronger than the bilinear
  clamp. For 420, chroma box lives on the subsampled lattice (SR back-projection form).
- **S6 1× repair mode:** same pipeline, scale=1 (deblock/dering only) as an imageflow filter.
  **DONE 2026-07-24 (first result):** dejpeg_1x (Compact nf64/nc16 s=1, A2c-body warm start,
  25 GPU-min) beats identity at every q on all metrics — q35/q50 worse-rate 2 %, q75 14 %,
  butteraugli 0.99 at q75. The interim ×2-up→down round-trip is retired (it lost to identity);
  s=1 runs natively (zero-channel head padding to the quad multiple). Next: severity bands (S2)
  + mozjpeg axis.
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
  **PILOT RESULT (2026-07-23, rt_distill_2x):** 45,156-param Compact (nf24/nc8 ×2)
  distilled 20 min from the span-2x teacher on 13.5k imazen-26 turbo-JPEG pairs kept
  **68 % of A2c's q35 SSIM2 gain over Lanczos** (+2.5 of +3.7) and beat Lanczos outright
  at q35 (+2.5 SSIM2) — but lost at q75/clean (severity-route those away). Speed
  **21.9 MP-out/s @12T / 3.1 @1T** (systems_bench 2026-07-23). Hypothesis SUPPORTED at
  heavy degradation; next rung = more pairs + longer schedule + nf32 + q-banded students.
  **Rungs 2+3 (same day): capacity FALSIFIED (nf32 = +0.5 SSIM2 for 2.7× compute); TEACHER
  CHOICE DOMINANT — the A2c-teacher rtc_distill_2x at the same 45K matches the teacher's
  butteraugli at every q, lands within 1.9 SSIM2 at q75, and halves the q35 worse-rate vs
  the span-teacher student. S-E = rtc. Next levers: q-banded data emphasis, more pairs.**
  Ops lesson baked into tools/train_distill.py: keep the train set GPU-resident
  (host gather was 10× slower); teacher outputs get a sanity gate before any training.

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
- **Storage (USER DIRECTIVE 2026-07-24): local disk OK for zensr data up to 400 GB total;**
  cleaning `target/` dirs of unclaimed repos is authorized when space runs low (zenjpeg +
  jxl-encoder targets = 291 G reclaimed same day; check `.workongoing` claims first).
- **Compute placement (USER 2026-07-25): compute jobs run on lianli or mac — the dev box is
  the operator.** lianli (.27, RTX 2080 + 24c) carries GPU training AND Rust eval jobs (zensr
  source + pinned eval slice at ~/data/imazen-26-eval + encoders + models provisioned
  2026-07-25); mac (M4 Pro 12c/24G) is the CPU spare. Shared boxes (node-2/node-3) only per
  explicit user direction, idle-verified, flags returned to-windows after.
- **Compute (USER DIRECTIVE 2026-07-22): LAN nodes only — no rented cloud compute.** CPU data
  generation and GPU training run on our own machines (this box: 7950X + RTX 5070 12 GB; plus
  the other LAN nodes — P1 inventories them). All multi-node orchestration goes through
  **zenfleet** (never hand-rolled); if zenfleet lacks a LAN/ssh provider for GPU training jobs,
  that capability gets built INTO zenfleet per the workspace mandate.
- **Storage reality check:** `/mnt/v` is 97 % full (≈68 GB free) — bulk HR + variants live
  **Tower-primary** (`/mnt/tower/output/zensr-training/`, 3.2 TB free) with R2 mirror
  (`s3://zentrain/zensr/`) and small hot caches locally; checkpoints to R2/Tower with pointer
  files in-repo (nothing > 30 KB binary in git).

## Training stack

- Base: **neosr** (Apache-2.0, local clone, supports SPAN + Compact archs, OTF degradations) —
  adapted with a real-encoder degradation hook (subprocess/FFI to pinned cjpeg + mozjpeg-rs;
  pre-generated variant pools to keep GPUs fed). Decision checkpoint P2: if adaptation fights
  us, fall back to a minimal in-repo trainer (basicsr-style, ~1 kLoC). Teacher pool for S9:
  permissively-licensed heavyweights (e.g. Apache SwinIR-class) run OFFLINE on LAN GPUs to
  produce distillation targets — teachers need not fit the op vocabulary, only ship models do.
- Export path stays: torch → fixed-order weight dump → zensr-micro golden verification (every
  trained model gets torch goldens at 64² + tiny shapes, same gates as SPANF today).
- Fidelity tier losses: Charbonnier + optional DISTS. Quality tier: + light UNet-GAN
  (Real-ESRGAN recipe) — clearly labeled, never the imageflow default.

**GPU-efficiency levers (mandatory; these models are data-pipeline-bound, not FLOP-bound):**
1. **Pre-packed training shards** — degraded pairs pre-generated and packed (LMDB/tar shards,
   pre-cropped), sequential reads; never decode+degrade per training step. RAM-cache the hot set.
2. **Warm-start everything** — bicubic-pretrained base checkpoint once per arch; S1/S2/S3/S7
   are FINE-TUNES (1–3 h) not scratch runs (8–24 h). ~5× matrix compression.
3. **Concurrent multi-model training** — a 0.15 M model uses <2 GB VRAM and a fraction of SM
   capacity; run 3–6 experiments per GPU concurrently (separate processes/MPS). Multiplies
   experiment throughput without new hardware.
4. **Cached teacher targets** — S9 teachers run ONCE over the corpus, targets stored; students
   train from disk. Distillation ≈ free per student.
5. bf16 autocast + channels_last + torch.compile (+CUDA graphs on static shapes) — launch
   overhead dominates tiny convs; fusion is worth more here than on big models.
6. **ASHA-style early kill** — mini-val ssim2 every N steps; dominated runs die at 25 %.
Net effect: the full S-matrix compresses from ~3–6 weeks to **~1–2 weeks on the single 5070,
days if more LAN GPUs exist**.

**Cloud burst (exists, user-gated, currently OFF per LAN-only directive):** zenfleet-vastai is
already built; live prices 2026-07: 4090 ≈ $0.33/h, 5090-32GB ≈ $0.40/h, 4090 spot from
~$0.14/h. The compressed program ≈ 150–300 GPU-h ⇒ **$50–120 total**, and 8 parallel boxes
turn the matrix into ~2–3 days wall-clock. The saving is calendar time, not money; flipping
this switch is the user's call, not a technical blocker. Managed AutoML platforms add nothing:
none support image-restoration training; the domain's "cookie-cutter" is the OSS
neosr/BasicSR stack we already adopted.

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
  <!-- status 2026-07-23: rt_distill_2x = 21.9 MP/s @12T but 3.1 @1T (single-thread gate
       NOT yet met — needs guard SIMD + maybe nf16 or ½-res shape); q35 ssim2 +2.5 ✓ at
       one band; q50 +0.5 ✗; see SYSTEMS.md S-E verdict. -->
  ≥ 0.7 dB PSNR-equiv and ≥ 4 ssim2 median at q∈{50,70} on ≥ 6/8 subcorpora; ≤ 4× Lanczos cost.
- G4: 4:2:0 chroma: measurable win from S5 or documented falsification.
- G5 (P5.5): at least one top-3 sub-track placement (runtime/params/overall) in a public
  challenge with shipped-identical models, or a documented near-miss analysis feeding P6.

## Adoption track (verified 2026-07-22; per-model licenses checked on live pages)

Ship-v0 and scoreboard baselines come from existing models — adopted, not trained:

| slot | model (author) | arch | license | note |
|---|---|---|---|---|
| **2× fast (our priority)** | 2xNomosUni_compact_multijpg(_ldl) + 2xNomosUni_span_multijpg(_ldl) (Phhofm) | Compact 64/16 · SPAN 48 | CC-BY-4.0 | the ONLY clean-license 2× web-JPEG photo models in existence |
| 4× quality | realesr-general-x4v3 + wdn pair (xinntao) | Compact 64/32 | BSD-3 | denoise-strength interpolation knob |
| 4× fast | 4xLSDIRCompactC3 / 4xNomosUni_span_multijpg (Phhofm) | Compact / SPAN | CC-BY-4.0 | JPEG 30–100 trained |
| 4× max-realism (needs RealPLKSR port) | 4xNomosWebPhoto_RealPLKSR (Phhofm) | RealPLKSR 64/28/17ks | CC-BY-4.0 | true web lifecycle: multi-round JPEG **and WebP** + LUDVAE noise; ~25× Compact FLOPs |
| 1× repair (needs RealPLKSR port) | 1xDeJPG_realplksr_otf (+_60) (Phhofm) | RealPLKSR | CC-BY-4.0 | pure dejpeg; the `_60` variant for lighter inputs |
| anime | realesr-animevideov3 (BSD-3) · 2xHFA2k_compact_multijpg (CC-BY-4.0) | Compact | | |

CC-BY-4.0 = commercial-OK with attribution (credit in docs/about). Blocked despite fit:
Kim2091's UltraSharpV2-Lite/ClearReality (NC), AnimeJaNai Compacts incl. the 0.4 MB
SuperUltraCompact (NC), alsa's quality-banded 1xJPG series (NC — but note: it independently
validates our S2 q-banding idea). Avoid `_dysample` RealPLKSR variants (grid_sample).
SPANPlus has zero released pretrains anywhere.

**Verified gaps = our moat (every one survives adoption):** nothing trains on mozjpeg —
every community pipeline bottoms out in cv2.imencode = libjpeg-turbo lineage (source-verified;
one popular tool even defaults 4:2:2, not web-dominant 4:2:0); progressive + trellis artifacts
unmodeled; no 2× multi-round web-lifecycle model; no quant-table/q-band conditioning anywhere;
no permissive 2× RealPLKSR photo model.

**Training-resource windfalls (fold into P1):** LUCID-CC0 v2 (2026-06, Phhofm) — 1.59 M tiles,
199 GB, CC0 SISR training set → Tower-primary pull, licensing-gate trivially passes; plus CC0
pretrain bases (musl 4x-realplksr-gan-pretrain, 4x-span-pretrain) for warm-starts.

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

**Budget & substrate (USER DIRECTIVE 2026-07-22):** the budget is *engineering time and AI
tokens* — effectively uncapped: run the science program exhaustively, many sessions, deep
ablations. Compute is **LAN nodes only** (CPU and GPU on our own hardware; no vast.ai/cloud
GPU rental). Currency is therefore **GPU-nights**: on the RTX 5070 (12 GB) alone, sub-0.2 M
models train in 2–8 h and Compact-class in 8–24 h → the full S1–S9 matrix (~30–40 runs) is
3–6 weeks of continuous single-GPU utilization, linearly divided by every additional LAN GPU
P1 finds. 12 GB VRAM comfortably fits every tier incl. the light-GAN quality runs (batch 32–64
at 256² crops) and offline heavyweight-teacher inference for the S9 distillation ladder.
The moat is degradation exactness + conditioning + the engine — not brute compute.

**Drive mechanics (how this reaches the end instead of parking):**
- Every phase has a Definition-of-Done gate below; a phase without its gate met does not yield
  to the next except by a written kill/pivot note in the appendix.
- Every experiment run lands in `benchmarks/` same-day with commit + config; conclusions edit
  PLAN.md in the same change (workspace DOCS discipline).
- The scoreboard is regenerated at every P≥3 milestone; regressions block.
- Kill criteria are pre-declared per science question (see S1–S9 falsification clauses) — a
  falsified branch is recorded and never blocks the critical path (which is: data → q-banded
  fast tier → realtime distillate → scoreboard → bench release → challenge entry).
- Standing cadence once P2 lands: **no idle GPU-night on any LAN node** — a persistent
  experiment queue (zenfleet-local/LAN) always holds the next S-list run; every morning has a
  result to log and a scoreboard delta or falsification to record. Machine-safety rules apply
  on shared boxes (run-heavy, capped dataloader workers, serialize with other agents' heavy
  jobs via the lockfile protocol).

## Phases

- **P0 — Foundation: ✅ SUBSTANTIALLY DONE 2026-07-23** (10h director session): engine additions
  (PReLU/nearest-residual/scale-param shuffle) landed; Compact + SPAN-48 ports golden-verified
  (7 adopted models, incl. stale-eval_conv branch-merge fix); guard layer (residual clamp /
  texture gate / round-trip fallback) property-tested; systems_eval with REAL turbo-jpeg axis;
  SYSTEMS.md defines the five deployable systems; S-E realtime distillation pilot launched.
  Remaining P0: zensim eval column (API is pipeline-shaped, deferred), G1 baseline report =
  the systems_eval output being generated now.
- **P1 — Data + LAN fleet:** HR corpus assembly + licensing gate; `zensr-degrade` + manifests
  + Tower/R2; frozen test splits; DATA.md + workspace provenance entry; **LAN compute
  inventory** (every node: cores/RAM/GPU/VRAM, ssh reachability) and zenfleet wiring for the
  training queue (build the LAN provider into zenfleet if missing).
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
  *Status 2026-07-26: the zenjpeg leg is LIVE — `zensr-zenjpeg::restore_jpeg` (probe →
  deblock policy → decode → guarded ×1 model → S10 projection, 444+420, YCbCr-native-aware)
  with a minimized snapshot-enforced API (`apidoc/PUBLIC_API.md`); product-crates rebuild
  1.56 s. See SYSTEMS.md "Production hardening pass".*
- **P5.5 — Challenge entries:** NTIRE ESR + real-world/real-time track submissions with
  shipping models (calendar-driven; prep starts when P4 models exist). Gate G5.
- **P6 — Frontier (ongoing):** S5 native-plane models, PLKSR-Rep-class large-kernel port,
  fusion-retry via per-tier `#[rite]` exp helpers, QAT-int8 if ever needed; scoreboard defense
  as competitors move.

**FORWARD PLAN: see `ROADMAP.md`** (2026-07-31) — open rungs ranked by EV, the
full CLOSED list (13 falsified directions with their evidence), infrastructure
debts, and the squintly human-judgment track. Read it before proposing work.

## Session state 2026-07-28 (for continuation — canon lives in SYSTEMS.md, read it first)

Everything below is COMMITTED with benchmarks/ TSVs; repo has NO remote — Tower bundle
`/mnt/tower/output/zensr-archive/zensr-2026-07-28.bundle` + models mirror are the backup.

**Production ladder (final, all golden-verified, f16 ship format cleared 3 levels):**
dejpeg7_graphics default / dejpeg_rt24d realtime (0.15 s/MP lianli) / dejpeg9_gfxycc
low-q-graphics route (chooser p>0.85 ∧ q≤60) / gates: q≥95 identity, Annex-K q≤9.5
Knusperli / chain ×1-before-SR when 420 ∨ q≲50. Reproducibility: every model dir has
repro.sh + full meta.repro; f16 baked into train_people.py export.

**Open tasks (task list + SYSTEMS.md sections have full detail):**
- #15 GPU spike: CUDA floor DONE (loses to CPU 4x naive; ladder to win = smem tiling,
  persistent buffers, pinned staging, f16). NEXT: wgpu leg on lianli(2080/vulkan) +
  mac(Metal) — `cargo build -p zensr-gpu-spike --features wgpu`; then decide tiled-kernel
  investment. Crate: crates/zensr-gpu-spike (API quirks solved: comptime scalars,
  from_raw_parts(handle,len), read_one(handle).unwrap()).
- CPU AI cores: fleet measured (Zen4 avx512_bf16+vnni; M4 SME/SME2 B16F32). Highest-EV:
  bf16-dot compute path (~2x GEMM density; weights already f16) — check magetypes bf16
  support first. Live landscape survey (NPUs/AVX10) pending via WebSearch.
- #13 feature-KD falsification rung (design in task; expect null — rt24d saturated).
- Winograd: CLOSED opt-in (3 rungs falsified, root cause instruction-bound IPC 5.0).
- node-2: rebooted to owner use (host key changed) — hands off until verified idle; node-3
  needs physical power button (WoL runbook in zenmetrics/NODES.md).

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
- **Consistency-only goldens (torch-reimpl ↔ Rust)**: agreed on a broken SPAN graph
  (missing input norm + inplace-SiLU concat semantics → constant gray) for a full eval +
  a poisoned 14k-pair teacher run (2026-07-23). Goldens MUST be cross-checked against the
  reference implementation (spandrel); `dump_adopted.py` now hard-gates this per model.
- Folding SPAN's `(x−mean)·255` input norm into merged conv_1: exact in the interior,
  wrong at image borders (official zero-pads AFTER norm ⇒ border=mean gray; folded
  zero-pad ⇒ border=black; measured 0.32 max err). Normalize explicitly instead.
