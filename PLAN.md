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

Budgets: fast tier ≤ ~0.20 M params (SPAN-class, ~4.5 MP-out/s/core today), quality tier
≤ ~1.3 M params (Compact-class, ~16× fast-tier compute). Weights ship f16 (proven transparent,
75–76 dB). Runtime perf discipline: **retime after every kernel-adjacent commit** (26x-incident
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
- G4: 4:2:0 chroma: measurable win from S5 or documented falsification.

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
- **P4 — Production models:** fast tier (q-banded per S2 outcome) + quality tier; ×1/×2/×4;
  f16 ship files + goldens; full eval reports. Gates G2, G3.
- **P5 — Product wiring:** imageflow/zenpipe integration (decode-metadata → band pick →
  tiled upscale), size diet toward sub-300 KB, docs; publish decision (user-gated).
- **P6 — Frontier (ongoing):** S5 native-plane models, PLKSR-Rep-class large-kernel port,
  fusion-retry via per-tier `#[rite]` exp helpers, QAT-int8 if ever needed.

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
