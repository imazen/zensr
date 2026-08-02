# zensr roadmap — what to try next, and what is already dead

**Read `SYSTEMS.md` first for the measured record; this file is the forward plan.**
Every item below is either (a) an open rung with a stated decision it informs, or
(b) explicitly listed as closed so nobody spends a night re-deriving it.

Status date: 2026-07-31. Production ladder + routing: see `README.md`.

---

## 0. Standing corrections that gate everything

1. **Eval reference contamination (found 2026-07-31, user-caught).** 39% of the
   pinned eval split has JPEG ground truth (all `unsplash-*` dirs + 23 files in
   `lilith/`). Absolute gains were understated: rt24g scores +3.82 at q35 on
   clean PNG references vs +1.37 on contaminated ones, and **+0.35 at q90 vs
   −1.04**. 60% of negative-file rows come from contaminated references (42% of
   rows). Training data is only 8% contaminated — much milder, still worth
   excluding.
   → **Every absolute claim needs the clean-reference number.** Relative
   model-vs-model comparisons are less affected (both tiers show the same
   pattern), but the identity-gate threshold was set on contaminated evidence
   and is being re-derived (§1.1).
2. **Report granularity matters.** "Zero negative cells" was true at
   encoder×q *mean* granularity and false per-file. Always report per-file
   negative counts and worst case alongside means.
3. **Metric ceiling.** Differences below ~0.3 ssim2 are at the edge of what our
   metrics resolve. Anything smaller needs human judgment (§4) rather than a
   bigger eval grid.

---

## 1. Open rungs, ranked by expected value

### 1.1 Re-derive the identity gate on clean references *(blocking a shipped constant)*
The realtime gate sits at q82 because q90 measured −0.23. On clean references
q90 is **+0.35**. The gate may be discarding real gains.
- Run the granular q sweep (q40…q85, both slack settings) filtered to clean
  references; find the true crossover.
- Decision: the shipped `RestoreOptions::realtime_tier()` threshold (currently
  82.0 / d≤1.0 in zenjpeg PR #191).

### 1.2 Training data scale-up *(strongest untested quality lever)*
200k steps × batch 48 over 24k pairs = **~400 epochs**. The corpus has 974
source files; we are re-showing the same crops hundreds of times.
- Build 100k+ crop datasets (CPU-only, generator already written; exclude
  JPEG-sourced files per §0.1).
- Retrain rt24-class at the same 200k budget; compare against rt24g.
- Decision: whether the 36.99 dB teacher-fidelity plateau was data-bound.

### 1.3 Better teacher *(raises every student's ceiling)*
Students are capped by what they imitate. dejpeg7 was only a 16k warm-start;
it has never had the f16-target + long-budget recipe that the students got.
- Retrain the quality tier with the validated recipe, then re-distil.
- Decision: whether the realtime tier's remaining gap is teacher-limited.

### 1.4 Feature/affinity KD (task #13, RUNNING 2026-08-02)
Output-KD plateaus the student ~33 dB from its teacher. Feature-level targets
give denser supervision. Case is *stronger* than when queued, because capacity
was falsified twice and the student is optimization-bound.
- Paired against output-KD at matched budget/seed/box.
- Design: `tools/run_fkd_pair.sh`. Arms differ by exactly one loss term —
  both take the output target from the SAME online teacher, so the only
  difference is the affinity supervision. Affinity (FAKD-style) rather than a
  projected feature match, because channel-normalised Gram products compare a
  24-wide student to a 64-wide teacher with no learned projector whose own
  capacity would confound the result.
- The weight is calibrated from a probe (the trainer prints `base`/`aff`/
  `aff_share`), not guessed: on random features the affinity term is ~30x the
  reconstruction loss, so an arbitrary weight silently replaces the objective
  instead of augmenting it.
- Box note: ian is a GTX 1660 Ti (6 GB, Turing cc 7.5), **not** the RTX 3070
  the fleet runbook guesses. The trainer's compute-capability gate correctly
  selects fp16+GradScaler there; inductor is unavailable (triton cannot build
  its CUDA shim), so both arms run eager via `ZENSR_COMPILE=0`.

### 1.5 Q-balanced sampling
The long cosine schedule traded q75 for q15 (89.5% → 67.8% of quality tier).
Balance the sampler across quality bands; expect both.

### 1.6 Generation-specialist models *(running 2026-07-31)*
Testing whether separate 1-generation and 2-generation models beat one
generalist. `dejpeg_rt24_gen2` trains on all-gen2 chains at the exact rt24f
recipe (64k, dejpeg7 teacher, f16 targets) so the comparison is clean.
- Evaluate both models on both input types (2×2).
- **Caveat that may kill the idea outright:** routing requires knowing the
  generation count at runtime, and §2.6 measured that detection has no usable
  conservative operating point. A gen2 specialist is only shippable if it also
  performs acceptably on gen1 input (i.e. it is a better *generalist*), or if
  provenance is known out-of-band (an imageflow pipeline that did the first
  encode itself).

### 1.7 Projection slack sized to p99, not max
Physics says slack must cover pre-FDCT rounding, which accumulates per
generation (turbo max violation 0.11 → 2.37 → 3.37 Q for gen1→2→3). Sizing to
**p99 (~1.0–1.1 Q)** rather than max captures most of the strict-box gain while
bounding the tail, and needs no generation detector. Gate on encoder family
first — trellis families (mozjpeg) already violate 9–15 Q at gen1.

---

## 2. CLOSED — do not re-attempt without new information

| # | Idea | Why it's closed |
|---|---|---|
| 2.1 | **Realtime capacity increase** (113k / 221k / 336k params) | Falsified 3×. 221KB scored −0.02 median vs 84KB at 2.2× the compute; bigger models also fit their teacher *worse* at equal steps. 43k saturates this topology. |
| 2.2 | **Input conditioning** (severity scalar, per-block damage map) | −3 dB on every degraded band. Gradient shortcut: the model reads the hint instead of the pixels. S10 belongs at the *output*, not the input. |
| 2.3 | **YCbCr-native as the general pipeline (S5b)** | Falsified under both bias regimes (warm-start and a from-scratch same-seed pair). Survives only as a low-q *graphics* trait inside dejpeg9_gfxycc. |
| 2.4 | **Lattice-aware chroma architecture above q35** | Oracle/lattice decomposition: ~100% of the chroma gap above q35 is subsampling loss, not recoverable quantization error. |
| 2.5 | **Guided chroma upsampling** | JBU with a *ground-truth* luma guide LOSES to bilinear (−2.25). Luma edges don't predict chroma edges on this corpus. Chroma direction closed with three concordant bounds. |
| 2.6 | **Multi-generation detection as a gate for tightening** | No conservative operating point: false-gen1 = 0 requires a threshold with gen1-recall = 0. Same-encoder equal-q recompression is near-invisible (9.4% recall); resampling destroys the DQ comb. Use detection to **loosen, never to tighten**. |
| 2.7 | **Edge/contrast-weighted loss** | Moved high-contrast +0.03 dB while flat regions collapsed (q75 +0.47 → −1.09). Reweighting starves the rest; it does not add edge skill. |
| 2.8 | **Winograd F(2×2,3×3)** | Correct but 1.17× slower even fully magetypes-tiered. perf: instruction-count-bound at IPC 5.0 with 2.1× the instruction count. Kept opt-in behind `ZENSR_WINOGRAD=1`. |
| 2.9 | **Naive GPU inference (wgpu/CUDA/Metal)** | All three backends lose to same-box CPU end-to-end at web image sizes; transfers + per-run allocation dominate. Deferred until a batch/server use case exists. |
| 2.10 | **int8 weights-only PTQ** | Catastrophic (0.05–0.24 output error in [0,1] units). Needs QAT or an int8-first architecture — a training program, not a conversion. |
| 2.11 | **Weight compression (zero-bias / zstd / byte-plane)** | Conv weights are near-max-entropy: zstd 1.07×, byte-planes 1.15×. zenpredict's zero-bias trick does not transfer (τ=0.002 corrupts at 100× the f16 noise floor). Ship plain f16. |
| 2.12 | **×1 repair via scale round-trip** | Loses to identity; superseded by native ×1 inversion. |
| 2.13 | **Boundary4Tap deblocking** | Dominated at every quality: inert at q96 (strength = 0.25·dc_quant), a net loss at q85–93, and 10–14× smaller than the model's gain at mid quality. |

---

## 3. Infrastructure debts

- **Eval harness must record ground-truth source type** (png vs jpg) per row and
  support filtering. Everything in §0.1 was invisible because it didn't.
- ~~The pinned eval split needs a clean-source subset~~ — **built 2026-08-02**:
  `/mnt/v/imazen-26-pristine` holds 74 downscaled-to-pristine PNG references
  (0 skipped) covering exactly the four JPEG-sourced subcorpora — unsplash-people
  28, lilith 23, unsplash-renders 13, unsplash-textures 10. Policy applied per
  file from zenjpeg's own probe: 3x below q90 (51 files), 2x at q90+ (23);
  sources ranged q89-100; smallest output 540x720; 310 MB. Measured residual
  bias at these factors is 0.11-0.22 ssim2 (`benchmarks/pristine_probe_*.tsv`),
  a twentieth of the effects being measured. **Still open**: rebuild the eval
  split on this corpus and re-run the ladder, so the absolutes stop carrying the
  39%-contaminated references.
- `slack_probe`'s `ZENSR_SLACK_RESIZE` mode emits zero rows (`decode_any`
  rejects `.ppm`). Fixed measurement lives in `gen_detect.rs`; the flag in
  `slack_probe` is still broken.
- A probe that produces **no rows must fail loudly**, not silently succeed.
- Kernel files get committed *before* iterating on them (a bad edit anchor
  amputated `simd.rs` mid-session).
- **Two dependencies are pinned to git revisions and must be unpinned when they
  publish**: `zenjpeg` at `e277e9c9` (0.9.0 — 0.8.4 keeps `DeblockMode`
  crate-private) and `zenanalyze` at `a7d8224` (0.2, the feature IDs the chooser
  was trained against). Until both are on crates.io, `zensr-zenjpeg` cannot be
  published. `zenjpeg` 0.9.0's own `zenanalyze` pin (`13d40c3`) differs from
  ours, so a `chooser` build compiles zenanalyze twice — harmless, but it goes
  away when both move to registry versions.
- **Long runs on boxes we do not exclusively own must sync checkpoints off-box
  while they run.** The 100k-crop student on the dual-boot box reached step
  22.5k of 200k (36.89 dB — the result that established the plateau was
  data-bound) and the box then rebooted into its other OS, stranding every
  checkpoint on a partition we cannot reach until it next boots back. The
  launcher only collected the *final* checkpoint, so an interrupted run kept
  nothing. `tools/pull_ckpts.sh` now pulls intermediates on a cadence; run it
  alongside any multi-hour training launch.
- The repo is **public** (github.com/imazen/zensr, AGPL-3.0-only OR
  LicenseRef-Imazen-Commercial). Anything committed here is world-readable:
  no LAN addresses, no box names, no corpus bytes. `tools/run_cond_ablation.sh`
  reads `ZENSR_BOXES` from the environment for exactly this reason.

---

## 4. Human-judgment track (squintly)

Our metrics cannot resolve the questions that now matter most. [Squintly](https://github.com/imazen/squintly)
is live (phone-first pairwise + threshold judgments, viewing conditions as
first-class data, ASAP active sampler, 84 imazen-26 sources already hosted).

Priority order for a restoration arm:
1. **Does the high-q identity gate match human judgment?** Our entire gate policy
   rests on ssim2/butteraugli calling the model harmful at high quality. The
   q90 case is the model adding plausible detail into smooth gradients — metrics
   score that as error; humans often prefer it. If observers prefer or can't
   distinguish, the gate is discarding gains rather than preventing harm.
2. **Is restoration noticeable at all, per quality band?** Uses the existing
   threshold flow (`notice` / `dislike` / `hate`) directly.
3. **Slack strategies** — last, and only with a focused design: differences are
   ~0.3 ssim2, which needs many trials even with active sampling.

Cost: new image set to R2, new strata, and a protocol amendment to the
pre-registered study. Not free, but it is the only instrument that can settle 1
and 2.
