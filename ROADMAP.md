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
   and has now been re-derived (§1.1).
   **Status 2026-08-02:** the clean corpus exists and the gate is re-derived,
   but the README's SOTA table and the SYSTEMS ladder still carry contaminated
   absolutes. First clean-reference numbers for the shipped realtime model
   (turbo 4:2:0, per-file median, n=64): **+6.88 / +3.36 / +2.02 / +1.15 at
   q15/35/55/75**, against the README's published +5.0/+2.8/+2.0/+0.8 — higher,
   in the predicted direction, and not methodologically identical (per-encoder
   median here vs a blended figure there), so they are not a drop-in swap.
1b. **File selection (found 2026-08-02).** "First N sorted" admitted training
   images whenever the directory listing differed from the one the split was
   frozen against — the `photos` subcorpus contributed 4 of 32 files, none in
   the pinned split. `dejpeg_eval` now selects from the pinned split by stem
   (`ZENSR_EVAL_PIN`) and warns loudly if no list is found. Same failure shape
   as the 2026-07-23 haberdoedas postmortem; it recurred because the fix then
   was a regenerated list rather than a harness that enforces it.
   **2026-08-03:** the SR eval (`eval.rs`) still had *both* defects — unpinned
   selection and no `gt_src` — and the selection one bit harder there, because
   it also drops images below the 512 crop and each drop slides one file deeper
   into training data. Fixed, and the helpers now live in `zensr_bench`
   (`pinned_stem` / `load_pinned` / `gt_src_of`) so the two evals cannot drift.
   For SR a JPEG reference biases the *opposite* way to restoration: detail the
   model correctly reconstructs was quantised out of the reference, so sharper
   output scores worse. **Any new eval binary must use the shared helpers.**
2. **Training-time metrics do not predict clean-reference quality
   (2026-08-02).** This one invalidated two conclusions in a single day. The
   100k-crop student improved `val_psnr_vs_teacher` by ~2 dB (36.89 -> 38.97)
   and went slightly BACKWARDS on clean references. The 100k-pair teacher
   improved its own val curve and did not improve its output either. Both had
   been written up as confirmations of the data scale-up on the strength of
   those curves alone.
   → **No rung is decided on a training curve.** Every verdict goes through
   `tools/model_ab.py` on the clean corpus, per file, before it is written
   down. `val_psnr_vs_teacher` in particular measures imitation of a target
   that is itself imperfect — the teacher beats the 84 KB student by +3.58
   ssim2 at q15, so converging on it is not the same as getting better.
3. **Report granularity matters.** "Zero negative cells" was true at
   encoder×q *mean* granularity and false per-file. Always report per-file
   negative counts and worst case alongside means.
   **2026-08-03, second instance — unpaired statistics invent and hide losses.**
   The README's SR table reports `median(spanf) − median(lanczos)`, comparing
   each image against the *distribution* of the other method instead of against
   itself. That produced the published "textures −1.7, SR loses on stochastic
   detail" — the paired median on the very same data is **+8.79 with SR
   winning** — while hiding the one subcorpus where the metrics genuinely
   disagree, `renders` (paired ssim2 −1.81 / win 0.38, but PSNR +0.92 on **8 of
   8** files and butteraugli −0.924 on 7 of 8), reporting it as a flat +5.93
   win. A plausible mechanism ("SR can't invent stochastic detail") kept the
   textures figure unchallenged for eleven days.
   → **Always pair, and pair every metric.** Median of per-file differences plus
   win fraction, never the difference of per-method medians. Pairing is also
   what made the encoder effect in §1.1b measurable at all. I made this exact
   mistake *inside the writeup of this finding* — quoting unpaired butteraugli
   medians (7.459 vs 7.397) as confirmation of the renders loss, when paired
   butteraugli says the opposite. It is easy to make and invisible without the
   per-file table. Full write-up: `benchmarks/sr_pinned_2026-08-03.md`. README
   correction pending review.
4. **Metric ceiling.** Differences below ~0.3 ssim2 are at the edge of what our
   metrics resolve. Anything smaller needs human judgment (§4) rather than a
   bigger eval grid.

---

## 1. Open rungs, ranked by expected value

### 1.1b Routing estimator validated on clean references — MEASURED 2026-08-03

Full report `benchmarks/clean_ladder_2026-08-03.md`; raw rows pointer-filed at
`benchmarks/clean_ladder_2026-08-03.pointer.md`. Grid: the 64 pinned files ×
{turbo, mozjpeg} × {4:2:0, 4:4:4} × q{15,35,55,75,85,90,94} (+ a q{94,96,98,100}
tail run), 100% PNG references, `gt_src` verified on every row. Analysis splits
30 calibrate / 34 validate **by image**.

**The shipped routing estimator survives and nothing beat it.** Scored on the
validate half by summing the actual measured delta on each cell a rule chose to
restore: shipped estimator **+2.0274 at 79% restored**, vs +1.9993 for the
subsampling-threshold gate, +1.8561 for always-restore, +1.8544 for a
per-encoder refit. Routing is worth its complexity (+0.17 over always-restore
while skipping 21% of the work) and the version already shipped is the best of
six. Expectation going in was that contaminated calibration would need redoing
wholesale; it did not.

**Fixed:** the far tail was wrong in sign. The 4:2:0 curve predicted **+0.15 at
q94 where clean references measure −0.27**, and **−0.17 at q100 where they
measure −1.89** — 11× too small. Anchors at q≥94 are now measured
(`G420`/`G444` in `crates/zensr-zenjpeg/src/api.rs`, landed in `4fc80c9`, whose
message describes only the eval fixes that rode along with it). At the default
`min_gain: 0.25` this changes 3 of 2000 sampled decisions; at `min_gain: 0.05`
it changes 45. Pinned by `gain_curve_is_negative_on_near_pristine_input` —
predicting gain on near-pristine input is the one error that spends cycles to
make an image worse, and it had survived the entire suite.

**Left alone deliberately:** anchors at q≤90 agree with clean measurement to
within 0.05 through q75 (5.70/5.66, 3.07/3.07, 1.80/1.75, 0.65/0.64). They read
~0.23 high at q85–90, but the calibrate/validate spread there is 0.23–0.64, so
the gap is inside the noise — and re-fitting the mid-range scored *worse*
held-out. The `S444_SHIFT = 4` approximation likewise over-predicts 4:4:4 gain
by 0.3–0.6 below q90; no decision depends on it and no re-fit improved it.
Recorded as a known bias in the source rather than silently corrected.

**NEGATIVE — per-encoder crossover thresholds are not identifiable at n=64.**
libjpeg-turbo images genuinely gain more from restoration than mozjpeg at every
q from 15 to 85, both subsamplings, paired sign test p from 1.4e-17 to 1.1e-5
(mozjpeg's trellis and tables leave less damage to repair). But the *crossover
point* swings across the split: mozjpeg 4:4:4 reads q75 on calibrate and q90 on
validate. Pooling all 64 files would have justified lowering that gate to ~q83.
Pairing makes the magnitude precise because it cancels image-to-image variance;
an unpaired cell median does not, and per-image spread runs −1.4..+4.2 in a
single cell. **Do not derive per-encoder thresholds from a 64-image ladder.**
Folding the effect in as a paired offset — the sound form — costs 0.011 ssim2
for 4 points less work, an order of magnitude under the metric floor.

### 1.1c jpegli/zenjpeg gain curve measured — FIXED 2026-08-03

The distance-quantised family (`CjpegliYcbcr`: cjpegli and zenjpeg) had **never
been measured on clean references**. `estimate_gain` converted butteraugli
distance to a pseudo-quality with `100 − 12·d` and read the IJG curves.

Measured (2,560 paired cells, 10 qualities × 2 subsamplings × 2 encoders, 64
pinned files, 100% PNG): the mapping is optimistic in **39 of 40 cells**, median
+0.68 and up to **+4.72** ssim2, with the **wrong sign in 9** — promising gain
from distance 0.3 to 1.3, where restoration costs quality. It fails worst
mid-range: distance 7.2 maps to "quality 13.6" and predicts +5.70 against a
true +1.31, because a jpegli file at distance 7.2 is far better than an IJG file
at q14 — the point of the encoder, and a linear inversion cannot express it.

Replaced by `DIST420`/`DIST444`, keyed on distance directly. Held-out:

| rule | mean ssim2 | restored | harmed |
|---|---|---|---|
| IJG curve via 100−12·d | +0.7998 | 0.60 | **0.17** |
| measured distance curve | +0.7940 | **0.30** | **0.03** |

Quality is a tie (0.006, far under the metric floor); the new curve gets it for
**half the cycles and one sixth the harm**. Full report
`benchmarks/clean_ladder_jpegli_2026-08-03.md`.

Secondary finding: distance-quantised encoders leave less to repair. At q75,
turbo +0.82 / mozjpeg +0.46 / cjpegli +0.19 / zenjpeg +0.19 — our own encoder is
the least improvable of the four, which is the right outcome.

### 1.1 Identity gate re-derived on clean references — MEASURED 2026-08-02

Corpus `/mnt/v/imazen-26-clean` (974 refs, 0 JPEG: native PNGs unioned with
downscaled-to-pristine replacements). Files selected from the **pinned split**,
8 per subcorpus, n=64 per cell, gate DISABLED so the ladder can see past it.
Summaries `benchmarks/pinned_gate_{main,s444,jpegli}_2026-08-02.tsv`; raw in
`/mnt/v/output/zensr/pinned-gate-2026-08-02/`.

Two defects had to be fixed to get here, both recorded under Infrastructure:
the gate short-circuited restoration so the first ladder measured *itself*, and
"first N sorted" selection admitted training images.

**Crossover (first q that is non-positive and stays so), per-file medians:**

| encoder | ss | crossover | q90 | q94 | q96 | q98 | q100 |
|---|---|---|---|---|---|---|---|
| turbo   | 420 | **q96** | +0.32 | +0.05 | -0.02 | -0.19 | -0.30 |
| mozjpeg | 420 | **q99** | +0.23 | +0.46 | +0.18 | +0.01 | -0.37 |
| zenjpeg | 420 | **q95** | +0.11 | +0.08 | -0.05 | -0.19 | -0.06 |
| jpegli  | 420 | none    | +0.40 | +0.16 | -0.04 | -0.05 | +0.20 |
| turbo   | 444 | **q92** | +0.05 | -0.37 | -0.80 | -1.43 | -2.01 |
| mozjpeg | 444 | **q90** | -0.14 | -0.38 | -0.72 | -1.19 | -2.06 |
| jpegli  | 444 | **q95** | +0.02 | +0.06 | -0.34 | -0.72 | -1.50 |
| zenjpeg | 444 | **q88** | -0.04 | -0.13 | -0.45 | -0.85 | -1.71 |

(jpegli/zenjpeg 4:4:4 measured on an 88-100 grid,
`benchmarks/pinned_gate_jpegli444_2026-08-02.tsv`. zenjpeg is already negative
at the lowest point sampled, so its floor is not yet located — a q40-88 sweep
is running.)

**The gate ignores chroma subsampling, and that is a shipping defect.** At
4:4:4 restoration is already negative at q90 (mozjpeg) / q92 (turbo) and gets
monotonically worse: by q100 the median is **-2.06 ssim2 with 91% of files
harmed** (mozjpeg 444 q99/q100: win_frac 0.06, harm_frac 0.91). The shipped
gate fires at `>= 94.5`, so the whole q90-94 band at 444 is unprotected and
already losing. At 420 the same constant is mildly early (turbo q96, zenjpeg
q95) or late (mozjpeg q99).

**Mechanism, measured rather than guessed.** It is *not* that the model never
saw 4:4:4 — `restore_jpeg` builds full-resolution RGB planes from the decode,
so the model's input format is identical either way. What differs is how
damaged that input is. Median `identity_off` ssim2 against the clean
reference (`--absolute identity_off`):

| q | 90 | 92 | 94 | 96 | 98 | 100 |
|---|---|---|---|---|---|---|
| turbo 420 | 85.54 | 86.82 | 88.04 | 89.68 | 91.10 | 91.69 |
| turbo 444 | 87.84 | 89.11 | 90.75 | 92.47 | 94.07 | 95.26 |

**4:4:4 at q90 is as clean as 4:2:0 at q94** (87.84 vs 88.04). The gate keys on
nominal quality, but the same nominal quality means a materially less damaged
image at 4:4:4 — so a threshold calibrated on 4:2:0 lets the model loose on
inputs it should already be skipping. This is the same F3 effect (the model
harms near-pristine input), reached by a different route.

That accounts for turbo's 4-point offset (crossover q96 vs q92). It does NOT
fully account for mozjpeg's (q99 vs q90), so damage-equivalence is part of the
story and not all of it. Do not present it as the whole explanation.

**The projection's value grows with quality, on both subsamplings** — the
opposite shape to the model's own contribution, and it survived the selection
fix at doubled n (`model_proj` minus `model_policy`):

| encoder | ss | q90 | q94 | q96 | q98 | q100 |
|---|---|---|---|---|---|---|
| turbo   | 420 | +0.39 | +0.48 | +0.72 | +0.96 | +1.31 |
| mozjpeg | 420 | +0.46 | +0.53 | +0.66 | +0.93 | +1.18 |
| turbo   | 444 | +0.32 | +0.39 | +0.43 | +0.62 | +1.09 |
| mozjpeg | 444 | +0.55 | +0.49 | +0.39 | +0.53 | +1.02 |

Monotone, no crossover anywhere in the grid. So the composite crossovers above
are the model degrading while the projection increasingly offsets it, and
restoration survives as far up the scale as it does *because* the
quantisation box constrains it. Direct support for `require_consistency` in
`docs/API_DESIGN.md` being load-bearing rather than merely a safety property.

**What changed (SHIPPED):** `policy_high_q_identity` now takes chroma
subsampling into account — 4:4:4 gates at q88 (IJG/mozjpeg scale) and at
distance 1.3 (Butteraugli scale), covered by a test that encodes the same
content both ways. The low-q sweep confirms the harmful band is bounded below:
at 4:4:4 the model is strongly positive at low quality (+2.2 to +3.3 median at
q40) and decays to zero around q85-88, so q88 is a floor and not a guess.

**Verified after the fix** (`benchmarks/gated_444_after_2026-08-02.tsv`, same
corpus and pinned selection, gate live): every 4:4:4 cell from q88 to q100 now
reads exactly +0.000 with harm_frac 0.00 for both families. The worst case
went from a -2.06 median with 91% of files harmed (mozjpeg q100) to no change
at all — the model is skipped and the decode returned, which is also the
cheaper path.

**What deliberately did NOT change:** the 4:2:0 thresholds. The measured
crossovers there (turbo q96, zenjpeg q95, mozjpeg q99, jpegli none) sit above
the shipped 94.5, so moving them would trade a small median gain for more
regressions — harm_frac at 4:2:0 runs 0.3-0.45 across exactly the band where
the median is still positive. `DoNoHarm` wants the current 94.5; `Fidelity`
wants the higher numbers. That is an `Intent` decision, not a constant to
pick unilaterally, so it stays as-is until the API split exists.

Caveats: n=64/cell, 512-crop, single generation, and the pristine references
are downscaled 2-3x so they run smaller than native inputs.

### 1.2 Training data scale-up — RESOLVED 2026-08-02: a low-q lever only

**Controlled test** (`benchmarks/dataab_100k_vs_24k_controlled_2026-08-02.tsv`).
Only `ZENSR_DATA` differs: the 24k arm is a SUBSET sliced from the 100k set,
so both share generation process, teacher, target dtype, crop size and the
identical val tail; same box, same bf16, same 120k steps, from scratch, no
restart.

| band | n | win% | median | sign-test p |
|---|---|---|---|---|
| turbo q15 | 64 | 0.73 | **+0.425** | 2.3e-04 |
| q15 both encoders | 128 | 0.66 | +0.235 | 5.2e-04 |
| low q (15+35) | 256 | 0.60 | +0.062 | 1.4e-03 |
| q>=55 | 640 | 0.47 | -0.007 | 0.22 (null) |
| overall | 896 | 0.51 | +0.007 | 0.53 (null) |

**4x the data buys quality only at the lowest quality**, and nothing from q55
up. The q15 effect is also the one result today whose magnitude (+0.425)
approaches the ~0.3 resolution threshold in §0, so it is plausibly a real
visible difference rather than only an established direction.

This resolves the two earlier confounded attempts rather than contradicting
them: both averaged over the whole q range, where a q15-only effect washes out.
Their "null" answered the wrong question.

**Consequence for how data is spent:** more crops of the same distribution pay
off only where damage is heaviest. If low-q is the target (it is — web
compression), scaling data is worth it; if the goal is the q75-q94 band, this
lever is spent and the effort belongs elsewhere.

Note the training metric misled a third time: the big arm's
`val_psnr_vs_teacher` was 38.42 vs 38.19, a gap spread across all q, which
predicted neither the size nor the location of the real effect.


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
- **The asymmetry is now concrete.** `dejpeg11_teacher`'s own `meta.json`
  records `dataset_meta.n = 24000`, while the student that imitates it trained
  on 99,488 crops — and the data scale-up was worth ~2 dB to that student
  (36.89 -> 38.97 fidelity-to-teacher). The student is being fit to a target
  produced by the *smaller-data* model, so the ceiling argument is not
  speculative. `dejpeg12_teacher_big` (120k steps on the 100k set, running on
  lianli) is the direct test.
- Compare teachers on the clean corpus once it exports, not on training loss:
  both are `models/adopted/` dirs and the eval already takes model names.
- **`dejpeg12_teacher_big` finished 2026-08-02** (120k steps, 100,000 pairs vs
  dejpeg11's 24,000, same nf=64/nc=16 architecture, 595,459 floats). Being
  compared to dejpeg11 per-file on the clean corpus, same grid the incumbent
  was already measured on. Judged on clean references from the start, NOT on
  fidelity-to-anything — that distinction is what the student comparison cost
  us today.
- Context for reading the result: the incumbent teacher is already far ahead
  of the 84 KB student it trains (median +3.58 ssim2 at q15, +0.61 at q75), so
  a teacher improvement only matters to the product if it survives
  distillation. A better teacher that the student cannot follow moves nothing.

### 1.4 Feature/affinity KD — REPLICATED at full budget 2026-08-03, but the recipe is not competitive

**Full 200k-step pair, both arms complete** (the 2026-08-02 result below ran to
22.5k). Seed 7 on the 3070; a second pair at seed 2 is running on the 5070 to
separate the effect from one seed's luck. Per file on the clean corpus, arm B
(affinity) minus arm A (output-KD only), pooled over encoders, `ZENSR_EVAL_QS`
15/35/55/75, 4:2:0, n=128 per cell:

| q | median | mean | signs +/− | p |
|---|---|---|---|---|
| 15 | **+0.497** | +0.488 | 95/33 | **3.8e-08** |
| 35 | +0.133 | +0.125 | 70/58 | 0.33 |
| 55 | +0.056 | +0.089 | 72/56 | 0.18 |
| 75 | −0.028 | +0.044 | 60/68 | 0.54 |
| ALL | +0.111 | +0.186 | 297/215 | 3.3e-04 |

**The effect narrowed as the budget grew.** At 22.5k steps it was significant
across the low-q band; at 200k it survives only at q15 — where it is now
**+0.497, above the ~0.3 metric floor** and so plausibly visible, not merely
directional. The natural reading is that affinity supervision mostly
*accelerates*, and output-KD catches up given enough steps, except in the
heaviest damage regime where the extra signal still buys something real.

**But neither arm is shippable.** Against the shipped `dejpeg_rt24g`, both lose
at every cell:

| | q15 | q35 | q55 | q75 | overall |
|---|---|---|---|---|---|
| affinity − rt24g (turbo) | −0.433 | −0.220 | −0.087 | −0.027 | **−0.109** (win 0.37) |
| output-KD − rt24g (turbo) | −0.898 | −0.315 | −0.127 | −0.128 | **−0.192** (win 0.34) |

So the rung's question — does affinity KD beat output-KD? — is answered **yes at
q15**, and affinity roughly halves the deficit to the shipped model there. The
follow-on question — is this KD recipe worth adopting? — is **no, not yet**: the
whole branch sits behind `dejpeg_rt24g`. Adopt the affinity term into a recipe
that is already competitive rather than shipping either arm.

Data: `~/tmp/kd_{fkd_a_outkd,fkd_b_affinity,dejpeg_rt24g}.tsv`, compared with
`tools/model_ab.py` per the pre-registered rule. Note the training metric was
again uninformative: final `val_psnr_vs_teacher` was 37.13 (A) vs 37.17 (B),
a 0.04 dB gap that predicts neither the q15 win nor the deficit to rt24g.

---

*(2026-08-02, 22.5k steps — superseded by the run above, kept for the trend.)*
**RESULT: passes the pre-registered rule, with the effect confined to low q.**
Per-file on the clean corpus, arm B minus arm A
(`benchmarks/fkd_affinity_vs_outkd_2026-08-02.tsv`):

| | q15 | q35 | q55 | q75 | q90 |
|---|---|---|---|---|---|
| turbo median | **+0.163** | +0.077 | +0.038 | -0.006 | -0.021 |
| turbo win% | **0.86** | 0.62 | 0.56 | 0.48 | 0.42 |
| mozjpeg median | **+0.153** | +0.058 | +0.041 | +0.002 | -0.012 |
| mozjpeg win% | **0.75** | 0.66 | 0.61 | 0.52 | 0.47 |

Sign test (two-sided exact binomial): turbo q15 p=3.5e-9, mozjpeg q15
p=7.7e-5, low-q band (q15+q35) 185/256 wins p=6.8e-13, overall 503/896
p=2.7e-4. From q75 up nothing reaches significance in either direction.

**Read it carefully.** The magnitudes (~0.1 ssim2) are an order of magnitude
below the ~0.3 single-comparison resolution in §0. What n=256 and p=6.8e-13
establish is the systematic DIRECTION at low q, not a difference a viewer
would notice on one image. And both arms sit at 25k steps against the shipped
model's 200k, so this says affinity KD helps *at a 25k budget* — per the scope
bound recorded before the run, that justifies a full-budget rerun rather than
a shipping decision.

**Next:** rerun the pair at the full 200k-step budget, since that is the only
way to learn whether the low-q gain survives to convergence or is an
early-training artifact that the output loss catches up on.

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
- **Verdict rule, fixed before the data lands (2026-08-02).** Both arms are
  scored per-file on the clean corpus with `tools/model_ab.py` (arm
  `model_proj`), NOT on `val_psnr_vs_teacher`. Today's student comparison
  showed those two answers can point opposite ways: ~2 dB better
  teacher-fidelity came with a -0.02 median and a 0.45 win rate against clean
  references. Affinity KD is a *feature-imitation* term, so it is exactly the
  kind of change that can improve teacher-imitation while making the product
  worse. Pass = positive per-file median with win_frac > 0.5 on the clean
  corpus, with low q (q15-35) weighted as the deciding band. Writing this down
  now so the metric is not chosen after seeing the result.
- **Scope bound, measured before the treatment arm finished.** Arm A (25k
  steps, online teacher) is well short of the shipped rt24g (200k steps,
  precomputed targets): per-file median -0.354, win_frac 0.24, and worst at
  low q (-2.23 turbo q15, -2.01 mozjpeg q15). So both arms sit far from
  convergence, and the rung answers "does affinity KD help at a 25k-step
  budget", NOT "at convergence". A positive result would justify a
  full-budget rerun; a null result does not rule out a benefit that only
  appears later in training. Stated up front so the conclusion is not
  over-read either way.
- Box note: node-3 is a GTX 1660 Ti (6 GB, Turing cc 7.5), **not** the RTX 3070
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

### 1.8 Estimated quality under-describes damage on downscaled input *(new 2026-08-02, from real-world e-commerce files)*

A real Amazon CDN sample probed at q82-87 — mid-to-high — yet reads visually
much worse. Measured, it is not blocking (blockiness 1.10-1.44 against a
calibration where q20 gives 3.61 and q95 gives 1.24) and not chroma
subsampling behaving oddly (the per-plane chroma deficits match a reference
encode to within 0.1 dB). It is **detail density**:

| | Y roundtrip PSNR @/2 |
|---|---|
| native 420 crop, q83 | 52.0 dB |
| same source downscaled to 420, q83 | 34.2 dB |
| downscaled to 270, q83 | 33.4 dB |
| **the Amazon files** | **21.0 - 34.9 dB** |

Aggressive downscaling (their filenames record 1500 -> 420 and 750 -> 270)
packs all content into the highest spatial frequencies — exactly the band the
quantiser coarsens. So **damage = quantiser x content spectrum**, and the
probe's scalar quality only measures the first factor. A q83 downscaled image
carries far more damage than a q83 native one.

**Why this matters to the pipeline:** every routing decision we have keys on
estimated quality — the identity gate, the deblock policy, the tier choice. On
downscaled input that estimate is systematically optimistic, so the gate will
skip files that would benefit. A detail-density signal is cheap (one
downscale-upscale roundtrip on Y) and is a concrete candidate for the damage
estimator `Budget::Adaptive` needs in `docs/API_DESIGN.md`.

Also worth noting for corpus building: these files probe as encoder family
`Unknown`, so `slack_for` falls back to the 0.15 round-to-nearest assumption.
If the CDN encoder trellises, the projection box is too tight on real traffic.

Tooling: `tools/jpeg_inspect.rs` (probe + policy + optional `--restore` delta).

---

### 1.11 Training hardware: the local 5070 is the fastest box, and WSL2 is not a tax

Matched microbenchmark — the real student (nf=24 nc=8), batch 48 at 96x96,
forward+backward+step, timed after warmup with a sync barrier (host timers lie
about async GPU work). `~/tmp/gpubench.py`.

| box | GPU | AMP | steps/min | note |
|---|---|---|---|---|
| this box (WSL2) | RTX 5070 cc12.0 | bf16 | **4,601** | idle, clean |
| lianli (native) | RTX 2080 cc7.5 | fp16 | 2,478 | idle, clean |
| jason (native) | RTX 3070 cc8.6 | bf16 | 1,331 | **CONTENDED — invalid** |

**WSL2 is not a handicap for CUDA compute.** The 5070 under WSL2 is 1.9x an
idle 2080. The known WSL2 problems are NVML under snap-docker and inbound UDP,
neither of which touches training throughput.

The 3070 figure was taken while jason was running the KD pair at 100% GPU, so
it is contention, not hardware — a 2080 does not beat a 3070. Re-measure when
jason is free before quoting any 5070:3070 ratio.

Second reason to prefer this box: 12 GB of VRAM against the 3070's 8 GB. The
dataset-placement logic falls back to host-side batches when the data exceeds
half of free VRAM, and the 100k-crop set (14.7 GB) triggered that on every
8 GB card — which is exactly what made the online-teacher runs slow.

### 1.9 Per-generation slack mapping *(blocks restoring `Provenance::Generations`)*

`Provenance::Generations(u8)` was implemented and then REMOVED from the 0.1
API, because nothing validated existed to do with it. The measurement we have
is "each generation adds roughly 1-2.5 quantizer steps of excess" — an
absolute figure, i.e. `slack_abs`. A first implementation wired it into
`slack_q`, a *fraction* of the quantizer, with an invented per-generation
coefficient; wrong units and a fabricated constant.

To restore the variant, measure the mapping directly: encode gen-1..gen-N
chains (`gen_eval` already builds them, `ZENSR_EVAL_GENS`), and for each
generation count fit the `slack_abs` that contains the p99 of the observed
coefficient excess. That yields a table, not a guess.

Until then `Provenance` carries only `Unknown` and `FreshEncode`. Keeping
`Generations(n)` while treating it as `Unknown` was rejected as decorative —
a variant that does nothing is the same defect as a parameter that is accepted
and ignored.

### 1.10 Tier routing: the gap is quality-dependent, and steeply

Measured per-file median gain on clean references, gap in multiples of the
~0.3 ssim2 the metric resolves:

| input | Quality | Realtime | gap | vs metric floor |
|---|---|---|---|---|
| q15 | +11.62 | +6.88 | 4.74 | **15.8x** |
| q35 | +6.18 | +4.14 | 2.04 | 6.8x |
| q55 | +4.02 | +3.06 | 0.96 | 3.2x |
| q75 | +2.05 | +1.58 | 0.47 | **1.6x** |

At q75 the quality tier's 16x compute buys a difference only 1.6x the metric's
own resolution. At q15 it buys nearly double the gain. A fixed tier choice is
therefore leaving a lot on the table in both directions, and a router keyed on
damage would pay for itself — which is the argument for `Budget::Adaptive`,
dropped from 0.1 only because the damage estimator does not exist (see 1.8;
input quality alone under-describes damage on downscaled files).

---

### 1.12 Content-aware routing — MEASURED 2026-08-03, biggest open lever

`tools/routing_headroom.py`. Ceilings on identical held-out cells (n=1360, all
rules fit on the calibrate half):

| rule | mean ssim2 | restored |
|---|---|---|
| always restore | +0.6573 | 1.00 |
| q-only curve (what ships) | +1.2969 | 0.35 |
| **+ graphic/photographic split (2 classes)** | **+1.4781** | **0.46** |
| + subcorpus (8 classes) | +1.4854 | 0.47 |
| per-image oracle (unreachable) | +1.7049 | 0.58 |

**These content-aware rows are ORACLE-LABEL CEILINGS, not achievable gains** —
class comes from the ground-truth subcorpus directory. A real classifier
misclassifies and lands lower; see the achievable-gain estimate below. The
ceiling is still the right thing to measure first, because it bounds what any
classifier could buy and says whether the work is worth starting.

**A binary content split captures 44% of the entire remaining headroom**, and
it restores *more* while doing so — it is not trading work for quality, it is
making better decisions in both directions. Eight classes add +0.007 over two,
so a binary classifier is enough; there is no case for fine-grained content
typing.

**zensr already has the classifier, and nothing calls it.**
`crates/zensr-zenjpeg/src/chooser.rs:76` — `classify_rgb8(rgb, w, h) ->
ChooserReport` with `ContentClass::{Photo, Graphics}` and `p_graphics: f32`,
fitted on 1023 train images against 192 pinned never-fit eval images
(`benchmarks/chooser_fit_2026-07-26.txt`), on center-512 crops of
compressed-then-decoded JPEGs — i.e. under restore-time conditions. It has
**zero call sites outside its own module**; the restore gate at `api.rs` still
routes on quality and subsampling alone.

**Achievable gain, and why the threshold must move.** Simulating the chooser as
independent per-image draws at its *measured* precision/recall (false-positive
rate derived as `FP = TP(1−P)/P` over this corpus's 24:40 class balance),
against the +0.1812 ceiling:

| threshold | P | R | mean ssim2 | share of ceiling gain |
|---|---|---|---|---|
| 0.85 (shipped) | 0.950 | 0.594 | +1.3667 | **39%** |
| 0.80 | 0.913 | 0.656 | +1.3831 | 48% |
| 0.75 | 0.901 | 0.760 | +1.4089 | **62%** |

Recall is the binding constraint, and the shipped 0.85 leaves most of the gain
on the floor — it was chosen precision-first for *model* routing, where a wrong
pick wastes a pass, not for the *restore gate*, where the asymmetry is
different (a missed graphics restore forgoes ~+3 ssim2; a wrongly-restored
art-scan inflicts −1.28). Re-fit it for this loss. This is an estimate from
measured operating points, not a measurement — confirm by running the real
chooser over the pinned split and substituting its actual labels.

Per-subcorpus graphics recall at t=0.80: documents 24/24, screen 21/24, maps
12/24, art-scans 6/24. Maps and art-scans are the weak classes, and art-scans
is exactly the one §1.13 shows being actively harmed.

Cheaper alternative worth measuring first:
`zenjpeg::detect::content::classify_from_luma_coefficients` runs on the JPEG's
own DCT coefficients — which zensr already has, since S10 projection works
against them — so it costs essentially nothing. If its zero-AC-block fraction
separates the corpus adequately, no pixel pass is needed at all.

Cost, if the pixel path is used: zenanalyze feature extraction is measured at
0.82–2.55 ms at 256² and 5.59–8.21 ms at 1 MP
(`zenanalyze/benchmarks/feature_cost_grid_2026-07-02.tsv`). The chooser runs on
a fixed center-512 crop, so its cost is bracketed by those rows and is
independent of input size — roughly 1.5–5% of a 1 MP realtime restore and well
under 1% of the quality tier. 512² itself is not measured; measure it before
putting it on the hot path.

zenanalyze itself has **no content-type classifier API** — the composite
likelihoods (`ScreenContentLikelihood`, `TextLikelihood`, `NaturalLikelihood`,
`LineArtScore`, ids 27–29/45) were deleted and their ids reserved. Its value
here is the underlying features, whose screen-vs-photo AUCs are recorded in
`zenanalyze/docs/calibration-corpus-2026-04-27.md`: `patch_fraction` 0.880,
`patch_fraction_fast` 0.852, `luma_histogram_entropy` 0.848,
`edge_slope_stdev` 0.844 (**Tier 1**), `flat_color_block_ratio` 0.838 (**Tier
1**). The current pin `a7d8224` already carries everything needed — no bump.

Why this survives n=64 when the per-encoder refit (§1.1b) did not: the content
effect is **~5 ssim2 of spread at every quality**, against 0.3–0.5 for the
encoder effect. Median gain at q75 runs documents +4.18, maps +3.27, screen
+2.83, then renders +0.30, photos +0.27, people +0.15, textures −0.01,
art-scans −1.28.

Where a content-aware rule disagrees with the shipped one at `min_gain: 0.25`:
**graphic content should be restored up to q94–96; photographic content should
be skipped from q75 up.** The pooled curve splits the difference and is wrong
for both.

Not an artifact of the `__pristine3x` corpus rebuild: native photographic
content (photos, art-scans) tracks the photographic group, and art-scans —
native PNG, never downscaled — is the worst cell of all.

**To build it** the router needs a runtime graphic/photographic signal. zensr
already has adjacent machinery (the graphics-specialist model, `gen_detect`,
`chooser_probe`), and zenanalyze exists precisely for cheap feature extraction.
Open question is cost: the classifier must be far cheaper than the restore pass
it decides against, or it eats its own margin.

### 1.13 The model actively harms art-scans — MEASURED 2026-08-03

At q75 the restore pass makes **91% of art-scan images worse** (median −1.28,
win 0.09, n=32), and it is negative from q75 all the way up. This is the one
content class where restoration is a net defect across almost the whole quality
range, and no gate currently knows it.

The corpus is Haeckel lithographs — engraved stippling and fine hatching. The
plausible mechanism is that the model reads high-frequency stipple as ringing
and smooths it, which is the same failure the `renders` metric disagreement
(§0.3) hints at. Worth confirming by eye before theorising further.

Cheapest fix is §1.12 — a content-aware gate would skip these. Alternative is a
training-side fix (more line-art/engraving in the mix), which is slower and may
trade against the classes that currently work.

### 1.14 Angles opened by the 2026-08-03 measurements, not yet run

- **Asymmetric routing objective.** The threshold sweep shows `min_gain` 0.10–0.15
  maximises mean quality (+1.3051) while the shipped 0.25 gives +1.2754 — but
  0.25 harms fewer images (5% of cells vs 6%). Mean ssim2 treats a missed gain
  and an inflicted harm as equal; users do not. An explicit asymmetric loss
  would make that trade a decision rather than an accident of the default.
- **Harm is concentrated and large.** At `min_gain: 0.25` the restores that go
  wrong average **−1.45 ssim2** — far above the metric floor, i.e. visible. A
  predictor of *harm probability* may be more useful than a predictor of median
  gain, and the two are not the same model.
- **Most routing decisions live under the metric floor.** Cell medians between
  q75 and q94 sit in ±0.5 ssim2 while per-image spread runs −4 to +4. Widening
  the eval grid cannot resolve these; only human judgment (§4, squintly) can.
  Worth pointing squintly at exactly the q85–94 band where the gate lives.
- **`CjpegliXyb` is unmeasured.** The jpegli distance curve (§1.1c) covers `CjpegliYcbcr`
  only. The XYB path is a separate family and may need its own curve.
- **Multi-generation jpegli.** Every clean-reference ladder so far is single
  generation.

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
- **A teacher must be evaluated in fp32, never the student's AMP dtype.** The
  64-wide/16-deep teacher overflows to all-NaN under fp16 autocast (measured on
  Turing: finite in fp32 with absmax 1.09, NaN in fp16), and because the
  GradScaler then skips every step, training looks *alive* — loss prints `nan`
  but the run continues and val sits frozen at its init value. bf16 boxes hid
  it entirely. Beyond the crash, a distillation target that varies with the
  student's precision is wrong on its own terms.
- **Running the teacher in fp32 costs ~10x on a card without an fp32 tensor
  path.** The correctness fix above is not free: on the GTX 1660 Ti (Turing,
  no TF32) the online 64-wide/16-deep teacher dropped throughput from ~150
  steps/min to ~12, which turns a 25k-step arm from 3 hours into 33. Ampere
  and later have TF32 and do not pay this. Schedule online-teacher work on
  Ampere+; a Turing card is fine for student-only training.
- **A measurement must not be able to observe its own gate.** The first clean
  ladder reported a q96 crossover that was entirely the shipped identity gate
  short-circuiting restoration. Any harness that evaluates a policy needs a
  documented way to disable it (`ZENSR_EVAL_NOGATE=1`).
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
