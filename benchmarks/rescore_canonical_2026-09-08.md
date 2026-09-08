# Re-scoring the existing checkpoints on the canonical split — 2026-09-08

The first measurement of any zensr checkpoint against a real held-out set. Every
previous number was produced on `/mnt/v/imazen-26` (the wrong corpus, since
deleted) under the "first-8-sorted ∪ 64 pinned" scheme, which was never a split
and leaked twice. Context: `docs/CORPUS-REPOINT-IMPACT.md`,
`docs/MODEL-PROVENANCE-AUDIT.md`.

**Headline: both shipped models survive. The published numbers were
unverifiable, not wrong.** On like-for-like references they are *lower at q15*
and *higher at q35-q90* than the README claims. The dejpeg9 graphics route is
confirmed and roughly twice as strong as published.

## Provenance

| | |
|---|---|
| corpus | `github.com/imazen/imazen-26` @ `187fbf3`, checked out at `~/work/imazen-26`; all 2,160 files sha256-verified against the manifest |
| held-out set | validate ∪ test of `eval_split/imazen26_effective_split.tsv` (`just split`) — canonical id rule + the near-duplicate same-bucketing |
| leakage check | **0 training files scored**; a spot audit of the smoke run resolved 120 validate + 6 test rows, 0 train |
| files | 63 = 3 per folder across **all 21 content classes** (the old basis was 8 per folder across 8) |
| references | 36 PNG / 27 JPEG; 27 photographic. **0 files skipped** — the new skip accounting reported none |
| grid | turbo × 4:2:0 × q ∈ {15,35,55,75,90} × 6 arms = 1,890 rows per model |
| arms | `identity_off` (plain decode, the baseline) … `model_proj` (**the shipped pipeline**, S10 projection on) |
| metric | ssim2; **paired** — median of per-file differences and win fraction, never a difference of medians |
| commit / host | `52caf412bd7f` / this workstation — WSL2, AMD Ryzen 9 7950X, 32T (NOT the 9950X3D box that the `dev` ssh alias reaches; the WSL hostname is also `dev`), 2,849 s wall under run-heavy |
| raw | `/mnt/v/zensr/rescore/2026-09-08/` (see the pointer file) |

The README's basis was 100% PNG references, so the like-for-like comparison is
the **PNG column**, not the aggregate.

## 1. The two shipped tiers, against their published claims

`model_proj`, gain in ssim2 over the plain decode, per file.

| model | source | q15 | q35 | q55 | q75 | q90 |
|---|---|---|---|---|---|---|
| dejpeg7_graphics | README (old corpus, PNG, n=64) | +10.65 | +5.64 | +3.53 | +2.03 | +0.95 |
| dejpeg7_graphics | **new, PNG refs (n=36)** | +7.78 | **+6.11** | **+4.60** | **+3.19** | **+1.06** |
| dejpeg7_graphics | new, all refs (n=63) | +8.29 | +5.19 | +2.76 | +1.59 | +0.49 |
| dejpeg_rt24g | README (old corpus, PNG, n=64) | +6.88 | +3.36 | +2.02 | +1.15 | +0.32 |
| dejpeg_rt24g | **new, PNG refs (n=36)** | +5.07 | **+4.44** | **+3.26** | **+2.85** | **+0.87** |
| dejpeg_rt24g | new, all refs (n=63) | +5.48 | +3.34 | +1.96 | +0.75 | +0.23 |

Win fractions on the full sample: dejpeg7 **97/100/100/94/84%**, rt24g
**98/100/94/84/67%**. These models help on nearly every individual file, which is
the claim that matters and the one a difference-of-medians would not support.

**The q15 regression is the one real divergence** — −27% for dejpeg7, −26% for
rt24g, consistently, on the like-for-like PNG population. Low-q is where web
traffic lives and where the routing constants are fitted, so it deserves an
explanation before the retrain rather than after.

Tier ratio (rt24g ÷ dejpeg7, all refs): 66 / 64 / 71 / 47 / 47%. The README's
"57–65% of quality tier" holds at low q and overstates at q75+.

## 2. The realtime tier really is behind the quality tier

Paired, `model_proj`, rt24g minus dejpeg7_graphics:

| q | median | win frac |
|---|---|---|
| 15 | −3.841 | 0.05 |
| 35 | −1.594 | 0.03 |
| 55 | −1.076 | 0.03 |
| 75 | −0.646 | 0.06 |
| 90 | −0.264 | 0.16 |

Unambiguous — rt24g loses on 94–97% of files at q15–q75. It is a 43k-parameter
model at 0.16 s/MP against 595k at 5.3 s/MP; this is the tier trade working as
designed, not a defect.

## 3. The graphics route is confirmed, and stronger than published

The README claims dejpeg9_gfxycc gives "+1.6/+0.7/+0.4 OVER dejpeg7 on graphics
at q15/35/55", flagged *not yet re-measured on clean references*. Re-measured,
paired, split by content label:

| q | graphic (n=27) | win% | photo/other (n=36) | win% |
|---|---|---|---|---|
| 15 | **+2.723** | 81% | +0.515 | 72% |
| 35 | **+1.451** | 81% | +0.107 | 61% |
| 55 | **+1.090** | 89% | −0.021 | 47% |
| 75 | **+0.779** | 81% | −0.117 | 36% |
| 90 | +0.205 | 67% | −0.105 | 33% |

The advantage on graphics is **1.7–2.7× larger than claimed** and extends to q75,
which the README did not claim. On non-graphic content it is neutral at q35 and
**negative from q55 up**, winning only 33–47% of files.

That is exactly what a content *route* should look like: a real gain where it is
routed, a real loss where it is not. It validates the routing design rather than
the model alone — and it means shipping dejpeg9 unrouted would be a regression on
most photographs.

## 4. The JPEG-reference effect, measured

rt24g at q90: **+0.867 on PNG references, −0.100 on JPEG ones** — it *loses*
against a JPEG-sourced reference, dragging the overall win fraction to 67%. At
high quality a JPEG reference already contains artifacts comparable to the q90
encode, so removing them moves the output away from the target. This is the
clean-reference rule (ROADMAP §0.1) as a number rather than an argument.

The effect inverts at q15, where JPEG references show *larger* gains (+6.07 vs
+5.07 for rt24g): a q15 encode is far worse than any reference, clean or not.

**Do not read this as a pure reference-quality effect.** In this corpus PNG/JPEG
is nearly collinear with graphic/photographic — 6 of 207 photographic files are
natively PNG — and graphic content is where dejpeg legitimately gains most. The
two columns are two populations, not a controlled comparison. Separating them
needs clean photographic references, i.e. the downscale-to-pristine treatment
(handoff §5 step 2). The warning is repeated in `tools/rescore_report.py` so it
travels with the output.

## 5. What this changes

- **The retrain is still justified**, but by the corpus swap and the corrupt-GT
  lineage (`docs/MODEL-PROVENANCE-AUDIT.md`), **not** by a demonstrated quality
  failure. dejpeg7_graphics works.
- **The q15 gap is the open question.** Both models lose ~27% of their published
  low-q gain on the new PNG population. Worth understanding first.
- **dejpeg9 should stay a route, not a default.** Its loss on photographs at
  q55+ is larger than its q90 gain on graphics.
- **The README numbers can now be replaced** with measured ones, on a corpus that
  exists, with a split that holds.

Reproduce: `just split`, then for each model
`ZENSR_EVAL_ENCODERS=turbo ZENSR_EVAL_SS=420 ./target/release/dejpeg_eval ~/work/imazen-26 out.tsv 3 12 <m> <m> <m>`
— the model must be passed **three** times or the `model_proj` arm is not emitted
and the run measures the raw model instead of the shipped pipeline.
