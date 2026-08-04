# Reusing the zensim/zenanalyze feature work for routing — 2026-08-04

Prompted by a fair challenge: months of zensim work produced parquets with
encoded variants, precomputed ssim2 and butteraugli, and hundreds of features.
Why hand-roll nine coefficient features and re-measure everything?

Two answers, and the second one is the useful one.

## 1. The canonical datasets — CORRECTED 2026-08-04, they are 77% usable

> **This section was wrong and is kept with its correction below.** I claimed
> "no `origin_id` → imazen-26 path map exists locally to filter by". It does:
> `clean-picker-corpus-2026-06-26/_provenance.tsv` carries `source_sha256` per
> rendition, and following it resolves **all 414 origins**. Doing so reverses
> the verdict — **319 origins (77%) are free of dejpeg training content, giving
> 3,452 usable renditions**. See "Addendum 3". The original reasoning follows.

### Original (superseded) reasoning

`/mnt/v/output/canonical-picker-2026-06-27/zenjpeg_lossy/` is genuinely rich:
1,484,010 rows (train/validate/test), 469 `feat_*` columns, `score_ssim2`,
`content_class`, encoded-variant R2 URLs, `encode_ms`/`decode_ms`.

It is unusable for dejpeg eval:

- Its README says it is built from "the all-origin clean-picker **imazen-26**
  sweep", and its origin ids sit in 1000–9999 — the imazen-26 range, per the
  sibling nonphoto corpus manifest which documents its own 10000–10501 as
  "disjoint from imazen-26 1000-9999".
- The dejpeg model trains on imazen-26. So these sources are very likely
  training data.
- The manifest's `leakage_check` reports `leaked: false`, but that covers
  **picker-internal train/val/test overlap** — a different question from
  overlap with a different model's training set.
- No `origin_id` → imazen-26 path map exists locally to filter by;
  `origin_split.py` derives the id from the filename and never records a source
  path.

Reusing it would reintroduce exactly the contamination this line of work exists
to remove. Two other details that would have limited it anyway: `content_class`
is `unknown` on every row, and the q grid has only 7 distinct values.

## 2. The precomputed metrics would not have helped much regardless

The stored `score_ssim2` is **(reference, encoded)** — the `identity_off` arm.
The metric this work needs is **(reference, restored)**, which requires running
the dejpeg model and which nobody has precomputed. Reuse would save one arm of
three plus the encode, not the expensive part.

Measured per call at 512² (`metric_cost` bin): ssim2 19.71 ms, butteraugli
16.30 ms, encode 1.99 ms. Model inference and decode dominate beyond those.

## 3. What DOES transfer: the feature set

The features in those parquets are **source-image** features — verified constant
across all 330 encodes of one reference. So they join to any encode at any
quality by filename, and the feature *design* transfers even though the data
does not. `source_features` now extracts them on any corpus:
**101 features on 913 images in 22 seconds**, one row per reference.

Against per-image restore gain on the clean 64-image corpus, the best
zenanalyze feature beats every hand-rolled one:

| feature | q35 | q55 | q75 | q85 | q90 |
|---|---|---|---|---|---|
| **`edge_slope_stdev`** | +0.75 | +0.79 | **+0.81** | +0.75 | +0.68 |
| `patch_fraction_fast` | +0.65 | +0.75 | +0.55 | +0.49 | +0.45 |
| `aq_map_std` | +0.67 | +0.71 | +0.70 | +0.60 | +0.51 |
| `colourfulness` | −0.58 | −0.66 | −0.54 | −0.56 | −0.55 |
| *`mean_abs_ac` (best hand-rolled)* | +0.56 | +0.61 | +0.67 | +0.57 | +0.49 |
| *(ground-truth content label)* | +0.61 | +0.69 | +0.58 | +0.51 | +0.47 |

`edge_slope_stdev` at q75 (+0.81) beats both the best coefficient feature
(+0.67) and the true content label (+0.58).

**And unlike `mean_abs_ac`, it survives the 20-split test:**

| router | mean over 20 splits | beats shipped |
|---|---|---|
| quality only | +1.2119 | 0/20 |
| binary zero-AC (**shipped**) | +1.3569 | — |
| oracle content label | +1.3730 | 16/20 |
| **linear `edge_slope_stdev`** | **+1.4424** | **20/20** |
| linear `edge_slope_stdev` + `aq_map_std` | +1.4339 | 20/20 |
| per-image oracle | +1.5343 | — |

It captures **71% of the per-image oracle headroom** against 45% for the shipped
binary, and beats perfect content labels — a per-image continuous signal
outperforms a perfect two-class one. Adding a second feature makes it slightly
worse, the same overfitting pattern seen everywhere else at this sample size.

## Cost

Marginal, on already-decoded pixels — which is what a router pays, since the
restore path decodes regardless:

| | ms/call |
|---|---|
| coefficient path (shipped) | 0.29 |
| `edge_slope_stdev` alone | **2.15** |
| all 101 zenanalyze features | 4.51 |

7× the shipped path for +0.086 ssim2. On a 512-crop realtime restore (~42 ms)
that is 5% overhead against 0.7%.

## Not shipped yet

20/20 splits at n=64 is strong, but n=64 has produced two false positives
already today (the per-encoder crossover, and `mean_abs_ac` at +1.5346 on one
split). The XL sweep lands 913 images within hours and settles it. Ship after
that confirms, not before.

If it holds, the shape is: replace the binary content class in
`Routing::Auto` with a per-cell linear model on `edge_slope_stdev`, keeping the
coefficient classifier as the fallback when features cannot be computed.

## Addendum — the split was at the wrong level

Prompted by a second challenge: zenmetrics has a canonical split rule
(`scripts/picker/origin_split.py`) whose header says *"Import this everywhere —
do not re-implement the rule."* I had written my own hash-of-filename split.

**Measured, that rule cannot apply here.** It keys on a *leading* numeric origin
stem (`o_1004…`, `1003_general_…`); zensr's corpora have none. It returns an id
for **2 of 64** pinned eval files and **0 of 913** XL files, against 390 of 390
on the picker corpus. So a separate rule is genuinely needed — but I should have
checked and said so, not silently invented one.

**Its underlying principle did apply, and I had missed it.** The rule is
origin-level so that *every* derivative of an origin shares a bucket. In the XL
corpus **58% of files share an origin** — patents is 357 files from 31
documents, up to 71 pages each; sci-figures 141 from 11. Two pages of one patent
share a scanner, typography and paper, so a per-file split puts near-duplicates
on both sides and "held out" stops meaning anything.

A first attempt at grouping proved the rule's other lesson the hard way:
stripping a trailing `-\d+` turned `pexels-photo-1029599` into `pexels-photo`
and **merged 37 distinct CID22 images into one origin**. That is exactly why
`origin_split.py` keys on a leading stem. The fix only groups on a trailing
index when the path corroborates it — the last component minus its index must
equal an earlier component.

`origin_of()` in `tools/routing_headroom.py` now implements this, and the tool
splits by origin:

| corpus | files | origins | in multi-file groups | largest |
|---|---|---|---|---|
| clean64 | 164 | 144 | 40 | 2 |
| **XL** | **913** | **419** | **554 (58%)** | **71** |

**The `edge_slope_stdev` result survives the correction:**

| router | per-FILE split | per-ORIGIN split (correct) |
|---|---|---|
| quality only | +1.2769 | +1.2599 |
| binary zero-AC (shipped) | +1.4322 | +1.4001 |
| **linear `edge_slope_stdev`** | +1.5010 (19/20) | **+1.4742 (20/20)** |

Small movement, as expected: the 64-image corpus has only 3 multi-file origins.
**It will matter enormously on XL**, where 58% of files share an origin — every
XL number must use the origin split or it will be optimistic.

## Addendum 2 — the canonical mechanic, and what it says at n=64

Corrected again: the canonical rule's key is the **trailing digit of the leading
integer** of the origin id. `split_of()`/`bucket_of()` in
`tools/routing_headroom.py` now implement exactly that, applied to the
*identifying component* of the origin.

That last part is forced, not a deviation. The corpus flattens nested paths with
`__`, so a corpus prefix leads — `CID22-512__training__1001682`. Taking the
first integer of the whole string picks up the `22` in `CID22` and hands **274
of 383 XL origins the same digit**, splitting 333/33/17. Reading the identifying
component instead gives **53/28/19 against the canonical 50/30/20**, with a
near-uniform digit histogram. Digitless origins (~11%: `haeckel` plates, some
NPS maps) get a deterministic hash fallback in the same proportions rather than
being dropped.

Fitting on `train` only, with `test` untouched until the moment of reporting:

| router | val | test |
|---|---|---|
| quality only | +1.1593 | +1.2834 |
| binary zero-AC (shipped) | **+1.1196** | +1.3632 |
| linear `edge_slope_stdev` | **+1.3426** | +1.3747 |
| per-image oracle | — | +1.5553 |

**This is less flattering than the 20-split result and the reason is sample
size.** The canonical 50/30/20 on 61 origins leaves a **9-origin test set** —
far too small to separate +1.3747 from +1.3632. It is one draw, which is exactly
the failure mode that killed `mean_abs_ac`.

Two things it does show:

- `edge_slope_stdev` wins on both buckets, decisively on val (+1.3426 vs
  +1.1196) and marginally on test.
- **The shipped binary content class LOSES to quality-only on val** (+1.1196 vs
  +1.1593). One draw, and it wins on test, so this is not a retraction — but it
  is a caution against treating the shipped +0.148 as settled.

At this corpus size the 20-random-origin-split result remains the better
evidence, because it averages over draws instead of trusting one. The canonical
split becomes the decisive protocol on XL, where 419 origins yield roughly
209/125/84 — a test set large enough to mean something.

**Both protocols now run on XL when it lands:** 20 random origin splits for
robustness, and the canonical fixed split for a reproducible number comparable
with other zen work.

## Addendum 3 — the mapping exists, and it reverses §1

I wrote that no origin→path map was available. `_provenance.tsv` in the corpus
directory carries `source_sha256` for every rendition. Following it resolves all
414 origins, and the verdict flips:

| | origins | |
|---|---|---|
| **dejpeg-safe** | **319 (77%)** | usable |
| exact training file | 81 (20%) | excluded |
| near-duplicate of training | 14 (3%) | excluded |

**3,452 of 4,497 renditions are leakage-free.**

Resolving it needs three different comparisons, and two obvious approaches fail
silently:

- Only 96 of 414 origins resolve against `/mnt/v/imazen-26`. The other 318 live
  in `/mnt/v/output/imazen-26-png` — a **larger, differently organised**
  collection (2,639 files in numbered dirs like `9226-lilith-ai-products`, vs
  1,068 in flat dirs like `lilith`).
- Comparing **directory names** across the two roots is wrong and flattering: it
  reads `2000-unsplash-people` as "not a trained subcorpus" when it is the same
  content as `unsplash-people`, which is trained on. I made this mistake first
  and it reported 325 safe for the wrong reason.
- Comparing **filename stems** finds **zero** matches — the schemes are disjoint.
- Comparing **bytes** also fails: imazen-26-png is re-encoded, so nothing there
  is byte-identical to an imazen-26 file.

So the 318 need a content fingerprint. A 16×16 luma thumbnail at mean
|Δ| < 3/255 catches re-encodes and format conversions of the same scene, which
is what leakage means — the model saw the picture, not the file. That finds 14
near-duplicates the exact test missed.

Audit: `tools/picker_leakage_audit.py`. Verified list:
`eval_split/picker_safe_origins_2026-08-04.txt`.

**Why this matters beyond the correction.** Those 3,452 renditions are
**size-diverse** — `scale36x64` upward through full size. The XL corpus is not:
every one of its 913 images is a 512 crop. The sweep discipline asks for 16-20
log-spaced sizes for anything a model is fitted on, and the routing curves are
fitted. This is the corpus's largest remaining gap and the picker renditions
close it, with per-rendition `score_ssim2` already computed for the identity arm.
