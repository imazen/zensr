# What the imazen-26 repoint actually changed

Measured 2026-09-07/08, after executing Step 1 of `CORPUS-REPOINT-HANDOFF.md`.
Everything here is counted from `CORPUS-MANIFEST.tsv` and the new split; nothing
is carried over from the pre-repoint record, which was fitted through the deleted
`/mnt/v/imazen-26` root and cannot be reproduced.

## 1. The headline: this is a corpus swap, not a path fix

The training set did not just move — **59% of it is content the model has never
seen**, and its centre of gravity moved from photographs to synthetic graphics.

| | old (invalid root) | new (canonical) |
|---|---|---|
| root | `/mnt/v/imazen-26` (deleted) | `~/work/codec-corpus/imazen-26` |
| corpus | 1,069 files, 8 flat subcorpora | 2,160 rows, 21 id-assigned folders |
| training files | ≤937 (not reproducible; upper bound = 1,001 in the eight trained subcorpora, less the 64 first-8-sorted) | **1,083** (measured) |
| held out | "first 8 sorted ∪ 64 pinned" | **657 val + 420 test**, origin-level |
| content classes | 8 | 21 folders → 13 labels |

## 2. Composition — the part that should change a decision

Training crops per pass over the pool, by folder. Every file clears 4× the 192px
crop area, so crop share equals file share; nothing is size-weighted away.

| folder | crops | share | |
|---|---|---|---|
| 9226-lilith-ai-products | 1,500 | **35.3%** | NEW |
| 8100-lilith-web-screenshots | 651 | 15.3% | |
| 1400-lilith-nature | 316 | 7.4% | |
| 7000-lilith-plots | 252 | 5.9% | NEW |
| 6000-lilith-scans-public-patents | 228 | 5.4% | NEW |
| 9000-lilith-ai-clipart | 172 | 4.0% | NEW |
| 9094-lilith-ai-illustrations | 152 | 3.6% | NEW |
| *(14 more, each ≤3.4%)* | 1,080 | 25.4% | |

Grouped by what the pixels actually are:

| | share of training crops |
|---|---|
| synthetic / graphic (AI renders, clipart, screenshots, plots) | **~66%** |
| photographic | **~20%** |
| scans and documents | ~14% |

**This is the finding that matters.** zensr restores *web JPEGs*. A model whose
training is two-thirds AI renders and screenshots, and a third one single folder,
is not obviously the model that was intended — and nobody chose it. It is what
falls out of "train on the whole canonical corpus", which is the only defensible
default once the old eight-subcorpus list is gone.

**Recommendation before Step 3 (retrain): cap per-folder contribution.** A cap
around 15% of pairs would take ai-products from 35% to 15% and leave every other
class untouched, at a cost of ~20% of the pool. That is a deliberate, recordable
choice; 35% by accident is not. The cap belongs in `make_distill_data.py` as an
explicit knob, and the ladder should be reported per content label either way —
an aggregate number over this mixture mostly measures AI-product renders.

## 3. The split got materially better

The old scheme — "first 8 sorted, union a hand-maintained list of 64" — was never
a split. It leaked twice: the `teresa` incident, where the runtime's "first 8
*usable*" slid past a 101 MP decode-skip and admitted file 9 into training.

The canonical corpus carries ids, so the canonical zenmetrics rule now applies
(trailing digit of the leading integer). `tools/corpus_split.py` imports that rule
rather than re-implementing it, and adds grouping: origins are `(folder,
descriptor)`, split on the group's **minimum** id, every member inheriting it.

2,160 files → **1,911 origins**; 1,083 train / 657 val / 420 test (50/30/19% against
a 50/30/20 target). There is no first-N rule left to slide.

**A correction to the handoff's own table.** It reported 1,884 origins and
1,097-648-415. Those are what the *naive* `(folder, descriptor)` key produces —
measured before its own Trap 2 was applied. The tell was in the table: it said
"largest group 7" while the naive key's largest group is 28. That 28 is
`2000-unsplash-people`, whose `descriptor` column is empty for all 28 rows, so the
naive key merges 28 unrelated photographs into one origin with minimum id 2000 →
**all 28 land in train, and neither val nor test contains a single photograph of a
person.** For a restoration model that is not a rounding error. Fixed: 14/9/5.

## 4. Two content classes moved, one is gone

- **`office-documents` has no canonical equivalent** — it did not survive
  curation. `canonical_for()` raises for it rather than returning an empty list,
  so a caller cannot silently train on a corpus missing a class it asked for.
- **`screen` is not the same pictures.** It maps onto two canonical folders, but
  half the old subcorpus (219 of its files) was never in the canonical corpus.
  Treat it as a different subcorpus that shares a content class.
- **New, never trained on:** patents, plots, AI clipart / illustrations /
  products, EPA and NOAA documents, Art Institute and Met Museum photography.

## 5. The eval corpus shrank, and two legs were wrong

XL corpus: **913 → 828 files.**

| leg | was | now | why |
|---|---|---|---|
| patents | 357 | **318** | 3 patent documents (39 pages) are now training data |
| noaa | 44 | **22** | rebuilt from the canonical folder, held-out buckets only |
| nasa | 24 | **0** | dropped — no canonical replacement |
| sci-figures, cid22, clic2025, gb82, gb82-sc | 538 | 538 | audited clean, 0 flagged |

The nasa leg is gone because the canonical corpus's 72 "nasa" rows are
`8100-lilith-web-screenshots` PNGs **of nasa.gov web pages** (source
`various-web`) — screenshots of a website, not NASA imagery. Substituting them
would have relabelled a content class while appearing to restore a leg. The 24
filenames are preserved in `eval_split/xl_nasa_leg_dropped_2026-09-07.txt` so it
can be re-acquired deliberately.

The corpus's `NO_PIN_REQUIRED` marker previously justified itself with "training
reads only these eight subcorpora of `/mnt/v/imazen-26`". That stopped being true
at the moment of the repoint and nothing forced anyone to notice, so the builder
now writes the marker itself and states which legs are filtered and how.

## 6. A measurement method that was over-reporting

`tools/leakage_audit.py`'s 16×16 luma fingerprint flagged 29 patent pages across
**six** documents. Only **three** of those documents are in the training corpus.

| | distance | agreed by name? | verdict |
|---|---|---|---|
| 20 pages, 3 documents | **0.00** | yes | true leak |
| 9 pages, incl. 3 other documents | 2.26–2.95 | no — matched *unrelated* patents | false positive |

Two of the false positives had fingerprint stdev ≈6.5: near-blank text pages,
which at 16×16 all look alike. 42 of 1,034 training fingerprints are near-blank,
so the collision surface is real. Acting on the fingerprint alone would have
discarded **87 clean pages**; the document-level exclusion built on it would have
dropped 126 instead of the correct 39.

Where a corpus carries identity — a patent number, a manifest id, a source hash —
the fingerprint is a **candidate generator** and identity is the verdict. The
final filter keys on the patent number against `CORPUS-MANIFEST.tsv`, not on the
fingerprint. This is recorded in the tool's own docstring, because the tool is
where the next session will look.

## 7. What is now provisional, and what survives

**Provisional — fitted through the wrong corpus.** Every constant in
`crates/zensr-zenjpeg/src/api.rs`: `G420`, `G444`, `DIST420`, `DIST444`,
`GRAPHIC420/444`, `PHOTO420/444`, `GRAPHIC_ZERO_AC_THRESHOLD`, the `estimate_gain`
calibration, and the identity gate (q ≥ 94.5 at 4:2:0, q ≥ 88 at 4:4:4). The
identity gate is the highest-stakes: ungated, the model *lost* up to 2.1 ssim2 and
harmed 91% of files. Re-derive it first and independently.

The 31 tests in that file assert **shape** — monotonicity, ordering,
interpolation — not fitted values, and all 31 still pass. They should survive
recalibration; a failure on new constants is a signal about the curve, not a test
to relax.

`edge_slope_stdev` (+0.81 correlation, 20/20 splits) was measured on the dead
corpus. Keep the hypothesis, discard the number, re-measure.

**Survives.** The zenjpeg integration (probe policy, deblock rule, S10 projection,
slack calibration — calibrated on coefficient statistics, not corpus content), the
runtime and its SIMD kernels, and the picker-corpus leakage work, whose corpus
never touched the invalid root. Its 24,137 unanalysed `picker_gain` cells are now
at `/mnt/v/zensr/picker-gain/2026-08-04/` behind a pointer file, off the
non-durable scratch they were sitting in.

## 8. Reference-provenance is now recorded, and it is a problem worth naming

20% of the canonical corpus is JPEG and 4% HEIC, concentrated in exactly the
photographic folders (`2000-unsplash-people` is 28/28 JPEG). A JPEG ground truth
penalises the model for removing artifacts that are present in the reference —
the defect that understated every gain in the 2026-07 record.

The train pool is **825 PNG / 210 JPEG / 46 HEIC**. `make_distill_data.py` now
records this breakdown in `meta.json` per run, so the ladder can be reported split
by reference kind — the column whose absence caused the original defect.

`ZENSR_CLEAN_GT=1` is **no longer a light filter** on this corpus: it would drop
nearly all photographic content and leave a corpus of screenshots, plots, AI
renders and scans, quietly turning the "photo" leg of the content-split curves
into a fiction. The real fix is the downscale-to-pristine treatment (handoff §5
Step 2), with `imazen-26-pristine` (74 files, still on disk) as the existing
prototype. The flag now carries that warning at its use site.
