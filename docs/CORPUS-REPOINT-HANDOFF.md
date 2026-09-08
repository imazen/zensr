# Handoff: repointing zensr onto the valid imazen-26

> **Step 1 DONE 2026-09-08, and this document's destination was wrong.** It names
> `~/work/codec-corpus/imazen-26` as the valid corpus throughout. That is the
> **pre-2026-08-23 location**: the corpus moved out of `imazen/codec-corpus` into
> its own repository, **`github.com/imazen/imazen-26`**, which is where the
> manifests, the canonical split, the variant registry and the generation tooling
> now live and version together. User directive 2026-09-07: *"nothing in
> /mnt/v/imazen* is valid"* — and the codec-corpus subdirectory is stale, not
> canonical. Read every "`~/work/codec-corpus/imazen-26`" below as
> "`github.com/imazen/imazen-26`, checked out at `~/work/imazen-26`".
>
> Consequences for §6 in particular: the split is **not** something to derive.
> The repo publishes it (`manifests/split_map.tsv`, 1,084/658/418). §6's
> hand-rolled grouping — and my own re-derivation of it — are superseded by
> `tools/corpus_split.py`, which reads the canonical buckets and adds only the
> near-duplicate same-bucketing the corpus repo itself prescribes.
> What actually changed: `docs/CORPUS-REPOINT-IMPACT.md`.

**Status: Step 1 complete; Steps 2-4 open. Nothing in this repo's measured record is trustworthy until they are done.**
Written 2026-09-07. Supersedes nothing; read alongside
`benchmarks/imazen26_contamination_audit_2026-08-05.md`, which has the per-file
evidence this summarises.

---

## 1. The situation in one paragraph

zensr trains and evaluates against a corpus called `imazen-26`. **There are two
different corpora with that name**, and zensr picked the wrong one. The valid one
is `~/work/codec-corpus/imazen-26` (curated, id-assigned, manifested). The one
zensr used is `/mnt/v/imazen-26`, the *pre-curation acquisition* corpus that the
curation pass was run against — 1,068 files to the canonical 2,067, of which 23%
never made it into the canonical corpus at all. Every dejpeg model, every ladder
number, and every routing constant now shipped in `crates/zensr-zenjpeg/src/api.rs`
was fitted through that root. They are provisional. The user's ruling, 2026-08-05:
*anything not based on `codec-corpus/imazen-26` is invalid.*

## 2. What zensr is, and where zenjpeg fits

zensr is a **CPU restoration and super-resolution engine for web JPEGs**: take a
JPEG that has already been compressed and lost detail, and give back a better
image than the plain decode. Two model tiers (a 595k-param quality tier, a 43k /
84 KB realtime tier), hand-written safe-Rust SIMD inference in `zensr-micro`, no
GPU path yet.

**zenjpeg is a separate crate and a separate repo** (`imazen/zenjpeg`) — the
pure-Rust JPEG codec. zensr depends on it; it does not vendor or fork it. The
whole integration is one crate, `crates/zensr-zenjpeg`, and the dependency is
currently pinned to a git rev because the API zensr needs is unpublished:

```toml
# crates/zensr-zenjpeg/Cargo.toml
zenjpeg = { git = "https://github.com/imazen/zenjpeg", rev = "e277e9c9" }
```

The pin exists because **zenjpeg 0.9.0 is not published** and 0.8.4 keeps
`DeblockMode` `pub(crate)`. A gitignored `.cargo` override (`just cargo-local`)
points at a local sibling tree when you have one. See the memory note
`caterr-release-train` — zenjpeg 0.9.0 is in a queued release train, and this pin
becomes a version requirement when it ships.

zensr uses exactly four things from zenjpeg, and each one is load-bearing:

| zenjpeg API | what zensr does with it |
|---|---|
| `detect::probe` → `JpegProbe` | **The router's entire input.** Recovers the encoder family (turbo / mozjpeg / cjpegli / zenjpeg / Unknown), the quality scale, and the quality value from the file's own DQT. Every routing decision below keys off this — no pixels needed. |
| `decoder::Decoder` + `DeblockMode` | Decode. Policy: `DeblockMode::Auto` (Knusperli, coefficient-domain) **only** for Annex-K-family files at probe q ≤ 9.5; AQ-family encoders never. |
| coefficient access (`CoeffView`) | Feeds the S10 **quantization-consistency projection** — the output is projected back into the box of images that re-encode to the file's own coefficients, so a restore provably cannot invent detail the bitstream excludes. Slack per encoder family, calibrated on 1M luma coefficients per cell. |
| `encoder::{EncoderConfig, ChromaSubsampling}` | Re-encode, for eval ladders and for the classification throughput work. |

So the relationship is: **zenjpeg tells zensr what kind of damage it is looking at,
and bounds what zensr is allowed to do about it.** The router is cheap precisely
because the probe is metadata-only — the shipped content classifier costs 0.29 ms
on already-decoded coefficients. This is also why the corpus defect propagates:
the probe is exact, but every *threshold* it feeds was fitted on the wrong images.

Sibling relationships worth knowing: `zensr-micro` is the inference engine
(no zenjpeg dependency), `zensr-micro-abi` is its cdylib FFI shell, `zensr-bench`
is the measurement harness where all the eval binaries live.

## 3. How the wrong corpus got picked

Not a typo — a migration that everything except dejpeg followed. `imazen-26` was
acquired to `/mnt/v/imazen-26`, then curated into `~/work/codec-corpus/imazen-26`:
files renamed to carry ids, a manifest written, licence-excluded material split
off. zensr's training script was written against the pre-curation path and never
moved. Meanwhile the picker corpora, squintly, and the PNG derivatives all
re-derived from the curated side — squintly's builder literally calls
`codec-corpus/imazen-26-png-v3` "the canonical corpus". zensr was the last consumer
on the old side, and because the two share 77% of their content, nothing ever
looked obviously wrong.

Root-cause detail: `benchmarks/imazen26_naming_break_2026-08-05.md`.

## 4. Disk state TODAY — this has changed since the audit

Re-measured 2026-09-07. **The invalid root has been deleted.** This removes the
option of doing nothing.

| path | 2026-08-05 | 2026-09-07 |
|---|---|---|
| `~/work/codec-corpus/imazen-26` (**valid**) | 2,563 files | **2,067 images / 2,160 manifest rows — present** |
| `/mnt/v/imazen-26` (invalid root) | 1,068 files | **DELETED** |
| `/mnt/v/imazen-26-clean` (derived) | 974 | **empty shell — 0 files** |
| `/mnt/v/imazen-26-clean-xl` (derived) | 913 | **empty shell — 0 files** |
| `/mnt/v/imazen-26-pristine` (derived) | 74 | 74 — still present |
| `~/work/codec-corpus/imazen-26 - Copy*` strays | 3 dirs | **gone** (audit fix-order step 5 already done) |
| `/mnt/v/output/imazen-26-png-v3` (valid-derived) | 2,639 | 2,639 — present |
| XL non-imazen sources (patents, sci-figures, CID22, clic2025, gb82, gb82-sc) | — | **all present**, 1,136 files total |
| `/mnt/v/output/clean-picker-corpus-2026-06-26` | 4,497 | 4,507 — present |

`/mnt/v` is at 90% (394 GB free), so assume this was disk pressure, not
deliberate curation. Consequence: **you cannot reproduce the old numbers even to
compare against.** Do not spend time trying. The comparison you want is
new-corpus-vs-new-corpus.

The valid-root count also moved (2,563 → 2,067 images). Reconcile against
`CORPUS-MANIFEST.tsv` (2,160 rows, every row's file verified present on disk) before
quoting a corpus size — the 2,563 figure in the audit is not reproducible today
and may have counted the stray `- Copy` dirs.

## 5. The work, in order

Steps 1–2 are mechanical. Step 3 is a retrain and is where the time goes. Do not
reorder — every step invalidates the ones after it.

### Step 1 — repoint the root, re-derive the pinned eval split

Three code sites, all flagged in-tree with `INVALID` comments (commit `15af08a`):

- `tools/make_distill_data.py:26` — `ROOT = env("ZENSR_ROOT", "/mnt/v/imazen-26")`.
  **This is the training set.** The default now points at a deleted path, so it
  fails loudly rather than silently training on the wrong thing.
- `tools/teacher_audition.py:27` — `SRC = "/mnt/v/imazen-26"`.
- `tools/build_xl_corpus.sh:45-46` — the `nasa` and `noaa` legs (68 of 913 XL files).
  Rebuild from `5300-noaa-hurricane-documents` and the NASA equivalent.
- `crates/zensr-bench/src/bin/gen_detect.rs:980` — cosmetic path display, fix while
  you are there.

`SUBS` in `make_distill_data.py` also needs remapping — the flat names
(`lilith`, `screen`, `office-documents`, …) do not exist in the canonical layout,
which is `1000-lilith-photos-general`, `8100-lilith-web-screenshots`,
`5200-epa-climate-impact-2021-report`, and so on. The mapping is not 1:1: the old
flat `lilith` spans four canonical dirs (photos-general / interiors / nature /
food), and the old `screen` — the subcorpus that was **half** outside the valid
corpus — maps onto `8000-lilith-mobile-screenshots` + `8100-lilith-web-screenshots`,
which are not the same images. Write the mapping down in the file; do not infer it
per-caller.

`eval_split/imazen26_eval_files.tsv` is keyed in the invalid layout and must be
re-derived, not translated. 54 of its 64 files exist in the valid corpus; 10 do
not. Re-derive from the split rule in §6 rather than trying to preserve the
old picks — the old "first-8-sorted + pinned" exclusion scheme is superseded by a
real origin split.

### Step 2 — rebuild the derived corpora

`imazen-26-clean` and `imazen-26-clean-xl` are empty shells; rebuild both from the
valid root. `build_xl_corpus.sh` still works for the six non-imazen legs.

**The clean-reference rule still governs (ROADMAP §0.1): PNG sources only.** A
JPEG ground truth penalises the model for removing artifacts that are present in
the reference — this was a user-caught defect that understated every gain in the
2026-07 record. The valid corpus is **20% JPEG (417 files) plus 90 HEIC**, and it
is concentrated exactly where you would not want it:

| subcorpus | images | jpg | heic | png |
|---|---|---|---|---|
| 1000-lilith-photos-general | 72 | 61 | 11 | 0 |
| 1200-lilith-interiors | 49 | 45 | 2 | 2 |
| 1400-lilith-nature | 154 | 71 | 75 | 8 |
| 1600-lilith-food | 41 | 39 | 2 | 0 |
| 2000-unsplash-people | 28 | 28 | 0 | 0 |
| 2200-unsplash-renders | 13 | 13 | 0 | 0 |
| 2400-unsplash-textures | 10 | 10 | 0 | 0 |
| 3000-art-institute-of-chicago-photos | 15 | 15 | 0 | 0 |
| 3300-met-museum-photos | 24 | 24 | 0 | 0 |
| 6000-lilith-scans-public-patents | 113 | 74 | 0 | 39 |
| 9226-lilith-ai-products | 749 | 36 | 0 | 713 |
| *(all other dirs)* | 799 | 1 | 0 | 798 |

**The entire photographic half of the corpus is JPEG-sourced.** Dropping it leaves
a corpus that is overwhelmingly screenshots, plots, AI renders and scans — which
would silently turn the "photo" leg of the content-split curves into a fiction.
So this is a real decision, not a filter:

- **HEIC (90 files, mostly `1400-lilith-nature`)** is the cheap win — decode to
  PNG and they are clean references.
- **The 417 JPEGs** need the downscale-to-pristine treatment (downscale until the
  original's artifacts are below the quantiser floor, then treat as clean) — the
  same treatment already noted as "a separate job" for sierra and art-cc0 in
  `build_xl_corpus.sh`. `imazen-26-pristine` (74 files, still on disk) is the
  existing prototype of this; look at how it was built before inventing a method.
- Whatever you choose, **record per-file whether the reference is native-PNG,
  HEIC-decoded, or downscaled-from-JPEG**, and report the ladder split by that
  provenance. The 2026-07 defect happened because that column did not exist.

### Step 3 — retrain, then re-measure

Every constant in `crates/zensr-zenjpeg/src/api.rs` is provisional: `G420`, `G444`,
`DIST420`, `DIST444`, `GRAPHIC420/444`, `PHOTO420/444`, `GRAPHIC_ZERO_AC_THRESHOLD`,
and the `estimate_gain` calibration. The 29 tests in that file assert *shape*
(monotonicity, ordering, interpolation) rather than fitted values, so they should
survive a recalibration — if one fails on new constants, that is a signal about
the curve, not a test to relax.

The identity gate (q ≥ 94.5 at 4:2:0, q ≥ 88 at 4:4:4) is the highest-stakes
number here: ungated, the model *lost* up to 2.1 ssim2 and harmed 91% of files.
Re-derive it first and independently.

### Step 4 — re-run both leakage audits

`tools/leakage_audit.py` and `tools/picker_leakage_audit.py` both define "training"
via `IMAZEN = "/mnt/v/imazen-26"` (hardcoded near the top of each). Their verdicts
— 319/81/14 on the picker corpus, and the synthetic-v2 99.6% — used the invalid
training set as their reference, so they inherit the question even though the
corpora they audited are themselves fine.

## 6. The split rule — this gets materially better after repointing

The old corpus had no ids, which is why zensr invented its own split and why the
canonical zenmetrics rule (`scripts/picker/origin_split.py`, whose header says
*"Import this everywhere — do not re-implement the rule"*) resolved only **2 of 64**
pinned files. That excuse expires. Canonical filenames lead with the id:

```
1000-lilith-photos-general/1016_general_hanging-glass-lamp_seattle-center-seattle_...jpg
5300-noaa-hurricane-documents/5326_noaa_nhc-al102023-idalia_p01_2550x3300.png
```

The rule is **trailing digit of the leading integer**: {0,2,4,6,8} train,
{1,3,5} val, {7,9} test. Two traps, both verified today:

**Trap 1 — the id is per-file, not per-origin.** noaa is 44 files with 44 distinct
ids; patents 113/113. But three of those noaa files are three pages of *one*
hurricane report, and pages of one document share a scanner, typography and paper.
Splitting on the per-file id puts near-duplicates on both sides of the split and
"held out" stops meaning anything. Group first, then split on the group's minimum id.
`CORPUS-MANIFEST.tsv` gives you the grouping directly — `(folder, descriptor)`.
Measured: 2,160 files → **1,911 origins**, 103 of them multi-file, 352 files
(16%) in a multi-file group, largest group 7.

Applying the canonical digit rule to each group's minimum id:

| bucket | files | share | canonical target |
|---|---|---|---|
| train | 1,083 | 50% | 50% |
| val | 657 | 30% | 30% |
| test | 420 | 19% | 20% |

**Corrected 2026-09-07.** This table previously read 1,884 origins / 104 multi /
380 files / 1,097-648-415. Those are the numbers the **naive** key produces —
they were measured before Trap 2 below was applied, and reproduce digit-for-digit
if you group on a bare `(folder, descriptor)`. The tell was left in the old table
itself: it said "largest group 7" while the naive key's largest group is 28,
because finding that 28-file group is *how* the trap was found. Implemented in
`tools/origin_split.py`.

That is the canonical split, on the canonical rule, with no hash fallback and no
invented mechanic. It is the single biggest methodological improvement available
from this repoint — take it.

**Trap 2 — `2000-unsplash-people` has an empty `descriptor` for all 28 rows**
(the descriptor lives in the filename: `by-alexander-aguero-...`). Grouping naively
on `(folder, descriptor)` merges 28 unrelated photographs into one origin and
dumps them all in one bucket. Fall back to the filename stem when `descriptor` is
empty. This is the same class of bug that once merged 37 distinct CID22 images by
stripping a trailing `-\d+` from `pexels-photo-1029599`.

Measured cost of getting it wrong: the merged origin's minimum id is 2000, digit
0, so **all 28 people photographs land in train and neither val nor test contains
a single one**. For a restoration model that is not a rounding error — faces and
skin are the most perceptually punishing content it handles, and the split would
have had no way to measure them. Fixed: 14 train / 9 val / 5 test.

## 7. Traps that have already cost time here

Every one of these is a mistake actually made in this repo — they are cheap to
repeat.

1. **Unpaired statistics.** Report the median of *per-file differences* and the
   win fraction. Never the difference of two per-method medians.
2. **n=64 produces false positives.** Two shipped-looking results died on
   re-testing: a `mean_abs_ac` router that scored +1.5346 on one split and *lost*
   over 20 splits (7/20), and a per-encoder crossover at q83 that the held-out
   half put at q90. Use the 20-random-origin-split protocol for anything you
   intend to ship, and treat a single split as a hypothesis.
3. **The metric floor is ~0.3 ssim2.** Differences below it are not resolvable
   and should not be argued about.
4. **`edge_slope_stdev` is a real but unconfirmed finding** — a zenanalyze source
   feature that beat the shipped binary content class 20/20 splits and captured
   71% of per-image oracle headroom, at 2.15 ms against 0.29 ms. It was held back
   pending XL confirmation and is now **measured on a dead corpus**. Re-measure
   before believing it; the hypothesis is worth keeping, the number is not.
5. **Do not compare directory names across the two roots.** `2000-unsplash-people`
   reads as "not a trained subcorpus" when it is the same content as the old flat
   `unsplash-people`, which was trained on. That mistake reported 325 origins safe
   for the wrong reason.
6. **Filename-stem and byte comparison both under-report overlap.** Between the
   old root and `imazen-26-png`, stems match 42% and bytes 59% — I once reported
   both as *zero*, the first from a `.sdr.png` double-extension bug and the second
   from never measuring it at all. Use the 16×16 luma fingerprint
   (mean |Δ| < 3/255) for any "is this the same picture" question.
7. **`pgrep -f` matches your own shell.** Use `pgrep -x`, a recorded PID, or a
   completion marker. This is now a global rule in `~/.claude/CLAUDE.md`.
8. **Check for existing data before generating any.** Three separate runs were
   launched and killed here after the user pointed out the answer already existed
   — 43,673 mozjpeg cells among them. Read `~/work/zen/DATA_PROVENANCE.md` first;
   `/mnt/v/output/canonical-picker-2026-06-27/` has 1.48M rows with encoded
   variants and precomputed ssim2.

## 8. What does NOT need redoing

- ~~**The picker-corpus leakage work.**~~ **WRONG — re-run 2026-09-08 and the
  verdict roughly halved: 171 safe origins → 1,865 renditions, against the 319 →
  3,452 claimed here.** The corpus never touched the invalid root, but the verdict
  is about its overlap with the TRAINING SET, which is precisely what the repoint
  changed. `eval_split/picker_safe_origins_2026-08-04.txt` is superseded by
  `..._2026-09-08.txt`. This is the "only its reference training set needs
  re-checking" caveat below turning out to be the whole point.
  These renditions are **size-diverse** (`scale36x64` upward), which is the XL
  corpus's largest gap — every XL image is a 512 crop, and the sweep discipline
  wants 16–20 log-spaced sizes for anything a model is fitted on.
- **`picker_gain`** (`crates/zensr-bench/src/bin/picker_gain.rs`) and its 24,137
  cells at `~/tmp/picker_gain.tsv` — **unanalysed, and `~/tmp` is not durable.**
  Move it to `/mnt/v/zensr/` with a pointer file before anything else.
- **The zenjpeg integration itself** — probe policy, deblock rule, S10 projection,
  slack calibration. Those were calibrated on coefficient statistics (1M luma
  coefficients per encoder/q cell), not on corpus content, and the DQT survey that
  motivated the widest-slack-for-unknown fix drew on 17,739 JPEGs from elsewhere.
- **The runtime and its SIMD kernels.** Nothing about the corpus touches them.

## 9. Parked, unrelated

- XL sweep outputs in `~/tmp/xl/` (turbo 158,863 rows; dist 317,725 rows) — never
  analysed, measured on the now-deleted corpus, and in non-durable scratch. Delete
  or archive; do not analyse.
- zenjpeg 0.9.0 publish unblocks dropping the git pin in
  `crates/zensr-zenjpeg/Cargo.toml`. Tracked in the `caterr-release-train` memory.
