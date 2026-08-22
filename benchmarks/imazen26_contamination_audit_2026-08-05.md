# imazen-26 contamination audit — 2026-08-05

**Rule (user directive, 2026-08-05): `~/work/codec-corpus/imazen-26` is the only
valid imazen-26. Anything based on another root is invalid.**

This audits every project for use of an invalid root.

## The roots on disk

| path | files | status |
|---|---|---|
| **`~/work/codec-corpus/imazen-26`** | **2,563** | **VALID — canonical, id-assigned** |
| `/mnt/v/output/imazen-26-png` / `-v2` / `-v3` | 2,639 ea | derived from valid — OK |
| `/mnt/v/output/imazen-26-hdr-2026-06-14` | 2,639 | derived from valid — OK |
| `/mnt/v/output/imazen-26-hdr-grid-2026-06-14` | 1,140 | derived from valid — OK |
| `/mnt/v/output/imazen-26-features` | 1,482 | derived from valid — OK |
| `~/work/codec-corpus/imazen-26-synth` | 10,734 | synthetic sibling, separate |
| `~/work/codec-corpus/imazen-26-not-pd` | 1,591 | licence-excluded sibling |
| `~/work/codec-corpus/imazen-26 - Copy (2)` | 2,563 | stray duplicate of valid |
| `~/work/codec-corpus/imazen-26 - Copy` / `(3)` | 94 / 193 | stray partial duplicates |
| **`/mnt/v/imazen-26`** | **1,069** | **INVALID — pre-curation acquisition corpus** |
| `/mnt/v/imazen-26-clean` | 974 | derived from INVALID |
| `/mnt/v/imazen-26-pristine` | 74 | derived from INVALID |
| `/mnt/v/imazen-26-clean-xl` | 913 | partly derived from INVALID (nasa + noaa) |

## Severity: how much of the invalid root is actually outside the valid one

Content-fingerprint comparison (16×16 luma, mean |Δ| < 3/255):

| | files | |
|---|---|---|
| in `/mnt/v/imazen-26` **and** present in the valid corpus | **818** | 77% |
| in `/mnt/v/imazen-26` and **absent** from it | **250** | **23%** |

The 250 outside-corpus files: `screen` 219, `lilith` 28, `unsplash` 2, `noaa` 1.
So the contamination is not uniform — it is overwhelmingly the `screen`
subcorpus, half of which does not exist in the canonical corpus at all.

**The pinned eval split fares better: 54 of 64 files are in the valid corpus,
10 are not.**

## Code contamination, by project

| project | file:line | reference | verdict |
|---|---|---|---|
| **zensr** | `tools/make_distill_data.py:26` | `ROOT = env("ZENSR_ROOT", "/mnt/v/imazen-26")` | **INVALID — this is the training set** |
| **zensr** | `tools/teacher_audition.py:27` | `SRC = "/mnt/v/imazen-26"` | **INVALID** |
| **zensr** | `tools/build_xl_corpus.sh:39-40` | `nasa`, `noaa` from `/mnt/v/imazen-26` | **INVALID — 68 of 913 XL files** |
| **zensr** | `crates/zensr-bench/src/bin/gen_detect.rs:980` | `strip_prefix("/mnt/v/imazen-26")` | cosmetic (path display only) |
| **zensr** | `eval_split/imazen26_eval_files.tsv` | dirs are the invalid flat layout | **INVALID as a key**, though 54/64 files are valid images |
| squintly | `scripts/build_demo_corpus.py:76` | `IMAZEN26 = Path("/mnt/v/imazen-26")` | **already correct** — documented as the offline fallback; the default path is `codec-corpus/imazen-26-png-v3`, named "the canonical corpus" |
| zenmetrics | `docs/CLEAN_PICKER_PROGRAM.md` | prose reference | doc only |
| zengif | `benchmarks/*.md` | prose reference | doc only |

Nothing outside zensr and squintly reads an invalid root, and squintly already
prefers the valid one.

## What this invalidates

Everything measured in zensr this cycle rests on the invalid root:

- **Every dejpeg model** — trained via `make_distill_data.py` on
  `/mnt/v/imazen-26` minus (pin ∪ first-8).
- **Every ladder and routing curve** — `imazen-26-clean` derives from it, so the
  shipped `G420`/`G444`, `DIST420`/`DIST444`, `GRAPHIC*`/`PHOTO*` constants and
  the whole 2026-08-03/04 benchmark set were fitted on it.
- **The XL corpus**, partly — 68 of its 913 files (nasa, noaa).
- **The `edge_slope_stdev` finding** — measured on `imazen-26-clean`.

What survives untouched: the picker-corpus work
(`picker_safe_origins_2026-08-04.txt`), `picker_gain` and its 24,137 cells, and
the synthetic-v2 mozjpeg audit — all keyed to corpora that never used the
invalid root, though their *leakage checks were run against a training set
defined by it*, so those verdicts inherit the same question.

## Fix order

1. **Repoint `ZENSR_ROOT`** to `~/work/codec-corpus/imazen-26` and re-derive the
   pinned eval split against its layout. This is the root fix; everything else
   follows.
2. **Rebuild the derived corpora** (`imazen-26-clean`, `-pristine`, and the
   nasa/noaa legs of `-clean-xl`) from the valid root.
3. **Retrain**, then re-measure. Every curve constant currently in
   `crates/zensr-zenjpeg/src/api.rs` is provisional until then.
4. **Re-run both leakage audits** against the corrected training set — the
   319/81/14 picker verdict and the synthetic-v2 99.6% both used the invalid
   training set as their reference.
5. Delete or clearly mark the stray `imazen-26 - Copy*` directories so no future
   session picks one.

Given step 3 invalidates every measurement in this repo, it is a deliberate
decision about scope and timing, not a mechanical fix.
