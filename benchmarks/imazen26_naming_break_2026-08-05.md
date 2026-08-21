# imazen-26 ↔ imazen-26-png: stem preservation is 42%, and two of my claims were wrong

`imazen-26-png` is supposed to preserve the source stems and root image ids.
Measured, it preserves them for **42% of files**, and the failure is not spread
evenly — six subcorpora preserve **none**.

This matters because the dejpeg training set and pinned eval split are defined
over `/mnt/v/imazen-26` (raw names, no ids), while every id-bearing dataset in
the workspace — picker, canonical, ext720 — is defined over `imazen-26-png`.
If names do not join, nothing downstream can cross-reference the two by name.

## First: two claims I published that are false

Both are in `benchmarks/zenanalyze_routing_2026-08-04.md`, the
`picker_leakage_audit.py` / `leakage_audit.py` docstrings, and the commit
messages for `f9e53fb` and `96b8be5`.

| claim | measured |
|---|---|
| "comparing filename **stems** finds **zero** matches" | **449 exact stem matches** |
| "no imazen-26-png file is **byte-identical** to any imazen-26 file" | **625 are byte-identical** |

The stem claim was a bug in my comparison, not a property of the data. These
files carry a double extension — `..._4032x3024.sdr.png` — and I split on the
last dot only, leaving `.sdr` glued to every imazen-26-png stem so none could
ever match. The byte claim was worse: I never measured it at all, I inferred it
from "imazen-26-png is re-encoded" and wrote it down as fact.

**The leakage verdicts are unaffected** — `picker_leakage_audit.py` maps origins
by `source_sha256` and classifies by content fingerprint, never by stem, and
re-running it reproduces 319 safe / 81 exact training / 14 near-duplicate
exactly. What was wrong was my stated *reason* for needing the fingerprint, not
the need itself.

## How the two corpora actually join

| test | coverage of imazen-26's 1,068 files |
|---|---|
| identical stems | 449 (42%) |
| byte-identical | 625 (59%) |
| content fingerprint | **813 (76%)** |

So the fingerprint is still the only test that finds most of the overlap — the
conclusion holds, on correct numbers now. The remaining 24% appear genuinely
absent from `imazen-26-png` rather than renamed.

## Where stem preservation breaks

| subcorpus | files | exact stem | none |
|---|---|---|---|
| office-documents | 31 | **31 (100%)** | 0 |
| nasa | 24 | **24 (100%)** | 0 |
| skitter | 21 | **21 (100%)** | 0 |
| lilith | 323 | 162 (50%) | 141 |
| screen | 438 | 211 (48%) | 227 |
| internet-archive-scans | 72 | **0** | 72 |
| national-park-service | 59 | **0** | 59 |
| noaa | 44 | **0** | 44 |
| unsplash-people | 28 | **0** | 28 |
| unsplash-renders | 13 | **0** | 13 |
| unsplash-textures | 10 | **0** | 10 |

Three subcorpora are perfect; six are total losses. The lost ones are not
renamed arbitrarily — they are **decomposed and reordered**:

```
imazen-26      haeckel_0007_cephalopods.png
imazen-26-png  6600_scans-illustrations_haeckel-cephalopods_plate0007_4988x7151.sdr.png
```

The id `0007` survives but moves to a `plate0007` suffix, `_` becomes `-`, the
descriptive term is re-ordered ahead of it, and a new origin id is prepended.
Every component is present; the *string* is not, so no substring or stem test
finds it. The 50%-preserving subcorpora (`lilith`, `screen`) are the mixed case:
some files renamed this way, some copied verbatim.

## The concrete cost

Of the **64 pinned dejpeg eval files**, only **15 are traceable by name** into
`imazen-26-png`:

| subcorpus | traceable |
|---|---|
| office-documents | 8/8 |
| screen | 5/8 |
| lilith | 2/8 |
| internet-archive-scans | 0/8 |
| national-park-service | 0/8 |
| unsplash-people / renders / textures | 0/24 |

**49 of 64 have no name trace.** Any tool that joins the dejpeg eval split to an
id-bearing dataset by name silently drops 77% of it — and silently, because a
missing join looks identical to "not present".

That is also why `origin_split.py` returns an id for only 2 of these 64 files:
the canonical origin ids live in `imazen-26-png`, and the eval split is defined
over a snapshot that does not carry them.

## What to do

1. **Do not join these corpora by name.** Use `source_sha256` where a manifest
   provides it (`imazen-26/*/MANIFEST.tsv` carries `sha256` per file), then a
   content fingerprint for the re-encoded remainder.
2. **If stem preservation is the intended contract, six subcorpora violate it**
   and the renaming is mechanical enough to be reversible — the components are
   all still there. A `plate0007`/`_0007_` normaliser would recover
   internet-archive-scans; the others look similar.
3. Better: publish an explicit `origin_id → source_path` table alongside
   `imazen-26-png`. Every consumer currently re-derives this by hashing, and I
   got it wrong twice doing so.

---

# Root cause: there are two corpora named "imazen-26", and dejpeg is on the wrong side of a curation

`imazen-26-png` **does** preserve stems. It preserves them from *its* source,
which is not the corpus dejpeg trains on.

| | path | dirs | files | naming |
|---|---|---|---|---|
| **acquisition** | `/mnt/v/imazen-26` | 17 flat (`lilith`, `screen`, …) | 1,068 | as downloaded — `20210605_072339.jpg` |
| **curated** | `~/work/codec-corpus/imazen-26` | 28 numbered (`1400-lilith-nature`, …) | 2,563 | id-assigned — `1401_nature_dolphin-in-ocean_…_20210605-072339_4000x3000.jpg` |

## The sequence

1. **`/mnt/v/imazen-26` is the acquisition corpus** (mtime 2026-06-03). Organised
   by source, files named as downloaded, with per-folder `PROVENANCE.md` mapping
   every file to its source URL and a 2026-05-28 URL re-verification campaign
   recorded at the top level. It is a provenance artifact, not a curated corpus.
2. **It was curated into `~/work/codec-corpus/imazen-26`** — reorganised into 28
   numbered categories, origin ids 1000–9999 assigned, files renamed
   descriptively, and `CORPUS-MANIFEST.tsv` written with `number` (the origin id)
   and `original_filename` (the pre-curation name).
3. **`imazen-26-png` was converted from the curated corpus** (2026-06-10). Its
   own reconvert report states v1 "inherited the source stem verbatim" — and it
   did. Stem preservation against *its* source is not the problem.
4. **Everything downstream rides the curated lineage**: the picker corpus, the
   canonical picker datasets, ext720, and `origin_split.py`'s id scheme.
5. **dejpeg's `ZENSR_ROOT` was never repointed.** `make_distill_data.py:26`
   still defaults to `/mnt/v/imazen-26` — the pre-curation acquisition corpus.

So the corpus did not lose its names. **dejpeg is on the far side of a curation
that every other consumer migrated across**, and the "naming break" I measured is
the seam between the two lineages.

## Why the earlier per-subcorpus table looked the way it did

The three subcorpora with 100% stem preservation (`office-documents`, `nasa`,
`skitter`) are ones whose acquisition filenames were already descriptive and
survived curation unchanged. The six with 0% are ones curation renamed —
`internet-archive-scans` → `6600-ia-scans-manuscript-illustrations` with
`haeckel_0007_cephalopods` → `6600_scans-illustrations_haeckel-cephalopods_plate0007`.
`lilith` and `screen` are split across several numbered categories, so only the
files whose names happened to carry through show up as matches.

## The bridge table exists, and is partial

`~/work/codec-corpus/imazen-26/CORPUS-MANIFEST.tsv` — 2,160 rows, columns
`path, folder, category, number, descriptor, location, width, height, format,
bytes, original_filename, source, license`. `number` is the origin id and
`original_filename` is the acquisition name, which is exactly the
`origin_id → source_path` table I recommended building. It already existed.

It joins **489 of 1,068** acquisition files (46%). The remainder are files
present in the acquisition corpus but not carried into curation — the membership
genuinely differs (1,068 vs 2,563), so no table can join what was never mapped.

## What this changes

- **Not a converter bug.** Nothing to fix in `hdr-corpus-convert`.
- **`ZENSR_ROOT` is the live question.** dejpeg trains and evaluates on a corpus
  that is outside the id scheme everything else uses. Repointing it to the
  curated corpus would make `origin_split.py` apply directly, make every dataset
  joinable by name, and remove the need for fingerprint-based leakage tracing —
  but it also **changes the training set** (2,563 files vs 1,068, different
  membership), so it invalidates every model and every ladder measured to date.
  That is a deliberate decision, not a config tidy-up.
- **Until then, content fingerprinting remains correct** for cross-lineage
  leakage checks, and `CORPUS-MANIFEST.tsv` should be the first join attempted
  (46% for free) before falling back to it.
