# Quantization-table survey — 17,739 JPEGs (2026-08-02)

Built to answer one question: **why does the probe report `Unknown` on real
traffic, and what is the cheapest fix?** `Unknown` silently falls back to the
round-to-nearest projection slack (0.15), so on any encoder that trellises or
uses adaptive quantization the consistency box is too tight.

Tool: `crates/zensr-bench/src/bin/dqt_survey.rs`. Raw data (13 MB, not in git):
`/mnt/v/output/zensr/dqt-survey-2026-08-02/`
— `corpus_dqt_files.tsv` sha256 `d079edc46a5853b4…`, one row per file
— `corpus_dqt_tables.tsv` sha256 `d1b39a9574f11741…`, one row per (file, table),
  all 64 coefficients de-zigzagged into natural order.

Corpus: `/mnt/v/output/corpus-builder` recursive, excluding `_repo_clones` and
`cc-index`. 18,240 JPEGs found, 17,739 parsed, 33,204 tables.

## Recognition today

| probe family | files | share |
|---|---|---|
| ImageMagick | 9,626 | 54.3% |
| LibjpegTurbo | 2,870 | 16.2% |
| **Unknown** | **2,649** | **14.9%** |
| Photoshop | 901 | 5.1% |
| WindowsImaging | 774 | 4.4% |
| **ProbeErr** | **579** | **3.3%** |
| Mozjpeg | 291 | 1.6% |
| Cjpegli (Ycbcr+Xyb) | 32 | 0.2% |

**18.2% unrecognized.** The ImageMagick majority is an artefact of how this
corpus was built, not of the web — the interesting population is the 3,228
unrecognized files.

## The corpus is far less diverse than it looks

539 distinct luma tables over 17,739 files, and they concentrate hard:

| top N luma tables | file coverage |
|---|---|
| 1 | 17.5% |
| 5 | 54.3% |
| 10 | 69.8% |
| 25 | 85.3% |
| 50 | 92.3% |
| 100 | 96.1% |

A ~50-entry lookup would cover 92% of everything. Table recognition is a small
problem, not an open-ended one.

## The key result: most unrecognized tables are STANDARD

Classifying every distinct table by whether its implied IJG quality is
consistent across frequency bands (Annex-K-shaped) or varies (custom base):

| shape | tables | files | share |
|---|---|---|---|
| Annex-K-shaped (implied-q spread ≤ 8) | 325 | 15,799 | **92.1%** |
| custom base table (spread > 8) | 214 | 1,357 | 7.9% |

And among the tables the probe calls `Unknown`/`ProbeErr`, the large majority
are Annex-K-shaped at **q84–98**:

| hash | files | DC/low | lo-mid | hi-mid | high | spread | diagnosis |
|---|---|---|---|---|---|---|---|
| `7901cdc07c3109d0` | 385 | 12.1 | 11.9 | 29.7 | 34.4 | 22.5 | custom |
| `f015efdfb51e88be` | 163 | 93.8 | 90.7 | 92.4 | 94.4 | 3.7 | **Annex K @ q94** |
| `cb348cef1f95fd85` | 156 | 97.9 | 95.8 | 95.8 | 96.2 | 2.1 | **Annex K @ q98** |
| `ac6cd4e3d5cd8adc` | 154 | 93.8 | 92.6 | 93.0 | 93.1 | 1.2 | **Annex K @ q94** |
| `ba2c7eeac39acc95` | 131 | 97.7 | 97.2 | 95.3 | 95.2 | 2.5 | **Annex K @ q98** |
| `29e79833997abc97` | 99 | 97.9 | 98.3 | 98.7 | 98.8 | 0.9 | **Annex K @ q98** |
| `de137a2edda3758b` | 75 | 84.1 | 82.8 | 88.7 | 86.0 | 5.9 | **Annex K @ q84** |
| `2cb06a4c3a5d996f` | 63 | 65.9 | 66.5 | 89.5 | 94.4 | 28.5 | custom |
| `d0f376970b4472a8` | 57 | 93.8 | 92.7 | 94.1 | 94.4 | 1.7 | **Annex K @ q94** |

**So the gap is not exotic encoders.** The probe is failing to map ordinary
Annex-K tables at high quality — spreads of 0.9–3.7 are unambiguous. That is
one generalisation (invert the IJG scale rather than exact-match a known table
set), not hundreds of lookup entries.

## The two custom tables worth adding by hand

`7901cdc07c3109d0` — 385 files, all from `SDWebImage` (iOS). DC is 8 while the
first AC jumps to 50, rising smoothly to 255. Not an Annex K shape at any
scale; a distinct base table with an anomalously fine DC.

```
   8   50   59   68   81   84   90  106
  50   50   68   75   84   90  106  115
  59   68   81   84   90  106  106  118
  68   68   81   84   90  106  115  125
```

`2cb06a4c3a5d996f` — 63 files across Pillow, nextcloud, openexr and
firefox-imagelib. Annex-K-like in the top-left corner, then every coefficient
past a diagonal is exactly 12 — a high-frequency clamp.

```
  12    8    8   12   17   21   24   17
   8    9    9   11   15   19   12   12
   8    9   10   12   19   12   12   12
  12   11   12   21   12   12   12   12
```

## Other findings

- **34.0% of the corpus is 4:4:4** (54.0% 4:2:0, 3.3% 4:2:2, 3.9% grayscale).
  Given the 4:4:4 gate defect fixed today, that is a large exposed population.
- **15.4% progressive.**
- 97 files carry **four components** (CMYK) and 19 use `4x1,1x1,1x1` sampling —
  both outside what the restoration path handles.
- Unrecognized files come overwhelmingly from real library outputs, not
  malformed inputs: nextcloud_server (645), SDWebImage (443), darktable (314),
  Pillow (245), sharp (142), libvips (93), thumbor. The `repro-images` tree is
  more useful than "GitHub issue repros" suggests — it is a broad sample of
  encoder *configurations*.

## What this corpus still lacks

Almost no real-world CDN traffic, which is exactly where `Unknown` matters for
zensr: the Amazon sample measured separately probes `Unknown` with a single
fixed table (`3a08a900b80dcaca`) reused across every rendition size. Common
Crawl's columnar index is already on disk (`cc-index/`, 10 parquets, ~74M URLs
with `content_mime_detected` + WARC offsets), which allows sampling archived
bytes by HTTP Range — header-only, stratified by registered domain — without
touching live sites.
