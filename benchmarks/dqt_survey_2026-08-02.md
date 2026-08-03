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

## CORRECTED classification — the first pass here was wrong

**A superseded claim, kept visible.** The first version of this report said
92.1% of files used "Annex-K-shaped" tables and that most unrecognized ones
were "Annex K at q84-98 the probe should map". That was an artefact of the
metric. Classifying by whether *implied IJG quality* is consistent across
frequency bands fails at high quality: there the quantizer values are 1-10, so
the inversion compresses every table into the 90-100 range and dissimilar
tables all look "consistent". A direct check killed it — the probe *does* map
freshly-encoded Annex K at q84-98, and the corpus tables it rejects differ from
those in 51-57 of 64 positions with deltas up to 7.

The correct test is the **absolute residual against the best-fitting IJG
table**, in quantizer units:

| class | tables | files | share |
|---|---|---|---|
| IJG exact (max abs delta <= 1) | 157 | 14,222 | 82.9% |
| IJG-like (max abs delta <= 3) | 54 | 662 | 3.9% |
| **custom base table (> 3)** | **328** | **2,272** | **13.2%** |

And among the files the probe calls `Unknown`, **65% use genuinely custom
tables** — not standard ones it should have mapped.

| hash | files | best-fit q | mean abs | max abs | class |
|---|---|---|---|---|---|
| `7901cdc07c3109d0` | 385 | 32 | 31.61 | 101 | custom |
| `f015efdfb51e88be` | 163 | 93 | 1.64 | 5 | custom |
| `cb348cef1f95fd85` | 156 | 96 | 0.84 | 3 | IJG-like |
| `ac6cd4e3d5cd8adc` | 154 | 93 | 1.11 | 5 | custom |
| `ba2c7eeac39acc95` | 131 | 95 | 0.11 | 1 | **IJG exact — a real probe miss** |
| `29e79833997abc97` | 99 | 98 | 0.52 | 2 | IJG-like |
| `de137a2edda3758b` | 75 | 87 | 2.86 | 12 | custom |
| `2cb06a4c3a5d996f` | 63 | 93 | 6.38 | 18 | custom |
| `9a5e1acec5b39d4e` | 55 | 93 | 0.08 | 1 | **IJG exact — a real probe miss** |
| `b2bab93a7a9cf0ea` | 49 | 98 | 0.53 | 1 | **IJG exact — a real probe miss** |

**So there are two separate fixes, with a 35/65 split of the unrecognized
population:**

1. **~35% are IJG-exact or within 3** (`ba2c7eeac39acc95`, `9a5e1acec5b39d4e`,
   `b2bab93a7a9cf0ea`, `29e79833997abc97`, `cb348cef1f95fd85` — 490 files in
   the top ten alone). These are genuine probe misses: matching the IJG scale
   with a small tolerance would recognise them.
2. **~65% are genuinely custom** and need lookup entries. They are not a long
   tail either — the top custom table alone is 385 files.

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
