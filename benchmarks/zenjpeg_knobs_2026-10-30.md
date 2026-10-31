# zenjpeg encode-knob sensitivity + rt24h_canon vs rt24g — 2026-10-30

Question (zenjpeg#189): which encoder-side parameters change what the dejpeg
model sees at an equal quality dial, i.e. what an APPn parameter record should
carry for restoration. Second question, same run: does the canonical-corpus
retrain (`dejpeg_rt24h_canon`) beat the shipped `dejpeg_rt24g`?

Host i265. Corpus: clean references built here with `make_pristine` from the
canonical imazen-26 (`~/data/zensr/imazen-26-clean-2026-10-30`, 2,105 refs — 47
fewer than dev's 2026-09-08 build because 47 originals 404 on R2). Selection:
effective validate+test split (`just split`), 3 files per class, 63 files, 512
crop. Grid: encoders below x {420,444} x q{15,35,55,75,90}; 30,240 rows per TSV.
All references are PNG here; `ref_kind` is in the corpus `PRISTINE_MANIFEST.tsv`
and was not joined into these rows.

Raw: `zenjpeg_knobs_rt24g_2026-10-30.pointer.md`, `zenjpeg_knobs_rt24h_canon_2026-10-30.pointer.md` (4.6 MB each, kept out of git).
Report: `python3 tools/knob_report.py <A.tsv> [B.tsv]`. Variants: `zjtool.rs`
(`zenjpeg:<variant>` in `ZENSR_EVAL_ENCODERS`). Model gain = ssim2(model_off) −
ssim2(identity_off), paired per file; deltas are medians of per-file differences
against `zenjpeg` default (= `EncoderConfig::ycbcr(q, ss)`: jpegli tables, AQ on,
deringing on, no trellis, baseline).

## Knobs, shipped model rt24g (delta of model gain vs default; win% of files)

| variant | header-visible? | q15 | q35 | q55 | q75 | q90 | identity ssim2 shift @q15 |
|---|---|---|---|---|---|---|---|
| baseline / progressive | scan script | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 (pixel-identical) |
| nodering | no | 0.00 (40%) | 0.00 | 0.00 | 0.00 | 0.00 | −0.8 |
| trellis = hybrid = hybridmax | no | +0.14 (61%) | +0.15 (71%) | +0.10 (69%) | +0.02 | −0.08 | +1.1 |
| auto_optimize | no | 0 (trellis only kicks in q50+) | 0 | +0.12 (73%) | +0.03 | −0.07 | 0 |
| noaq | no | **+0.52 (87%)** | **+0.32 (80%)** | +0.20 (79%) | +0.09 | −0.06 | **+4.3** |
| noaq_nodering | no | **+0.58 (92%)** | **+0.37 (90%)** | +0.26 (88%) | +0.14 (78%) | −0.05 | +3.7 |
| blur σ0.4 | no | −0.19 (32%) | −0.24 (27%) | −0.21 (21%) | −0.16 | −0.08 | −2.9 |
| mozbase (Robidoux tables, trellis, no AQ) | DQT | **+1.52 (87%)** | **+1.25 (90%)** | +0.62 (87%) | +0.27 | +0.25 | **−25** |
| turbo (for scale) | DQT | +2.34 | +1.63 | +0.94 | +0.68 | +0.30 | −15 |

Findings:
- **Quant-table family dominates** everything else, and it is already in the
  file. But the fingerprinter fails on zenjpeg's Robidoux output: family
  `Unknown`, quality estimated **25 / 36 / 59 / 84 for dials 35 / 55 / 75 / 90**
  (IJG scale, 10-15 points low). That mis-estimate feeds `estimate_gain` directly.
  A record carrying the dial (native scale) fixes it for our files; a **32-bit
  DQT hash in the record** lets the decoder confirm the tables it sees are the
  ones the record describes (survives metadata-preserving copies, fails closed on
  transcodes), and the same hash is a cheap exact-match key for a known-table
  registry that would fix the `Unknown` classification for third-party files too.
- **AQ on/off is the one invisible knob the model can measure**: +0.3-0.6 ssim2
  more gain without AQ at q15-35 (80-92% of files), and AQ costs ~4 ssim2 of
  identity fidelity at the same dial (it spends bits perceptually, not on SSIM).
  Worth a field (on/off + strength class), and worth a q-band in training data.
- **Trellis / hybrid presets**: +0.1-0.15, 60-70% win — at the metric floor
  (ROADMAP §0.4: <0.3 is unresolved). Record it as a flag for recompression use,
  but it does not justify model routing. `auto_optimize` is trellis at q50+ only.
- **Deringing: no effect on gain** (median 0.00). Not a restoration field.
- **Pre-blur**: −0.2 gain and −3 identity. Worth a byte (σ in 1/16 units); it
  shifts what the model should do (the image is softer than the tables say).
- Baseline vs progressive: pixel-identical, nothing to record beyond the header.
- Probe on jpegli-lineage zenjpeg: `CjpegliYcbcr`, distance 13.0/6.6/4.7/2.6/1.2
  for dials 15/35/55/75/90 — stable and monotone; the dial itself is what's lost.

So for #189 the restoration-relevant payload is: encoder id + semver, quality
dial (scale enum + value), quant-table source enum, **AQ on/off (+strength)**,
pre-blur σ, trellis flag (+ dc flag), colour front-end, chroma subsampling as
set, 32-bit DQT hash. Deringing, scan script and Huffman strategy can ride along
as flags but have no measured restoration effect.

## Model A/B: rt24h_canon minus rt24g (model_off, paired per file)

| encoder | q15 | q35 | q55 | q75 | q90 |
|---|---|---|---|---|---|
| turbo | **+1.66 (95%)** | +0.82 (89%) | +0.55 (83%) | +0.26 (76%) | +0.16 (74%) |
| zenjpeg default | +0.80 (79%) | +0.45 (74%) | +0.35 (75%) | +0.25 (74%) | +0.19 (71%) |
| zenjpeg:mozbase | +1.08 (81%) | +0.29 (70%) | +0.31 (76%) | +0.30 (72%) | +0.17 (73%) |
| zenjpeg:noaq | +1.00 (83%) | +0.44 (76%) | +0.34 (78%) | +0.20 (74%) | +0.18 (76%) |

Negative-gain cells (of 126 per cell), A→B: turbo 1→0 / 2→2 / 6→2 / 16→8 / 46→30;
zenjpeg 13→7 / 23→8 / 34→16 / 35→21 / 62→44. Worst file B: turbo −2.44 at q90
(gated region), zenjpeg −2.5 at q90, −2.48 at q35. Pooled over turbo+zenjpeg:
median +0.41, 79% win, n=1260.

**rt24h_canon wins on every cell and halves the negative count**, on the same
recipe with only the data swapped (canonical corpus, 48k pairs, turbo+zenjpeg
encoders — no mozjpeg/jpegli binaries on this box; that mix is non-canonical and
recorded in the dataset meta). This is a clean-reference, per-file verdict, not
a val-curve one. Still open before it can ship: q90 gate re-derivation
(handoff step 3 says first and independently), a mozjpeg leg (not encodable
here), and the per-`ref_kind` split.
