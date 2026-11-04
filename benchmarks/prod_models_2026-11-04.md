# Production dejpeg model decision + handoff — 2026-11-04

Everything a second agent needs to finish shipping `dejpeg_rt24i_canon4`
as the realtime tier and the zenjpeg APP10 record work. Handoff step 3 of
`docs/CORPUS-REPOINT-HANDOFF.md` (retrain → re-measure → recalibrate) is at
its final stage.

## Data on disk (raw results, not in git)

- `/home/lilith/data/zensr/benchmarks/prod_dejpeg_rt24g_nogate_2026-11-04.tsv`
  (60,480 data rows) — shipped model, gate OFF.
- `/home/lilith/data/zensr/benchmarks/prod_dejpeg_rt24i_canon4_nogate_2026-11-04.tsv`
  — candidate model, same grid.
- Grid: 8 encoder legs (`turbo,mozjpeg,jpegli,zenjpeg,zenjpeg:auto,
  zenjpeg:hybrid,zenjpeg:mozbase,zenjpeg:noaq`) × 420/444 ×
  q ∈ {15,35,55,75,85,88,90,92,94,96} × 6 arms
  (`identity_off/auto, model_off/auto, model_policy, model_proj`) × 63 files.
- Corpus: `/home/lilith/data/zensr/imazen-26-clean-2026-10-30`, canonical
  validate+test effective split (`eval_split/imazen26_effective_split.tsv`,
  `just split`). `gt_src` column = reference provenance.
- Earlier 5-q grid8 TSVs (rt24h_canon, rt24i_canon4, rt24z_zen specialist,
  knob study) in the same dir. Knob study writeup:
  `benchmarks/zenjpeg_knobs_2026-10-30.md` (+ raw TSV beside it in data dir).
- Models (gitignored except the shipped one): `models/adopted/{dejpeg_rt24g,
  dejpeg_rt24h_canon,dejpeg_rt24i_canon4,dejpeg_rt24z_zen}`. `repro.sh` +
  `meta.json` in each dir. Training datasets: `~/tmp/zensr-dejpeg-v4-{canon,
  canon4,zen}` (~4.5 GB npy each, regenerable via `make_dejpeg_data4.py`).
- Eval harness: `dejpeg_eval` parallelized (`5426731f`), `ZENSR_EVAL_JOBS`,
  per-cell inference cache, TSV order unchanged; byte-identical parity proven.
  `model_proj` = `restore_jpeg` full pipeline (guards+projection); `model_policy`
  = model + decode-source policy; `model_off` = model on plain decode.
- Analysis: `tools/model_ab.py` (per-file paired delta), `tools/gate_recalibrate.py`
  (needs a ladder.tsv; shipped curves duplicated inside it — update or generalize).

## Decisions reached

- **Ship `dejpeg_rt24i_canon4`** (canonical 4-encoder retrain: turbo+mozjpeg+
  jpegli+zenjpeg). Beats shipped `rt24g` in all 160 encoder×ss×q cells,
  per-file median +0.323 ssim2 overall (win 81%); turbo 420: +1.43/+0.76/
  +0.46/+0.23/+0.16..+0.21 at q15..q96, win 70–97%. `rt24h_canon`
  (turbo+zenjpeg-only retrain) is superseded — canon4 ≈ h on turbo and
  decisively better on mozjpeg/jpegli/mozbase (med +0.1..+0.3, win 73–94%).
- **No zenjpeg specialist model.** `rt24z_zen` is not worth it: vs canon4 only
  small median edges (≤+0.15) with negative means/tails, worse on turbo
  (−0.30 med q15), same speed (not a fast mode). One generalist + record
  routing is the policy.
- **AQ-off record delta already integrated** (`record.rs`, `5cb0d558`).

## Gate re-derivation (unfinished — the work to do)

`model_proj` − `identity_off` medians (canon4), first harmful cell:

| encoder | 420 | 444 |
|---|---|---|
| turbo | + at q94 (+0.39); q96 −0.02 | q94 +0.02; q96 −0.36 (84% harm) |
| mozjpeg | + at q96 (+0.43) | q94 +0.10; q96 −0.20 (65%) |
| jpegli | + at q96 (+0.15) | + at q96 (+0.12) |
| zenjpeg(+variants) | ~0 at q94–96 | ~0 at q92, − at q96 (−0.10..−0.24) |

Current shipped gate: q≥94.5 @420, q≥88 @444 (+ Butteraugli-distance
equivalents), `policy_high_q_identity*` in `crates/zensr-zenjpeg/src/lib.rs`,
fitted on rt24g turbo/mozjpeg. For canon4 it now LEAVES GAIN on the table
(turbo444 q88: +0.48 med/86% win) while q96 turbo444 is clearly harmful.
The gate may become family-aware (record/probe family is available after
`record::apply_record`). **Method required** (handoff §step 3): calibrate/
validate split BY IMAGE; harm-fraction criterion like the original fit
(original gated where ungated harm was severe, e.g. 84–91% at q96 444);
update shape-tests constants in api.rs/lib.rs only if shape assertions fail —
a failure is a signal about the curve, not a test to relax.
Also recalibrate G420/G444/DIST420/DIST444/GRAPHIC/PHOTO curves +
`GRAPHIC_ZERO_AC_THRESHOLD` + slack in api.rs from `model_off`/`model_proj`
gains at every q (they were fit pre-canonical-corpus).

## Commit checklist for the model swap

- Weights ship like rt24g did: add `!models/adopted/dejpeg_rt24i_canon4/…`
  exception lines in `.gitignore` (weights_f16.raw + gold_in/out *.raw +
  meta.json + repro.sh — NOT weights.raw; goldens generated THROUGH the f16
  roundtrip). Then point shipped references (`dejpeg_rt24g` default args in
  zensr-bench bins, adopted.rs tests, docs README/SYSTEMS) at the new dir or
  rename dir→the shipped name; keep rt24g files until docs decide.
- Regenerate goldens for rt24i_canon4 with the shipped f16 (however
  rt24g's goldens were produced — check its meta.json/repro).
- Update: `benchmarks/retrain_rt24h_canon_2026-10-30.md` (mark superseded),
  README realtime tier, SYSTEMS.md, api.rs/lib.rs constants + comments,
  CHANGELOG.
- `cargo test -p zensr-zenjpeg -p zensr-micro` must pass; shape tests in
  api.rs are contracts — recalibration failures mean the curve is wrong.

## zenjpeg APP10 record (PR #253, draft; waits a day then fix CI)

- Branch `encode-record-189` @ `24b6d3fb`; PR imazen/zenjpeg#253 draft,
  assigned lilith. zensr main deps on that rev. Do not merge: public API.
- Known CI fixes (diagnosed): (a) wasm/no-`metadata` build — `detect` mod is
  cfg'd on `metadata` but encoder emits record unconditionally; record encode
  path must compile without `metadata` (crate-internal module, public path
  stays under `metadata`), emission stays on. (b) clippy job: `insert_record`
  dead without `recompress` feature → `#[cfg(feature="recompress")]`.
  (c) ubuntu test: `ld signal 7 bus error` = runner infra, retry.
  Test-job logs: `~/tmp/pr253_{win-arm,ubuntu-arm,ubuntu,windows,macos,clippy}.log`.
- After merge: repoint zensr `Cargo.toml` zenjpeg rev off the branch tip,
  drop the zencodec `[patch]` when a zencodec release with the needed API
  exists (comment at Cargo.toml:35 explains).

## Open follow-ups (not blocking the swap)

- Durable codec-history carrier in `zencodec::Metadata` → EXIF/APPn; stripper
  survival matrix (jpegtran, exiftool -all=, ImageMagick -strip, libvips,
  canvas, zenpipe decode→encode). zenpipe recompress history-append plumbing.
- Optional turbo-weighted retrain (40/20/20/20 mix) to close the tiny residual
  turbo gap vs rt24h; ~25 min CPU data + 65 min GPU.
- 47 R2 files 404 (24 train) — imazen/imazen-26 issue; corrupted train PNG
  already filed as imazen/imazen-26#2.
- IJG libjpeg-9 leg never measured (not packaged on Ubuntu; same tables as
  turbo, different DCT/downsampler).
- Generation-loss chains with record (q60→q90, strip/tamper cases).
- Host facts: i265 RTX 4080, torch venv `~/.venvs/zensr`; binaries at
  `~/.local/bin` (`cjpegli` wrapper→r7900x… verify), mozjpeg/cjpeg present;
  disk / ~6.6 GB free — keep artifacts small, datasets are already written.
