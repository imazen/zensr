# Retrain of the realtime dejpeg tier on the canonical corpus — IN PROGRESS 2026-10-30

Handoff step 3 (`docs/CORPUS-REPOINT-HANDOFF.md`): first retrain of the shipped
`dejpeg_rt24g` recipe against the canonical imazen-26 (github.com/imazen/imazen-26,
bytes synced from R2 with `tools/fetch_corpus_bytes.py`). Host i265 (RTX 4080 16 GB).

- Corpus sync: 2,113 of 2,160 files fetched; **47 return 404 on R2** (24 in the
  train bucket, 23 validate/test — list in `~/tmp/zensr-fetch-*.log` on i265;
  filed nowhere yet). Python urllib is refused with 403 (User-Agent); curl works.
- Data: `make_dejpeg_data4.py 48000 8`, `ZENSR_ENCODERS=turbo,zenjpeg` (no mozjpeg /
  cjpegli binary on i265 — non-canonical mix, recorded in dataset meta), policy-mix
  zenjpeg decode, 5% clean, q~U(5,96). Pool 1,094 train / 57 val files
  (provenance png 892 / jpg 211 / heic 46 / other 2). Output
  `~/tmp/zensr-dejpeg-v4-canon` (4.7 GB). The generator crashed at save time on a
  pre-existing `SUBS` NameError after writing the arrays; `pairs.tsv`/`meta.json`
  were replayed from the seeded rng and verified (sampled crops bit-equal, clean
  rows lr==hr). Bug fixed in this change.
- Train: `ZENSR_SCALE=1 ZENSR_NF=24 ZENSR_NC=8 ZENSR_QBOOST=3 ZENSR_OUT=dejpeg_rt24h_canon
  python3 tools/train_people.py 200000 48 3e-4` (exactly the rt24g repro.sh, new data).
  torch 2.15.0+cu132. Log `~/tmp/zensr-runs/dejpeg_rt24h_canon.log`, PID file beside it.
  Checkpoints every 5k steps in the data dir. ~50 steps/s.
- Verdict rule: NOT the val curve (ROADMAP §0.2). Score with `tools/model_ab.py`
  against `dejpeg_rt24g` on the clean-reference pinned split, per file, paired, by
  `ref_kind`, before anything is written into README/SYSTEMS or `api.rs` constants.

**DONE 2026-10-30 23:10**: 200k steps, final val PSNR 33.15, exported to
`models/adopted/dejpeg_rt24h_canon`. Verdict (clean refs, paired per file):
`zenjpeg_knobs_2026-10-30.md` §"Model A/B" — beats rt24g on every encoder x q
cell, median +0.41 ssim2, 79% win, negatives halved.
