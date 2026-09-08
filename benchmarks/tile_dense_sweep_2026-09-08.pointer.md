# Dense tile sweep, 2026-09-08 — raw data pointer

The full sweep is 51 KB, over the repo's 30 KB limit for non-source files, so it
lives in block storage. The per-cell summary the analysis actually cites is
committed beside this file as `tile_dense_sweep_summary_2026-09-08.tsv`.

- **Block storage:** `/mnt/v/output/zensr/tile-sweep-2026-09-08/tile_dense_sweep_2026-09-08.tsv`
- **sha256:** `93668722f6f3899f3507f67958a16a0bc0992e8e7e8d43904736d664041e8acc`
- **Rows:** every aligned tile from 36 to 560 (tile + 2*halo ≡ 0 mod 16), plus
  the shipped default, for 2 models x 6 sizes (128-768px) x 4 thread counts
  (4/8/12/28), 2 reps each, median reported.
- **Produced by:** `zensr-bench/src/bin/tile_probe` with `ZENSR_TP_TILES`,
  on `dev` (WSL2 on a 7950X), release build, no `-C target-cpu=native`.
- **Analysis:** `tile_ladder_2026-09-08.md`, section "What is left, measured".

Not mirrored to Tower: it is cheap to regenerate (about 90 minutes) and the
summary carries every number the analysis rests on.
