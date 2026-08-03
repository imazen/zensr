# clean_ladder_2026-08-03 — raw rows

Analysis and conclusions: `clean_ladder_2026-08-03.md` (same directory).

The row-level TSV is 627 KB, past the 30 KB in-repo threshold, so it lives in
block storage. Several sibling ladders of this size are committed directly
under `benchmarks/`; if that is the convention you want here too, the file can
be moved in — this pointer is the conservative default, not a judgement that
the others are wrong.

| | |
|---|---|
| path | `/mnt/v/zensr/ladders/2026-08-03/clean_ladder_full.tsv` |
| sha256 | `1b891818c12b825e32cae85b6a71dbc27345d567d37a4fcb6c44b7857840c945` |
| bytes | 641,321 (5,376 data rows + header) |
| R2 mirror | not yet synced |
| Tower mirror | not yet synced |

## Provenance

- Produced by `crates/zensr-bench/src/bin/dejpeg_eval.rs` at commit `0a43a70`
  on host `lilith`, 2026-08-03.
- Command:
  ```
  ZENSR_EVAL_QS="15,35,55,75,85,90,94" ZENSR_EVAL_ENCODERS="turbo,mozjpeg" \
  ZENSR_EVAL_SS="420,444" ZENSR_EVAL_ARMS=policy \
    run-heavy --mem 16G --jobs 6 -- \
    dejpeg_eval /mnt/v/imazen-26-clean <out.tsv> 8 6 \
      dejpeg_rt24g dejpeg_rt24g dejpeg_rt24g
  ```
- Corpus `/mnt/v/imazen-26-clean`: 974 PNG references, **0 JPEG-sourced**.
  Every row carries `gt_src`, verified `png` for all 5,376.
- Scored files are exactly the 64 in `eval_split/imazen26_eval_files.tsv`;
  training excludes that list ∪ first-8-sorted, so there is no leakage.

## Columns

`sub  file  encoder  ss  q  arm  psnr  ssim2  butter_n3  probe_family  probe_q  gt_src`

Reproduce the analysis with:

```
python3 tools/gate_recalibrate.py /mnt/v/zensr/ladders/2026-08-03/clean_ladder_full.tsv
```
