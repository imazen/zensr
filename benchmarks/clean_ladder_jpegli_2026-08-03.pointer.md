# clean_ladder_jpegli_2026-08-03 — raw rows

Analysis: `clean_ladder_jpegli_2026-08-03.md` (same directory).

| | |
|---|---|
| path | `/mnt/v/zensr/ladders/2026-08-03/clean_jpegli.tsv` |
| sha256 | `fe3f10495e38d8d4f3b5c864c1f22c27d8400f99f534f24f8c82bc84cc25be87` |
| bytes | 939111 (7,681 rows incl. header) |
| R2 mirror | not yet synced |
| Tower mirror | not yet synced |

Produced by `dejpeg_eval` at commit `8697384` on host `lilith`:

```
ZENSR_EVAL_QS="15,35,55,75,85,90,94,96,98,100" ZENSR_EVAL_ENCODERS="jpegli,zenjpeg" \
ZENSR_EVAL_SS="420,444" ZENSR_EVAL_ARMS=policy ZENSR_EVAL_NOGATE=1 \
  run-heavy --mem 16G --jobs 6 -- \
  dejpeg_eval /mnt/v/imazen-26-clean <out.tsv> 8 6 \
    dejpeg_rt24g dejpeg_rt24g dejpeg_rt24g
```

Columns: `sub file encoder ss q arm psnr ssim2 butter_n3 probe_family probe_q gt_src`.
Note `probe_q` here is a **butteraugli distance**, not a quality — the probe
reports these files on the `ButteraugliDistance` scale.
