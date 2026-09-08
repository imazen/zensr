# Raw TSVs for the 2026-09-08 canonical re-scoring

Three dejpeg_eval runs, 1,890 rows each (63 held-out files × turbo × 4:2:0 ×
q{15,35,55,75,90} × 6 arms). 270 KB each — block storage, not git.

| | |
|---|---|
| path | `/mnt/v/zensr/rescore/2026-09-08/` |
| `dejpeg7_graphics.tsv` | `0594d77ec0ed9aa5049946fa67680ac98ebff1e32f84dd89103b67100609673f` |
| `dejpeg9_gfxycc.tsv` | `16c750a647a902528d700fec7b383f645d767c23ba247bcc8ea283ebc9328c27` |
| `dejpeg_rt24g.tsv` | `7640719dc47a88bb443914c3cc024a2d36c419f9f1ffd84afd2a35413c61b117` |

Columns: `sub file encoder ss q arm psnr ssim2 butter_n3 probe_family probe_q gt_src`.
Analysis: `tools/rescore_report.py` (per-file gain, split by `gt_src`) and
`tools/model_ab.py` (paired model-vs-model). Findings:
`benchmarks/rescore_canonical_2026-09-08.md`.
