# Affinity-KD two-seed pair — raw rows

Analysis and verdict: `ROADMAP.md` §1.4.

Five eval runs at 179 KB each (895 KB total), past the 30 KB in-repo threshold,
so they live in block storage.

| | |
|---|---|
| path | `/mnt/v/zensr/kd/2026-08-03/` |
| checksums | `/mnt/v/zensr/kd/2026-08-03/SHA256SUMS` |
| R2 mirror | not yet synced |
| Tower mirror | not yet synced |

| file | model | seed |
|---|---|---|
| `kd_fkd_a_outkd_2026-08-03.tsv` | output-KD only (control) | 7 |
| `kd_fkd_b_affinity_2026-08-03.tsv` | + affinity KD | 7 |
| `kd2_fkdlocal_a_outkd_2026-08-03.tsv` | output-KD only (control) | 2 |
| `kd2_fkdlocal_b_affinity_2026-08-03.tsv` | + affinity KD | 2 |
| `kd_dejpeg_rt24g_2026-08-03.tsv` | shipped realtime tier, for reference | — |

All five share a grid so any pair is directly comparable: 64 pinned files ×
{turbo, mozjpeg} × 4:2:0 × q{15,35,55,75}, clean PNG references, gate disabled.

```
ZENSR_EVAL_QS="15,35,55,75" ZENSR_EVAL_ENCODERS="turbo,mozjpeg" ZENSR_EVAL_SS="420" \
ZENSR_EVAL_ARMS=policy ZENSR_EVAL_NOGATE=1 \
  run-heavy --mem 16G --jobs 6 -- \
  dejpeg_eval /mnt/v/imazen-26-clean <out.tsv> 8 6 <model> <model> <model>
```

The model name is passed in all three slots deliberately: they are the
`m_off`/`m_auto`/`m_policy` **arms**, not competing models. Passing three
different models there evaluates none of them head-to-head — with
`ZENSR_EVAL_ARMS=policy` only the third is used.

Compare with `tools/model_ab.py <A.tsv> <B.tsv> --arm model_policy`.

Training: `tools/run_fkd_pair.sh` (seed 7) and `tools/run_fkd_local.sh` (seed 2),
both 200k steps, batch 48, lr 3e-4, `ZENSR_FKD_W` 0 vs 0.027, online fp32
teacher `dejpeg11_teacher_100000.pth`.
