# benchmarks/ — what each measurement is, and what it is worth

Summaries live here; raw per-cell TSVs above ~30 KB live in
`/mnt/v/output/zensr/<run>/` with sha256s recorded in the summary's provenance
block, so this directory stays small enough to clone.

## Reading these honestly

- **Per file, not per cell.** Cell means hid a real defect for weeks: a model
  can win on the mean while losing on most individual files. `median` and
  `harm_frac` are the columns that decide things; `mean` is shown only so the
  two can be compared.
- **Check the reference.** Anything measured before 2026-08-01 used an eval
  split that was 39% JPEG-sourced, which understates every restoration gain.
  Anything before 2026-08-02 selected files by sorted order, which admitted
  training images. Both are fixed; older absolutes are not comparable to newer
  ones.
- **n bounds the claim.** At n=64 differences below ~0.1 ssim2 are not
  resolvable. Non-monotonic wiggles at that scale are noise.

## 2026-08-02 — identity gate re-derived on clean references

Corpus `/mnt/v/imazen-26-clean` (974 refs, 0 JPEG), pinned eval split, 8 files
per subcorpus, n=64/cell, gate disabled unless stated. Raw:
`/mnt/v/output/zensr/pinned-gate-2026-08-02/`.

| file | what it answers |
|---|---|
| `pinned_gate_main_2026-08-02.tsv` | 4:2:0 turbo+mozjpeg: where restoration stops helping, and what the projection alone contributes |
| `pinned_gate_s444_2026-08-02.tsv` | the same at 4:4:4 — the run that found the gate ignores subsampling |
| `pinned_gate_jpegli_2026-08-02.tsv` | 4:2:0 jpegli+zenjpeg (both probe as ButteraugliDistance) |
| `pinned_gate_jpegli444_2026-08-02.tsv` | 4:4:4 jpegli+zenjpeg — confirms the harm is not family-specific |
| `gated_444_after_2026-08-02.tsv` | the fix, verified: every 4:4:4 cell reads +0.000, harm_frac 0.00 |

Conclusion and full context: `SYSTEMS.md` "CHROMA SUBSAMPLING BREAKS THE
HIGH-Q GATE", `ROADMAP.md` 1.1.

## Tools

- `tools/gate_crossover.py` — per-file deltas, crossover, and `--absolute` for
  how damaged the input itself is.
- `tools/model_ab.py` — per-file comparison of two models measured in separate
  eval runs.
