# Delivered-pixel projection diagnostic — 2026-11-06

Four deliberately selected worst cases from the already-inspected
`decoded/budget-0.005` report; this is not a representative quality estimate.
The tool runs the same canon4 model, guards and decode policy with projection
Off/Auto, disabling routing and high-quality gating in both model arms.
All four Auto RGB8 outputs match the retained study outputs byte for byte.

| Case | Identity SSIM2 | Projection off | Projection on | Luma violations off → on |
|---|---:|---:|---:|---:|
| 0 | 8.163414 | -10.370994 | -10.370994 | 0 → 0 |
| 1 | 52.254919 | 42.880397 | 42.879192 | 107 → 65 |
| 2 | 20.280769 | 14.913845 | 14.913845 | 0 → 0 |
| 3 | -7.147712 | -12.446691 | -12.446691 | 0 → 0 |

Case 0 is the known turbo 420 q6 line plot. Projection on/off pixels are
identical in cases 0, 2 and 3: these failures already satisfy the chosen bounds.
In case 1, maximum delivered luma excess falls from 14.158140 to 0.829691
coefficient units. Projection reduces violations but does not eliminate them.
The identity decode itself has 1,413 luma violations in that case.

The independent f64 DCT checker operates on final RGB8 converted to YCbCr,
using box sampling for 420, edge replication and the same family/record-corrected
slack as restoration. It counts excess above 0.001 coefficient units. It checks
visible blocks, not wholly padded MCU blocks. This is consistency under that
specified operator, not a claim about any encoder's exact sampling pipeline.
SSIM2 compares delivered RGB8 with exact retained pristine RGB8 references.

Reproduce under `run-heavy`:

```sh
just projection-check-inputs <worst-cases.json> <artifact-root> <new-input-dir> 4
just projection-check <new-input-dir>/inputs.tsv <new-output-dir>
just projection-check-tests
```

`projection_check` saves all three RGB8 outputs, input manifest and per-component
residual counts/severity. The preparer records selected rows and JPEG/reference
hashes. Missing inputs fail explicitly. Archived raw aliases and zstd artifacts
are distinguished by magic bytes, not filename. No encoder or model was trained,
no thresholds changed, and the cancelled broad artifact audit was not resumed.

Verified on i265. Supporting artifacts and logs are preserved at
`/home/lilith/data/zensr-array-archive/projection-diagnostic-20261106-v1/`.
The initial release build/test/run wrapper reported:
`run-heavy: done rc=0 55s | peak-RSS 1.41GiB | min-avail 20666MiB | peak-load 2.79`.
That includes compilation and is not a runtime benchmark.

Tool/source commit: `577c4daf728f`. Decode reuse landed as `f1de5e4aa2db`;
all four ablation tables and saved outputs remained byte-identical afterward.
Sampling-factor geometry correction: `dc12fad581a8`. The reserved vector table's
minimum short edge is 201 pixels; the reproduced tiny-padding failure is outside
that measured geometry. Follow-up logs and profiler captures are in the artifact
folder's `follow-up-verification/`, with a build-identity/hash manifest. Profiling
is diagnostic; no controlled speedup claim is made.
