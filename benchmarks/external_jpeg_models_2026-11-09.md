# Public MeanFlow JPEG repair comparison — 2026-11-09

The public one-step checkpoint is useful as a research candidate for heavily
compressed inputs; these measurements do not support a default replacement.
Its small q5–20 mean advantage after projection comes with substantially more
harm. The two-step schedule performs worse here. Higher-quality probes are
outside the checkpoint’s stated QF5–20 training range.

## Scope and method

Source/availability pins and protocol: [external-model study](../docs/research/EXTERNAL_MODELS_2026-11-09.md).
Runner revision: `788de6a83dff06297ab3a3421bfe42ff09dc7c88`.
Corrected native loader: `a01962d6a104655790ed233023ae7ccfa15f7e55`.

All 2,960 fixed JPEG inputs completed, with both author schedules and both
raw/projected outputs. Primary: 2,720 cases, 17 origins (one per category),
turbo/mozjpeg/jpegli/zenjpeg, 420/444, q5–100 every five, long side 512.
Secondary: 240 encodings of two origins at 128/512/1024. Reused encodings
are not independent origins or a fresh held-out certification. Author-training
overlap is unknown. No checkpoint, subset or noise seed was selected by score.

Input decoding uses zenjpeg with deblocking off for MeanFlow; the native
dejpeg7 control uses the existing production Auto policy. Both are pipeline
comparisons, not an isolated architecture comparison with identical preprocessing.
MeanFlow one/two steps use the same per-input noise draw. RGB8 conversion is
shared `(value*255+0.5).clamp(0,255)`; this differs from the author’s PIL truncation.
The optional adapter applies the existing guard and JPEG coefficient projection.
No new router was fitted or selected. Canon4’s existing calibrated routing is
not revalidated by this experiment.

Harm is a loss strictly greater than 0.5 SSIM2 at baseline ≤90, falling linearly
to 0.1 at baseline 100. Subtraction and boundaries use decimal arithmetic.
SSIMULACRA2 and Butteraugli are score-only; decoded/restored RGB exists only in
RAM/pipes. Retained inputs, scalar records, output hashes and identities are
outside Git. No diagnostic maps were generated.

## Quality bands

Each entry is mean SSIM2 gain / harmed cases. Baseline is the unchanged decode.

| Pipeline | q5–20 (544) | q25–60 (1,088) | q65–100 (1,088) | Worst loss |
|---|---:|---:|---:|---:|
| dejpeg7-raw | +7.142 / 51/544 (9.38%) | +2.795 / 112/1088 (10.29%) | +0.043 / 279/1088 (25.64%) | -15.846 |
| dejpeg7-projected | +6.958 / 32/544 (5.88%) | +2.849 / 68/1088 (6.25%) | +0.837 / 157/1088 (14.43%) | -6.628 |
| meanflow-1-raw | +6.371 / 139/544 (25.55%) | +0.399 / 508/1088 (46.69%) | -5.985 / 1041/1088 (95.68%) | -25.366 |
| meanflow-1-projected | +7.397 / 95/544 (17.46%) | +1.757 / 259/1088 (23.81%) | -1.202 / 767/1088 (70.50%) | -10.908 |
| meanflow-2-raw | +1.563 / 238/544 (43.75%) | -2.693 / 803/1088 (73.81%) | -8.912 / 1075/1088 (98.81%) | -33.194 |
| meanflow-2-projected | +4.131 / 173/544 (31.80%) | +0.219 / 460/1088 (42.28%) | -2.018 / 903/1088 (83.00%) | -14.293 |

## Category × quality

One-step MeanFlow with projection compared with native dejpeg7 with projection.
Each cell is **MeanFlow / dejpeg7** mean SSIM2 gain. Each category has 32 low-band
and 64 middle/high-band encodings of one source image.

| Category | q5–20 | q25–60 | q65–100 |
|---|---:|---:|---:|
| 1000-lilith-photos-general | +9.672 / +9.401 | +2.673 / +3.165 | -0.880 / +1.247 |
| 1200-lilith-interiors | -0.832 / +2.770 | -0.769 / +0.820 | -1.477 / +0.367 |
| 1400-lilith-nature | +1.150 / +4.193 | +1.116 / +2.293 | -0.420 / +1.139 |
| 1600-lilith-food | +3.081 / +5.704 | +0.439 / +1.722 | -1.201 / +0.566 |
| 2000-unsplash-people | +11.857 / +9.916 | +2.849 / +3.257 | -1.681 / +0.483 |
| 2200-unsplash-renders | +14.473 / +10.511 | +3.569 / +3.909 | -0.842 / +0.905 |
| 2400-unsplash-textures | +13.383 / +5.541 | +3.453 / +1.850 | -1.673 / +0.193 |
| 3000-art-institute-of-chicago-photos | +30.078 / +20.622 | +7.290 / +7.362 | -0.773 / +1.172 |
| 3300-met-museum-photos | +9.734 / +10.004 | +1.123 / +2.502 | -2.073 / +0.427 |
| 5000-national-park-service-brochures | -2.489 / -2.270 | -1.430 / +0.156 | -1.401 / +0.041 |
| 5300-noaa-hurricane-documents | +0.421 / +2.678 | -0.434 / +1.739 | -1.391 / +0.946 |
| 7000-lilith-plots | +2.937 / +4.261 | -0.903 / +1.222 | -2.669 / +0.158 |
| 8000-lilith-mobile-screenshots | +6.078 / +6.712 | +1.678 / +2.872 | -0.559 / +1.180 |
| 8100-lilith-web-screenshots | +7.023 / +5.017 | +2.068 / +4.395 | -0.463 / +2.553 |
| 9000-lilith-ai-clipart | +8.181 / +9.380 | +3.573 / +5.843 | -0.778 / +1.659 |
| 9094-lilith-ai-illustrations | +2.825 / +4.977 | +0.160 / +1.408 | -1.486 / +0.183 |
| 9226-lilith-ai-products | +8.177 / +8.866 | +3.415 / +3.925 | -0.670 / +1.010 |

The one-step low-band advantage is concentrated in particular sources: museum
art +30.078 versus +20.622, textures +13.383 versus +5.541, renders +14.473
versus +10.511. Interiors instead lose 0.832 versus a native gain of 2.770;
brochures lose 2.489 versus 2.270. These are source-specific results, not
category-wide guarantees. The full scalar report also contains category × exact
quality, encoder × sampling × band, harm frequency and worst-case IDs.

## Size evidence

One-step projected MeanFlow, two-origin secondary cohort; mean SSIM2 gain.

| Long side | q5–20 (16) | q25–60 (32) | q65–100 (32) |
|---|---:|---:|---:|
| 128 | +3.010 | +2.060 | +1.104 |
| 512 | +2.271 | -0.248 | -1.075 |
| 1024 | +2.697 | -0.294 | -1.463 |

The author model runs at these sizes, but this small cohort cannot establish
arbitrary-resolution quality. No CPU/tract export or inference-latency advantage
has been measured. The author package has 65,323,201 parameters and a
261,352,510-byte checkpoint. It is an external PyTorch oracle, not a new native
runtime backend or a bundled default.

## Evidence and verification

All eight full-run controls reproduce the pilot cells exactly.
Eight encoder/sampling controls reproduce cached native raw/projected RGB8 hashes
and SSIM2 values after the binary16 fix; all 595,459 expanded model weights match
NumPy byte for byte. Projected native pixels also match the serving pipeline.
The exhaustive binary16 test and product suite pass (84 passed, five existing
explicit-input replays ignored). No existing expectation was relaxed.

Full evaluation on i265, six CPU scorers and one serialized RTX 4080 worker:
`run-heavy: done rc=0 411s | peak-RSS 1.77GiB | min-avail 19381MiB | peak-load 6.87`.
This is whole-study execution, including scoring, not model inference latency.

Artifact root on i265: `/mnt/v/output/zensr/external-models-20261109/`.
Full records: `jpeg-evaluation-v1/`; grouped scalars: `jpeg-summary-v1/`.
Input manifest SHA-256: `61ff80a858f910ce4eb92f2df1289ce27455c4a72e8a40ad56e65933497bb1c3`.
Grouped rows SHA-256: `6f9d373482c754243cf4692ee0298d5a95e2d67838c32680aa3a991b9848e65a`.

The product fix changes how subnormal weights are interpreted. Historical
runtime and routing measurements keep their original identities; this bounded
comparison does not retroactively certify those studies with the corrected loader.
