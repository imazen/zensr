# Public ×4 SR model comparison — 2026-11-09

ResShift’s bicubic checkpoint is the strongest new high-JPEG-quality candidate
on this reused cohort. Its licensing and task limits prevent treating this as
a product-default recommendation. ResShift real v3 trades more low-quality gain
for more harm; S3Diff does not improve the fidelity frontier here.

## Fixed comparison

[Protocol, source pins, licensing and availability](../docs/research/EXTERNAL_MODELS_2026-11-09.md).
Runner revision: `6ff0d3470cf4b990c34861b0baac4e1a4445b654`.

All three arms completed all 697 fixed inputs: 17 source/category representatives,
680 zenjpeg inputs at q5–100 every five in 420/444, and 17 clean inputs. Crop
right/bottom by at most three pixels to make the 512-long-side HR reference
divisible by four; downscale with zenresize CatmullRom using its default
linear-light RGB8 configuration. The unchanged baseline upsamples the decoded
LR pixels with the same filter. The native comparison uses the retained
NomosUni SPAN ×4 model, raw inference, corrected f16 expansion. It is one
existing baseline, not an exhaustive ranking of retained SR models.

This is a web-input fidelity test, not matched authors’ RealSR or MATLAB bicubic
testing. The source references are reused, one per category; encodings are not
independent source observations. Training overlap with the public models is
unknown. No router, checkpoint or noise seed was selected by these scores.
SSIMULACRA2 and Butteraugli use full RGB output with no border shave, no metric
maps, and a fixed input-ID-derived seed. SR outputs are scored against HR
references. They are not pooled with the [×1 JPEG repair results](external_jpeg_models_2026-11-09.md).
No JPEG coefficient projection is applied on the HR lattice.

Harm uses the owner’s curve against the CatmullRom baseline: loss >0.5 SSIM2
through baseline 90, linearly decreasing to >0.1 at baseline 100. The table
reports empirical case frequencies; these are not population risk estimates.

## Quality bands

Cells are mean SSIM2 gain / harmed inputs. Counts: 17 clean, 136 q5–20,
272 q25–60 and 272 q65–100.

| Model | Clean | q5–20 | q25–60 | q65–100 |
|---|---:|---:|---:|---:|
| Native SPAN (raw) | +1.144 / 7/17 (41.18%) | +2.508 / 8/136 (5.88%) | +2.604 / 80/272 (29.41%) | +0.530 / 123/272 (45.22%) |
| S3Diff | -35.017 / 17/17 (100.00%) | +1.647 / 70/136 (51.47%) | -5.253 / 190/272 (69.85%) | -19.923 / 255/272 (93.75%) |
| ResShift real v3 | -6.473 / 15/17 (88.24%) | +4.159 / 27/136 (19.85%) | +2.288 / 100/272 (36.76%) | -1.103 / 168/272 (61.76%) |
| ResShift bicubic | -0.418 / 10/17 (58.82%) | -2.513 / 101/136 (74.26%) | -1.243 / 202/272 (74.26%) | +3.897 / 73/272 (26.84%) |

For q65–100, ResShift bicubic improves both mean gain and harm frequency over
raw native SPAN, while reducing worst loss within that band from 15.338 to
8.377 SSIM2. It still harms 73/272 inputs. At q5–20, native SPAN and ResShift
real v3 present a gain/risk tradeoff; in the middle band SPAN has both higher
mean gain and less harm. These descriptive comparisons do not select a router.

## Butteraugli and tails

Butteraugli p-norm-3 deltas: negative is improvement.

| Model | Clean | q5–20 | q25–60 | q65–100 | Worst SSIM2 gain, all cases |
|---|---:|---:|---:|---:|---:|
| Native SPAN (raw) | -0.510 | -0.024 | -0.154 | -0.307 | -15.338 |
| S3Diff | +1.916 | +2.016 | +2.316 | +2.149 | -69.123 |
| ResShift real v3 | +0.073 | +1.367 | +0.824 | +0.299 | -24.466 |
| ResShift bicubic | -0.302 | +0.618 | +0.049 | -0.849 | -14.581 |

ResShift real v3’s positive low-quality SSIM2 gain accompanies a worse mean
Butteraugli score. ResShift bicubic improves mean Butteraugli at q65–100 as
well as SSIM2. These metrics do not substitute for a human-preference study;
none was performed. The owner-cancelled pixel-artifact audit was not restarted.

## Category × quality

Each cell is mean SSIM2 gain / harm percentage. Categories have one source
each, with 8 low-band and 16 middle/high-band encodings. These tables must not
be read as category-wide generalization estimates.

### q05-20

| Category | Native SPAN | S3Diff | ResShift real v3 | ResShift bicubic |
|---|---:|---:|---:|---:|
| 1000-lilith-photos-general | +0.915 / 0.00% | -1.078 / 62.50% | +2.049 / 0.00% | -2.195 / 100.00% |
| 1200-lilith-interiors | -0.433 / 37.50% | -1.169 / 75.00% | -0.499 / 37.50% | -0.321 / 25.00% |
| 1400-lilith-nature | +0.168 / 0.00% | -1.841 / 75.00% | -0.976 / 62.50% | -0.709 / 50.00% |
| 1600-lilith-food | +0.475 / 0.00% | -2.257 / 87.50% | +0.643 / 0.00% | -1.387 / 87.50% |
| 2000-unsplash-people | +1.110 / 0.00% | -5.343 / 100.00% | +2.121 / 0.00% | -4.218 / 100.00% |
| 2200-unsplash-renders | +6.461 / 0.00% | +10.453 / 0.00% | +11.951 / 0.00% | -5.578 / 100.00% |
| 2400-unsplash-textures | -0.105 / 0.00% | -0.226 / 12.50% | +4.114 / 0.00% | -0.435 / 37.50% |
| 3000-art-institute-of-chicago-photos | +4.979 / 0.00% | +2.464 / 12.50% | +9.320 / 0.00% | -5.793 / 100.00% |
| 3300-met-museum-photos | +0.335 / 12.50% | -2.043 / 100.00% | -0.942 / 62.50% | -2.702 / 100.00% |
| 5000-national-park-service-brochures | +0.079 / 12.50% | -4.432 / 100.00% | -0.803 / 62.50% | +0.514 / 0.00% |
| 5300-noaa-hurricane-documents | +1.311 / 0.00% | -19.348 / 100.00% | -15.150 / 100.00% | -0.122 / 37.50% |
| 7000-lilith-plots | +4.610 / 0.00% | +16.533 / 0.00% | +13.077 / 0.00% | -2.820 / 100.00% |
| 8000-lilith-mobile-screenshots | +1.264 / 0.00% | -2.306 / 87.50% | +3.330 / 0.00% | -2.487 / 100.00% |
| 8100-lilith-web-screenshots | +9.616 / 0.00% | +7.470 / 0.00% | +8.220 / 0.00% | -4.692 / 75.00% |
| 9000-lilith-ai-clipart | +8.232 / 0.00% | +17.882 / 0.00% | +22.901 / 0.00% | -1.692 / 100.00% |
| 9094-lilith-ai-illustrations | -0.490 / 37.50% | -1.977 / 62.50% | -0.034 / 12.50% | -0.736 / 50.00% |
| 9226-lilith-ai-products | +4.114 / 0.00% | +15.215 / 0.00% | +11.375 / 0.00% | -7.347 / 100.00% |

### q25-60

| Category | Native SPAN | S3Diff | ResShift real v3 | ResShift bicubic |
|---|---:|---:|---:|---:|
| 1000-lilith-photos-general | +1.754 / 0.00% | -5.748 / 100.00% | +2.101 / 0.00% | -2.277 / 100.00% |
| 1200-lilith-interiors | -3.843 / 100.00% | -8.395 / 100.00% | -2.081 / 100.00% | -1.172 / 93.75% |
| 1400-lilith-nature | +0.566 / 0.00% | -12.129 / 100.00% | -6.241 / 100.00% | -2.113 / 100.00% |
| 1600-lilith-food | -0.422 / 43.75% | -7.553 / 100.00% | +0.062 / 18.75% | -1.728 / 81.25% |
| 2000-unsplash-people | +0.305 / 12.50% | -13.779 / 100.00% | +0.231 / 18.75% | -4.626 / 100.00% |
| 2200-unsplash-renders | +4.692 / 0.00% | +2.549 / 18.75% | +5.938 / 0.00% | -4.487 / 100.00% |
| 2400-unsplash-textures | +1.246 / 6.25% | -1.001 / 56.25% | +11.431 / 0.00% | -2.508 / 100.00% |
| 3000-art-institute-of-chicago-photos | +5.618 / 0.00% | +0.861 / 25.00% | +6.578 / 0.00% | -5.084 / 100.00% |
| 3300-met-museum-photos | -3.405 / 100.00% | -11.879 / 100.00% | -4.321 / 100.00% | -3.993 / 100.00% |
| 5000-national-park-service-brochures | -2.525 / 93.75% | -10.832 / 100.00% | +0.213 / 43.75% | +6.551 / 0.00% |
| 5300-noaa-hurricane-documents | +0.231 / 18.75% | -30.748 / 100.00% | -19.717 / 100.00% | +1.701 / 0.00% |
| 7000-lilith-plots | +1.590 / 25.00% | +12.126 / 0.00% | +8.043 / 0.00% | -2.837 / 100.00% |
| 8000-lilith-mobile-screenshots | +4.932 / 0.00% | -4.148 / 100.00% | +2.576 / 12.50% | -1.345 / 62.50% |
| 8100-lilith-web-screenshots | +11.033 / 0.00% | -3.174 / 81.25% | -0.339 / 50.00% | +8.226 / 0.00% |
| 9000-lilith-ai-clipart | +19.644 / 0.00% | +15.759 / 0.00% | +26.641 / 0.00% | +1.598 / 25.00% |
| 9094-lilith-ai-illustrations | -4.222 / 100.00% | -15.654 / 100.00% | -1.967 / 81.25% | -2.286 / 100.00% |
| 9226-lilith-ai-products | +7.075 / 0.00% | +4.446 / 6.25% | +9.742 / 0.00% | -4.745 / 100.00% |

### q65-100

| Category | Native SPAN | S3Diff | ResShift real v3 | ResShift bicubic |
|---|---:|---:|---:|---:|
| 1000-lilith-photos-general | +1.710 / 0.00% | -19.812 / 100.00% | -0.858 / 43.75% | +2.174 / 31.25% |
| 1200-lilith-interiors | -6.193 / 100.00% | -29.665 / 100.00% | -4.771 / 100.00% | +1.885 / 12.50% |
| 1400-lilith-nature | +0.002 / 37.50% | -29.291 / 100.00% | -9.329 / 100.00% | +0.550 / 43.75% |
| 1600-lilith-food | -3.007 / 93.75% | -20.730 / 100.00% | -2.114 / 81.25% | +3.788 / 0.00% |
| 2000-unsplash-people | -0.698 / 62.50% | -26.973 / 100.00% | -0.530 / 31.25% | -0.411 / 43.75% |
| 2200-unsplash-renders | -1.365 / 62.50% | -14.715 / 100.00% | -2.132 / 62.50% | -0.125 / 56.25% |
| 2400-unsplash-textures | +3.833 / 0.00% | -21.608 / 100.00% | +6.930 / 18.75% | +3.354 / 37.50% |
| 3000-art-institute-of-chicago-photos | +4.600 / 0.00% | -3.353 / 81.25% | +0.476 / 31.25% | +1.650 / 37.50% |
| 3300-met-museum-photos | -8.462 / 100.00% | -36.917 / 100.00% | -10.806 / 100.00% | -3.167 / 81.25% |
| 5000-national-park-service-brochures | -9.067 / 100.00% | -19.980 / 100.00% | -5.896 / 100.00% | +14.661 / 0.00% |
| 5300-noaa-hurricane-documents | -0.629 / 50.00% | -39.180 / 100.00% | -5.694 / 100.00% | +5.487 / 6.25% |
| 7000-lilith-plots | -1.141 / 62.50% | -1.830 / 50.00% | -3.544 / 50.00% | -1.317 / 62.50% |
| 8000-lilith-mobile-screenshots | +3.766 / 0.00% | -12.209 / 100.00% | -2.193 / 56.25% | +3.307 / 6.25% |
| 8100-lilith-web-screenshots | +10.216 / 0.00% | -5.315 / 100.00% | -2.351 / 75.00% | +15.980 / 0.00% |
| 9000-lilith-ai-clipart | +19.448 / 0.00% | -3.834 / 62.50% | +24.417 / 0.00% | +10.239 / 0.00% |
| 9094-lilith-ai-illustrations | -10.198 / 100.00% | -44.735 / 100.00% | -6.665 / 100.00% | +1.990 / 25.00% |
| 9226-lilith-ai-products | +6.203 / 0.00% | -8.541 / 100.00% | +6.313 / 0.00% | +6.198 / 12.50% |

The high-quality bicubic gain is particularly large on the brochure source
(+14.661 versus native −9.067) and web screenshot (+15.980 versus +10.216).
It still loses on people, renders, museum photography and plots in that band.
The real v3 checkpoint gains +24.417 on the clipart source in the high band,
but loses on most natural-photo sources. These differences rule out using a
pooled mean to justify a universal replacement.

## Runtime and integration

| Model | Instantiated parameters / stored scalars | Checkpoint bytes used | Inference configuration |
|---|---:|---:|---|
| Native SPAN | 426,288 runtime weight scalars | 852,576 | Native f16 weights expanded to f32, ×4 |
| S3Diff + degradation estimator | 1,326,767,133 | 5,307,509,362 | SD-Turbo + degradation-conditioned LoRA, f32, one diffusion step with positive/negative UNet passes, author wavelet/PIL color fix |
| Either ResShift checkpoint + shared autoencoder | 173,916,365 | 699,281,769 | Four diffusion steps, f32 weights and author fp16 autocast |

These are external PyTorch oracles, not new native/tract/wgpu backends. The
current adopted-model runtime accepts Compact and SPAN; these graphs cannot
be loaded through that weight-container path. Large-image tiling parity, CPU
export, inference-only latency and production GPU resource behavior are not
established. The tested SR sizes fit single tiles.

Measured complete-run wall time and maximum process RSS from `/usr/bin/time -v`,
including initialization, native baselines and scalar scoring on i265 / RTX 4080:

| Arm, 697 inputs | Wall time | Maximum RSS (KiB) | Exit |
|---|---:|---:|---:|
| S3Diff | 5:14.70 | 5,039,584 | 0 |
| ResShift real v3 | 3:12.35 | 2,182,424 | 0 |
| ResShift bicubic | 3:14.93 | 2,186,108 | 0 |

Supervisor memory record: `run-heavy: done rc=0 702s | peak-RSS 4.81GiB | min-avail 21858MiB | peak-load 4.68`.
This measures whole-study execution, not comparative inference latency.

## Verification and artifacts

The unchanged S3Diff author inference loop and the tensor adapter produce
bit-identical outputs at LR 128×72 and 65×71, including the recommended PIL
quantization and wavelet color fix. The check intercepts the author’s final
PIL save in RAM. Both ResShift checkpoints similarly match the unchanged
author `sample_func` at those sizes.

All 697 CatmullRom/native-SPAN baseline score records and pixel hashes match
across the three runs. Fourteen S3Diff/ResShift-v3 pilot cells reproduce the
full-run records exactly. Scalar harm-boundary regressions and all-target
Rust Clippy pass; no existing test was weakened. No restored RGB or maps were
saved. Only HR/LR reference inputs, encoded JPEGs, scalar scores and output
hashes are retained.

Artifact root on i265: `/mnt/v/output/zensr/external-models-20261109/`.
Three runs: `sr-s3diff-v1/`, `sr-resshift-v3-v1/`, `sr-resshift-bicubic-v1/`.
Grouped report: `sr-summary-v1/`; exact per-quality/category rows remain outside Git.
SR input manifest SHA-256: `840380d9155c0906bd62f51fd691ac7d834edbe5b6c3f5a3fb60bfdf49a28319`.
Grouped rows SHA-256: `06fdb4c8a840dd7bc7bfd1d4870022bd8ac06a4c409127f12398f3f84684d282`.

## Support recommendation

Keep the public checkpoints as reproducible research oracles. Prioritize the
ResShift bicubic result as an architecture/training reference for high-quality
web SR, subject to the noncommercial code license; do not bundle it in the
ordinary product without separate permission. ResShift real v3 is a risky
low-quality gain candidate. S3Diff does not earn fidelity-focused product
integration from this cohort and requires a much larger base-model chain.

The JPEG MeanFlow checkpoint has a narrow q5–20 research case but materially
higher harm than dejpeg7. FreeTransformSR and PixelIR remain blocked on a public
implementation/checkpoint. This study changes no default model or router.
