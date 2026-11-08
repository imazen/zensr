# dejpeg7_graphics by JPEG quality and image category — 2026-11-08

Forced native float repair with deployed f16-expanded weights; guard/coefficient projection applied.
Routing and the near-pristine gate are off. All deltas are SSIMULACRA2 against the decoded JPEG, not against canon4.

**Coverage: one source image in each of 17 categories, long side 512.** Each image has four encoders × two sampling modes × 20 quality settings = 160 cases; each category/quality-band cell has 32 cases. Repeated encodings do not supply independent category-level evidence. Inputs are reused, and legacy training overlap is not established.

Quality bands contain four settings each: 5/10/15/20, 25/30/35/40, 45/50/55/60, 65/70/75/80 and 85/90/95/100. Encoder q values are nominal settings, not equal perceptual severity across encoders. Means and harm percentages balance origins within each cell. Harm is loss strictly above 0.5 SSIM2 through identity score 90, tapering linearly to 0.1 at 100, with decimal boundary arithmetic.

## Mean gain / harm percentage

Each cell is **mean gain / harm %**; positive gain is better. Negative means are retained.

| Image category (source ID) | 5–20 | 25–40 | 45–60 | 65–80 | 85–100 |
|---|---:|---:|---:|---:|---:|
| photos general (1017) | +9.40 / 0.0% | +3.99 / 0.0% | +2.34 / 0.0% | +1.51 / 0.0% | +0.98 / 12.5% |
| interiors (1237) | +2.77 / 0.0% | +1.02 / 0.0% | +0.62 / 0.0% | +0.43 / 0.0% | +0.30 / 12.5% |
| nature (1517) | +4.19 / 0.0% | +2.53 / 0.0% | +2.05 / 0.0% | +1.59 / 0.0% | +0.69 / 12.5% |
| food (1619) | +5.70 / 0.0% | +2.17 / 0.0% | +1.27 / 0.0% | +0.82 / 0.0% | +0.31 / 12.5% |
| people (2017) | +9.92 / 0.0% | +4.30 / 0.0% | +2.21 / 0.0% | +1.02 / 0.0% | -0.06 / 18.8% |
| renders (2209) | +10.51 / 0.0% | +4.84 / 0.0% | +2.98 / 0.0% | +1.61 / 0.0% | +0.20 / 15.6% |
| textures (2407) | +5.54 / 0.0% | +2.37 / 0.0% | +1.33 / 0.0% | +0.58 / 12.5% | -0.19 / 37.5% |
| art institute of chicago photos (3007) | +20.62 / 0.0% | +9.66 / 0.0% | +5.06 / 0.0% | +2.25 / 0.0% | +0.09 / 28.1% |
| met museum photos (3309) | +10.00 / 0.0% | +3.38 / 0.0% | +1.62 / 0.0% | +0.86 / 0.0% | -0.01 / 25.0% |
| national park service brochures (5039) | -2.27 / 71.9% | +0.06 / 50.0% | +0.25 / 50.0% | +0.50 / 43.8% | -0.42 / 46.9% |
| noaa hurricane documents (5307) | +2.68 / 15.6% | +1.89 / 18.8% | +1.59 / 12.5% | +1.47 / 6.2% | +0.42 / 18.8% |
| plots (7109) | +4.26 / 6.2% | +1.46 / 40.6% | +0.98 / 40.6% | +0.43 / 46.9% | -0.12 / 40.6% |
| mobile screenshots (8027) | +6.71 / 0.0% | +3.24 / 0.0% | +2.50 / 0.0% | +1.77 / 0.0% | +0.59 / 12.5% |
| web screenshots (8157) | +5.02 / 6.2% | +4.68 / 0.0% | +4.11 / 0.0% | +3.37 / 0.0% | +1.73 / 0.0% |
| ai clipart (9059) | +9.38 / 0.0% | +6.32 / 0.0% | +5.37 / 0.0% | +3.29 / 0.0% | +0.03 / 34.4% |
| ai illustrations (9137) | +4.98 / 0.0% | +1.74 / 0.0% | +1.07 / 0.0% | +0.59 / 0.0% | -0.22 / 31.2% |
| ai products (9779) | +8.87 / 0.0% | +4.94 / 0.0% | +2.91 / 0.0% | +1.63 / 0.0% | +0.39 / 21.9% |

## Worst per-case gain

Minimum gain in each of the same 32-case cells; negative values are losses.

| Image category | 5–20 | 25–40 | 45–60 | 65–80 | 85–100 |
|---|---:|---:|---:|---:|---:|
| photos general | +2.86 | +0.85 | +1.09 | +0.75 | -1.44 |
| interiors | -0.36 | -0.20 | -0.07 | -0.06 | -1.78 |
| nature | +1.85 | +1.62 | +1.38 | +1.00 | -1.27 |
| food | +1.57 | +0.79 | +0.54 | +0.28 | -1.44 |
| people | +3.82 | +1.85 | +0.97 | +0.32 | -1.75 |
| renders | +4.58 | +2.67 | +1.56 | +0.80 | -0.96 |
| textures | +2.69 | +0.50 | -0.41 | -1.63 | -2.09 |
| art institute of chicago photos | +9.49 | +4.36 | +1.97 | +0.92 | -2.08 |
| met museum photos | +2.31 | +1.10 | +0.59 | +0.19 | -1.36 |
| national park service brochures | -6.63 | -3.59 | -2.82 | -2.08 | -2.35 |
| noaa hurricane documents | -4.96 | -0.99 | -1.45 | -0.82 | -2.41 |
| plots | -3.60 | -4.82 | -5.46 | -2.32 | -1.79 |
| mobile screenshots | +2.58 | +1.69 | +1.46 | +0.78 | -1.16 |
| web screenshots | -3.18 | +0.21 | +1.73 | +2.00 | +0.05 |
| ai clipart | +6.08 | +4.46 | +3.55 | +1.77 | -1.95 |
| ai illustrations | +1.49 | +0.73 | +0.57 | +0.24 | -1.37 |
| ai products | +3.44 | +2.97 | +1.64 | +0.91 | -0.98 |

## Every nominal JPEG quality

Each row covers all 17 source images and eight encoder/sampling variants per image (136 cases). Median is the case median; mean balances origins.

| q | Mean gain | Median gain | Harm cases / 136 | Harm % | Worst gain |
|---:|---:|---:|---:|---:|---:|
| 5 | +7.627 | +7.282 | 12/136 | 8.82% | -6.628 |
| 10 | +8.579 | +7.947 | 6/136 | 4.41% | -6.336 |
| 15 | +6.379 | +5.369 | 8/136 | 5.88% | -6.076 |
| 20 | +5.245 | +4.634 | 6/136 | 4.41% | -4.041 |
| 25 | +4.325 | +3.569 | 7/136 | 5.15% | -3.724 |
| 30 | +3.476 | +2.881 | 9/136 | 6.62% | -3.187 |
| 35 | +3.201 | +2.729 | 8/136 | 5.88% | -2.910 |
| 40 | +2.785 | +2.613 | 11/136 | 8.09% | -4.819 |
| 45 | +2.630 | +2.422 | 6/136 | 4.41% | -2.821 |
| 50 | +2.280 | +2.110 | 9/136 | 6.62% | -3.098 |
| 55 | +2.163 | +1.946 | 9/136 | 6.62% | -3.170 |
| 60 | +1.936 | +1.763 | 9/136 | 6.62% | -5.455 |
| 65 | +1.798 | +1.656 | 9/136 | 6.62% | -2.153 |
| 70 | +1.533 | +1.468 | 10/136 | 7.35% | -2.312 |
| 75 | +1.195 | +1.230 | 7/136 | 5.15% | -2.319 |
| 80 | +1.057 | +0.959 | 9/136 | 6.62% | -1.881 |
| 85 | +0.807 | +0.742 | 7/136 | 5.15% | -1.904 |
| 90 | +0.552 | +0.551 | 8/136 | 5.88% | -1.785 |
| 95 | +0.196 | +0.117 | 15/136 | 11.03% | -1.945 |
| 100 | -0.443 | -0.916 | 92/136 | 67.65% | -2.406 |

## Encoder and chroma sampling

Each cell has 68 cases (17 images × four qualities); mean gain / harm %.

| Encoder / sampling | 5–20 | 25–40 | 45–60 | 65–80 | 85–100 |
|---|---:|---:|---:|---:|---:|
| jpegli / 420 | +5.36 / 5.9% | +2.50 / 0.0% | +2.02 / 0.0% | +1.36 / 0.0% | +0.66 / 11.8% |
| jpegli / 444 | +5.84 / 4.4% | +2.87 / 0.0% | +2.10 / 0.0% | +1.30 / 0.0% | +0.13 / 25.0% |
| mozjpeg / 420 | +7.35 / 7.4% | +3.86 / 11.8% | +2.30 / 10.3% | +1.40 / 10.3% | +0.51 / 11.8% |
| mozjpeg / 444 | +7.00 / 7.4% | +3.32 / 13.2% | +1.94 / 13.2% | +0.98 / 19.1% | -0.13 / 32.4% |
| turbo / 420 | +10.09 / 2.9% | +5.77 / 0.0% | +3.63 / 0.0% | +2.36 / 0.0% | +0.71 / 10.3% |
| turbo / 444 | +10.06 / 2.9% | +5.45 / 0.0% | +3.39 / 0.0% | +2.11 / 0.0% | +0.12 / 30.9% |
| zenjpeg / 420 | +4.85 / 8.8% | +1.74 / 13.2% | +1.24 / 14.7% | +0.81 / 11.8% | +0.36 / 22.1% |
| zenjpeg / 444 | +5.11 / 7.4% | +2.07 / 13.2% | +1.40 / 10.3% | +0.85 / 10.3% | -0.15 / 35.3% |

## Reproduction and scope

Computed from the completed precision study without rerunning inference or changing selection. The complete exact-q × category × encoder × sampling table is retained outside Git alongside the experiment; each row includes cases, origins, mean, median, harm count/rate and worst gain.

Results SHA-256: `1b5186ff30104101f3da6c6e454af44b4a33ad7794fc61945e8dc9c2a3f2270d`.
Input manifest SHA-256: `61ff80a858f910ce4eb92f2df1289ce27455c4a72e8a40ad56e65933497bb1c3`.

```sh
just quality-int8-stratify EVALUATION OUTPUT.md --inputs INPUTS.tsv --detail-output DETAILS.tsv
```

[Complete study and limitations](quality_int8_2026-11-08.md). The automatic router is calibrated for canon4, not this model. These descriptive tables do not select a router or establish quality across arbitrary resolutions.
