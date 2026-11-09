# Canon4 → SPANF/DSCF matched ×4 comparison — 2026-11-09

Routed canon4 repair improves both upscalers on this reused cohort. It restores
249/697 inputs; all 249 gain final-output SSIMULACRA2, while the 448 skipped
inputs retain exactly the direct-SR tensor and output pixels. Butteraugli
improves on 248/249 restored SPANF cases and 247/249 restored DSCF cases.
The exceptions and remaining losses against CatmullRom are retained below.
No model, routing threshold, coefficient or runtime default was selected here.

All 697 inputs complete all four arms: 17 source/category representatives,
zenjpeg q5–100 every five in 420/444, plus 17 clean inputs. HR long side is
512; LR long side is 128. These are reused sources with one image per category,
not an independent population-risk or generalization estimate.
[Frozen protocol](../docs/research/SR_CASCADE_2026-11-09.md).

## Overall final-image quality

All gains and harm counts in this table use the common CatmullRom HR baseline.
Negative Butteraugli p-norm-3 change is better. Harm follows the existing
quality-dependent SSIM2-loss curve. All-input means weight each origin equally
because every origin contributes 41 cells. Conditional restored means below
are cell-weighted and have different origin coverage.

| Arm | Mean SSIM2 gain | Median gain | Harm / 697 | Worst gain | Mean BA-n3 change |
|---|---:|---:|---:|---:|---:|
| SPANF direct | +2.878 | +0.504 | 219 | -8.551 | -0.406 |
| canon4 → SPANF | +4.442 | +3.230 | 91 | -8.551 | -0.571 |
| DSCF direct | +2.419 | +0.164 | 244 | -10.526 | -0.364 |
| canon4 → DSCF | +4.008 | +2.724 | 107 | -10.526 | -0.535 |

Repair-first adds **+1.563 mean SSIM2 / −0.164 mean BA-n3** for SPANF and
**+1.589 / −0.171** for DSCF relative to each model’s direct path. Overall paired
medians are zero because most inputs are unchanged. Among the 249 restored
inputs, the SSIM2 mean/median additions are **+4.376/+3.524** for SPANF and
**+4.447/+3.660** for DSCF. All 249 report coefficient projection applied at LR
(`Consistency::Certified`); that API label is not a final RGB8 re-encoding or
perceptual-improvement guarantee.

Zero incremental SSIM2 losses here does not mean the entire SR pipeline is
harmless: the repaired arms still harm 91/697 and 107/697 inputs versus
CatmullRom. The current ×1 router was used unchanged, including all skips.

## Paired repair effect by JPEG quality

Each pair is **mean ΔSSIM2 / mean ΔBA-n3**, repair→SR minus direct SR.
The all-input column includes identity skips; the restored column conditions
on the existing production router applying repair. No SSIM2 loss occurs in
either paired arm on this cohort. A dash is an empty restored subset.

| Band | Restored / inputs | SPANF all | SPANF restored | DSCF all | DSCF restored |
|---|---:|---:|---:|---:|---:|
| Clean | 0/17 | +0.000 / +0.000 | — / — | +0.000 / +0.000 | — / — |
| q5–20 | 86/136 | +2.798 / -0.458 | +4.425 / -0.724 | +2.832 / -0.482 | +4.478 / -0.762 |
| q25–60 | 106/272 | +1.962 / -0.161 | +5.035 / -0.413 | +1.997 / -0.168 | +5.123 / -0.432 |
| q65–100 | 57/272 | +0.645 / -0.031 | +3.077 / -0.148 | +0.658 / -0.030 | +3.142 / -0.142 |

## Chroma breakdown

Paired all-input means, with the same units and baselines as above.

| Sampling / band | Restored / inputs | SPANF ΔSSIM2 / ΔBA | DSCF ΔSSIM2 / ΔBA |
|---|---:|---:|---:|
| 420 / 05-20 | 50/68 | +3.273 / -0.571 | +3.298 / -0.599 |
| 420 / 25-60 | 66/136 | +2.397 / -0.211 | +2.434 / -0.221 |
| 420 / 65-100 | 37/136 | +0.824 / -0.044 | +0.837 / -0.042 |
| 444 / 05-20 | 36/68 | +2.323 / -0.344 | +2.365 / -0.365 |
| 444 / 25-60 | 40/136 | +1.528 / -0.111 | +1.560 / -0.115 |
| 444 / 65-100 | 20/136 | +0.465 / -0.018 | +0.480 / -0.018 |
| clean / clean | 0/17 | +0.000 / +0.000 | +0.000 / +0.000 |

## Category × quality

Paired all-input repair effects; each category has one origin. The final two
columns are absolute mean BA-n3 after the routed pipeline, lower is better.
Repeated encodings do not establish category-wide confidence.

### Clean

| Category | Restored / inputs | SPANF ΔSSIM2 / ΔBA | DSCF ΔSSIM2 / ΔBA | SPANF final BA | DSCF final BA |
|---|---:|---:|---:|---:|---:|
| General photos | 0/1 | +0.000 / +0.000 | +0.000 / +0.000 | 7.405 | 7.259 |
| Interiors | 0/1 | +0.000 / +0.000 | +0.000 / +0.000 | 8.318 | 8.330 |
| Nature | 0/1 | +0.000 / +0.000 | +0.000 / +0.000 | 8.262 | 8.264 |
| Food | 0/1 | +0.000 / +0.000 | +0.000 / +0.000 | 9.054 | 8.718 |
| People | 0/1 | +0.000 / +0.000 | +0.000 / +0.000 | 5.081 | 5.284 |
| Renders | 0/1 | +0.000 / +0.000 | +0.000 / +0.000 | 4.032 | 4.009 |
| Textures | 0/1 | +0.000 / +0.000 | +0.000 / +0.000 | 5.320 | 5.179 |
| Museum art (AIC) | 0/1 | +0.000 / +0.000 | +0.000 / +0.000 | 3.454 | 3.535 |
| Museum art (Met) | 0/1 | +0.000 / +0.000 | +0.000 / +0.000 | 5.044 | 4.412 |
| Brochures | 0/1 | +0.000 / +0.000 | +0.000 / +0.000 | 9.339 | 9.401 |
| Weather documents | 0/1 | +0.000 / +0.000 | +0.000 / +0.000 | 3.735 | 5.489 |
| Plots | 0/1 | +0.000 / +0.000 | +0.000 / +0.000 | 4.642 | 4.610 |
| Mobile screenshots | 0/1 | +0.000 / +0.000 | +0.000 / +0.000 | 8.274 | 8.595 |
| Web screenshots | 0/1 | +0.000 / +0.000 | +0.000 / +0.000 | 10.098 | 10.006 |
| AI clipart | 0/1 | +0.000 / +0.000 | +0.000 / +0.000 | 5.721 | 5.840 |
| AI illustrations | 0/1 | +0.000 / +0.000 | +0.000 / +0.000 | 5.602 | 5.612 |
| AI products | 0/1 | +0.000 / +0.000 | +0.000 / +0.000 | 4.183 | 4.276 |

### JPEG q5–20

| Category | Restored / inputs | SPANF ΔSSIM2 / ΔBA | DSCF ΔSSIM2 / ΔBA | SPANF final BA | DSCF final BA |
|---|---:|---:|---:|---:|---:|
| General photos | 4/8 | +1.081 / -0.351 | +1.083 / -0.365 | 12.072 | 12.271 |
| Interiors | 7/8 | +0.280 / -0.449 | +0.284 / -0.477 | 12.708 | 12.728 |
| Nature | 8/8 | +0.581 / -0.409 | +0.596 / -0.434 | 15.723 | 15.745 |
| Food | 8/8 | +1.597 / -0.776 | +1.633 / -0.828 | 13.831 | 13.822 |
| People | 8/8 | +4.657 / -0.744 | +4.668 / -0.750 | 10.272 | 10.292 |
| Renders | 8/8 | +11.751 / -0.951 | +11.745 / -0.965 | 9.837 | 9.863 |
| Textures | 6/8 | +0.317 / -0.509 | +0.315 / -0.514 | 9.314 | 9.318 |
| Museum art (AIC) | 8/8 | +10.118 / -0.612 | +10.296 / -0.635 | 9.869 | 9.881 |
| Museum art (Met) | 2/8 | +0.719 / -0.050 | +0.706 / -0.048 | 7.406 | 7.388 |
| Brochures | 0/8 | +0.000 / +0.000 | +0.000 / +0.000 | 13.137 | 13.411 |
| Weather documents | 0/8 | +0.000 / +0.000 | +0.000 / +0.000 | 8.320 | 8.588 |
| Plots | 2/8 | +2.491 / -0.438 | +2.520 / -0.435 | 11.656 | 11.648 |
| Mobile screenshots | 8/8 | +3.417 / -0.776 | +3.445 / -0.807 | 13.985 | 14.009 |
| Web screenshots | 0/8 | +0.000 / +0.000 | +0.000 / +0.000 | 13.016 | 13.324 |
| AI clipart | 4/8 | +4.074 / -0.734 | +4.153 / -0.700 | 15.791 | 16.189 |
| AI illustrations | 7/8 | +0.560 / -0.254 | +0.570 / -0.264 | 11.847 | 11.862 |
| AI products | 6/8 | +5.921 / -0.725 | +6.122 / -0.973 | 10.551 | 10.522 |

### JPEG q25–60

| Category | Restored / inputs | SPANF ΔSSIM2 / ΔBA | DSCF ΔSSIM2 / ΔBA | SPANF final BA | DSCF final BA |
|---|---:|---:|---:|---:|---:|
| General photos | 4/16 | +0.900 / -0.079 | +0.910 / -0.108 | 9.372 | 9.362 |
| Interiors | 2/16 | +0.159 / -0.038 | +0.166 / -0.026 | 10.853 | 10.870 |
| Nature | 9/16 | +1.135 / -0.184 | +1.169 / -0.189 | 11.670 | 11.752 |
| Food | 14/16 | +2.700 / -0.371 | +2.797 / -0.393 | 10.832 | 10.784 |
| People | 14/16 | +3.777 / -0.302 | +3.807 / -0.306 | 8.102 | 8.124 |
| Renders | 16/16 | +8.011 / -0.489 | +8.123 / -0.504 | 7.484 | 7.493 |
| Textures | 8/16 | +0.799 / -0.070 | +0.801 / -0.075 | 7.365 | 7.450 |
| Museum art (AIC) | 14/16 | +7.190 / -0.343 | +7.343 / -0.352 | 7.508 | 7.518 |
| Museum art (Met) | 0/16 | +0.000 / +0.000 | +0.000 / +0.000 | 6.219 | 6.150 |
| Brochures | 0/16 | +0.000 / +0.000 | +0.000 / +0.000 | 10.656 | 10.718 |
| Weather documents | 0/16 | +0.000 / +0.000 | +0.000 / +0.000 | 6.713 | 6.947 |
| Plots | 0/16 | +0.000 / +0.000 | +0.000 / +0.000 | 9.183 | 9.112 |
| Mobile screenshots | 16/16 | +4.586 / -0.373 | +4.519 / -0.390 | 10.594 | 10.668 |
| Web screenshots | 0/16 | +0.000 / +0.000 | +0.000 / +0.000 | 10.719 | 10.740 |
| AI clipart | 4/16 | +2.478 / -0.316 | +2.604 / -0.330 | 11.774 | 11.953 |
| AI illustrations | 2/16 | +0.163 / -0.032 | +0.163 / -0.029 | 8.768 | 8.798 |
| AI products | 3/16 | +1.460 / -0.140 | +1.540 / -0.156 | 7.232 | 7.331 |

### JPEG q65–100

| Category | Restored / inputs | SPANF ΔSSIM2 / ΔBA | DSCF ΔSSIM2 / ΔBA | SPANF final BA | DSCF final BA |
|---|---:|---:|---:|---:|---:|
| General photos | 4/16 | +0.785 / -0.029 | +0.805 / -0.036 | 7.491 | 7.288 |
| Interiors | 0/16 | +0.000 / +0.000 | +0.000 / +0.000 | 9.043 | 8.938 |
| Nature | 1/16 | +0.137 / -0.013 | +0.141 / -0.014 | 9.188 | 9.222 |
| Food | 5/16 | +0.847 / -0.075 | +0.859 / -0.081 | 9.268 | 9.131 |
| People | 13/16 | +2.116 / -0.059 | +2.125 / -0.068 | 6.129 | 6.160 |
| Renders | 5/16 | +1.826 / -0.086 | +1.846 / -0.089 | 5.283 | 5.251 |
| Textures | 2/16 | +0.184 / -0.009 | +0.186 / -0.006 | 5.486 | 5.633 |
| Museum art (AIC) | 14/16 | +3.551 / -0.151 | +3.672 / -0.127 | 4.835 | 4.853 |
| Museum art (Met) | 0/16 | +0.000 / +0.000 | +0.000 / +0.000 | 5.591 | 5.269 |
| Brochures | 0/16 | +0.000 / +0.000 | +0.000 / +0.000 | 9.919 | 9.914 |
| Weather documents | 0/16 | +0.000 / +0.000 | +0.000 / +0.000 | 5.230 | 5.817 |
| Plots | 0/16 | +0.000 / +0.000 | +0.000 / +0.000 | 7.246 | 7.039 |
| Mobile screenshots | 12/16 | +1.408 / -0.101 | +1.453 / -0.082 | 8.730 | 8.911 |
| Web screenshots | 0/16 | +0.000 / +0.000 | +0.000 / +0.000 | 10.177 | 10.178 |
| AI clipart | 0/16 | +0.000 / +0.000 | +0.000 / +0.000 | 8.249 | 8.174 |
| AI illustrations | 1/16 | +0.105 / -0.004 | +0.105 / -0.002 | 6.355 | 6.410 |
| AI products | 0/16 | +0.000 / +0.000 | +0.000 / +0.000 | 4.712 | 4.737 |

## SPANF versus DSCF

This closes the earlier matched-quality gap: the comparator is native SPANF,
not the distinct NomosUni SPAN model. SPANF’s direct path leads DSCF by
0.459 mean SSIM2 and 0.043 mean BA-n3; after routed repair the leads are
0.434 SSIM2 and 0.036 BA-n3. After repair, SPANF wins SSIM2 on 529/697 cells;
DSCF wins 168/697. Butteraugli prefers SPANF on 408/697 and DSCF on 289/697.
DSCF retains category strengths, including food, plots and museum-Met across
all quality bands by category-mean BA-n3. These results do not measure CPU
latency of the complete cascade or establish subjective model preference.

A per-input perfect chooser after routed repair would add 0.067472 mean SSIM2
over fixed SPANF when choosing by SSIM2. A separate perfect BA chooser would
improve BA-n3 by 0.051462. These are two different oracle ceilings with
reference access, not one practical policy or two simultaneous attainable gains.
No chooser or blend was fitted.

## Tails and visual witnesses

The sole paired SPANF BA regression is mobile screenshot 8027, q80/444:
+0.049542 BA-n3. DSCF’s two are AIC source 3007, q90/444 (+0.019085) and
q95/444 (+0.004301). Their SSIM2 still improves. Reported scores and images
are retained, rather than filtered as outliers.

Six feature-selected 420 witnesses were visually inspected against HR:
general photo and plot at q10/q90, weather document and texture at q50.
The q10 photo’s block grid and q10 plot’s ringing/color spread are reduced
after repair. The q50 texture has less JPEG-grid disruption. Fine source
filaments, thin plot whiskers and document text remain lost or distorted;
the positive metrics do not mean true fine detail was recovered. The q90 plot
and q50 document are routed out and identical between direct/repaired paths.
This is a bounded visual witness check, not a blinded preference study.

Artifacts include `WITNESSES-v1.json`, four independently decoded lossless HR
reference PNGs and `witnesses.html`, linking original PNGs at full size. No IQA
maps were generated.

## Reproduction, verification and archive

Numerical helper/driver/statistics revision: `9041cc3f31bf232066c3469f0f591ee226eba810`.
The full run uses the frozen helper binary; the later RGB reference renderer
is separate from its numerical identity. Models are canon4 f16, native SPANF
mixed f16 weights/f32 biases, and pinned author DSCF26 GPU FP32. Float pixels
pass directly from the public default `Restorer` into SR. No intermediate JPEG
encode, RGB8 rounding, changed routing threshold or HR coefficient projection
is applied. The source/current model overlap and split limitations remain.

Both the 7-input pilot and full comparison pass finite geometry, input/tensor
hash and unchanged-path gates. All 697 direct DSCF outputs plus their scalar
scores match the earlier retained study. Independent Pillow decode matches
every scored PNG hash and the four HR reference hashes. A failed first pilot
had a copied-binary executable-mode error and produced zero cells; it is kept
separate and does not supply results.

Artifact root verified on i265: `/mnt/v/output/zensr/sr-cascade-20261109/`.
The verified NAS mirror is `output/zensr/sr-cascade-20261109/` in the project
archive store. `ARCHIVE_VERIFIED-v1.json` records the identity/table/model/binary
checks and seeded-random scored/reference PNG checks. Open `witnesses.html`
from either artifact root for the six-case visual comparison.
`summary-v1/rows.tsv` retains 2,788 arm records; `summary-v1/summary.json` retains
all predefined paired/grouped summaries. `run-v1/cells/` gives routes, input
and pixel hashes, exact scalar scores and PNG locations. The scored outputs,
controls and LR previews occupy **3,282 unique PNGs / 394,265,002 bytes**, plus
the four HR reference PNGs for the visual witness set. Duplicate outputs share
a lossless artifact. The frozen input JPEGs/references remain in the earlier
external-model archive and are named by hash in `inputs.jsonl`.

| Evidence | SHA-256 |
|---|---|
| `run-v1/IDENTITY.json` | `e71151895e27da332bf6422989eae75e4ad9ad25efe9fffa0656ec2435224ad8` |
| `summary-v1/rows.tsv` | `447c99dd64d49e22b19ab6a3b551834fc59cfa0566e7b8f1769fd4e3d803c297` |
| `summary-v1/VERIFIED.json` | `4e6e2c9164e8f1e60383628203d4c9631a52eb8efab724d4072493551093cd52` |
| `canon4-f16.raw` | `c3db9373791270bbe5493d3ee98cd4f62296510fe6931073e2966f433e9a5cf4` |
| `spanf-f16.raw` | `955f7bfe7d7d503aac8f5e98177cd6a87f4b0235ec0b1fa42d013e874242fe1a` |
| `bin/superres` | `27ca11f81e802fa54c73128b3d4925280bd1b597bad0b7ea62d3854fd8ab7d0c` |

Reproduction: `just external-sr-cascade` with the frozen inputs/weights and
`just external-sr-cascade-report --dscf-control` with the earlier retained DSCF
run. Exact command lines and numerical identities are in the run logs. All
27 Python analysis/contract tests and five Rust storage/transport/PNG tests pass.

The quality run is not an inference latency benchmark. Resource record:

```text
run-heavy: done rc=0 81s | peak-RSS 1.34GiB | min-avail 21289MiB | peak-load 5.26
```
