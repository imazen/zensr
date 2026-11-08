# Quality-model precision and inference comparison — 2026-11-08

The full quality cohort is complete. The four-thread timing studies are still
running. This report contains all quality results and the completed
single-thread timing blocks on both CPUs.
The observed integer path is slower, but every process has a resource-gate
warning: **zenbench marks confirmation invalid**. These are descriptive
shared-host measurements, not a clean confirmed performance claim.

## Complete quality comparison

**The tested PTQ activation formats lose substantial quality. Retaining one or
two original float output convolutions does not recover the native result.**
This is PTQ on the larger historical model; no activation-QAT training or
learned post-refiner was performed here.

All 2,960 required inputs completed, with 26,640 paired model/input rows.
Every row's identity and geometry match the frozen input manifest. The 512-pixel
primary cohort has 2,720 cases from 17 origins: four encoders, 420/444 and q5–100
in steps of five. The secondary cohort contains 240 cases from two origins at
128, 512 and 1024 pixels. These cohort origin counts are reported separately.

Repair is forced on every input, with routing unused and the near-pristine gate
off. Each arm is scored before and after the guard/coefficient projection.
Quality inference ran on the Core Ultra 7 270K Plus with eight single-thread
workers. SSIMULACRA2 changes are origin-balanced. Intervals resample 17 origin means
10,000 times with seed 20261108. Analysis uses NumPy 2.5.2. Harm is loss against
the decoded JPEG above 0.5 SSIM2 through identity score 90, tapering linearly to
0.1 at score 100. It is counted with decimal boundary arithmetic.

| Model | Raw ΔSSIM2 vs native | Projected ΔSSIM2 [95% origin interval] | Projected gain vs decode | Harm cases / 2720 | Added / removed harm vs native |
|---|---:|---:|---:|---:|---:|
| Native float | +0.00000 | +0.00000 [+0.00000, +0.00000] | +2.86600 | 257 | 0 / 0 |
| Tract float | -0.00017 | +0.00005 [-0.00035, +0.00048] | +2.86605 | 256 | 0 / 1 |
| Rounded-weight float | -0.76793 | -0.28821 [-0.58341, -0.04288] | +2.57779 | 341 | 93 / 9 |
| QLinearConv | -7.80043 | -4.71186 [-6.28446, -3.27946] | -1.84586 | 1725 | 1473 / 5 |
| QLinearConv + 1 F32 | -7.04832 | -4.23329 [-5.74360, -2.86279] | -1.36729 | 1463 | 1211 / 5 |
| QLinearConv + 2 F32 | -7.00672 | -4.17339 [-5.64755, -2.84671] | -1.30739 | 1490 | 1240 / 7 |
| ConvInteger | -8.78029 | -4.61554 [-6.01228, -3.27942] | -1.74954 | 1762 | 1509 / 4 |
| ConvInteger + 1 F32 | -7.12345 | -3.72688 [-4.98894, -2.54885] | -0.86089 | 1491 | 1238 / 4 |
| ConvInteger + 2 F32 | -6.80470 | -3.57714 [-4.78307, -2.45457] | -0.71114 | 1461 | 1208 / 4 |

These are forced-repair harm rates. The approved routed-default measurements
remain in their [separate comparison](routing_feature_ablation_2026-11-06.md).

### RGB precision

The actual first input quantizer preserves all 768 tested RGB8 codes exactly.
The errors below arise later in the network. Errors compare each arm with the
native float arm; percentages count channel values, weighted by image area.
The complete JSON retains exact 0/1/2/3-or-more code histograms and normalized
float errors for both output stages.

| Model | Raw max RGB8 error | Projected max RGB8 error | Projected changed values (%) | Projected error >1 code (%) | Projected normalized float MAE |
|---|---:|---:|---:|---:|---:|
| Native float | 0 | 0 | 0.000000 | 0.000000 | 0.00000000 |
| Tract float | 1 | 1 | 0.000635 | 0.000000 | 0.00000003 |
| Rounded-weight float | 82 | 59 | 22.173032 | 1.852327 | 0.00102924 |
| QLinearConv | 132 | 92 | 66.513720 | 36.307025 | 0.00626028 |
| QLinearConv + 1 F32 | 132 | 94 | 64.330986 | 34.466858 | 0.00576379 |
| QLinearConv + 2 F32 | 129 | 94 | 60.029440 | 31.123191 | 0.00531680 |
| ConvInteger | 128 | 102 | 69.148697 | 38.859060 | 0.00612513 |
| ConvInteger + 1 F32 | 128 | 102 | 65.977975 | 32.124369 | 0.00530954 |
| ConvInteger + 2 F32 | 129 | 103 | 60.908624 | 30.104408 | 0.00493897 |

### Loss severity and the second IQA metric

Lower Butteraugli p-norm-3 is better. Worst SSIM2 losses are per-case minima;
they need not occur on the same input as one another.

| Model | Projected mean ΔButteraugli p-norm-3 vs native | Worst projected ΔSSIM2 vs native | Worst projected SSIM2 gain vs decode |
|---|---:|---:|---:|
| Native float | +0 | +0.00000 | -6.62849 |
| Tract float | +1.767671e-07 | -0.12290 | -6.63477 |
| Rounded-weight float | +0.01127282 | -7.77147 | -10.04395 |
| QLinearConv | +0.4455652 | -74.37165 | -65.18937 |
| QLinearConv + 1 F32 | +0.4156244 | -74.72445 | -65.45762 |
| QLinearConv + 2 F32 | +0.3791684 | -73.11458 | -64.73942 |
| ConvInteger | +0.3214443 | -22.04351 | -19.48898 |
| ConvInteger + 1 F32 | +0.1965862 | -20.84552 | -17.62249 |
| ConvInteger + 2 F32 | +0.193241 | -20.62612 | -17.40309 |

The QLinearConv worst loss against decode occurs on source 7109, turbo 444 q5:
−65.18937 SSIM2 after projection. Keeping two original float output layers still
loses 64.73942 there. The ConvInteger format with two float output layers has a
worst loss of 17.40309, on source 8157, mozjpeg 444 q50. These numeric tails remain
in the report; no new pixel-artifact audit was performed.

Explicit quantizer-input bounds are exceeded by 3 of 555,017,011,200 values for
QLinearConv and 7 of the same denominator for full ConvInteger. The float-head
variants have no observed exceedances at these instrumented inputs. Saturation
inside QLinearConv's own output quantizer is not instrumented. Small explicit
input-range counts therefore do not establish absence of internal clipping.

### Secondary sizes

Projected origin-balanced ΔSSIM2 against native float; two origins and 80 cases
at each size. These are limited size checks on reused inputs.

| Model | 128 pixels | 512 pixels | 1024 pixels |
|---|---:|---:|---:|
| Native float | +0.00000 | +0.00000 | +0.00000 |
| Tract float | -0.00058 | +0.00057 | -0.00062 |
| Rounded-weight float | +0.17561 | -0.14615 | -0.19838 |
| QLinearConv | -2.81742 | -2.93386 | -4.03218 |
| QLinearConv + 1 F32 | -2.52309 | -2.47614 | -3.34092 |
| QLinearConv + 2 F32 | -1.91995 | -2.54425 | -3.38326 |
| ConvInteger | -2.19456 | -3.53191 | -4.88680 |
| ConvInteger + 1 F32 | -1.72877 | -2.79893 | -3.84952 |
| ConvInteger + 2 F32 | -1.32156 | -2.69644 | -3.68085 |

### Replay and independent reference

All 54 rows from the six source/size replays match the full table in every one
of their 41 fields. Those replays independently compare original models and
range counters with the shared graph, and the native adapter with `restore_jpeg`.

On the fixed `[1,3,17,13]` input, ONNX 1.23.2's reference evaluator matches both
variants with 18 integer convolutions bit for bit; PReLU and residual addition
remain float. All eight exported variants have
identical RGB8 results to the reference; maximum normalized float difference
is 7.1525574e-7. Official opset adaptation from 13 to 19 leaves every node,
constant and input/output declaration unchanged. The reference uses NumPy
2.5.4. This one-input interoperability check supplements the full Tract study.

```text
run-heavy: done rc=0 15108s | peak-RSS 6.58GiB | min-avail 17338MiB | peak-load 11.80
```

This resource observation covers the complete quality run and its eight-worker
pool, not one model. Fixed input chunks leave a final worker active after other
chunks finish; wall time is not a per-model inference benchmark.

Results SHA-256: `1b5186ff30104101f3da6c6e454af44b4a33ad7794fc61945e8dc9c2a3f2270d`.
Input manifest SHA-256: `61ff80a858f910ce4eb92f2df1289ce27455c4a72e8a40ad56e65933497bb1c3`.
Frozen evaluator binary: `36fc3945933349af7f5385a95ab8cab95abc020a968fccbe226a910e6e9fda12`.
Full summaries, all rows, input identities and reference/replay proofs remain in
the retained experiment. The model asset bundle includes the measurements and
summary. [Asset layout](../docs/QUALITY_INT8_ASSETS.md).

## Original Core Ultra 7 265K timing block

Retained `dejpeg7_graphics`, 595,459 parameters, RGB ×1. The original f16
artifact is expanded to f32 for native and Tract float execution. The
[protocol](../docs/QUALITY_INT8.md) defines the fixed train-only calibration,
exports, controls and reused evaluation inputs.

All four sizes completed five measured rounds in each of three fresh processes.
Values are arithmetic means of process means, in milliseconds. CPU placement is
root-cgroup enforced on performance core 2, one execution thread. Models and
input tensors are loaded before measured calls; output extraction/copy/drop is
included. Inputs are the same deterministic nonsmooth RGB texture for every arm.

| Pixels per side | Native float | Tract float | Rounded-weight float | QLinearConv INT8 | Last conv f32 | Last 2 convs f32 | Native serving tiles |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 197.277 | 210.012 | 209.379 | 598.709 | 592.313 | 565.845 | 198.085 |
| 256 | 805.335 | 1126.346 | 1127.954 | 3170.648 | 3102.951 | 2997.404 | 807.245 |
| 512 | 3255.796 | 4851.988 | 4833.511 | 13209.086 | 12999.640 | 12505.403 | 3261.908 |
| 1024 | 13197.959 | 20186.807 | 20130.362 | 61422.098 | 60183.890 | 57838.335 | 14363.969 |

Rounded-weight float uses INT8-rounded weights with float computation. It is a
precision control. Last-convolution controls retain original float weights in
those layers; they are not trained post-refiners.

The process resource flags are retained: 14, 14 and 10 flagged rounds. No process
or arm is removed to obtain a favorable summary. No within-process confidence
interval is presented as independent-process confirmation.

```text
run-heavy: done rc=2 6680s | peak-RSS 1.59GiB | min-avail 18860MiB | peak-load 3.10
```

This resource observation covers the complete experiment, including model
preparation and every arm. It does not identify per-model runtime memory.

Producer: `28ed1d81cebfbaf87b1ce1fedc0b18953eee0243`.
Binary SHA-256: `e6273d2927bcf6a6e04e3d8181462beccf0c8327b60210de8074bf98932dd1bc`.
Frozen original export plan SHA-256:
`1aa0b0855ef329df1066a454a4a4eb67696cd0e6e42a7ceb917d8a46aae50668`.

The later ConvInteger block includes new native, Tract-float and QLinearConv
controls because its build enables the multithreaded matrix capability.
Comparisons between graph representations belong within that block; it does
not replace these original measurements.

## Core Ultra 7 265K: conversion-reduced follow-up

The follow-up enables the multithreaded matrix capability at build time but
explicitly executes one thread, pinned to performance core 2. Three fresh
processes completed five measured rounds at all four sizes. All process
reliability flags remain in the retained report; confirmation is unconfirmed.
Comparisons between graph formats use these controls from the same build.

```text
run-heavy: done rc=2 6885s | peak-RSS 1.41GiB | min-avail 21865MiB | peak-load 17.70
```

Observed arithmetic means of process means, in milliseconds:

| Side | Native float | Tract float | QLinearConv | ConvInteger | ConvInteger + last F32 | ConvInteger + last 2 F32 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 202.163 | 220.839 | 630.750 | 498.094 | 490.951 | 475.633 |
| 256 | 827.001 | 1152.474 | 3158.389 | 2560.501 | 2522.676 | 2449.025 |
| 512 | 3326.484 | 5060.393 | 13792.843 | 10829.920 | 10705.134 | 10324.575 |
| 1024 | 13108.450 | 20631.249 | 62799.175 | 48997.322 | 48154.125 | 46428.804 |

Retained report: `latency-i265-convinteger.json`. The wrapper memory observation covers the
complete matrix and setup; it is not per-model memory.

## Ryzen 9 7950X under WSL2: single thread

Three fresh processes ran the nine-arm matrix on guest core 2. This is a WSL2
observation, not a bare-metal measurement. The 1800-second per-group wall cap
can reduce the completed round count; the retained report records actual
counts for every arm/process. Resource/noise flags make confirmation invalid.

```text
run-heavy: done rc=2 10485s | peak-RSS 1.88GiB | min-avail 5672MiB | peak-load 11.00
```

Observed arithmetic means of process means, in milliseconds:

| Side | Native float | Tract float | QLinearConv | ConvInteger | ConvInteger + last F32 | ConvInteger + last 2 F32 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 214.580 | 464.507 | 838.394 | 457.216 | 451.369 | 437.231 |
| 256 | 905.631 | 2051.786 | 3866.835 | 2468.279 | 2439.898 | 2445.278 |
| 512 | 3762.106 | 11148.657 | 22198.396 | 15440.346 | 15198.954 | 14754.043 |
| 1024 | 15807.805 | 49668.635 | 126068.369 | 93764.254 | 92842.990 | 85111.778 |

Retained report: `latency-wsl-original.json`. The wrapper memory observation covers the
complete matrix and setup; it is not per-model memory.

## Input precision and executed kernels

The actual first quantizer selected from the frozen QLinearConv model preserves
all 256 RGB8 codes in each of the three channels: 768 exact output codes on the
Core Ultra 7 270K Plus worker. This does not test the network's internal
activation or weight precision; the complete nine-arm study measures final RGB
and perceptual changes before and after the guard/coefficient projection.

[Executed VNNI profiles](quality_int8_profile_2026-11-08.md) show actual
256-bit AVX-VNNI and AVX-512-VNNI execution, with different kernel/packing shares
on the two CPUs. Profile shares are not latency multipliers.

## Retention and limits

On i265, `/mnt/v/output/zensr/quality-tract-int8-20261108/` retains frozen
executables, source snapshots, exports, raw process/round evidence,
`latency-i265-original.json`, probe records and complete logs. Generated data
and model binaries remain outside git.

This historical model retains the training/provenance limitations in
[Models](../docs/MODELS.md). Reused quality inputs do not establish unseen-origin
generalization. No new serving backend, model or routing default is selected.
