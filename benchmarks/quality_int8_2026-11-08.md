# Quality-model precision and inference comparison — 2026-11-08

The full quality and threaded studies are still running. This report contains
the completed single-thread timing blocks on both CPUs.
The observed integer path is slower, but every process has a resource-gate
warning: **zenbench marks confirmation invalid**. These are descriptive
shared-host measurements, not a clean confirmed performance claim.

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
