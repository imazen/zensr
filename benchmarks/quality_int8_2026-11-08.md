# Quality-model precision and inference comparison — 2026-11-08

The full quality, additional-format and threaded studies are still running.
This report currently contains the completed original six-arm timing block.
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
