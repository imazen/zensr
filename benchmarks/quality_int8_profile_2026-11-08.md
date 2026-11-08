# Quality-model INT8 execution profile — 2026-11-08

AVX-512 VNNI executes in the Tract integer path on the Ryzen 9 7950X under
WSL2. Convolution input packing and quantization operations consume substantial
CPU time around it. Smaller serialized weights alone do not establish faster
inference. This profile does not establish a latency multiplier.

## Frozen workload

The retained `dejpeg7_graphics` f16 baseline has SHA-256
`7ac1b1f9bf7952336fb27ad0132378fd02b5e77b6b6958ea4ced6612317f0b39`.
[Protocol and limits](../docs/QUALITY_INT8.md).

Both probes use the same deterministic 128 × 128 RGB float input and 20 forward
calls. Tract is pinned to 0.23.10. The software `cpu-clock` event was sampled at
499 Hz with call graphs. Sampling covers the whole probe process, including
model loading and graph preparation; it does not isolate an already warmed
forward call. The profiles ran separately from latency measurement.

```sh
run-heavy --mem 4G --jobs 1 -- perf record -e cpu-clock -F 499 -g \
  -o "$PROFILE" -- "$BINARY" probe "$EXPORTS" "$ARM" 128 128 20
perf report --stdio --no-children -i "$PROFILE"
```

Set `ARM=int8` for QLinearConv or `ARM=integer-f32` for ConvInteger followed
by float scaling, bias and PReLU. The profile executable came from the retained
`source-v3` snapshot; later reader/quality-tool fixes do not change these probes.

## Observed samples

Values are self sample shares, not time saved or predicted speedups.

| Sampled operation | QLinearConv | ConvInteger with float scaling |
|---|---:|---:|
| Actual `avx512vnni_mmm_i32_8x8_0_23_10` assembly | 9.21% | 18.15% |
| Convolution input patch packing | 20.75% | 38.81% |
| Float multiplication | 8.14% | 14.41% |
| Float-to-U8 activation quantization | 6.49% | 11.30% |

QLinearConv additionally spends 12.46% in its quantized `Scale` operation and
11.71% in `QSumB` correction. Those operations are absent from the leading rows
of the ConvInteger profile. Packing remains its largest sampled operation.
The `rusty` wrapper found during graph inspection calls the actual assembly
kernel; its name does not identify a scalar fallback.

The QLinearConv recording has 5,442 samples and the ConvInteger recording
3,185; both report zero lost samples. Resource-wrapper observations:

```text
QLinearConv: run-heavy: done rc=0 13s | peak-RSS 0.04GiB | min-avail 8888MiB | peak-load 2.12
ConvInteger: run-heavy: done rc=0 7s | peak-RSS 0.05GiB | min-avail 9079MiB | peak-load 2.65
```

These are whole-process observations. Do not divide their duration by 20 to
publish inference latency. The independent zenbench experiment retains model
preparation outside measured calls and records its own reliability flags.

## Core Ultra 7 265K execution

A subsequent probe directly records execution of the 256-bit
`avxvnni_mmm_i32_8x8_0_23_10` assembly on i265. It uses the same 128-pixel
input and 20-call procedure, pinned to performance core 2, with the preserved
v7 executable. That build supports multithreaded matrix multiplication; these
probes explicitly use one execution thread. Original timing blocks retain
their own build identities and controls.

| Sampled operation | QLinearConv | ConvInteger with float scaling |
|---|---:|---:|
| Actual AVX-VNNI assembly | 51.17% | 65.61% |
| Convolution input patch packing | 15.86% | 19.78% |
| Float-to-U8 activation quantization | 3.01% | 3.59% |
| Float multiplication | 1.33% | 1.68% |

QLinearConv also records 6.82% in `QSumB`. These shares show that the integer
kernel contributes much more to the Intel probe than to the WSL2 probe;
packing/conversion overhead alone does not explain both machines. A function's
sample share does not separate its arithmetic throughput from its load/stall
costs, or establish a speed ratio between CPUs.

There are 6,008 and 4,711 samples respectively, with zero lost samples.

```text
QLinearConv: run-heavy: done rc=0 13s | peak-RSS 0.04GiB | min-avail 24387MiB | peak-load 1.69
ConvInteger: run-heavy: done rc=0 10s | peak-RSS 0.04GiB | min-avail 23566MiB | peak-load 2.05
```

These remain whole-process profiles, not inference latency measurements.
The WSL2 observations above do not establish bare-metal 7950X performance.

## Retention

Raw profiles and complete logs are retained outside git under the experiment's
`profiles/` directory; snapshots, models and executable hashes bind the probes.
The quality study and latency comparison remain separate evidence. No serving
model, policy or backend was selected from these profiles.
