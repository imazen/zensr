# Browser SR and JPEG repair — 2026-11-09

These are functional wall-clock runs on i265 (Core Ultra 7 265K), Chrome
153.0.8010.12, ONNX Runtime Web 1.31.1, Rust 1.99.1. They include tile inference,
PNG encoding, native-detail tile retention and OPFS writes. They are single runs
with shared-machine activity, not isolated zenbench measurements or native CPU
speed comparisons. The models, operation graphs, halos and stored weights are
identified by `prepare.py`; the browser verifies their hashes.

Native product code is unchanged. Decoder: `8ba10f01`; relaxed build: `95a5f0f9`;
viewer: `cc954890`; pipeline integration: `5a3c85be`. Starting source: `a2479fae`, following the
browser worker-pool change `1b58fad3`. [Integration and reproduction](../docs/BROWSER_DEMO.md).

## Concurrency and 12 MP coverage

All large inputs are the retained 4000×3000 synthetic fixture. JPEG variants
were encoded with zenjpeg. These measurements establish execution and pixel
parity on this fixture, not perceptual quality on 12 MP photographs.

| Input / model / output | CPU workers | SR + encoding seconds | Whole processing seconds |
|---|---:|---:|---:|
| JPEG q40, SPANF ×4, strict repair routed off | 1 | 132.234480 | 132.984080 |
| Same JPEG and settings | 16 | 12.846715 | 13.848810 |
| PNG, DSCF ×4 | 12 | 15.324280 | 16.021220 |
| JPEG q40, zenjpeg decode, SPANF ×4, repair off | 12 | 14.798840 | 15.531410 |
| JPEG q5, strict repair, DSCF ×2 | 8 | 19.509855 | 908.461595 |
| Same q5 JPEG, relaxed repair, DSCF ×2 | 8 | 20.431260 | 45.210725 |

The matched SPANF pair is 10.3× faster during SR/encoding, 9.6× for the full
processing interval. Its 164,212,014-byte PNG is byte-identical between one and
sixteen workers. Moving ordinary JPEG decoding to zenjpeg also preserves this
PNG exactly. The DSCF ×4 output is 16000×12000; ×2 is 8000×6000. Neither path
assembles a full float output image.

## Why strict WASM repair was slow

The CNN's SIMD128 kernel uses magetypes 0.9.30 `mul_add_portable` and
`mul_sub_portable`. On WASM these enforce one-rounding fused arithmetic using
software f64 expansion. `+simd128,+relaxed-simd` alone does not change those
portable calls. A Chrome CPU profile concentrates samples in
`conv3x3_packed_dispatch`; this is a convolution arithmetic bottleneck.

The isolated relaxed build copies owned source, checks the SIMD file's SHA-256,
and replaces exactly 15 CNN portable multiply-add/sub calls with engine
multiply-add/sub. It enables `+simd128,+relaxed-simd`. No projection, routing,
model or native runtime source is changed. The source and generated file hashes
are retained in `RELAXED_SOURCE.json`. Strict remains the reference/default;
relaxed is an explicit experimental browser option. Engine-dependent rounding
means observed parity is not a cross-platform guarantee.

- A controlled small three-pair call test measured strict 0.682825, 0.635695,
  0.640520 seconds versus relaxed 0.035110, 0.016770, 0.016080 seconds.
- The explicit matrix covers 136 JPEGs: 17 origins/categories × q5/q20/q50/q90 ×
  420/444. Repair applied in 60 cases and routing bypassed 76. All 4,758,528 RGB8
  values match strict output exactly; dimensions and routing outcomes match too.
- Full 12 MP q5 strict repair took 888.233585 seconds. A direct relaxed call
  took 22.925385 seconds; all 36,000,000 RGB8 values
  match the retained strict checkpoint SHA-256
  `0dcf4712181e55f10c07139869ee4e4f31e858dbbc08c7fa42bc330293b2202b`.
- Through the real worker pipeline, relaxed repair took 23.929080 seconds
  versus strict 888.233585 seconds (37.1×);
  zenjpeg decoding took 0.126360 seconds. The complete repair + DSCF ×2 job
  took 45.210725 seconds. Both its 16,282,590-byte 1× checkpoint and
  50,957,284-byte SR PNG are byte-identical to the strict run.

The direct call excludes module setup and worker transfers; use the pipeline
phase comparison for matching timing boundaries. These measurements do not
isolate every shared-machine scheduling effect.

The original long repair exposed an OPFS writer opened before preprocessing.
That run failed on its first write after repair completed. Writers now open
after preprocessing; a separate 1× PNG checkpoint completes before SR begins.
The regression refuses early writer creation and verifies checkpoint persistence.
The failed run and its logs remain retained.

## Memory and validation limits

Each CPU tile worker is capped at 64 MiB WASM memory with an 80 MiB reservation.
The observed 16-worker SPANF pool uses 647,430,144 WASM bytes; its configured
reservation is 1280 MiB. Relaxed and strict 12 MP repair both reach 880,672,768
WASM bytes under the separate 1 GiB cap. JPEG decoding is capped at 512 MiB and
PNG encoding at 128 MiB. These phase workers exit before the SR pool begins.
Native-detail viewing uses a 32 MiB decoded tile LRU. Canvas, browser and driver
allocations are outside those caps: this is not a tab-RSS guarantee.

Required wrapper observations (peak RSS is the wrapper-reported maximum process
RSS, **not aggregate Chrome process-tree RSS**):

| Run | run-heavy peak RSS | Minimum available RAM | Peak load |
|---|---:|---:|---:|
| SPANF q40 serial | 0.51 GiB | 20,827 MiB | 2.88 |
| SPANF q40 sixteen workers | 0.57 GiB | 19,794 MiB | 6.14 |
| DSCF ×4 twelve workers | 0.57 GiB | 20,642 MiB | 4.36 |
| Strict q5 repair + DSCF ×2 | 0.31 GiB | 17,357 MiB | 5.75 |
| Direct relaxed 12 MP repair | 0.18 GiB | 19,638 MiB | 5.70 |
| Relaxed q5 repair + DSCF ×2 | 0.31 GiB | 20,491 MiB | 5.69 |

Tiny/odd full-frame versus halo-tiled ONNX inference is float-exact in both
models, including true boundaries. Independent Pillow PNG decoding validates
outputs, and twelve retained native-detail tiles match their full-output crops
exactly. Decoder tests cover all eight EXIF orientations and an identity sRGB
ICC conversion. Repair separately matches the native Restorer oracle and rejects
profiled JPEG repair explicitly; ordinary profiled JPEG SR uses zenjpeg CMS.
Cancellation terminates repair and tile workers; restart, preserved downloads,
reload, malformed-input rejection, viewer controls and gallery comparisons pass.

WebGPU functional checks use **SwiftShader software rendering** with actual GPU
buffer tracking. Physical GPU WebGPU speed is not measured. No IQA maps are
produced; no new perceptual-quality or publication-generalization claim is made.

## Retained evidence and commands

Artifacts live outside git under `/mnt/v/output/zensr/browser-demo-20261109/`.
`PAIRED_OUTPUTS.json` binds case stats and PNG/checkpoint hashes. Each case retains
`result.json`, full logs, screenshots and encoded outputs. `relaxed-cohort.json`
identifies every matrix input by source and byte hash. CPU profiles, strict and
relaxed WASM packages and generated source are retained separately from git.
The project-storage archive is bound by `ARCHIVE_VERIFIED-v1.json`: 444 files,
1,363,321,963 bytes, copied and read back with matching SHA-256. The rebuildable
`relaxed-target-v2` Cargo directory is excluded; its generated source, logs and
loadable WASM package are retained. Archive wrapper: 21 seconds, peak RSS
0.02 GiB, minimum available RAM 22,981 MiB, peak load 1.76.

Reproduction recipes include `browser-build`, `browser-prepare`,
`browser-relaxed-build`, `browser-test`, `browser-repair-matrix`,
`browser-repair-large-compare`, `browser-decode-test`, `browser-writer-test`,
`browser-parity`, `browser-lifecycle`, `browser-viewer-test`,
`browser-gallery-test` and `browser-verify`. Full commands and source/model
identities remain in the corresponding artifact logs/manifests. Set
`ZENSR_REPAIR=1 ZENSR_REPAIR_MATH=relaxed` for the full relaxed pipeline test;
`ZENSR_CPU_WORKERS` fixes pool concurrency.
