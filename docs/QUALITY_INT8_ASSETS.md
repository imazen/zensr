# Quality-model PTQ comparison assets

These exports compare numerical representations of the retained
`dejpeg7_graphics` RGB ×1 model: 64 channels, 16 middle blocks and 595,459
parameters. They are experimental PTQ exports. The separate canon4 study
contains the QAT-trained models.

The historical quality model retains the target/provenance limitations in
[Models](https://github.com/imazen/zensr/blob/aea369361e7b531577f1897b7517f4a91a9d814b/docs/MODELS.md).
Reused evaluation inputs do not establish unseen-origin quality. No serving
model or backend is adopted by this package.

The complete 17-origin, 2,720-case primary comparison finds mean SSIM2 losses
of 4.71186 for QLinearConv and 4.61554 for ConvInteger against native float after
guard/coefficient projection. Retaining two original float output layers
reduces those losses to 4.17339 and 3.57714. The tested integer paths also take
longer in the observed single- and four-thread measurements on both CPUs;
resource/noise flags leave formal confirmation invalid. These are forced-repair
results. The [complete report](https://github.com/imazen/zensr/blob/main/benchmarks/quality_int8_2026-11-08.md)
retains harm frequency, worst losses and RGB errors. Native float remains the
serving implementation.

## Files

| ONNX file | Convolution representation | Bytes |
|---|---|---:|
| float.onnx | F32 baseline | 2384621 |
| weight-int8.onnx | F32 with INT8-rounded weights | 2384627 |
| int8.onnx | 18 QLinearConv | 614764 |
| final-f32.onnx | 17 QLinearConv + 1 F32 conv | 619676 |
| final2-f32.onnx | 16 QLinearConv + 2 F32 convs | 729685 |
| integer-f32.onnx | 18 ConvInteger + F32 scaling/bias | 614446 |
| integer-final-f32.onnx | 17 ConvInteger + 1 F32 conv | 619320 |
| integer-final2-f32.onnx | 16 ConvInteger + 2 F32 convs | 729351 |

All graphs accept normalized RGB F32, NCHW `[1,3,H,W]`, and return RGB F32 at
the original dimensions. The input residual is already added. Guard and JPEG
coefficient projection are outside the graph. PReLU and residual addition use
F32; QLinearConv quantizes its convolution output before dequantization/PReLU.
ConvInteger retains INT32 accumulation through F32 scaling and bias. Float-head
variants retain original float weights; they are not learned post-refiners.

`weight-int8.onnx` performs float computation on weights rounded to the INT8
code grid. Its serialized weights are F32. It is a precision control.

- `baseline/weights_f16.raw`: exact retained baseline, 1,190,918 bytes, SHA-256
  `7ac1b1f9bf7952336fb27ad0132378fd02b5e77b6b6958ea4ced6612317f0b39`.
- `models/weights.raw`: that baseline expanded to little-endian F32, in native
  Compact layer order. `gold-*.raw` hold the 17 × 13 F32 input/output golden.
- `models/PLAN.json`: frozen exports, calibration ranges, IDs and source hashes.
- `calibration/`: the original calibration plan and goldens required by re-export.
- `historical-reference/`: original unrounded F32 weights and retained metadata;
  these weights were not evaluated in this comparison.
- `quality/`: complete measurements, summary, input identity and RGB8 input-grid
  check, plus the independent ONNX reference comparison on the fixed golden
  input. The original input manifest records staging paths for provenance.
- `latency/`: observed process means, actual sample counts and reliability flags.
- `EVALUATION-INPUT-CATALOG.tsv`: source/rendition identities with pixel paths omitted.

The package excludes JPEG/reference pixels, raw profiler captures, machine
logs, auxiliary range-counter/shared graphs and raw timing-round objects.
Those remain in the retained experiment. `MODEL-CATALOG.json` records exclusions.
It also identifies each retained auxiliary graph/map by relative path, byte count
and SHA-256 so those files can be retrieved separately.

## Reproduce an export

Clone zensr at source revision `aea369361e7b531577f1897b7517f4a91a9d814b`.
The frozen calibration/export environment records PyTorch
2.15.0+cu132, NumPy 2.5.4 and ONNX 1.23.2.
Re-export uses the stored ranges and runs on CPU without recalibration or
optimizer steps. From the cloned repository, with the bundle unpacked:

```sh
ZENSR_BUNDLE=/path/to/unpacked-bundle
python3 tools/quality-int8/reexport.py \
  --weights "$ZENSR_BUNDLE/baseline/weights_f16.raw" \
  --calibration "$ZENSR_BUNDLE/calibration" --output new-export
```

The exporter verifies original graph hashes. Compare all regenerated model
hashes with `MODEL-CATALOG.json`. The frozen ranges avoid needing calibration
pixels to reproduce these exports. Training this historical model from scratch
is a separate provenance question.

The standalone Rust harness builds with the committed lockfile:

```sh
cargo build --release --locked --manifest-path tools/quality-int8/runtime/Cargo.toml
```

The [protocol](https://github.com/imazen/zensr/blob/aea369361e7b531577f1897b7517f4a91a9d814b/docs/QUALITY_INT8.md)
and [comparison report](https://github.com/imazen/zensr/blob/main/benchmarks/quality_int8_2026-11-08.md)
define coverage and limits. The quality comparison forces repair and reports
both raw and guarded/projected outputs; it does not measure the routed default.
Timing summaries retain unconfirmed resource/noise flags where present.
