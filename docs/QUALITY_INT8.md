# Quality-model integer inference comparison

Frozen before quality or latency results on 2026-11-08. Baseline is the actual
shipped f16 bytes of `dejpeg7_graphics`, Compact RGB x1, 64 channels and 16 middle
blocks, 595,459 parameters. SHA-256:
`7ac1b1f9bf7952336fb27ad0132378fd02b5e77b6b6958ea4ced6612317f0b39`.
This historical model has the target/provenance limitations in MODELS.md; this
experiment compares numerical implementations, not a new training-lineage claim.

## Frozen arms

- Native float inference using f16-expanded baseline weights.
- Tract float on exactly those weights.
- Tract float on per-channel INT8-rounded weights (weight-only control).
- Tract explicit QLinearConv W8A8, per-channel weights, fixed calibration
  activation ranges, float PReLU and residual addition.
- Same integer body with the final convolution in f32.
- Same integer body with the final two convolutions in f32.

Calibration uses 256 fixed evenly spaced optimizer pairs from the existing
published-training-ID pair assembly; internal validation and external evaluation
rows do not select ranges. Min/max ranges include zero. Input RGB quantization
uses the exact 0..255 code grid. No test-output parameter or checkpoint tuning.

Evaluate every arm on the same retained 2,960 JPEG/reference inputs selected for
the canon4 QAT study: 2,720 primary and 240 secondary size cells. This is reused
data, not proof of unseen-origin quality for this historical larger model.
Record float max/mean absolute RGB error, RGB8 changed-channel fraction and
integer-code error histogram, SSIMULACRA2, Butteraugli, clipping and harmful
changes. Compare raw model and guarded/coefficient-projected outputs. A float
final layer restores precision at the output, but cannot by assumption recover
information already lost in the integer body; the comparison measures that.

Latency uses zenbench, shuffled interleaved arms, in-memory identical inputs,
preloaded models, equal single-thread execution and uniform CPU placement on
i265. Measure 128, 256, 512 and 1024 square inputs; report execution separately
from loading/export and from JPEG/metric work. All conversion/activation costs
stay inside each inference call. Verify integer graph lowering and profile its
actual selected integer microkernel. No speed multiplier is inferred from file
size or hardware arithmetic throughput. Thread scaling and tiled serving costs
will be reported separately from a single-core implementation comparison.

Sources: [Tract quantized convolution](https://github.com/sonos/tract/blob/v0.23.10/core/src/ops/cnn/conv/conv.rs),
[ONNX importer](https://github.com/sonos/tract/blob/v0.23.10/onnx/src/ops/nn/mod.rs).
The standalone harness pins Tract 0.23.10 and keeps the serving graph unchanged.
