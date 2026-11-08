# Quality-model integer inference comparison

Frozen before quality or latency results on 2026-11-08. Baseline is the retained
f16 bytes of `dejpeg7_graphics`, Compact RGB x1, 64 channels and 16 middle
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
The quality comparison forces repair, with the near-pristine gate disabled;
no router is fitted or used to hide numerical regressions. It is not a new
measurement of the routed serving default.

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

## Repetition-budget amendment

The initial 15-round run was stopped after its first 128-pixel block. That
partial run is retained as exploratory and excluded from final confirmation.
The replacement retains all four resolutions and three fresh processes, with
up to five rounds per arm/size within the 1,800-second group wall limit.
Report actual completed rounds for every process and size. This changes
measurement duration, not models,
calibration, evaluation inputs, quality thresholds or the comparison family.

## Additional integer export representation

An additional format control uses explicit ConvInteger (INT8 inputs/weights,
INT32 accumulation), then f32 scaling, original f32 bias and PReLU. Activations
are quantized once per block. In contrast, the original QLinearConv export
quantizes convolution output before PReLU and quantizes again for the next
block. The ConvInteger controls retain the same fixed ranges and weights, with
all-integer convolutions or one/two final f32 convolutions. This amendment
follows the first 128-pixel timing, before inspecting pilot quality scores.
Original ONNX exports must remain byte-identical. All nine native/Tract arms
will be included in quality evaluation; format and threaded latency are
additional comparisons and do not replace the original timings.

## Hardware amendment

The owner requested a Ryzen 9 7950X / WSL2 comparison after asking whether
VNNI was enabled. The Core Ultra 7 265K graph selects AVX-VNNI; the second CPU
exposes AVX-512-VNNI. Record both selected and executed microkernels, separate
all host timings, and disclose VM/host scheduling limits. Build natively on
the older system-library platform without target-cpu=native. Quality workers
on another eligible host require a matching model/input/binary identity and
replay checks; their quality scores are explicitly attributed to that host.

## Quality execution optimization

Quality scoring shares identical pure graph nodes across the eight Tract arms.
The timed ONNX graphs remain separate and byte-identical. Synthetic goldens
gate the independent models. Full-size, independently executed output/counter
parity gates the shared quality graph. This avoids recomputing common network prefixes without omitting
any candidate or input. Range counters use int32 because each map has at most
64 * 1024 * 1024 elements; the runner enforces the counter domain. This avoids
Tract's symbolic-dimension representation of ONNX int64 tensors.

## Observed timing limits

The first 7950X / WSL2 process completed only one measured 1024-pixel round
within the group wall limit. Do not describe the configured maximum of five
as five completed rounds, or treat a one-round within-process interval as
an uncertainty estimate. The three independent processes remain required.
Quietness-gate flags remain part of the retained evidence; flagged runs do
not establish a clean confirmation, regardless of the observed speed gap.

[Executed VNNI and packing/quantization profiles](../benchmarks/quality_int8_profile_2026-11-08.md)
explain the observed integer overhead. Profile shares do not predict latency;
the independent timing runs retain their actual sample counts and gate flags.

## Threaded replay budget

The threaded runner can explicitly use `ZENSR_THREAD_PARITY=first-process`.
With pinned zenbench 0.1.11, child zero checks exact one-thread/four-thread
output equality for all nine models at both 512 and 1024 pixels before timing.
The parent must finish that child successfully before launching either later
child. The mode requires the three-process subprocess context and a retained
first-child result. Models, binary, inputs and CPU boundary remain frozen.
The default still checks each process. This avoids repeating the same full-size
numerical test in every fresh timing process; all three timing processes remain.

`rgb-grid EXPORTS OUTPUT.json` checks the first input quantizer selected from
the actual frozen QLinearConv graph against every RGB8 code in every channel.
This is separate from the table's native normalization/roundtrip field;
internal weight/activation quantization and final RGB errors remain full-study
measurements. Keeping an original float head is not a trained post-refiner.
