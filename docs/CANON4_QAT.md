# Canon4 quantization-aware fine-tuning

This experiment measures INT8 **weight-only** QAT on the exact deployed canon4
f16 artifact. Activations, biases and PReLU slopes remain f32; execution uses
the existing float CPU convolutions. It is a storage experiment, not an integer
compute speedup. The shipped default is unchanged.

The model definition is reused from `tools/train_people.py`. Each convolution
uses PyTorch's native per-output-channel fake quantization with its
straight-through gradient. Scales are detached channel absmax/127, recomputed
on every forward. Export recovers the native operator's integer codes and
requires bit-identical decoded weights; division-based rounding alone can
differ at a tie. Biases and PReLU slopes stay f32.

References: [native fake quantization](https://docs.pytorch.org/docs/2.15/generated/torch.fake_quantize_per_channel_affine.html),
[PyTorch QAT overview](https://pytorch.org/blog/quantization-aware-training-in-torchao-ii/).
The current TorchAO QATConfig targets Linear/Embedding; this CNN uses the native
convolution weight primitive directly rather than silently leaving convolutions
unquantized.

## Frozen primary design

- Baseline: canon4 f16 SHA-256
  `c3db9373791270bbe5493d3ee98cd4f62296510fe6931073e2966f433e9a5cf4`.
- Arms: unchanged baseline, untrained INT8 PTQ, float fine-tuning and INT8 QAT.
- Three paired seeds: 7, 17, 29. Float and QAT arms see the same batches and use
  identical optimizer/schedule/loss. All convolutions are quantized in QAT.
- 10,000 steps per trained arm, batch 16, AdamW without weight decay, initial
  learning-rate ceiling 1e-5, 250-step warmup then cosine to 2e-7, gradient norm
  ceiling 1. FP32 compute, deterministic algorithms, TF32/AMP disabled.
- Training loss: ground-truth Charbonnier plus 0.25 times Charbonnier to frozen
  deployed canon4 outputs. This anchors behavior while adapting quantized weights.
- Fine-tune on the existing filtered 39,503 pair rows whose IDs belong to the
  published imazen-26 training split. The 392 internal-validation rows are also
  from published training IDs and are source-disjoint from optimizer rows.
  Source labels are independently joined to imazen-26 revision
  `187fbf338ce08e8e6654db7f04ddae58d5263da2`, `manifests/split_map.tsv`.
- Select each run's checkpoint by minimum internal-validation GT Charbonnier,
  measured every 500 steps and at the last step. External evaluation is opened
  only after selection. There is no test-set checkpoint selection.
- Evaluate the complete Rust repair pipeline using the frozen canon4 routing
  policy for all arms, plus forced repair to distinguish quantization damage
  from routing coverage. Report paired SSIMULACRA2 changes, gain versus identity,
  the existing quality-dependent harm curve and loss severity, with per-origin
  summaries. No recalibration or default adoption is implicit.
- Evaluation sources must be from the published validation/test IDs and absent
  from the entire original 48,000-pair manifest, including its validation tail.
  Use retained JPEG/reference bytes where available. Low qualities receive the
  same quality-grid density as high qualities; do not regenerate old encodes.

The original baseline inherited sources reassigned across the published split.
Filtering new fine-tuning does not undo that exposure. The selected external
sources exclude those known origins; wider family independence is not claimed.
See [split authorities](SPLIT_DEFINITIONS.md).

## Running and artifacts

`just qat-tests` tests native quantization gradients, deployed baseline goldens,
exact packed/dequantized operands and malformed weight data. `just qat-train`
accepts explicit dataset, split-map, baseline and new output directory paths.
The run records input/source hashes, package/numerical settings, exact batches'
RNG state, model/optimizer checkpoints, losses and selected-step identities.
A bounded pilot may test execution before the primary run; pilot results are
not the primary estimate. Outputs and checkpoints stay outside git.

`weights_q8pc.bin` has an eight-byte `ZSRQ8PC\0` magic, then little-endian u32
version=1, nf=24, nc=8, scale=1. Each convolution stores f32 scales by output
channel, signed INT8 weights in original OIHW order, f32 biases, then f32 PReLU
slopes except after the final convolution. Codes are -127..127. Scales must be
finite and positive. No entropy compression is applied. The exported
`weights.raw` is the exact f32 expansion for existing runtimes, not the compact
storage-size claim. This experimental format is not yet a `.zensr` encoding.

## Numerical-control amendment, before reading external candidate scores

The explicit CPU/GPU conversion check found identical initial INT8 codes but
seven differing f32 scale words: 1,025 expanded coefficients differ, with a
maximum absolute difference of 2.9802322387695312e-8. Add an untrained
GPU-exported PTQ control using the unchanged baseline. Evaluate it on the same
frozen inputs, with the same runtime/dependency graph, threads, routing and
metrics. Retain the original CPU PTQ control as well. No training, checkpoint
selection, evaluation-source selection or routing calibration changes.

The extra evaluator run repeats the baseline. Combining results requires exact
agreement of its baseline scores and control fields with the original run;
source binary identities remain separate. This isolates conversion rounding
from QAT's measured effect rather than attributing both to fine-tuning.
