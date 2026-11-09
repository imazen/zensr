# Canon4 repair followed by ×4 super-resolution

Frozen comparison: direct native SPANF, default-routed canon4 → native SPANF,
direct author DSCF26, and default-routed canon4 → author DSCF26. No model,
routing threshold, mixing coefficient or image subset is selected from outcomes.

Reuse all 697 inputs from `sr-inputs-v1/inputs.jsonl`: 17 source/category
representatives, zenjpeg q5–100 every five in 420/444, plus 17 clean inputs.
References and JPEG bytes stay unchanged. HR references have longest side 512;
the LR lattice has longest side 128. These are reused inputs with one source
per category, not an independent held-out publication certification. External
model training overlap is unknown, and the canon4 training split has the
limitations described in [split definitions](../SPLIT_DEFINITIONS.md).

## Pipelines

- Direct arms consume the plain zenjpeg RGB decode used by the earlier study.
- Repair arms call the current public `Restorer` API with deployed canon4 f16
  weights, Realtime budget, default Fidelity intent/default routing and one
  worker. No routing or high-quality gate override is applied. Clean RGB
  inputs pass through unchanged.
- Guard and JPEG coefficient projection remain in that API at LR dimensions.
  SR consumes its planar f32 output directly, without RGB8 rounding or a new
  JPEG encode. The report retains the repair/skip decision and projection status.
- SPANF uses the retained mixed f16-weight/f32-bias artifact and native ×4
  implementation, matching the earlier CPU configuration. DSCF uses its pinned
  author GPU FP32 implementation with deterministic settings. No SR guard or
  coefficient projection is added on the HR lattice.

## Measurements and retention

Score the final HR RGB8 image using SSIMULACRA2 and Butteraugli overall/p-norm-3,
without border shaving. Retain the exact scored outputs as lossless PNG with
pixel/file hashes. Retain LR repair previews separately from the f32 input
hashes; a preview is not the unquantized tensor. IQA maps stay off.

Report all four arms against the common CatmullRom baseline, and pair each
repair arm with its own direct-SR arm. Include mean/median gain, win frequency,
harm frequency, worst loss and Butteraugli changes by JPEG quality band,
category and chroma. Harm uses the existing baseline-quality curve. Show repair
coverage and conditional results for the actually restored inputs, alongside
all-input values; routed-out inputs must retain exact direct-path pixels.

Cross-model winners describe this cohort only. A per-input best-of-models
summary, if shown, is an oracle ceiling rather than an implementable router.
No production policy is fitted or changed by this experiment.

## Gates

Use a separate frozen helper/identity/output directory. Verify input and weight
bytes, finite output geometry, transport hashes and unchanged-path equality.
A bounded pilot precedes the full run. Compare direct DSCF outputs and scalar
scores against the retained earlier study. Independently decode retained PNGs
and match their scored pixel hashes. Keep failed/incomplete attempts separate.
Archive the completed results and source/model identities outside Git.
