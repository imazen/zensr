# External restoration candidates — 2026-11-09

Status: the full JPEG comparison is complete; separate ×4 SR evaluation is running.
No runtime or default-model change has been selected.

| Candidate | Verified availability | Evaluation status |
|---|---|---|
| [FreeTransformSR](https://arxiv.org/abs/2609.05912v5) | The paper's [cited repository](https://github.com/HJiLi/FreeTransformSR) returns GitHub API 404; exact-name repository and Hugging Face model searches return no matching release | No author implementation/checkpoint available to run; paper claims are not our measurements |
| [PixelIR](https://arxiv.org/abs/2608.30782) | No implementation/checkpoint link found in the paper; GitHub repository search returns unrelated names and Hugging Face model search returns none | No author implementation/checkpoint available to run; no surrogate implementation is counted as this model |
| [jpeg-restore](https://github.com/stefanasandei/jpeg-restore) | Public MIT code and [MIT-tagged inference weights](https://huggingface.co/asandeistefan/jpeg-restore-meanflows) | Both schedules, raw and projected: all 2,960 inputs scored; [results](../../benchmarks/external_jpeg_models_2026-11-09.md) |
| [S3Diff](https://github.com/ArcticHare105/S3Diff) | Public Apache-2.0 code and adapters; SD-Turbo base weights have separate terms | Seven-input ×4 pilot passes; full 697-input comparison running |
| [ResShift](https://github.com/zsyOAOA/ResShift/tree/journal) | Public code and four-step v3/bicubic checkpoints; code has a noncommercial S-Lab license | Seven-input v3 pilot and exact author-sampler parity pass; full ×4 comparison queued in the same sweep |

GitHub code revision: `cb68b163e43bf06ee98a6dd6d84e63e0b777def6`.
Author inference-package revision: `33315831b6b342d3b8ddbdc70aa11cb84e55c66b`.
The published checkpoint is 261,352,510 bytes, LFS SHA-256
`fbb573706c64ae90da85e47d1ff3377734264142460770c54ebadc50a4ba4caa`.
The package is used as an external evaluation oracle, not a product dependency
or a source of training targets.

## ×4 SR comparison

S3Diff source: `6a2e0a47676f2f01fccdfd4e940077d3038d953e`; adapters:
`zhangap/S3Diff` at `41b5e9b6258829d379de50322fb843a97cacba27`;
SD-Turbo: `b261bac6fd2cf515557d5d0707481eafa0485ec2`.
ResShift uses its current default `journal` branch at
`bb03b7d21614cace01787e097c8a6ab6b945227d`, not the older master instructions.
The v2.0 release supplies the four-step real-world v3 and bicubic models and
shared VQ autoencoder. All acquired files have retained URLs, byte sizes and
SHA-256 records outside Git.

The fixed SR set uses the same 17 primary clean references, cropping at most
three pixels from right/bottom to make dimensions divisible by four. Downscale
with zenresize CatmullRom in its default linear-light RGB8 mode, then encode
with zenjpeg at q5–100 every five, 420/444: 680 JPEG cases plus 17 clean inputs.
Compare against CatmullRom upsampling and the retained native NomosUni SPAN ×4
model. This is a web-input fidelity probe, not the authors' RealSR degradation
benchmark or ResShift's MATLAB bicubic benchmark. JPEG ×1 and SR ×4 gains have
different baselines and are never pooled. JPEG coefficient projection is not
applied on the larger HR lattice.

S3Diff retains author defaults, float32 weights, recommended wavelet color fix
and PIL quantization. ResShift retains four-step sampling and author fp16
autocast, with exact sampling parity against unchanged `sample_func` at 128×72
and 65×71 for both checkpoints. Source modules are unchanged; the S3Diff adapter
loads inference leaf modules without importing unrelated training registries.
The tested SR cohort fits single tiles; large-image tile parity and production
GPU integration are not established. Compatibility pins are in
[`requirements-sr.txt`](../../tools/external-models/requirements-sr.txt).

[ResShift's license](https://github.com/zsyOAOA/ResShift/blob/bb03b7d21614cace01787e097c8a6ab6b945227d/LICENSE)
permits noncommercial use and asks commercial users to contact the contributors.
[SD-Turbo's pinned license](https://huggingface.co/stabilityai/sd-turbo/blob/b261bac6fd2cf515557d5d0707481eafa0485ec2/LICENSE.md)
is the Stability AI Community License, including commercial registration and
revenue conditions. These weights are research oracles, not bundled product
assets. Checkpoint quality alone does not settle distribution eligibility.

## Fixed comparison

Reuse the [precision study's](../../benchmarks/quality_int8_2026-11-08.md)
hash-bound JPEG/reference inputs: 2,720 primary cases, 17 category representatives,
four encoders, 420/444 and q5–100 every five. Report the full quality × category
grid, harm curve and worst losses. Each category has only one source image;
repeated encodings do not establish category-wide generalization. The reused
inputs are not a new held-out certification. Secondary size evidence has two
origins and 240 cases at 128/512/1024.

Measure the author's one- and two-step schedules with deterministic, per-input
noise seeds selected before scoring. Report raw model output and the optional
zensr guard/coefficient-projection adapter separately. The trained checkpoint
targets QF5–20; higher-quality bands are explicit out-of-training-range probes.
Compare existing native quality-model and canon4 pipeline controls on matching
input IDs. Bind decoding, RGB8 conversion, model/source hashes and numerical
settings. No score-based subset, seed or checkpoint selection; report both
schedules. Support recommendations are exploratory, not a runtime-default change.

Admission found and corrected a binary16 subnormal-expansion bug in the native
weight loader. After the fix, all 595,459 expanded quality-model weights match
NumPy's conversion byte for byte, and all eight encoder/sampling controls match
the retained native raw/projected RGB8 hashes and SSIM2 scores. Projection also
matches the actual serving pipeline. The exhaustive binary16 regression and
product tests pass; historical runtime measurements are not retroactively
relabelled as measurements of the corrected loader.

Use score-only SSIMULACRA2 and Butteraugli, retaining scalars and output hashes.
Decoded/restored pixels stay in memory or pipes and are dropped after scoring;
no diagnostic maps or image-output sweep is requested. Geometry and resource
probes are bounded separately. Benchmark or profiler results must identify
hardware and timing boundaries and cannot be inferred from paper timings.
