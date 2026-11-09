# Codec-excluded ×4 CPU inference protocol — 2026-11-09

Measure complete synchronous SR calls from decoded, planar RGB f32 in memory.
Exclude file reads, model/checkpoint loading, weight packing, tract planning,
RNG reseeding, output hashing/destruction, codecs and IQA. Include inference,
scratch/output allocation, default native tiling/worker dispatch, and each
external model's required resize/conditioning/color fix. S3Diff retains its
author PIL quantization and wavelet color fix; native models return raw f32.
No optional native guard is applied.

## Frozen matrix

Primary: 64×64 → 256×256 and 128×128 → 512×512, each with requested thread
counts 1 and 4. Twelve configurations:

- Native NomosUni SPAN ×4.
- Native SPANF with its retained f16-weight/f32-bias artifact.
- Native Compact general-x4v3, general-WDN-x4v3 and animevideo-x4v3.
- Retained EFDN baseline, NanoSR, TSR and SPANF ONNX exports through tract.
- S3Diff, ResShift real v3 and ResShift bicubic through CPU PyTorch.

Supporting native matrix: the first nine configurations at 256×256 →
1024×1024, requested threads 1 and 4. These sizes are measured separately;
no speed is extrapolated to unmeasured image sizes.

All arithmetic is f32. Native adopted weights are expanded from retained f16;
tract runs the retained f32 ONNX graphs. ResShift CPU disables the GPU study's
fp16 autocast, so this is an explicitly different precision configuration.
No INT8, torch.compile, BF16 or CPU-specific quantization is enabled.

The inputs use source 7109 from the existing SR cohort. The smaller cases use
the clean 128×128 LR buffer (center crop for 64). The 256 case samples every
second pixel from its retained clean 512×512 reference. These are timing
fixtures, not new quality/degradation evidence. Input hashes are frozen.

## Placement, work and correctness

Host: i265, Intel Core Ultra 7 265K. Zenbench resolves performance cores 0–7
and verifies an enforced uniform cpuset. The benchmark uses that OS boundary,
nice 19, idle I/O priority and a 14 GiB hard memory limit with no swap. The
user-scope launcher did not expose an enabled cpuset controller; the measured
run therefore uses a system scope, validated by zenbench before timing.
No unrelated workload is stopped or reprioritized.

Each matrix uses zenbench's external-process Iterate protocol: three fixed,
fresh-process blocks; three internal latency samples per invocation; automatic
A/A clone of the native-SPAN baseline. Two full seeded calls first verify
repeatability and warm the process. Native tiled paths additionally verify
exact output equality against one requested thread. Tract/PyTorch verify their
own repeated outputs; this does not assert cross-backend/thread equivalence.
The comparison check binds input identity and output geometry across different
trained models, whose output values are intentionally different.

The primary plan requires 156 invocations; the larger native plan requires 60.
These are planned counts, not runtime estimates. Pilot attempts remain
separate and are not pooled into the reported matrix. Results are descriptive
CPU observations, not a Confirm acceptance, a quality ranking or a production
baseline promotion. Retain all samples and resource observations without
outlier trimming or selecting the fastest process.

The report requires complete three-block coverage, preserved stdout/stderr
hashes, finite positive timing vectors and matching fresh-process output
hashes. It checks the placement fields and resolved topology: an enforced
cpuset wholly within the highest-ranked core class. Diagnostic prose that
merely contains a placement status is not evidence of enforcement.

Native SPANF's default core tile is 128. Its 64/128 cases use one tile, so a
request for four threads still executes one tile worker. The 256 case supplies
four tiles. Adopted models retain their existing adaptive default tiling;
tract/PyTorch use their explicitly configured internal pools. Report requested
threads and these saturation limits rather than implying every model uses four
busy cores at every size.

## CPU adaptation and artifacts

The external snapshots retain their original bytes. For S3Diff CPU imports,
the test adapter changes only `.cuda()` conversions and exact `cuda` device
literals in three author modules; CUDA visibility is empty. Both external
adapters check CPU placement and f32 parameter dtype. Successful pilots cover
the full CPU calls, finite output geometry and seeded exact repeatability.
This does not add a product CPU backend or revalidate GPU/subjective quality.

Python: Torch 2.15.0+cu132 in CPU mode, one inter-op thread, oneDNN enabled.
Tract: 0.23.10. Native code uses ordinary runtime SIMD dispatch, without
`target-cpu=native`. Model/source/input bytes and resolved launch settings are
bound in the zenbench plans and packets. The venv interpreter is passed as an
argument to `/usr/bin/time -v`, preserving its environment while retaining
per-worker RSS. Model outputs are hashed in RAM and discarded.

Artifact root on i265: `/mnt/v/output/zensr/sr-cpu-20261109/`.
Source tools: `tools/external-models/cpu_plan.py`, `cpu_worker.py`, `sr_models.py`
and `tools/quality-int8/runtime/src/bin/cpu_sr.rs`. Reproduction entry points:
`just external-sr-cpu-build`, `just external-sr-cpu-plan`, then the retained
zenbench pair plan inside the recorded CPU boundary. Full stdout/stderr,
per-process samples, checkpoints and compressed packets remain outside Git.
