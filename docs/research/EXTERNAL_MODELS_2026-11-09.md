# External restoration candidates — 2026-11-09

Status: availability checks complete; measured evaluation is in preparation.
No runtime or default-model change has been selected.

| Candidate | Verified availability | Evaluation status |
|---|---|---|
| [FreeTransformSR](https://arxiv.org/abs/2609.05912v5) | The paper's [cited repository](https://github.com/HJiLi/FreeTransformSR) returns GitHub API 404; exact-name repository and Hugging Face model searches return no matching release | No author implementation/checkpoint available to run; paper claims are not our measurements |
| [PixelIR](https://arxiv.org/abs/2608.30782) | No implementation/checkpoint link found in the paper; GitHub repository search returns unrelated names and Hugging Face model search returns none | No author implementation/checkpoint available to run; no surrogate implementation is counted as this model |
| [jpeg-restore](https://github.com/stefanasandei/jpeg-restore) | Public MIT code and [MIT-tagged inference weights](https://huggingface.co/asandeistefan/jpeg-restore-meanflows) | Runnable author package selected for testing |

GitHub code revision: `cb68b163e43bf06ee98a6dd6d84e63e0b777def6`.
Author inference-package revision: `33315831b6b342d3b8ddbdc70aa11cb84e55c66b`.
The published checkpoint is 261,352,510 bytes, LFS SHA-256
`fbb573706c64ae90da85e47d1ff3377734264142460770c54ebadc50a4ba4caa`.
The package is used as an external evaluation oracle, not a product dependency
or a source of training targets.

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
settings. No score-based subset, seed, model or routing-policy selection.

Use score-only SSIMULACRA2 and Butteraugli, retaining scalars and output hashes.
Decoded/restored pixels stay in memory or pipes and are dropped after scoring;
no diagnostic maps or image-output sweep is requested. Geometry and resource
probes are bounded separately. Benchmark or profiler results must identify
hardware and timing boundaries and cannot be inferred from paper timings.
