# zensr

JPEG artifact repair and CPU super-resolution in Rust.

JPEG repair returns RGB pixels at the original resolution. The compact canon4
model uses **86,406 bytes of f16 weights**. Routing can leave an image unchanged;
repair uses a tiled CNN followed by JPEG coefficient constraints.

[Models and training](docs/MODELS.md) covers the larger quality models,
realtime retrains, SR models, and how canon4 was trained.

## JPEG repair

Load the model once, then reuse the restorer:

```rust
use zensr_zenjpeg::model_file::ModelBundle;

let restorer = ModelBundle::from_file("canon4.zensr")?.into_restorer()?;

let outcome = restorer.restore(&jpeg_bytes).run()?;
let pixels = outcome.pixels();
let rgb8 = pixels.to_rgb8(); // Interleaved RGB, pixels.width() × pixels.height().
```

[Single-file models](docs/MODEL_CONTAINER.md) bundle weights, routing and repair
settings; larger uncalibrated models require an explicit routing choice.

The [runnable example](crates/zensr-zenjpeg/examples/repair.rs) loads the included
weights (or accepts `--model path.zensr`) and writes raw RGB8, printing the dimensions:

```sh
cargo run --release --locked -p zensr-zenjpeg --example repair -- input.jpg output.rgb8
```

Use the committed lockfile. This is a source integration; the crates are not yet
published as a release. [Dependencies and integration](docs/PRODUCTION_DEPENDENCIES.md).

## Quality and limits

The approved default uses decoded-pixel and coefficient features to decide
whether to repair. On the **512-pixel** comparison cohort it achieved
**+2.417 mean SSIMULACRA2**, **0.566% observed harm**, and **79.75% repair coverage**.
The final policy was checked on a previously inspected cohort; these are not
fresh held-out estimates or per-image guarantees. The worst observed loss was
18.534 SSIM2. [Measurements and harm definition](benchmarks/routing_feature_ablation_2026-11-06.md).

Those routing measurements predate the November 9 weight-loader and November 10
partial-block projection corrections; the full routing cohort has not been
rescored. [Projection fix and qualification](benchmarks/projection_fix_2026-11-10.md).

Arbitrary input resolutions are supported, but quality across resolutions and
end-to-end deployment latency remain unmeasured. Coefficient projection does
not guarantee exact re-encoding or better perceptual quality.
[Routing behavior](docs/DEFAULT_ROUTING.md) · [Readiness](docs/REPAIR_READINESS.md).

## Super-resolution

The CPU runtime also supports ×2 and ×4 models. Upscaling is separate from the
×1 JPEG repair API. Quality and timing coverage varies by model; GPU comparisons
use research adapters.

[Public-model evaluation](docs/research/EXTERNAL_MODELS_2026-11-09.md) covers
MeanFlow JPEG repair and separate S3Diff/ResShift/DSCF ×4 research comparisons.
[Codec-excluded CPU timings](benchmarks/sr_cpu_2026-11-09.md) compare native
SPAN/SPANF, the retained built-in SR models and those external ×4 models.
[Repair followed by SR](benchmarks/sr_cascade_2026-11-09.md) compares direct
SPANF/DSCF with routed canon4 repair before upscaling.
[8K GPU measurements](benchmarks/sr_8k_gpu_2026-11-09.md) cover PyTorch CUDA
latency, VRAM and transfers for SPANF and DSCF.
[Memory experiments](benchmarks/sr_memory_2026-11-09.md) measure integer output
storage and earlier activation release with exact paired output checks.

[Browser workbench](docs/BROWSER_DEMO.md): upload JPEG/PNG locally, compare
SPANF/DSCF ×2 or ×4, optionally repair JPEG artifacts, and download a streamed PNG.
[Browser versus author PyTorch](benchmarks/browser_torch_2026-11-09.md) records
per-category SSIM2 differences and the overview-resolution correction.
CPU tile workers and experimental WebGPU have explicit memory caps. The viewer
supports zoom, pan, flicker and native output tiles; 12 MP inputs are supported.

## Development

```sh
cargo test --locked -p zensr-micro -p zensr-zenjpeg
cargo clippy --locked -p zensr-micro -p zensr-zenjpeg --all-targets -- -D warnings
```

[Documentation](docs/README.md) · [Changelog](CHANGELOG.md).

## License

[AGPL-3.0](LICENSE-AGPL3) or a [commercial license](LICENSE-COMMERCIAL).
