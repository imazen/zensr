# zensr

JPEG artifact repair and CPU super-resolution in Rust.

JPEG repair returns RGB pixels at the original resolution. The compact canon4
model uses **86,406 bytes of f16 weights**. Routing can leave an image unchanged;
repair uses a tiled CNN followed by JPEG coefficient constraints.

## JPEG repair

Load the model once, then reuse the restorer:

```rust
use zensr_micro::{adopted::AdoptedModel, decode_all_f16};
use zensr_zenjpeg::api::{Budget, Restorer};

let weights = decode_all_f16(&weights_f16);
let model = AdoptedModel::load_compact(&weights, 24, 8, 1)?;
let restorer = Restorer::new().with_tier(Budget::Realtime, model);

let outcome = restorer.restore(&jpeg_bytes).run()?;
let pixels = outcome.pixels();
let rgb8 = pixels.to_rgb8(); // Interleaved RGB, pixels.width() × pixels.height().
```

The [runnable example](crates/zensr-zenjpeg/examples/repair.rs) loads the included
weights and writes raw RGB8, printing the dimensions:

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

Arbitrary input resolutions are supported, but quality across resolutions and
end-to-end deployment latency remain unmeasured. Coefficient projection does
not guarantee exact re-encoding or better perceptual quality.
[Routing behavior](docs/DEFAULT_ROUTING.md) · [Readiness](docs/REPAIR_READINESS.md).

## Super-resolution

The CPU runtime also supports ×2 and ×4 models. Upscaling is separate from the
×1 JPEG repair API. General SR quality results are historical; there is no
production GPU backend.

## Development

```sh
cargo test --locked -p zensr-micro -p zensr-zenjpeg
cargo clippy --locked -p zensr-micro -p zensr-zenjpeg --all-targets -- -D warnings
```

[Documentation](docs/README.md) · [Changelog](CHANGELOG.md).

## License

[AGPL-3.0](LICENSE-AGPL3) or a [commercial license](LICENSE-COMMERCIAL).
