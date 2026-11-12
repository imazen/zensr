# Single-file runtime models

A `.zensr` file holds portable weights, architecture, color space and the JPEG
runtime policy. Load it once; the existing CPU dispatcher unpacks and repacks
weights at load time, then reuses them for requests. There is no Python, JSON
parser, ONNX runtime or model-directory lookup in the inference path. This does
not claim a new inference speedup; CPU kernels and automatic tiling are unchanged.

```rust,no_run
use zensr_zenjpeg::model_file::ModelBundle;

let restorer = ModelBundle::from_file("canon4.zensr")?.into_restorer()?;
let outcome = restorer.restore(&jpeg_bytes).run()?;
let rgb8 = outcome.pixels().to_rgb8();
```

`from_bytes` supports embedded/downloaded bytes. `model()` exposes the loaded
`AdoptedModel`, including separate ×2/×4 super-resolution; `into_restorer()`
requires ×1. The stored budget becomes the request default. Explicit caller
routing, intent, consistency and thread settings remain available.

## Creating a file

The standard-library-only packer reads an adopted model directory (`meta.json`,
plus `weights_f16.raw` when `f16_goldens` is true, otherwise `weights.raw`):

```sh
python3 tools/model_container.py pack models/adopted/dejpeg_rt24i_canon4 canon4.zensr \
  --policy canon4 --budget realtime
python3 tools/model_container.py inspect canon4.zensr
cargo run --release --locked -p zensr-zenjpeg --example repair -- \
  input.jpg output.rgb8 --model canon4.zensr
```

Outputs are created exclusively: an existing file is an error. Put generated
files outside the checkout. The packer includes only the model name and runtime
fields, not training paths, host information or the source metadata's repro block.
Its `inspect` command checks framing/checksum and prints a summary; the Rust
loader performs complete semantic validation.

Three policy modes are explicit:

| Packer policy | Runtime behavior |
|---|---|
| `canon4` | Approved metadata-independent decoded router, plus coefficient fallback and explicit-override calibration. Requires `chooser`. |
| `canon4-coefficients` | Coefficient gain routing and near-pristine gates; loads without `chooser`. This is a different policy from the decoded default. |
| `manual` | No fitted routing claim. JPEG requests must set `Routing::Always` or `Routing::Never`; implicit routing and `Auto` fail. Model-specific near-pristine gating is disabled. |

The built-in canon4 profiles require the exact committed f16 weight hash and
RGB Compact 24/8 ×1 layout. Larger models must use `--policy manual --budget
quality` until their own routing is calibrated. For example:

```sh
python3 tools/model_container.py pack /path/to/dejpeg7_graphics quality.zensr \
  --policy manual --budget quality
cargo run --release --locked -p zensr-zenjpeg --example repair -- \
  input.jpg output.rgb8 --model quality.zensr --routing always
```

`Always` explicitly opts into restoration without a fitted quality gate. It is
not a quality guarantee. Manual files retain their stored generic guard,
deblock and projection settings. ×2/×4 models are loaded for separate SR, not
JPEG repair. Supported architectures are Compact (channels divisible by four,
4–64 channels) and Span48 (48 channels, six blocks, ×2/×4 only), with f16/f32
little-endian weight storage.

[Verified preview files and runtime checks](../benchmarks/model_container_2026-11-08.md)
record exact sizes and upload digests.

## Format version 1

All integers and IEEE floats are little-endian. Readers reject unknown versions,
unsupported pipeline/feature identities, trailing bytes, truncated sections,
non-finite values, invalid dimensions/scales and calibration identity mismatches.
The complete file is limited to 64 MiB, metadata to 4 KiB and policy to 64 KiB.

The 64-byte header is:

| Offset | Value |
|---|---|
| 0 | Eight bytes `ZSRMODEL` |
| 8 | u32 format version, currently 1 |
| 12 | u32 metadata length |
| 16 | u64 policy length |
| 24 | u64 weights length |
| 32 | 32-byte SHA-256 of header bytes 0–31 followed by the complete body |

The body is metadata, policy, then unchanged weight bytes. A string is a u16
byte length followed by UTF-8, without NUL. Metadata contains name, u8
architecture (1 Compact, 2 Span48), u8 color space (0 RGB, 1 YCbCr), u8 scale,
u16 channels, u16 blocks, u8 budget (0 realtime, 1 quality) and u8 weight
encoding (1 f16, 2 f32).

The policy prefix contains, in order:

- u8 mode (0 manual, 1 decoded, 2 coefficients), u32 pipeline ABI (1), u32 tile
  size (0 automatic).
- f32 maximum deblock quality and residual clamp; u8 texture-gate flag; f32
  texture minimum alpha, texture k and roundtrip MAE ceiling.
- Six f32 projection allowances: relative plain-round, mozjpeg, cjpegli,
  unknown; absolute cjpegli, other. Family interpretation follows pipeline ABI 1.
- f32 minimum estimated gain and conservative margin.
- SHA-256 of the exact metadata bytes followed by weight bytes. Calibrated modes
  validate this separately from the whole-file checksum.

Calibrated modes append f32 graphic zero-AC threshold, six f32 near-pristine
gates (IJG420/444, Mozjpeg420/444, distance420/444), and 14 curves. Each curve is
a u16 point count followed by ascending f32 x/y pairs. Curve order is pooled,
graphic, photo for quality420, quality444, distance420 and distance444, then
AQ-off corrections for 420 and 444.

Decoded mode additionally contains the SHA-256 of the frozen feature-definition
source, f64 arrays mean[37], scale[37], gain[47], risk[47], followed by f64 gain
bias, risk bias, risk penalty and decision threshold. Three categorical groups
follow, each a u8 count then strings (encoder, quality scale, subsampling),
totaling ten categories. The stored penalty and threshold participate in every
native-window decision. There are no per-request string allocations to unpack
model coefficients or categories.

Pipeline ABI 1 fixes native analysis windows, feature ordering and qualified
feature identities, rounding, coefficient statistics, gain interpolation,
family mapping and repair/projection algorithms. Changes to those semantics
require a new ABI rather than silently reinterpreting existing files. Ordinary
floating-point execution still depends on the runtime build/CPU and existing
kernel options. A container is data for a compatible runtime, not an executable.
SHA-256 detects corruption and binds calibration; it is not a signature or a
publisher-authentication mechanism.

The original [quality limitations](REPAIR_READINESS.md), provenance and tail
losses still apply. Packaging does not produce new quality evidence or establish
that a model/policy is optimal.
