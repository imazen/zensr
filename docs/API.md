# Repair entry points

The repository has two repair entry points. They share `zensr-micro` inference,
guards and coefficient projection, but their request types are distinct.

| Caller | Entry point | JPEG ownership |
|---|---|---|
| Applications loading `.zensr` files or raw models | `zensr_zenjpeg::model_file::ModelBundle` and `zensr_zenjpeg::api::Restorer` | `zensr-zenjpeg` parses and decodes JPEG bytes. |
| Codecs supplying decoded samples and coefficients | `zensr::api::Restorer` and `zensr::source::Source` | The codec owns parsing, limits, decoding, color management and orientation. |
| Native zenjpeg/browser preview | Draft zenjpeg `restoration` feature calling `zensr` | Pinned review integration; not on zenjpeg main. |

## Application API

`ModelBundle::from_file` or `from_bytes` validates the container and loads its
weights. `into_restorer` requires a ×1 model; `model()` also exposes separately
loaded ×2/×4 models. Reuse the resulting restorer. See the
[format and loading guide](MODEL_CONTAINER.md) and the runnable
[`repair` example](../crates/zensr-zenjpeg/examples/repair.rs).

`Restorer::new().with_tier(...)` remains available for raw caller-loaded models.
Requests support intent, provenance, budget, explicit routing, consistency and
thread settings. `Outcome::Unchanged` is a normal result. Manual containers
require explicit `Routing::Always` or `Routing::Never`; other weights cannot
silently borrow canon4's fitted router.

## Codec engine

A codec implements `Source` to provide `JpegInfo`, requested decoded RGB8 and
borrowed coefficient views. `EncodedRgb8` (`new`, `new_strided`) includes width,
height and byte stride. Samples are in physical JPEG orientation and coding
color space; ICC conversion, orientation and crop belong to the codec after
repair. `Restorer::restore(&mut source)` executes the existing repair policy.
The optional chooser consumes the existing `zenanalyze-api` feature contract.

### Strided pixel, `zenpixels`, and DCT block views

- **Pixel input (`zensr::source::EncodedRgb8`):** `EncodedRgb8::new` and
  `EncodedRgb8::new_strided` validate packed or row-padded RGB8 slices. Under
  the optional `zenpixels` feature (`zensr/zenpixels`), `EncodedRgb8` bridges
  zero-copy with `zenpixels::PixelSlice` (`from_pixel_slice`, `From`/`TryFrom`),
  tagging outgoing slices with `PixelDescriptor::RGB8` (`TransferFunction::Unknown`)
  because samples remain in JPEG coding RGB prior to ICC conversion.
- **Pixel output (`Pixels` and `PixelsRef`):** `write_rgb8_strided` and
  `write_rgb_f32_strided` write interleaved output directly into caller-owned
  strided slices without touching row padding or allocating on the heap. Under
  `zenpixels`, `write_pixel_slice` writes into typed or type-erased
  `PixelSliceMut` (`U8` or `F32`), and `to_pixel_buffer_rgb8` /
  `to_pixel_buffer_rgb_f32` allocate owned `PixelBuffer`s.
- **Borrowed JPEG DCT blocks (`JpegDctBlocks` and `CoefficientsView`):**
  `zensr::source::JpegDctBlocks` (`{ coeffs: &[i16], blocks_wide, blocks_high, quant_table }`)
  matches the zero-copy 64-element zigzag block layout used by `zenjpeg`
  (`ComponentCoefficients`) and sibling vision crates (`zensally::JpegDctBlocks`).
  `Coefficients::quant_tables` stores DQT tables in 8×8 natural (row-major)
  order; `quant_table_natural` borrows natural order while `quant_table_zigzag`
  permutes into 64-element zigzag order. Codecs can override
  `Source::coefficients_view` (`CoefficientsView`) to return up to four
  component views on the stack without allocating a `Vec<Component>`.
- **Zero-allocation workspaces (`RepairWorkspace` and `ModelWorkspace`):**
  `RepairWorkspace` (`new` or `with_capacity`) holds reusable planar buffers,
  `zensr_micro::ModelWorkspace` tile scratch, `GuardScratch`, `ChromaScratch`,
  router window scratch and report storage. `Request::run_into_workspace(&mut ws)`
  returns a borrowed `WorkspaceOutcome` (`PixelsRef` + `&Report`) with zero
  warm heap allocations on single-threaded execution (aside from external
  `zenanalyze` `OwnedOffer` allocation when default `chooser` `Fidelity` routing
  runs without an explicit `Routing` override); `Request::workspace(&mut ws).run()`
  reuses the same internal scratch while returning an owned `Outcome`.

The extracted engine is on zensr main. The full zenjpeg adapter remains a draft:
all native layouts/row entry points, arbitrary ICC, crop, cancellation and
resource limits have not been qualified. The old `zensr-zenjpeg` path remains
supported; completing its replacement is outside the delivered extraction.
Container loading currently belongs to `zensr-zenjpeg`, not the new engine.
The preview also resolves local and Git copies of `zensr-micro`; their model
types are distinct Cargo identities. Sharing a loaded model across these APIs
requires aligned sources or an explicit bridge.

## Meaning of the result

`Fidelity` is the default request intent; callers may choose `Conservative`.
Neither guarantees perceptual improvement. `Provenance::FreshEncode` expresses
caller knowledge; generation count is not inferred. `Consistency::Certified`
means projection was applied to all three components with calibrated slack.
It does not certify final RGB8 bounds, exact re-encoding or absence of harm.
422/440 chroma projection is unavailable; consistency requests report that.

These are unreleased source APIs. The removed August design proposal and July
API dumps described superseded signatures and are not compatibility contracts.
[Routing](DEFAULT_ROUTING.md), [dependencies](PRODUCTION_DEPENDENCIES.md) and
[readiness](REPAIR_READINESS.md) describe the current implementation and limits.
