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
borrowed coefficient views. `EncodedRgb8` includes width, height and byte
stride. Samples are in physical JPEG orientation and coding color space;
ICC conversion, orientation and crop belong to the codec after repair.
`Restorer::restore(&mut source)` executes the existing repair policy.
The optional chooser consumes the existing `zenanalyze-api` feature contract.

The extracted engine is on zensr main. The full zenjpeg adapter remains a draft:
all native layouts/row entry points, arbitrary ICC, crop, cancellation and
resource limits have not been qualified. The old `zensr-zenjpeg` path remains
supported; completing its replacement is outside the delivered extraction.
Container loading currently belongs to `zensr-zenjpeg`, not the new engine.

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
