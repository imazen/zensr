# Decoded-feature default

The owner selected `decoded/budget-0.005` after reviewing its reserved scatter.
The name records a 0.5% development selection target. Reserved observed harm is
0.5490%, with +2.4260 origin-balanced mean SSIM2, on 290 references / 269 origins.
The image model is canon4; these measurements do not calibrate arbitrary weights.
Full results and the cancelled artifact-audit limitation are in the
[reserved report](../benchmarks/publication_primary_reserved_2026-11-05.md).

`api::Restorer::restore(...).run()` uses this policy for default Fidelity requests
on three-component 420/444 JPEGs whose longest edge is 512 pixels and shortest
edge is at least eight pixels. Other geometries and explicit caller routing keep
the existing rules. Conservative intent also retains its existing policy. The
public API signatures and `Routing::Auto` threshold behavior are unchanged.
`Routing::default()` remains the explicit legacy Auto configuration; omitting
the request override selects the learned default. The lower-level `restore_jpeg`
function remains available for explicitly configured restoration.

The native predictor exports the frozen `decoded-linear-1` ridge gain head and
logistic harm head, fitted transforms and categorical vocabularies as Rust
floating-point constants. It restores when `predicted_gain - 32 * predicted_risk
>= 0`. It does not additionally apply the old high-quality gate, which would
change the selected operating point. Explicit consistency requirements continue
to apply. A rejected image returns the plain decoded pixels.

Inputs are the same 37 numeric features and three categorical fields as the
study. Pixel features use the plain decoder output and original analysis request;
coefficient features use the shared frozen kernel. Twelve-decimal serialization
of numeric study inputs is reproduced. Feature-definition identities are checked
once per process; a mismatch is an explicit unsupported-model error. The fitted
policy has no reference-image score or post-restoration outcome as an input.

The `chooser` feature now defaults on to supply zenanalyze's pixel analysis. Its
historical graphics classifier is not used by the new linear router. Builds with
`--no-default-features` retain the coefficient/quality policy. Use the committed
lockfile: the workspace's unpublished dependency patches are described in the
[dependency chain](PRODUCTION_DEPENDENCIES.md).

Reproduction tools:

- `tools/publication_routing_export_linear.py` exports constants and independent
  sklearn prediction/decision vectors from the frozen selection.
- `just decoded-router-replay <vectors.tsv>` replays all 140,940 reserved decisions.
- `just decoded-router-jpeg-replay <vectors-with-jpeg-paths.tsv>` exercises decoder,
  coefficient and pixel features on retained JPEGs, including near-boundary cases.
- Built-in small goldens exercise numeric inference without external data.
- API tests compare default output pixels with the selected plain/restored path
  and verify explicit Never and Auto overrides.

The Python artifact is 5,660 bytes; that is not a Rust deployment-size measurement.
The image weights are separate, 86,406 bytes. Pixel feature extraction adds work
before the restoration decision; deployment latency has not been confirmed by a
valid runtime comparison. The worst reserved loss remains 18.5344 SSIM2.
