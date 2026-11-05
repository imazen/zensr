# Decoded-feature default

The owner selected `decoded/budget-0.005` after reviewing its reserved scatter.
The name records a 0.5% development selection target. Reserved observed harm is
0.5490%, with +2.4260 origin-balanced mean SSIM2, on 290 references / 269 origins.
The image model is canon4; these measurements do not calibrate arbitrary weights.
Full results and the cancelled artifact-audit limitation are in the
[reserved report](../benchmarks/publication_primary_reserved_2026-11-05.md).

`api::Restorer::restore(...).run()` uses this policy for default Fidelity requests
on three-component 420/444 JPEGs at arbitrary resolutions whose shortest edge
is at least eight pixels. Unsupported layouts and explicit caller routing keep
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

## Native-resolution analysis

The restoration CNN is fully convolutional: its ×1 output has the input's
width and height. JPEG block damage has a fixed pixel scale. Resizing a large
image to 512 for routing would change that damage, so routing never resizes.

For images fitting inside 512×512, the router analyzes the whole image. Larger
images use up to five deduplicated windows: center and four corners, each at
most 512×512. Starts align to 16 pixels, preserving both luma DCT and 420 MCU
phase. Coefficient
statistics come from the corresponding native luma blocks; an axis spanning
the whole image retains its MCU padding. Pixel features come from those same
windows. The predictor receives window dimensions, locally estimated legacy
gain, and the original file's bits per pixel. Restoration requires every sampled
window's gain-minus-risk score to pass. Restoration itself still covers the
whole image at original dimensions, with the existing exact-halo tiled runtime.

The original 512-pixel path is unchanged. All 140,940 independent numeric replay
vectors match; retained JPEG replays also match feature inference and both
selected pixel paths. Native-size feature tests exercise sizes through
4096×2160, extreme aspect ratios and flat images. API output tests exercise odd
dimensions, sizes above 512, explicit overrides and exact selected-path pixels.
These checks establish execution and parity, not a quality frontier at new sizes.

**Limits:** the frozen model was fitted on whole images resized to a 512-pixel
longest edge before JPEG encoding. Native windows, small images and a minimum
across multiple predictions change that distribution. Five positions can miss
localized damage. Whole-file bits per pixel include metadata and need not
represent each window. The high-quality identity gate remains disabled for learned
routing at all sizes; its benefit or harm outside 512 is unmeasured. Neither the
0.5490% observed harm nor +2.4260 mean gain is established at other sizes.
The [VSR review](VSR_REVIEW.md) records independent critique and improvement
candidates without treating them as measured improvements.

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
