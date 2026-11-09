# Decoded-feature default

The owner selected the metadata-independent 34-input linear router on
2026-11-06. At its frozen 0.5% development harm target, the secondary comparison
on the reused 512-pixel reserved cohort gives +2.4173 origin-balanced mean SSIM2
and 0.5662% observed harm on 290 references / 269 origins, restoring 79.75% of
cells. The owner accepted that tradeoff for metadata stability. This is not a
fresh untouched validation or a harm guarantee. The image model is canon4;
these measurements do not calibrate arbitrary weights. See the
[feature comparison](../benchmarks/routing_feature_ablation_2026-11-06.md).
The original 37-input policy and its published 0.5490% result remain preserved
as the [historical baseline](../benchmarks/publication_primary_reserved_2026-11-05.md).

These results predate the November 9 binary16 subnormal-loader correction
(`a01962d6`). The study executable expanded canon4 weights through that loader.
The corrected decoder is exhaustively verified and product tests pass, but the
full routing cohort has not been rescored with corrected weights. Numeric
predictor-vector replay alone does not revalidate delivered-pixel quality.

`api::Restorer::restore(...).run()` uses this policy for default Fidelity requests
on three-component 420/444 JPEGs at arbitrary resolutions whose shortest edge
is at least eight pixels. Unsupported layouts and explicit caller routing keep
the existing rules. Conservative intent also retains its existing policy. The
public API signatures and `Routing::Auto` threshold behavior are unchanged.
`Routing::default()` remains the explicit legacy Auto configuration; omitting
the request override selects the learned default. The lower-level `restore_jpeg`
function remains available for explicitly configured restoration.

The native predictor exports the development-frozen `no_bpp_geometry` ridge gain head and
logistic harm head, fitted transforms and categorical vocabularies as Rust
floating-point constants. It restores when `predicted_gain - 32 * predicted_risk
>= 0`. It does not additionally apply the old high-quality gate, which would
change the selected operating point. Explicit consistency requirements continue
to apply. A rejected image returns the plain decoded pixels.

The heads use 34 numeric features and three categorical fields: file bpp and
image dimensions were removed before refitting. The native compatibility vector
keeps 37 slots with zero coefficients for those three fields. Pixel features use
the plain decoder output and original analysis request;
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
phase. Coefficient statistics come from the corresponding native luma blocks; an axis spanning
the whole image retains its MCU padding. Pixel features come from those same
windows. The predictor receives locally estimated legacy gain and native
coefficient/pixel features. Window dimensions and whole-file bpp occupy compatibility slots
but have zero weight in both heads. Restoration requires every sampled
window's gain-minus-risk score to pass. Restoration itself still covers the
whole image at original dimensions, with the existing exact-halo tiled runtime.

All 140,940 independent numeric replay vectors match for both the new policy
and the historical baseline; retained JPEG replays also match feature inference
and both selected pixel paths. Native-size feature tests exercise sizes through
4096×2160, extreme aspect ratios and flat images. API output tests exercise odd
dimensions, sizes above 512, explicit overrides and exact selected-path pixels.
These checks establish execution and parity, not a quality frontier at new sizes.

**Limits:** the frozen model was fitted on whole images resized to a 512-pixel
longest edge before JPEG encoding. Native windows, small images and a minimum
across multiple predictions change that distribution. Removing direct dimension
inputs does not establish quality generalization. Five positions can miss
localized damage. The high-quality identity gate remains disabled for learned
routing at all sizes; its benefit or harm outside 512 is unmeasured. Neither the
0.5662% observed harm nor +2.4173 mean gain is established at other sizes.
The [VSR review](VSR_REVIEW.md) records independent critique and improvement
candidates without treating them as measured improvements.

The `chooser` feature now defaults on to supply zenanalyze's pixel analysis. Its
historical graphics classifier is not used by the new linear router. Builds with
`--no-default-features` retain the coefficient/quality policy. Use the committed
lockfile: the workspace's unpublished dependency patches are described in the
[dependency chain](PRODUCTION_DEPENDENCIES.md).

Reproduction tools:

- `tools/routing_candidate_export.py` exports the approved 34-input constants
  and independent prediction vectors. The original exporter preserves the baseline.
- `just decoded-router-candidate-replay <vectors.tsv>` replays the current policy.
- `just decoded-router-metadata-sensitivity <jpeg-vectors.tsv> <report.tsv>` verifies
  active prediction invariance and records historical metadata sensitivity.
- `just decoded-router-replay <vectors.tsv>` replays all 140,940 reserved decisions.
- `just decoded-router-jpeg-replay <vectors-with-jpeg-paths.tsv>` exercises decoder,
  coefficient and pixel features on retained JPEGs, including near-boundary cases.
- Built-in small goldens exercise numeric inference without external data.
- API tests compare default output pixels with the selected plain/restored path
  and verify explicit Never and Auto overrides.

The Python artifact is 5,492 bytes; that is not a Rust deployment-size measurement.
The image weights are separate, 86,406 bytes. Pixel feature extraction adds work
before the restoration decision; the request now shares coefficient data and
reuses plain decoded pixels whenever the restoration decode mode is Off. Auto
deblocking retains a distinct pixel decode. Deployment latency has not been
confirmed by a valid runtime comparison. The worst reserved loss remains 18.5344 SSIM2.
