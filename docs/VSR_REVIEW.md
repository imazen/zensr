# Intel Video Super Resolution source review — 2026-11-05

Reviewed by the requested Claude Opus 5.5 lane at xhigh effort, with findings
checked against source by Codex. Upstream revision:
[`ea37e77bcf705a5f8ffeab56b83424bed6ea2461`](https://github.com/OpenVisualCloud/Video-Super-Resolution-Library/tree/ea37e77bcf705a5f8ffeab56b83424bed6ea2461).
This was a source review; no upstream benchmarks or zensr quality experiments
were run. No upstream code, weights or dependencies were adopted.

## Useful ideas

Upstream applies RAISR filters independently to each frame. It hashes a local
luma gradient tensor into angle, strength and coherence buckets, then selects
an 11×11 filter. It also blends the refined result back toward the cheap
upscale according to local census disagreement. See
[hash selection](https://github.com/OpenVisualCloud/Video-Super-Resolution-Library/blob/ea37e77bcf705a5f8ffeab56b83424bed6ea2461/Library/Raisr.cpp#L814-L876),
[census blend](https://github.com/OpenVisualCloud/Video-Super-Resolution-Library/blob/ea37e77bcf705a5f8ffeab56b83424bed6ea2461/Library/Raisr_AVX256.cpp#L68-L166)
and [per-frame integration](https://github.com/OpenVisualCloud/Video-Super-Resolution-Library/blob/ea37e77bcf705a5f8ffeab56b83424bed6ea2461/ffmpeg/vf_raisr.c#L285-L311).

The most relevant experiments for ×1 JPEG restoration are:

1. **Local structure safeguards.** Compare the existing 16-pixel-cell Laplacian
   texture guard with a structure-tensor coherence guard and a quantization-aware
   local disagreement measure. Thin lines and stochastic texture need different
   treatment. The known low-quality line-plot loss is a diagnostic case, not an
   untouched evaluation set. A changed guard changes the router's outcome labels;
   adoption requires refitting on development data and fresh held-out evaluation.
   Census disagreement alone can reject intended deblocking, because repairing a
   flat block changes local ordering. That is a hypothesis requiring measurement.
2. **Pre-JPEG resize diversity.** Upstream's
   [training notes](https://github.com/OpenVisualCloud/Video-Super-Resolution-Library/blob/ea37e77bcf705a5f8ffeab56b83424bed6ea2461/filters_2x/filters_highres/filternotes.txt)
   mix area, bicubic, nearest and Lanczos degradation. Our data generator imports
   native crop generation from `tools/make_dejpeg_data2.py`. Test a paired retrain
   using our resize and encoder implementations before JPEG encoding, retaining
   pristine fidelity targets. Upstream's sharpened targets are not our objective.
3. **JPEG block-phase conditioning.** Explore absolute position modulo eight as
   extra input to a separately trained model. This differs from damage/severity
   conditioning and would need an explicit model-format change. Tile coordinates
   must preserve absolute phase, with whole/tiled pixel parity as a required gate.

These are proposals, not measured gains or reasons to replace canon4 today.

## Independent critique of native-window routing

The ×1 compact CNN is fully convolutional and its tiled path has bit-exact
coverage. That establishes arbitrary-size execution, not constant perceptual
quality across resolutions. Artifact block size is fixed in pixels; content,
border frequency and image-wide guards still vary with geometry.

The current five-window extension keeps the native luma grid and avoids resizing
away block damage. The reviewer recommended retaining the measured 512-only
default until quality evidence exists elsewhere. The owner explicitly requested
arbitrary-resolution operation; the runtime extension implements it, with these
unresolved quality limitations:

- The model was fitted on whole 512-longest-edge images, not crops from larger
  images. Small images also extrapolate outside that longest-edge constraint.
- Requiring all sampled windows to pass is an uncalibrated aggregation rule.
  The number of windows changes with geometry, and fixed positions miss regions.
- Global file bpp includes metadata and may not describe a particular window.
  Changing this feature only at runtime would break the frozen model contract;
  any replacement must be matched in development features and fitting.
- The reviewer identified that eight-pixel starts could change 420 chroma MCU
  phase. Starts now align to 16 pixels for both 420 and 444.
- Disabling the old high-quality gate preserves the measured 512 operating point,
  but its effect at other sizes has not been measured.
- The reviewed implementation repeated pixel decoding and coefficient parsing.
  The implemented follow-up below shares compatible data; bounded window count
  is still not a measured end-to-end latency bound.

The existing retained native-crop score grid and completed development size-sweep
cells could support a development diagnostic without regenerating restoration
pixels. Check origin overlap and exact model/pipeline provenance first. Neither
source is a substitute for fresh held-out resolution evidence. No cancelled
artifact audit was resumed for this review.

## Source and reuse boundaries

The implementation's [license](https://github.com/OpenVisualCloud/Video-Super-Resolution-Library/blob/ea37e77bcf705a5f8ffeab56b83424bed6ea2461/LICENSE)
is BSD-3-Clause; the earlier PLAN grouping as license-blocked was inaccurate.
This establishes the source license, not a conclusion about other rights.
The review did not establish independent quality gains, weight reproducibility
or suitability of its IPP dependency, approximate hashing, fast-math build
settings and border fallback for zensr's exact-output contracts.

## Follow-up assessment — 2026-11-06

Source inspection at `816a43ce7b2f` confirms the opportunities below. Its remote
CI completed successfully. This follow-up ran no quality or speed experiment;
priorities express engineering judgment, not estimated gains.

| Priority | Improvement | Current evidence | Acceptance condition |
|---|---|---|---|
| 1 | Reuse parsed coefficients and compatible decoded pixels | `api.rs::run` classifies from coefficients; `decoded_router::for_jpeg` decodes pixels and coefficients again; `restore_jpeg_routed` decodes pixels and parses coefficients for projection | Preserve decisions, output pixels and reports; profile and measure before claiming a speedup |
| 2 | Make bpp describe the primary image, independently of metadata | `decoded_router.rs` uses `data.len()*8/(w*h)` | Matched feature extraction and refit; adding non-pixel APP/COM payload must not change the new feature or decision |
| 3 | Select windows using spatial damage and structure | `windows()` selects only center/corners; coefficient extraction already examines the luma block map | Compare position-only and structure/damage-aware selection at the same window budget; fit aggregation on development data and report gain, harm, severity and coverage by size |
| 4 | Evaluate local structure safeguards with a true unguarded control | `dejpeg_eval.rs::run_x1` always calls `guarded_merge`, including `model_off`; the texture map uses a 16-pixel-cell mean absolute Laplacian | Isolate clamp, texture gate and fallback effects; evaluate a coherence-aware candidate on final projected output; refit routing if adopted |
| 5 | Diversify pre-JPEG resizing and smooth-region training | `make_dejpeg_data4.py` imports native `gen_crops`; that generator rejects grayscale standard deviation below eight | Paired training with matched origins/seeds and our resize/encoder path; report native/resized and smooth/texture/line strata separately |
| 6 | Test absolute JPEG block-phase conditioning | Current compact graph accepts three pixel channels; previous rejected conditioning supplied damage/severity, not absolute phase | Separate model experiment; no quality claim without matched evaluation and exact whole/tiled output |

Decode reuse must respect policy differences: routing sees deblocking Off,
while restoration may use Auto. Reuse those pixels only when the selected
restoration decode is identical. A rejected file can reuse its plain decode.
Coefficient reuse is independent of that pixel-deblocking choice. Keep the
public API unchanged; pass an internal prepared-input context through the call
chain rather than introducing a second restoration implementation.

For spatial routing, high AC energy alone is insufficient: useful candidates
include both structured edges/lines and regions with unusually erased AC.
Keep a representative region as well. Blank margins can be useful evidence of
smooth content, so blindly dropping every flat region is also unjustified.
Compare the current minimum-window-score rule with a calibrated aggregation
that accounts for window count and evidence. A local gate may retain gains in
safe regions instead of rejecting an entire mixed-content image, but it changes
the restored output and therefore requires new outcome labels.

Additional controls matter:

- At ×1, the guard's round-trip operation reduces to output-versus-input mean
  absolute deviation. It is not independent evidence that a correction is wrong.
  Its image-wide mean can dilute a localized change. Compare a local version
  jointly with the texture/clamp controls; do not assume every large residual
  is damage.
- The exported selection deliberately has `domain_guard = false`
  (`publication_routing_export_linear.py` checks it). Absence of range abstention
  is not a missing implementation of that frozen choice. Small-image handling,
  fitted ranges and high-quality gating should be compared as candidate policies.
- Clamping dimensions into the fitted range invents a different feature vector.
  Prefer matched native-window development inputs when refitting; retain the
  measured 512 policy as a comparison.
- Metadata-only additions provide a strict feature-invariance control when the
  image and encode record are unchanged. White-margin additions change content
  and geometry: measure their effect, rather than imposing decision invariance
  as a correctness guarantee.
- The retained native-crop grid is a useful diagnostic, but its 12,096 cells are
  the combined grid, not all untouched validation data. Its old harm tables use
  a loss threshold of 0.1; recompute using the owner's quality-dependent curve
  before comparing them to the publication frontier. Check origin overlap with
  router fitting inputs before labeling any subset held out.

The next engineering change with the clearest compatibility gate is internal
decode/coefficient reuse. The next quality comparison should pair corrected
bpp and spatial-window selection on existing development artifacts. Preserve
arbitrary-resolution support and the original frozen 512 results throughout.

## Implemented follow-ups — 2026-11-06

The matched delivered-RGB8 projection diagnostic is implemented. See
[its measured four-case results](../benchmarks/projection_diagnostic_2026-11-06.md).
It reproduces the known line-plot loss unchanged with projection Off/On: that
repair already lies inside the chosen bounds. This motivates local safeguards;
coefficient consistency alone does not establish perceptual safety.

The request now shares parsed coefficients across content classification,
learned routing and projection. Plain Off decoding is shared across routing and
restoration/skip. Auto deblocking remains a separate decode and releases the
cached Off buffer. A new parity regression checks odd dimensions, 420/444,
low/high qualities, deblock Off/Auto, and routed skips using actual canon4;
floating-point output bits, RGB8 and reports must match fresh decoding.
The existing independent full prediction and retained-JPEG replays remain gates.
Profiling captures were recorded before and after, but no controlled deployment
latency comparison has been completed; no speedup is claimed.

Metadata-independent features, spatial selection/refitting, new guards and
training changes remain experiments; none was silently applied to the frozen
model or its recorded quality estimates.

## Feature and spatial follow-up — 2026-11-06

The [measured feature ablation](../benchmarks/routing_feature_ablation_2026-11-06.md)
reproduces the original baseline fit exactly, then removes bpp and optionally
width/height. The 34-input candidate avoids direct dependence on those three
fields and has native prediction/replay and APP15 invariance checks. It trades
slightly lower gain and slightly higher observed harm on the reused reserved
cohort for metadata stability. The owner approved adopting it as the default
on 2026-11-06. All frozen baseline expectations remain preserved.

A test-only structure/AC selector samples an off-center synthetic structure
missed by fixed windows. It changes scores but no decisions in the bounded
eight-cell native-size diagnostic. That is insufficient evidence to replace
production window selection. No spatial quality improvement is claimed.
