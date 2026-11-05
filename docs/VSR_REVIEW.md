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
- Analysis can repeat pixel decoding and coefficient parsing before restoration.
  Bounded window count is not a measured end-to-end latency bound.

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
