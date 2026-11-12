# Retained SR blend evaluator

An unpublished experiment tool. It scores mixtures of retained SPANF/DSCF RGB8
outputs and zenresize upscales against the original HR pixels. No inference,
per-candidate image decoding or IQA-map generation occurs in the scoring loop.

`prepare.py` copies hash-verified originals, LR sources and existing lossless SR
PNGs into a new artifact directory. The frozen subset has 17 sources, one per
category, at clean/q5/q20/q50/q90, with JPEG 420. Both models use direct SR;
JPEG repair is off. This is exploratory data, not an untouched holdout.

The evaluator loads each original, LR image and both SR PNGs once per case.
It precomputes zensim's reference and reuses scoring scratch. Each zenresize
filter is generated once per scale and reused for all intensities/models.
Four cases may run concurrently in the caller's bounded Rayon pool. Results
contain scalar scores and pixel hashes. `PROGRESS.json`, `IDENTITY.json`, and
`COMPLETE.json` record coverage, executable identity and result hashes.

The grid is ×2/×4, Triangle/Mitchell/CatmullRom/Lanczos2 and SR intensity
0/25/50/75/100%. Resizing and blending use encoded sRGB float samples; only the
completed mixture is rounded/clamped to RGB8. SR starts from retained RGB8,
so these measurements do not establish a blend before the original SR float
quantization. ×2 downsamples stored ×4 SR with Box; its HR reference uses
linear-light CatmullRom. The target construction is disclosed rather than
claimed independent of the resize method.

Metrics:

- **PSNR RGB**, encoded RGB8 dB, from unchanged zenmetrics source modules.
  Exact matches remain infinity (`"inf"` in JSON); no finite cap is substituted.
- **zensim-A**, explicitly pinned to the public repo's adopted revision-7 A
  model at `6060039d8c0ace71d47c722ea76233bb2d745a22`. This is distinct from the
  old research checkout's deprecated legacy A. No default profile alias is used.

## Source reuse

PSNR currently lives inside zenmetrics-cli's private classical module. To use
its actual implementation without bringing the complete codec/GPU CLI graph
into this tool, `src/vendor/classical.rs` and
`src/vendor/classical/psnr_kernels.rs` are unchanged copies from
[imazen/zenmetrics](https://github.com/imazen/zenmetrics/tree/a3329089d95c1d0bd546dfc63213b220c03c6e82).
Their existing tests are retained. The test-only `studio601_gray` helper is
copied from that revision's metrics module. No sibling repository is modified.
The code retains the destination's AGPL/commercial license; it is Imazen code.

The SIMD dependency matches zenmetrics' pinned archmage revision
`da0ed5deca40ac60df32dcf4f6e7602658b2f959`; zensim's separate stable SIMD
runtime keeps its own declared dependency. No SIMD types cross that boundary.

## Run

Use the root `sr-blend-prepare`, `sr-blend-build`, `sr-blend-test` and
`sr-blend-evaluate` recipes. Set `TMPDIR` to disk-backed scratch, and run builds
and scoring through the workspace resource wrapper. Inputs/outputs must live
outside Git. An existing output directory is an error; partial results remain
available if scoring fails. No missing input is silently skipped.
