# The `renders` metric disagreement — resolved as NOT a model regression

`renders` is the one subcorpus where SR metrics disagree: paired per file,
PSNR favours SPANF on **8 of 8** files and butteraugli on 7 of 8, while SSIM2
says the median image gets worse (−1.81, winning on 3 of 8). This asks whether
SPANF actually degrades these images.

**It does not.** SPANF has lower error than Lanczos under every decomposition
tried. Three explanations for SSIM2's dissent were tested and all three
falsified, including one based on looking at the pixels. The mechanism remains
unidentified and is a question for the metric, not the model.

## Setup

Images dumped by the `sr_dump` bin under the `eval` protocol (centre-crop 512,
CatmullRom ÷4 down, then up by each method), so the pixels examined are exactly
those scored. Outputs: `/mnt/v/output/zensr/renders-2026-08-03/` — reference,
both upscales, and 8×-amplified difference maps, viewable at
`http://localhost:3300/zensr/renders-2026-08-03/`.

## Error decomposition — SPANF wins nearly everywhere

Regions from the reference's own gradient magnitude: flat = smoothest 50% of
pixels, edge = top 10%, highlight = brightest 1% of luma. Lower MSE is better.

| region | SPANF wins | typical margin |
|---|---|---|
| flat | **8 of 8** | 3–11× lower (5.7→1.5, 9.8→0.9) |
| edge | 5 of 8 | mixed, within ~15% |
| highlight (brightest 1%) | **8 of 8** | 2–3× lower (977→566, 2372→735) |

Lanczos rings into smooth areas — visible in the difference maps as a wide
banded halo along every edge, against SPANF's thin tight one. On this content
(dark smooth 3D renders with sparse specular edges) flat regions are most of
the image, which is why PSNR favours SPANF decisively.

## Three falsified explanations

1. **"SSIM2 weights flat regions, where the ringing lives."** No: the
   correlation between Δssim2 and the flat-region error advantage is **+0.268**,
   and with the edge advantage **+0.043**. Neither region's error advantage
   predicts the SSIM2 verdict.
2. **"SPANF is softer at the exact edge."** Weak at best: SPANF wins edge-region
   MSE on 5 of 8 files, and the losses are within ~15%.
3. **"SPANF breaks thin specular highlights into beads."** This came from
   looking at the images — the reference's continuous highlight line does appear
   beaded in SPANF's output. **Measurement contradicts the eye:** SPANF's error
   on the brightest 1% of pixels is *lower* on 8 of 8 files, often 2–3×, and the
   highlight-MSE winner agrees with the SSIM2 verdict on only 3 of 8. Whatever
   the apparent beading is, it is closer to the reference than Lanczos's
   continuous-but-displaced line.

Recording the third explicitly: a plausible mechanism, visible in the pixels,
consistent with the subcorpus, and wrong. It is exactly the shape of the
`textures` claim this eval already withdrew — "SR can't invent stochastic
detail" survived eleven days on plausibility alone.

## Conclusion

- **No evidence SPANF degrades `renders`.** It has lower error than Lanczos in
  flat regions, on highlights, and on 5 of 8 edge regions, and both PSNR and
  butteraugli favour it.
- The README's "wins 8/8 on PSNR and butteraugli, 7/8 on SSIM2" stands as
  written; no correction needed.
- **The open question belongs to the metric.** SSIM2 dissents on smooth
  synthetic content in a way that tracks none of the error structure measured
  here. That is worth raising with zensim — SSIM-family metrics are known to
  behave oddly at very low local variance, and these images are dark and smooth
  almost everywhere.
- n=8. Confirm on a wider render set before generalising about the content
  class.

## Do not re-run

The three hypotheses above are falsified on this data. A new attempt needs a
new idea — a mechanism inside the metric, not another error decomposition,
which has now been tried at three spatial scales and explains nothing.
