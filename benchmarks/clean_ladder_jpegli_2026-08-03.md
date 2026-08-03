# jpegli / zenjpeg restore ladder on clean references — 2026-08-03

The distance-quantised encoder family had **never been measured on clean
references**. The 2026-08-03 ladder covered libjpeg-turbo and mozjpeg only, so
everything the router did with a cjpegli or zenjpeg file rested on an
unvalidated conversion from butteraugli distance onto the IJG quality axis.

It was wrong, and it was wrong in the expensive direction.

## Provenance

Harness `dejpeg_eval` at `8697384`, host `lilith`, model `dejpeg_rt24g`.
Corpus `/mnt/v/imazen-26-clean` (974 PNG refs, 0 JPEG-sourced), the 64 pinned
eval files, gate disabled. Grid: `{cjpegli, zenjpeg} × {4:2:0, 4:4:4} ×
q{15,35,55,75,85,90,94,96,98,100}` = 2,560 paired cells, 7,681 rows, all
`gt_src=png`. Raw: `~/tmp/clean_jpegli.tsv` (see the pointer file).

Both encoders probe as encoder family `CjpegliYcbcr` on the
`ButteraugliDistance` quality scale. Encoded quality maps to probed distance as:

| -q | 15 | 35 | 55 | 75 | 85 | 90 | 94 | 96 | 98 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| distance | 14.2 | 7.2 | 5.2 | 3.0 | 1.8 | 1.3 | 0.7 | 0.5 | 0.3 | 0.0 |

## 1. The `100 − 12·d` mapping was optimistic in 39 of 40 cells

The old code converted distance to a pseudo-quality and read the IJG curves.
Against clean measurement:

| enc | ss | -q | dist | q_eff | estimated | measured | error |
|---|---|---|---|---|---|---|---|
| cjpegli | 420 | 35 | 7.2 | 13.6 | +5.70 | +1.31 | **+4.39** |
| cjpegli | 420 | 55 | 5.2 | 37.6 | +2.90 | +0.81 | +2.09 |
| cjpegli | 420 | 75 | 3.0 | 64.0 | +1.28 | +0.19 | +1.10 |
| cjpegli | 420 | 90 | 1.3 | 84.4 | +0.47 | **−0.09** | +0.56 |
| cjpegli | 420 | 94 | 0.7 | 91.6 | +0.05 | **−0.36** | +0.41 |
| zenjpeg | 420 | 35 | 7.2 | 13.6 | +5.70 | +0.98 | **+4.72** |
| zenjpeg | 420 | 100 | 0.0 | 100.0 | −1.89 | −1.71 | −0.18 |

Median error **+0.68**, mean **+1.27**, and the sign is wrong in **9 of 40**
cells — all in the distance 0.3–1.3 band, where the estimator promised gain and
restoration actually costs quality.

The mid-range is where it breaks worst. Distance 7.2 maps to "quality 13.6",
so the estimator treats a cjpegli `-q 35` file as nearly the most damaged input
it has ever seen and predicts +5.70; the truth is +1.31. A jpegli file at
distance 7.2 is enormously better than an IJG file at q14 — that is the entire
point of the encoder — and a linear inversion cannot express it.

## 2. Fix: a curve keyed on distance directly

Fitted on 30 calibrate images, validated on the 34 held out:

| rule | mean ssim2 | restored | harmed |
|---|---|---|---|
| always restore | +0.0312 | 1.00 | 0.49 |
| **shipped: IJG curve via 100−12·d** | **+0.7998** | 0.60 | 0.17 |
| **measured distance curve** | **+0.7940** | **0.30** | **0.03** |
| per-image oracle (ceiling) | +1.0795 | 0.51 | 0.00 |

Realized quality is a tie — the 0.006 gap is an order of magnitude below the
~0.3 ssim2 metric floor. The difference is everywhere else: the measured curve
does **half the work** and harms **one sixth as many images** (3% of cells vs
17%).

The old rule bought its score by restoring almost everything and winning big on
the few heavily damaged files, while quietly degrading 17% of the corpus. For
web images — where cycles cost money and handing back a worse image is the one
unacceptable outcome — that is a bad trade at equal quality.

Shipped as `DIST420` / `DIST444` in `crates/zensr-zenjpeg/src/api.rs`. Final
anchors are refit on all 64 images (the held-out test validated the *form*);
cjpegli and zenjpeg are pooled because they share a quant law and agree within
0.1 ssim2 at every distance below 5.2.

## 3. Distance-quantised encoders leave less to repair

At matched nominal quality, gains run well below libjpeg-turbo's:

| -q | turbo 4:2:0 | mozjpeg 4:2:0 | cjpegli 4:2:0 | zenjpeg 4:2:0 |
|---|---|---|---|---|
| 15 | +6.68 | +4.26 | +4.02 | +3.15 |
| 35 | +3.35 | +2.73 | +1.31 | +0.98 |
| 55 | +1.74 | +1.76 | +0.81 | +0.51 |
| 75 | +0.82 | +0.46 | +0.19 | +0.19 |

Same ordering as the turbo-vs-mozjpeg effect and for the same reason: a better
encoder leaves less damage to recover. zenjpeg — our own encoder — is the least
improvable of the four, which is the outcome to want.

## Open

- Only one generation. Multi-generation jpegli was not measured.
- The distance→quality table above is from this corpus at 512-crop; a different
  content mix will move the probe distances, though the curve is keyed on the
  probed distance rather than on `-q`, so that is handled.
- `CjpegliXyb` (the XYB colour path) is untested — it is a separate family and
  may not share this curve.
