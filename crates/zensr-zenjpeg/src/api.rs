//! The restoration API.
//!
//! Callers declare *intent* and what they *know about the file*; the library
//! owns policy. Nothing here exposes a research constant, because a caller has
//! no way to reason about one — they cannot know whether a projection slack of
//! 0.15 is right, but they can know whether they just encoded the file.
//!
//! Design and the measurements forcing each decision: `docs/API_DESIGN.md`.

use crate::{restore_jpeg, Projection, RestoreConfig, RestoreError};

/// What the caller is optimising for.
///
/// There is deliberately no `Default`: the honest answer differs per product,
/// and picking one for the caller is how a library ends up making a
/// quality/safety tradeoff nobody chose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Intent {
    /// Maximise measured closeness to the pre-compression original.
    ///
    /// Restores wherever the median gain is positive. Individual files can
    /// still come out worse — see [`Intent::Conservative`] for the numbers.
    Fidelity,
    /// Prefer leaving a file alone when the expected gain is small.
    ///
    /// This reduces regressions; it does **not** eliminate them, and it is not
    /// named `DoNoHarm` for that reason. Measured across the band where the
    /// median gain is still positive at 4:2:0, **30-45% of individual files
    /// are made worse by more than 0.1 ssim2**. No configuration today is
    /// never-worse. If you need that guarantee, do not restore.
    Conservative,
}

/// What the caller knows about where these bytes came from.
///
/// This cannot be measured from the file. Generation detection was
/// implemented and falsified: there is no operating point where a
/// conservative answer exists (zero false-gen1 implies zero gen1 recall), and
/// same-encoder equal-quality recompression is essentially invisible at 9.4%
/// recall. So it is an input, and absence of evidence is never read as
/// "freshly encoded".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Provenance {
    /// Third-party file of unknown history. The safe default.
    Unknown,
    /// This process encoded it, once, from an uncompressed source.
    FreshEncode,
    // `Generations(u8)` was here and has been REMOVED, not deferred.
    //
    // Two separate reasons, and the second is the disqualifying one:
    //
    // 1. Generation count is not recoverable from a file. Detection was
    //    implemented and falsified — no operating point exists where a
    //    conservative answer is possible, and resizing (which CDNs do
    //    constantly) destroys the double-quantisation periodicity it relies
    //    on. So only a caller could supply it, and few can.
    // 2. Nothing validated existed to DO with it. The measurement is "each
    //    generation adds roughly 1-2.5 quantizer steps of excess" — an
    //    ABSOLUTE figure, i.e. `slack_abs`. A first implementation here wired
    //    it into `slack_q`, which is a FRACTION of the quantizer, with an
    //    invented per-generation coefficient. Wrong units and a fabricated
    //    constant.
    //
    // Treating it as `Unknown` would have been safe but decorative: identical
    // behaviour under a different name. Restore the variant when the
    // n -> slack_abs mapping is measured end to end (ROADMAP 1.9).
}

/// Compute budget. The library picks the model; callers should never name one.
///
/// `Adaptive` is deliberately absent in 0.1: it needs a damage estimator, and
/// the obvious candidate — the probe's quality — is not one. A q83 file that
/// was aggressively downscaled carries far more damage than a q83 native file,
/// because downscaling packs content into the frequencies the quantiser
/// coarsens.
///
/// **The tiers are far apart on damaged input and close together on clean
/// input**, which is the opposite of how compute budgets usually behave.
/// Measured per-file median gain on clean references, and the gap expressed
/// in multiples of the ~0.3 ssim2 the metric can actually resolve:
///
/// | input quality | Quality | Realtime | gap | gap vs metric floor |
/// |---|---|---|---|---|
/// | q15 | +11.62 | +6.88 | 4.74 | **15.8x** |
/// | q35 | +6.18 | +4.14 | 2.04 | 6.8x |
/// | q55 | +4.02 | +3.06 | 0.96 | 3.2x |
/// | q75 | +2.05 | +1.58 | 0.47 | **1.6x** |
///
/// So at q75 the 16x compute buys a difference only 1.6x the metric's own
/// resolution — hard to justify. At q15 it buys nearly twice the gain. If you
/// are choosing one tier for mixed traffic, `Realtime` is defensible above
/// roughly q50 and `Quality` earns its cost below it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Budget {
    /// ~0.16 s/MP at 12 threads, 84 KB of weights.
    Realtime,
    /// ~2.7 s/MP, 1.16 MB — 16x the compute. Worth it on heavily damaged
    /// input, marginal above ~q75; see the table above.
    Quality,
}

/// Why a file was returned unmodified.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum SkipReason {
    /// The input is near-pristine for its chroma subsampling, where the model
    /// is measured to lose. The threshold depends on subsampling, not quality
    /// alone: 4:4:4 at q90 is as clean as 4:2:0 at q94.
    NearPristine,
    /// Consistency was required but cannot be certified for this file's
    /// chroma geometry.
    ConsistencyUnavailable(NotCertified),
    /// [`Routing::Auto`] estimated the benefit below the caller's threshold,
    /// or [`Routing::Never`] was set. Carries the estimate so the decision is
    /// auditable rather than opaque.
    NotWorthIt { estimated_gain: f32 },
    /// The colour model is not one restoration understands — 4-component
    /// (CMYK/YCCK) files are decoded and returned untouched.
    UnsupportedColorModel,
}

/// Why the quantisation box could not be certified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum NotCertified {
    /// 4:2:2 and 4:4:0 chroma are not back-projected today.
    ChromaGeometry,
    /// The caller opted out.
    CallerDisabled,
}

/// What can be certified about the output.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Consistency {
    /// Every output coefficient lies inside the interval the file's own
    /// quantisation tables certify, so re-encoding with those tables
    /// reproduces the input's coefficients.
    ///
    /// This is not only a safety property. The projection's measured
    /// contribution *grows* with quality — +0.39 to +1.31 ssim2 from q90 to
    /// q100 at 4:2:0, monotone — so at high quality it is carrying the gain.
    Certified,
    /// Projection was not applied; `why` is specific.
    NotEnforced(NotCertified),
}

/// Chroma subsampling, as a routing input rather than trivia: it decides
/// whether restoration helps at all at a given quality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Chroma {
    /// 4:4:4 — full chroma resolution. Least damaged at a given quality, so
    /// the near-pristine threshold is *lower* here.
    Full,
    /// 4:2:0 — chroma halved in both axes. The common web case.
    HalfBoth,
    /// 4:2:2, 4:4:0, or anything else. Not back-projected today.
    Other,
}

/// Why the library did what it did. Every field answers "why did it do that",
/// which is what was missing during two weeks of measurement.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Report {
    /// Encoder family as the probe names it, from markers and structure.
    ///
    /// Note this describes provenance, not the quantiser: measured over 17,739
    /// files the probe reports `ImageMagick` for 9,626 of which only 180 use
    /// the ImageMagick table. Use `quality` for quantiser behaviour.
    pub encoder_family: String,
    /// Estimated quality on that family's own scale.
    pub quality: f32,
    /// Which scale `quality` is on — IJG quality, or a Butteraugli distance.
    pub quality_scale: String,
    /// Chroma subsampling; see [`Chroma`].
    pub chroma: Chroma,
    /// Whether the coefficient-domain deblock ran.
    pub deblocked: bool,
    /// Fraction of projected coefficients that were clamped, per component.
    pub clamped_fraction: Vec<f32>,
}

/// The result of a restoration request.
///
/// `Unchanged` is a first-class outcome, not an error and not a flag: on
/// high-quality traffic it is the common path, and a CDN can skip re-encoding
/// entirely when it sees one.
#[derive(Debug)]
pub enum Outcome {
    /// Policy determined restoration would not help, or could not be applied
    /// safely. `pixels` is the plain decode.
    Unchanged {
        pixels: Pixels,
        why: SkipReason,
        report: Report,
    },
    /// Restoration applied.
    Restored {
        pixels: Pixels,
        consistency: Consistency,
        report: Report,
    },
}

impl Outcome {
    /// The pixels, whichever arm this is.
    #[must_use]
    pub fn pixels(&self) -> &Pixels {
        match self {
            Outcome::Unchanged { pixels, .. } | Outcome::Restored { pixels, .. } => pixels,
        }
    }

    /// The report, whichever arm this is.
    #[must_use]
    pub fn report(&self) -> &Report {
        match self {
            Outcome::Unchanged { report, .. } | Outcome::Restored { report, .. } => report,
        }
    }
}

/// Decoded pixels, planar f32 in 0..=1, three components.
#[derive(Debug)]
pub struct Pixels {
    planes: Vec<f32>,
    width: usize,
    height: usize,
}

impl Pixels {
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }
    /// Planar f32, three planes of `width * height`, values in 0..=1.
    #[must_use]
    pub fn planes(&self) -> &[f32] {
        &self.planes
    }
    /// Interleaved 8-bit RGB.
    #[must_use]
    pub fn to_rgb8(&self) -> Vec<u8> {
        let plane = self.width * self.height;
        let mut out = vec![0u8; plane * 3];
        for i in 0..plane {
            for c in 0..3 {
                out[i * 3 + c] = (self.planes[c * plane + i] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
        out
    }
}

/// Estimated benefit of restoring a file, in ssim2 points, from header
/// information alone — no decode, no model run.
///
/// Fitted on 4,096 per-file measurements across the clean corpus. Only two
/// features are used because only two carry most of the signal: the probe's
/// quality and the chroma subsampling. Subsampling matters as much as quality
/// does, because 4:4:4 at q90 is as clean as 4:2:0 at q94 and the model loses
/// on clean input.
///
/// This predicts a **median over files at that (quality, subsampling)**, not
/// the outcome for one image. Per-file spread is wide; the estimate is for
/// deciding whether to spend cycles, not for reporting quality.
///
/// **This is the quality-only estimate, and the library's own routing is
/// better than it.** [`Routing::Auto`] additionally classifies the image as
/// graphic or photographic from its coefficients, which is worth about +0.15
/// ssim2 of realized routing quality. That signal needs the compressed bytes,
/// not just a header probe, so it cannot be offered through this signature.
/// Expect `Auto` to make decisions this function would not.
/// Median restore gain by 4:2:0 quality, in ssim2 points.
///
/// Anchors are measured medians; between them the curve is smooth and monotone,
/// so linear interpolation is honest.
///
/// Values at q<=90 come from the original calibration. Values at q>=94 were
/// re-measured on 2026-08-03 against references that are 100% PNG
/// (`benchmarks/clean_ladder_2026-08-03.md`); the originals had been fit partly
/// against references that were themselves JPEGs, which flatters restoration
/// and understated the loss badly up here — the old q100 anchor read -0.17
/// where clean references measure -1.89.
///
/// The q<=90 anchors were re-checked against the same clean run and left alone:
/// they agree to within 0.05 through q75 (5.70/5.66, 3.07/3.07, 1.80/1.75,
/// 0.65/0.64). At q85-90 they read ~0.23 high, but a calibrate/validate split by
/// image puts the uncertainty there at 0.23-0.64, so that gap is not resolvable
/// at 64 images and re-fitting it scored WORSE on held-out images than leaving
/// it. The tail is resolvable: the same split puts uncertainty at q94-100
/// between 0.02 and 0.21.
const G420: [(f32, f32); 10] = [
    (15.0, 5.70),
    (35.0, 3.07),
    (55.0, 1.80),
    (75.0, 0.65),
    (85.0, 0.46),
    (90.0, 0.26),
    (94.0, -0.27),
    (96.0, -0.48),
    (98.0, -1.01),
    (100.0, -1.89),
];

/// Same, for 4:4:4, measured directly from q90 up — the range where it matters
/// and where it turns negative sooner than 4:2:0 does.
const G444: [(f32, f32); 5] = [
    (90.0, -0.24),
    (94.0, -0.62),
    (96.0, -1.10),
    (98.0, -1.89),
    (100.0, -3.10),
];

/// Median restore gain by butteraugli distance, for encoders that quantise on a
/// distance scale (cjpegli and zenjpeg, which share a quant law).
///
/// This used to map distance onto the IJG quality axis with `100 - 12·d` and
/// reuse [`G420`]/[`G444`]. That mapping is not what damage does: measured
/// against clean references on 2026-08-03 it was optimistic in **39 of 40**
/// cells, by a median of +0.68 and up to +4.72 ssim2, and it had the wrong sign
/// in 9 — predicting gain from distance 0.3 to 1.3, where restoration actually
/// costs quality. It breaks worst in the middle: distance 7.2 maps to "quality
/// 13.6" and predicts +5.70 where the truth is +1.31, because a jpegli file at
/// distance 7.2 is far less damaged than an IJG file at q14.
///
/// Held-out (30 calibrate / 34 validate images, split by image), against the
/// mapping it replaces: same realized quality (+0.7940 vs +0.7998 ssim2, a gap
/// an order of magnitude under the metric floor) for **half the work** (restores
/// 30% of cells vs 60%) and **one sixth the harm** (3% of cells made worse vs
/// 17%). Same quality, half the cycles, far less damage.
///
/// Distance ascends with damage, so gain ascends with distance — the opposite
/// direction to the quality curves above.
///
/// Source: `benchmarks/clean_ladder_jpegli_2026-08-03.md`, n=64 per cell,
/// cjpegli and zenjpeg pooled (they agree within 0.1 ssim2 below distance 5).
const DIST420: [(f32, f32); 10] = [
    (0.0, -1.77),
    (0.3, -1.17),
    (0.5, -0.71),
    (0.7, -0.38),
    (1.3, -0.11),
    (1.8, 0.08),
    (3.0, 0.19),
    (5.2, 0.73),
    (7.2, 1.08),
    (14.2, 3.39),
];

/// Gain curves split by content class, IJG quality axis, measured 2026-08-03 on
/// clean references (`benchmarks/content_routing_2026-08-03.md`). Same corpus,
/// grid and method as [`G420`]/[`G444`], which are these two pooled together.
///
/// The split is worth roughly +0.15 ssim2 of realized routing quality on
/// held-out images — an order of magnitude above the metric floor, and by far
/// the largest lever found after quality itself. Graphic content stays worth
/// restoring up to about q96; photographic content stops paying at about q75.
/// Pooling them splits the difference and is wrong for both.
// The q15 anchor is 6.28, a measured median that happens to land near tau.
// Coincidence, not a constant — do not "fix" it by substituting one.
#[allow(clippy::approx_constant)]
const GRAPHIC420: [(f32, f32); 10] = [
    (15.0, 6.28),
    (35.0, 5.81),
    (55.0, 4.75),
    (75.0, 3.77),
    (85.0, 2.58),
    (90.0, 1.93),
    (94.0, 0.94),
    (96.0, 0.35),
    (98.0, -0.15),
    (100.0, -0.33),
];

const GRAPHIC444: [(f32, f32); 10] = [
    (15.0, 6.10),
    (35.0, 5.66),
    (55.0, 4.74),
    (75.0, 3.51),
    (85.0, 2.44),
    (90.0, 1.79),
    (94.0, 0.86),
    (96.0, 0.24),
    (98.0, -0.24),
    (100.0, -0.52),
];

const PHOTO420: [(f32, f32); 10] = [
    (15.0, 3.69),
    (35.0, 1.89),
    (55.0, 0.88),
    (75.0, 0.10),
    (85.0, -0.21),
    (90.0, -0.47),
    (94.0, -0.60),
    (96.0, -1.04),
    (98.0, -1.58),
    (100.0, -2.30),
];

const PHOTO444: [(f32, f32); 10] = [
    (15.0, 4.08),
    (35.0, 1.94),
    (55.0, 0.85),
    (75.0, -0.09),
    (85.0, -0.43),
    (90.0, -0.83),
    (94.0, -1.24),
    (96.0, -1.82),
    (98.0, -2.62),
    (100.0, -3.71),
];

/// Same, for 4:4:4. Full chroma is cleaner at equal distance, so it turns
/// negative sooner — the same ordering the quality curves show.
const DIST444: [(f32, f32); 10] = [
    (0.0, -3.00),
    (0.3, -1.77),
    (0.5, -1.09),
    (0.7, -0.77),
    (1.3, -0.26),
    (1.8, -0.06),
    (3.0, 0.23),
    (5.2, 0.55),
    (7.2, 1.54),
    (14.2, 3.50),
];

#[must_use]
pub fn estimate_gain(probe: &zenjpeg::detect::JpegProbe) -> f32 {
    estimate_gain_for(probe, None)
}

/// Whether an image is synthetic (flat regions, sharp edges, limited palette)
/// or photographic (stochastic detail, grain, stippling).
///
/// This is the single largest signal the router has after quality. Measured
/// 2026-08-03 on clean references: at q75 the median restore gain is +3.77 on
/// graphic content and +0.10 on photographic — a gap of nearly 4 ssim2 that a
/// quality-only curve averages away, restoring photos it should skip and
/// skipping graphics it should restore.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Content {
    Graphic,
    Photographic,
}

/// Fraction of luma blocks with no AC coefficients, above which an image is
/// treated as graphic.
///
/// Selected on the calibrate half of the pinned split by maximising realized
/// ssim2, then reported on the held-out half — 0.25 sits in the middle of a
/// flat 0.20–0.30 plateau, so it is not balanced on a peak. Deliberately NOT
/// zenjpeg's `SCREENSHOT_ZERO_AC_THRESHOLD` (0.10): that one is calibrated for
/// choosing a deblock strategy, a different decision with a different loss.
pub(crate) const GRAPHIC_ZERO_AC_THRESHOLD: f32 = 0.25;

/// Classify from the file's own luma coefficients.
///
/// Chosen over the pixel-domain chooser (`crate::chooser`) on measured value,
/// not on accuracy — the chooser is slightly more accurate and **16x** more
/// expensive. Held-out on the pinned split, as realized routing quality:
///
/// | router | ssim2 | share of oracle-label gain | marginal cost |
/// |---|---|---|---|
/// | quality only | +1.2969 | 0% | 0 |
/// | this, threshold 0.25 | +1.4452 | 82% | **0.29 ms** |
/// | chooser at t=0.75 | +1.4511 | 85% | 4.54 ms |
/// | oracle labels | +1.4781 | 100% | — |
///
/// The 0.006 ssim2 the chooser buys is an order of magnitude below the metric
/// floor, against 4.25 ms — about 11% of a 512-crop realtime restore. Full
/// record: `benchmarks/content_routing_2026-08-03.md`.
pub(crate) fn classify_content(jpeg: &[u8]) -> Option<Content> {
    let coeffs = zenjpeg::decoder::Decoder::new()
        .decode_coefficients(jpeg, enough::Unstoppable)
        .ok()?;
    let luma = coeffs.components.first()?;
    let nblocks = luma.coeffs.len() / 64;
    if nblocks == 0 {
        return None;
    }
    let zero_ac = luma.coeffs[..nblocks * 64]
        .chunks_exact(64)
        .filter(|b| b[1..].iter().all(|&c| c == 0))
        .count();
    Some(
        if zero_ac as f32 / nblocks as f32 >= GRAPHIC_ZERO_AC_THRESHOLD {
            Content::Graphic
        } else {
            Content::Photographic
        },
    )
}

/// [`estimate_gain`] with an optional content class. `None` reproduces the
/// quality-only estimate exactly.
pub(crate) fn estimate_gain_for(
    probe: &zenjpeg::detect::JpegProbe,
    content: Option<Content>,
) -> f32 {
    let full = crate::chroma_full_of(probe) == Some(true);
    let scale = format!("{:?}", probe.quality.scale);
    if scale == "ButteraugliDistance" {
        // The distance curves are not split by content yet — the jpegli ladder
        // measured encoders, not content classes. Falls back to the pooled
        // distance curve rather than guessing a split that was never measured.
        return gain_at_distance(full, probe.quality.value);
    }
    let q = probe.quality.value;
    match content {
        None => gain_at(full, q),
        Some(Content::Graphic) => lerp(if full { &GRAPHIC444 } else { &GRAPHIC420 }, q),
        Some(Content::Photographic) => lerp(if full { &PHOTO444 } else { &PHOTO420 }, q),
    }
}

/// Gain curve for distance-quantised encoders, separated from probing so it can
/// be tested at exact distances.
fn gain_at_distance(full: bool, d: f32) -> f32 {
    lerp(if full { &DIST444 } else { &DIST420 }, d)
}

/// Linear interpolation over an ascending table, clamped at both ends.
fn lerp(pts: &[(f32, f32)], x: f32) -> f32 {
    if x <= pts[0].0 {
        return pts[0].1;
    }
    for w in pts.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if x <= x1 {
            let t = if x1 > x0 { (x - x0) / (x1 - x0) } else { 0.0 };
            return y0 + t * (y1 - y0);
        }
    }
    pts[pts.len() - 1].1
}

/// The gain curve itself, separated from probing so it can be tested at exact
/// qualities.
///
/// Going through a real encode to reach it does not test this: the probe
/// recovers a quality a point or two off what the encoder was asked for, and
/// that slack is indistinguishable from a mistyped anchor. Both are worth
/// testing — this is the half that says what the calibration claims.
fn gain_at(full: bool, q: f32) -> f32 {
    if full && q >= 90.0 {
        // Measured directly: this is where 4:4:4 turns negative.
        lerp(&G444, q)
    } else if full {
        // Below q90 only 4:2:0 was measured. 4:4:4 at q is about as damaged as
        // 4:2:0 at q+4 (measured: 4:4:4 q90 decodes as cleanly as 4:2:0 q94),
        // so read the 4:2:0 curve 4 points higher. Shifting the anchors down
        // by 4 is the same thing and keeps the table sorted.
        let mut pts = G420;
        for a in &mut pts {
            a.0 -= 4.0;
        }
        lerp(&pts, q)
    } else {
        lerp(&G420, q)
    }
}

/// When to spend cycles on restoration.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Routing {
    /// Restore whenever the near-pristine gate allows it.
    Always,
    /// Never restore; decode and return. Useful for A/B and for kill switches.
    Never,
    /// Restore only when [`estimate_gain`] reaches `min_gain` ssim2.
    ///
    /// Held-out measurement (predictor fitted on one half of the images,
    /// evaluated on the other half), per-file averages over the whole set:
    ///
    /// | `min_gain` | files skipped | gain forgone | harm avoided | net |
    /// |---|---|---|---|---|
    /// | 0.00 | 25% | 0.043 | 0.342 | **+0.299** |
    /// | 0.10 | 50% | 0.218 | 0.522 | **+0.304** |
    /// | 0.25 | 72% | 0.348 | 0.655 | **+0.307** |
    /// | 0.50 | 84% | 0.438 | 0.710 | +0.271 |
    /// | 1.00 | 88% | 0.478 | 0.722 | +0.244 |
    ///
    /// Every threshold is **net positive on quality while also saving
    /// cycles** — skipping avoids more harm than the gain it gives up. That is
    /// not a tradeoff curve; it means restoring everything is worse than
    /// restoring selectively on both axes at once.
    ///
    /// `0.25` skips roughly seven files in ten for the best measured net, and
    /// drops the regression rate among the files still processed from 41% to
    /// 23%. Above ~1.0 the net starts falling again as real gains are lost.
    Auto { min_gain: f32 },
}

impl Default for Routing {
    /// [`Routing::Auto`] at 0.25 — the best measured net, and it saves ~72%
    /// of restoration cycles.
    fn default() -> Self {
        Routing::Auto { min_gain: 0.25 }
    }
}

/// A reusable restorer. Build once, use for many images.
///
/// Holds one model per [`Budget`] tier the caller chooses to load. Requesting
/// a tier that was not loaded is an error rather than a silent downgrade —
/// quietly serving 84 KB weights to a caller who asked for the 1.16 MB tier
/// would misreport quality with no way to notice.
pub struct Restorer {
    realtime: Option<zensr_micro::adopted::AdoptedModel>,
    quality: Option<zensr_micro::adopted::AdoptedModel>,
}

impl Restorer {
    /// An empty restorer; add tiers with [`Restorer::with_tier`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            realtime: None,
            quality: None,
        }
    }

    /// Load weights for a tier. Which tier a set of weights belongs to is the
    /// caller's to track; it is not derivable from the bytes.
    #[must_use]
    pub fn with_tier(mut self, tier: Budget, model: zensr_micro::adopted::AdoptedModel) -> Self {
        match tier {
            Budget::Realtime => self.realtime = Some(model),
            Budget::Quality => self.quality = Some(model),
        }
        self
    }

    fn tier(&self, b: Budget) -> Option<&zensr_micro::adopted::AdoptedModel> {
        match b {
            Budget::Realtime => self.realtime.as_ref(),
            Budget::Quality => self.quality.as_ref(),
        }
    }

    /// Begin a restoration request.
    #[must_use]
    pub fn restore<'a>(&'a self, jpeg: &'a [u8]) -> Request<'a> {
        Request {
            restorer: self,
            jpeg,
            intent: Intent::Fidelity,
            provenance: Provenance::Unknown,
            budget: Budget::Realtime,
            routing: Routing::default(),
            require_consistency: false,
            threads: 0,
        }
    }
}

/// A pending restoration request.
pub struct Request<'a> {
    restorer: &'a Restorer,
    jpeg: &'a [u8],
    intent: Intent,
    provenance: Provenance,
    budget: Budget,
    routing: Routing,
    require_consistency: bool,
    threads: usize,
}

impl Default for Restorer {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Request<'a> {
    /// What to optimise for. Defaults to [`Intent::Fidelity`].
    #[must_use]
    pub fn intent(mut self, intent: Intent) -> Self {
        self.intent = intent;
        self
    }

    /// What is known about the file's history. Defaults to
    /// [`Provenance::Unknown`], which is the safe reading.
    #[must_use]
    pub fn provenance(mut self, p: Provenance) -> Self {
        self.provenance = p;
        self
    }

    /// Compute budget. Defaults to [`Budget::Realtime`].
    #[must_use]
    pub fn budget(mut self, b: Budget) -> Self {
        self.budget = b;
        self
    }

    /// Return [`Outcome::Unchanged`] rather than emit pixels whose
    /// quantisation box cannot be certified. Some pipelines would rather do
    /// nothing than emit uncertifiable output; that is a policy only the
    /// caller can set.
    #[must_use]
    pub fn require_consistency(mut self, on: bool) -> Self {
        self.require_consistency = on;
        self
    }

    /// When to spend cycles at all. Defaults to [`Routing::Auto`] at 0.25,
    /// which held-out measurement shows skips ~72% of files while *improving*
    /// average quality — skipping avoids more harm than the gain it forgoes.
    #[must_use]
    pub fn routing(mut self, r: Routing) -> Self {
        self.routing = r;
        self
    }

    /// Worker threads; 0 means choose automatically.
    #[must_use]
    pub fn threads(mut self, n: usize) -> Self {
        self.threads = n;
        self
    }

    /// Run it.
    pub fn run(self) -> Result<Outcome, RestoreError> {
        let model = self
            .restorer
            .tier(self.budget)
            .ok_or(RestoreError::TierNotLoaded)?;

        // Routing decides before any model work. The probe is cheap (header
        // only), so a skip here costs a decode and nothing more.
        let probe_for_routing =
            zenjpeg::detect::probe(self.jpeg).map_err(|e| RestoreError::Probe(format!("{e:?}")))?;
        // Content class comes from the file's own luma coefficients, and is
        // worth ~+0.15 ssim2 of realized routing quality — the largest lever
        // after quality itself. It costs a coefficient parse (~0.29 ms on a
        // 512 crop) even on files that then skip, which is why it is computed
        // only when the decision can actually turn on it. `None` falls back to
        // the pooled curve, so an unparseable file routes exactly as before.
        let content = match self.routing {
            Routing::Auto { .. } => classify_content(self.jpeg),
            _ => None,
        };
        let predicted = estimate_gain_for(&probe_for_routing, content);
        let skip = match self.routing {
            Routing::Always => false,
            Routing::Never => true,
            Routing::Auto { min_gain } => predicted < min_gain,
        };

        // Slack, gate thresholds and deblock policy are derived from the probe
        // and the caller's declarations — never surfaced.
        let mut cfg = RestoreConfig::default()
            .with_threads(self.threads)
            .with_high_q_identity(true)
            .with_projection(Projection::Auto);

        // Conservative gates earlier. That is the only mechanism measured to
        // reduce per-file regressions; the margin is deliberately modest
        // because harm_frac falls gradually rather than off a cliff.
        cfg = cfg.with_high_q_margin(match self.intent {
            Intent::Fidelity => 0.0,
            Intent::Conservative => 6.0,
        });

        // Provenance currently affects only whether the strict box is
        // available. `FreshEncode` is the one claim that unlocks anything, and
        // even that is left to the family calibration today rather than
        // hard-coded here.
        let _ = self.provenance;

        // A routed-out file still gets a faithful decode; the caller needs
        // pixels either way, and Unchanged is a first-class outcome.
        if skip {
            cfg = cfg.with_projection(Projection::Off);
        }
        let r = restore_jpeg(self.jpeg, model, &cfg)?;
        let chroma = match r.report.chroma_full {
            Some(true) => Chroma::Full,
            Some(false) => Chroma::HalfBoth,
            None => Chroma::Other,
        };
        let report = Report {
            encoder_family: r.report.encoder_family.clone(),
            quality: r.report.est_quality,
            quality_scale: r.report.quality_scale.clone(),
            chroma,
            deblocked: r.report.used_deblock_auto,
            clamped_fraction: r.report.projection.iter().map(|p| p.clamped_frac).collect(),
        };
        let pixels = Pixels {
            planes: r.planes,
            width: r.width,
            height: r.height,
        };
        if skip {
            return Ok(Outcome::Unchanged {
                pixels,
                why: SkipReason::NotWorthIt {
                    estimated_gain: predicted,
                },
                report,
            });
        }
        if r.report.skipped_model_high_q {
            return Ok(Outcome::Unchanged {
                pixels,
                why: SkipReason::NearPristine,
                report,
            });
        }
        let consistency = if report.clamped_fraction.is_empty() {
            Consistency::NotEnforced(NotCertified::ChromaGeometry)
        } else {
            Consistency::Certified
        };
        if self.require_consistency {
            if let Consistency::NotEnforced(why) = consistency {
                return Ok(Outcome::Unchanged {
                    pixels,
                    why: SkipReason::ConsistencyUnavailable(why),
                    report,
                });
            }
        }
        Ok(Outcome::Restored {
            pixels,
            consistency,
            report,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(w: usize, h: usize) -> Vec<u8> {
        let mut v = vec![0u8; 3 * w * h];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                v[i] = ((x * 7 + y * 3) % 256) as u8;
                v[i + 1] = ((x * 2 + y * 11) % 256) as u8;
                v[i + 2] = ((x * 13 + (y * y) % 97) % 256) as u8;
            }
        }
        v
    }

    fn jpeg_at(q: f32, ss: zenjpeg::encoder::ChromaSubsampling) -> Vec<u8> {
        let (w, h) = (64usize, 64usize);
        let rgb = synth(w, h);
        let px: &[rgb::Rgb<u8>] = bytemuck::cast_slice(&rgb);
        zenjpeg::encoder::EncoderConfig::ycbcr(q, ss)
            .encode(px, w as u32, h as u32)
            .expect("encode")
    }

    /// Asking for a tier that was never loaded must fail loudly. Silently
    /// serving the other tier would misreport quality with no way to notice.
    #[test]
    fn requesting_an_unloaded_tier_is_an_error() {
        let r = Restorer::new();
        let jpg = jpeg_at(70.0, zenjpeg::encoder::ChromaSubsampling::Quarter);
        let out = r.restore(&jpg).budget(Budget::Quality).run();
        assert!(
            matches!(out, Err(RestoreError::TierNotLoaded)),
            "expected TierNotLoaded, got {out:?}"
        );
    }

    /// Intent must actually change behaviour. It was briefly accepted and
    /// ignored, which is worse than not offering it — a caller would ask for
    /// caution and silently get none.
    #[test]
    fn conservative_intent_gates_strictly_more_than_fidelity() {
        use zenjpeg::detect::probe;
        // Sweep the band either side of the 4:2:0 threshold and count how many
        // qualities each intent would skip.
        let (mut fid, mut cons) = (0usize, 0usize);
        for q in [80.0f32, 85.0, 88.0, 90.0, 92.0, 94.0, 96.0] {
            let jpg = jpeg_at(q, zenjpeg::encoder::ChromaSubsampling::Quarter);
            let p = probe(&jpg).expect("probe");
            if crate::policy_high_q_identity_with_margin(&p, 0.0) {
                fid += 1;
            }
            if crate::policy_high_q_identity_with_margin(&p, 6.0) {
                cons += 1;
            }
        }
        assert!(
            cons > fid,
            "conservative skipped {cons} and fidelity {fid}; conservative must \
             gate strictly more or the intent is decorative"
        );
    }

    /// The gate margin must move the threshold in the right direction on BOTH
    /// scales. IJG quality rises with fidelity while Butteraugli distance
    /// falls, so a sign error would make a conservative caller restore MORE.
    #[test]
    fn gate_margin_direction_is_correct_on_both_scales() {
        for (q, ss) in [
            (92.0f32, zenjpeg::encoder::ChromaSubsampling::Quarter),
            (92.0, zenjpeg::encoder::ChromaSubsampling::None),
        ] {
            let jpg = jpeg_at(q, ss);
            let p = zenjpeg::detect::probe(&jpg).expect("probe");
            let strict = crate::policy_high_q_identity_with_margin(&p, 6.0);
            let loose = crate::policy_high_q_identity_with_margin(&p, 0.0);
            assert!(
                strict || !loose,
                "a positive margin must never gate LESS ({:?} at {q})",
                p.quality.scale
            );
        }
    }

    /// Chroma subsampling must reach the report. It decides whether
    /// restoration helps at all at a given quality, so a caller that cannot
    /// see it cannot reason about the outcome. Tested at the mapping rather
    /// than end-to-end so it does not depend on weight files being present.
    #[test]
    fn chroma_mapping_distinguishes_the_cases_policy_depends_on() {
        use zenjpeg::encoder::ChromaSubsampling as Cs;
        for (ss, want) in [
            (Cs::None, Some(true)),
            (Cs::Quarter, Some(false)),
            (Cs::HalfHorizontal, None), // 4:2:2 — not back-projected
        ] {
            let jpg = jpeg_at(70.0, ss);
            let p = zenjpeg::detect::probe(&jpg).expect("probe");
            assert_eq!(
                crate::chroma_full_of(&p),
                want,
                "{ss:?} mapped wrongly (probe saw {:?})",
                p.subsampling
            );
        }
    }

    /// Above q94 restoration LOSES quality against a clean reference, in both
    /// subsamplings. Until 2026-08-03 the curve predicted a gain there (+0.15
    /// at 4:2:0 q94, where clean references measure -0.27), because it had been
    /// fit partly against references that were themselves JPEGs.
    ///
    /// That sign error survived the whole suite, so it is pinned here: a
    /// prediction of gain on near-pristine input is the one error that makes
    /// the library burn cycles to make an image worse.
    #[test]
    fn gain_curve_is_negative_on_near_pristine_input() {
        for full in [false, true] {
            for q in [94.0f32, 96.0, 98.0, 100.0] {
                let g = gain_at(full, q);
                assert!(
                    g < 0.0,
                    "full_chroma={full} q{q} predicts {g:+.3} — restoring \
                     near-pristine input costs quality, so a positive estimate \
                     spends cycles to make the image worse"
                );
            }
        }
    }

    /// More damage never predicts less gain. Linear interpolation between
    /// anchors preserves monotonicity, so a violation means an anchor was
    /// mistyped or pasted out of order — the failure mode of a hand-edited
    /// calibration table, and the reason the anchors are swept at 0.5 steps
    /// here rather than only at the anchor points themselves.
    #[test]
    fn gain_curve_never_increases_with_quality() {
        for full in [false, true] {
            let mut prev = f32::INFINITY;
            let mut qi = 10.0f32;
            while qi <= 100.0 {
                let g = gain_at(full, qi);
                assert!(
                    g <= prev + 1e-4,
                    "full_chroma={full} q{qi}: gain rose to {g:+.3} from {prev:+.3}"
                );
                prev = g;
                qi += 0.5;
            }
        }
    }

    /// 4:4:4 is cleaner than 4:2:0 at the same nominal quality, so it always
    /// has less to gain from restoration. If this inverts, the two curves have
    /// been swapped.
    #[test]
    fn full_chroma_never_gains_more_than_subsampled() {
        let mut qi = 10.0f32;
        while qi <= 100.0 {
            assert!(
                gain_at(true, qi) <= gain_at(false, qi) + 1e-4,
                "q{qi}: 4:4:4 {:+.3} > 4:2:0 {:+.3}",
                gain_at(true, qi),
                gain_at(false, qi)
            );
            qi += 0.5;
        }
    }

    /// Distance-quantised encoders get their own curve, and it must not repeat
    /// the sign errors the `100 - 12·d` mapping made. Measured against clean
    /// references, restoration costs quality from distance 0.0 to about 1.3
    /// (cjpegli -q 100 down to -q 90); the old mapping predicted gain there.
    #[test]
    fn distance_curve_is_negative_on_near_pristine_input() {
        for full in [false, true] {
            for d in [0.0f32, 0.3, 0.5, 0.7, 1.3] {
                let g = gain_at_distance(full, d);
                assert!(
                    g < 0.0,
                    "full_chroma={full} distance {d} predicts {g:+.3} — near-pristine \
                     jpegli input loses quality when restored"
                );
            }
        }
    }

    /// Gain rises with distance, because distance rises with damage. This is the
    /// opposite direction to the quality curves, which is exactly why the two
    /// must not share a table.
    #[test]
    fn distance_curve_never_decreases_with_distance() {
        for full in [false, true] {
            let mut prev = f32::NEG_INFINITY;
            let mut d = 0.0f32;
            while d <= 16.0 {
                let g = gain_at_distance(full, d);
                assert!(
                    g >= prev - 1e-4,
                    "full_chroma={full} distance {d}: gain fell to {g:+.3} from {prev:+.3}"
                );
                prev = g;
                d += 0.1;
            }
        }
    }

    /// A cjpegli file must reach the distance curve, not the quality curve. If
    /// the scale match regresses, the raw distance (0.0..~15) is read as a
    /// quality and every near-pristine file is treated as maximally damaged.
    #[test]
    fn cjpegli_probe_routes_to_the_distance_curve() {
        use zenjpeg::encoder::ChromaSubsampling as Cs;
        // zenjpeg quantises on distance and probes as ButteraugliDistance, so
        // its own output exercises the branch without needing cjpegli present.
        for ss in [Cs::None, Cs::Quarter] {
            let jpg = jpeg_at(100.0, ss);
            let p = zenjpeg::detect::probe(&jpg).expect("probe");
            if format!("{:?}", p.quality.scale) != "ButteraugliDistance" {
                continue; // encoder built without the distance path
            }
            let full = crate::chroma_full_of(&p) == Some(true);
            let g = estimate_gain(&p);
            assert_eq!(
                g,
                gain_at_distance(full, p.quality.value),
                "{ss:?}: estimate_gain did not use the distance curve"
            );
            assert!(g < 0.0, "{ss:?} pristine predicted {g:+.3}");
        }
    }

    /// Graphic content is worth restoring further up the quality range than
    /// photographic content — that ordering IS the routing signal, and if it
    /// ever inverts the two curve pairs have been swapped.
    #[test]
    fn graphic_content_always_predicts_more_gain_than_photographic() {
        for full in [false, true] {
            let mut q = 10.0f32;
            while q <= 100.0 {
                let (g, p) = (
                    lerp(if full { &GRAPHIC444 } else { &GRAPHIC420 }, q),
                    lerp(if full { &PHOTO444 } else { &PHOTO420 }, q),
                );
                assert!(
                    g > p,
                    "full_chroma={full} q{q}: graphic {g:+.3} <= photographic {p:+.3}"
                );
                q += 0.5;
            }
        }
    }

    /// Both content curves fall as quality rises, for the same reason the
    /// pooled curve does: less damage leaves less to recover.
    #[test]
    fn content_curves_never_increase_with_quality() {
        for pts in [&GRAPHIC420, &GRAPHIC444, &PHOTO420, &PHOTO444] {
            let mut prev = f32::INFINITY;
            let mut q = 10.0f32;
            while q <= 100.0 {
                let g = lerp(pts, q);
                assert!(
                    g <= prev + 1e-4,
                    "q{q}: gain rose to {g:+.3} from {prev:+.3}"
                );
                prev = g;
                q += 0.5;
            }
        }
    }

    /// Passing no content class must reproduce the quality-only estimate
    /// exactly. A caller who cannot classify has to get the old behaviour, not
    /// a silently different one.
    #[test]
    fn absent_content_reproduces_the_pooled_estimate() {
        use zenjpeg::encoder::ChromaSubsampling as Cs;
        for ss in [Cs::None, Cs::Quarter] {
            for q in [20.0f32, 50.0, 75.0, 90.0, 100.0] {
                let jpg = jpeg_at(q, ss);
                let p = zenjpeg::detect::probe(&jpg).expect("probe");
                assert_eq!(estimate_gain_for(&p, None), estimate_gain(&p));
            }
        }
    }

    /// The classifier must run on real files without panicking and must return
    /// a class for ordinary baseline JPEGs. A synthetic gradient is
    /// photographic-ish by construction (dense AC), which also checks the
    /// threshold is not so low that everything reads as graphic.
    #[test]
    fn content_classifier_handles_real_files() {
        use zenjpeg::encoder::ChromaSubsampling as Cs;
        for q in [30.0f32, 75.0, 95.0] {
            let jpg = jpeg_at(q, Cs::Quarter);
            assert_eq!(
                classify_content(&jpg),
                Some(Content::Photographic),
                "q{q}: synthetic noise field should not classify as graphic"
            );
        }
        // Truncated input must decline to classify rather than panic; routing
        // then falls back to the pooled curve.
        assert_eq!(classify_content(&[0xFF, 0xD8, 0xFF]), None);
        assert_eq!(classify_content(&[]), None);
    }

    /// A flat image is nearly all zero-AC blocks, so it must read as graphic.
    /// This is the positive half of the classifier check.
    #[test]
    fn flat_content_classifies_as_graphic() {
        let (w, h) = (64usize, 64usize);
        let mut rgb = vec![200u8; 3 * w * h];
        // A few hard edges, as a UI screenshot would have — still overwhelmingly
        // flat, which is exactly the signal the threshold keys on.
        for y in 20..24 {
            for x in 0..w {
                let i = (y * w + x) * 3;
                rgb[i] = 20;
                rgb[i + 1] = 20;
                rgb[i + 2] = 20;
            }
        }
        let px: &[rgb::Rgb<u8>] = bytemuck::cast_slice(&rgb);
        let jpg = zenjpeg::encoder::EncoderConfig::ycbcr(
            85.0,
            zenjpeg::encoder::ChromaSubsampling::Quarter,
        )
        .encode(px, w as u32, h as u32)
        .expect("encode");
        assert_eq!(classify_content(&jpg), Some(Content::Graphic));
    }

    /// The probe path reaches the curve. A real q100 encode has enough margin
    /// over probe-inversion slack that this stays true regardless of how many
    /// points the recovered quality is off by.
    #[test]
    fn probe_path_reaches_the_curve_on_a_pristine_encode() {
        use zenjpeg::encoder::ChromaSubsampling as Cs;
        for ss in [Cs::None, Cs::Quarter] {
            let jpg = jpeg_at(100.0, ss);
            let p = zenjpeg::detect::probe(&jpg).expect("probe");
            let g = estimate_gain(&p);
            assert!(g < 0.0, "{ss:?} q100 predicted {g:+.3}");
        }
    }
}
