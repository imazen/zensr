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
#[must_use]
pub fn estimate_gain(probe: &zenjpeg::detect::JpegProbe) -> f32 {
    // Anchors are measured medians; between them the curve is smooth and
    // monotone, so linear interpolation is honest.
    const G420: [(f32, f32); 9] = [
        (15.0, 5.70),
        (35.0, 3.07),
        (55.0, 1.80),
        (75.0, 0.65),
        (85.0, 0.46),
        (90.0, 0.26),
        (94.0, 0.15),
        (96.0, 0.05),
        (100.0, -0.17),
    ];
    const G444: [(f32, f32); 6] = [
        (90.0, -0.04),
        (92.0, -0.13),
        (94.0, -0.38),
        (96.0, -0.78),
        (98.0, -1.38),
        (100.0, -2.04),
    ];
    // 4:4:4 was only measured from q90 up, where it is the interesting case.
    // Below that it tracks 4:2:0 shifted about 4 points cleaner, which is the
    // measured damage equivalence; extrapolating the negative branch downward
    // would invent a loss that was never observed.
    let full = crate::chroma_full_of(probe) == Some(true);
    let scale = format!("{:?}", probe.quality.scale);
    // Butteraugli distance runs the other way; map it onto the quality axis
    // rather than maintaining a second curve.
    let q = if scale == "ButteraugliDistance" {
        (100.0 - probe.quality.value * 12.0).clamp(1.0, 100.0)
    } else {
        probe.quality.value
    };
    let interp = |pts: &[(f32, f32)]| -> f32 {
        if q <= pts[0].0 {
            return pts[0].1;
        }
        for w in pts.windows(2) {
            let ((x0, y0), (x1, y1)) = (w[0], w[1]);
            if q <= x1 {
                let t = if x1 > x0 { (q - x0) / (x1 - x0) } else { 0.0 };
                return y0 + t * (y1 - y0);
            }
        }
        pts[pts.len() - 1].1
    };
    if full && q >= 90.0 {
        // Measured directly: this is where 4:4:4 turns negative.
        interp(&G444)
    } else if full {
        // Below q90 only 4:2:0 was measured. 4:4:4 at q is about as damaged as
        // 4:2:0 at q+4 (measured: 4:4:4 q90 decodes as cleanly as 4:2:0 q94),
        // so read the 4:2:0 curve 4 points higher. Shifting the anchors down
        // by 4 is the same thing and keeps `interp` reading a sorted table.
        let mut pts = G420;
        for a in &mut pts {
            a.0 -= 4.0;
        }
        interp(&pts)
    } else {
        interp(&G420)
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
        let predicted = estimate_gain(&probe_for_routing);
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
}
