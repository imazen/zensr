//! The restoration API.
//!
//! Callers declare *intent* and what they *know about the file*; the library
//! owns policy. Nothing here exposes a research constant, because a caller has
//! no way to reason about one — they cannot know whether a projection slack of
//! 0.15 is right, but they can know whether they just encoded the file.
//!
//! Design and the measurements forcing each decision: `docs/API_DESIGN.md`.

use crate::calibration::*;
use crate::{Projection, RestoreConfig, RestoreError};

/// What the caller is optimising for.
///
/// The enum has no `Default` implementation. Requests start with Fidelity and
/// the routing policy selected for this crate; callers can override the intent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Intent {
    /// Maximise measured closeness to the pre-compression original.
    ///
    /// Default requests use decoded-feature routing over native-resolution
    /// analysis windows. Explicit overrides retain coefficient/quality routing.
    /// Individual files can still come out worse; this is not a guarantee.
    Fidelity,
    /// Prefer leaving a file alone when the expected gain is small.
    ///
    /// Uses the existing coefficient/quality policy with an earlier identity
    /// gate. This reduces coverage compared with that policy's Fidelity mode;
    /// it does not promise lower harm than the decoded-feature default.
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

/// Select a caller-loaded model tier. The library does not load weights itself.
/// Current routing calibration targets canon4; other model weights need their
/// own quality measurements. Runtime cost depends on the selected model and host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Budget {
    /// Compact tier; canon4 uses 86,406 bytes of f16 weights.
    Realtime,
    /// The caller's quality-oriented model.
    Quality,
}

/// Why a file was returned unmodified.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum SkipReason {
    /// The input is near-pristine for its chroma subsampling, where the model
    /// is measured to lose frequently. Thresholds depend on the corrected
    /// probe family, quality scale, and subsampling.
    NearPristine,
    /// Consistency was required but cannot be certified for this file's
    /// chroma geometry.
    ConsistencyUnavailable(NotCertified),
    /// The default policy declined restoration, [`Routing::Auto`] estimated
    /// benefit below the caller's threshold, or [`Routing::Never`] was set.
    /// Carries the policy score: risk-adjusted gain for the decoded default,
    /// or the legacy estimated gain for an explicit Auto override.
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
    /// Projection was applied to all three YCbCr components using the
    /// family-calibrated slack. This is not a bit-exact re-encoding guarantee:
    /// slack expands the quantisation intervals, and RGB conversion and
    /// rounding can move coefficients after projection.
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

/// Estimated projected restoration gain for the canon4 realtime model, in
/// SSIM2 points. Uses the record-corrected header probe without model inference.
///
/// Curves are monotone fits to calibration-image medians, not guarantees for
/// individual files. Auto routing also measures the luma zero-AC fraction.
/// See `benchmarks/ship_rt24i_canon4_2026-11-04.md` for held-out results and
/// the quality/work/harm tradeoff. Other models require separate calibration.
#[must_use]
pub fn estimate_gain(probe: &zenjpeg::detect::JpegProbe) -> f32 {
    let mut corrected = probe.clone();
    crate::record::apply_record(&mut corrected);
    estimate_gain_for(&corrected, None)
}

/// Coefficient-domain content class used by the calibrated routing curves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Content {
    Graphic,
    Photographic,
}

/// Classify from luma zero-AC fraction, with a threshold selected on the
/// calibration images only. Missing coefficients fall back to pooled curves.
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
        .as_chunks::<64>()
        .0
        .iter()
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
        // The AQ term is zero without a verified record (record.rs).
        return gain_at_distance(full, probe.quality.value, content)
            + crate::record::aq_off_gain_delta(probe);
    }
    if scale != "IjgQuality" && scale != "MozjpegQuality" {
        // An uncalibrated scale is not an IJG quality dial.
        return 0.0;
    }
    let q = probe.quality.value;
    match content {
        None => gain_at(full, q),
        Some(Content::Graphic) => lerp(if full { GRAPHIC444 } else { GRAPHIC420 }, q),
        Some(Content::Photographic) => lerp(if full { PHOTO444 } else { PHOTO420 }, q),
    }
}

/// Gain curve for distance-quantised encoders, separated from probing so it can
/// be tested at exact distances.
fn gain_at_distance(full: bool, d: f32, content: Option<Content>) -> f32 {
    let pts = match (content, full) {
        (Some(Content::Graphic), false) => GRAPHIC_DIST420,
        (Some(Content::Graphic), true) => GRAPHIC_DIST444,
        (Some(Content::Photographic), false) => PHOTO_DIST420,
        (Some(Content::Photographic), true) => PHOTO_DIST444,
        (None, false) => DIST420,
        (None, true) => DIST444,
    };
    lerp(pts, d)
}

/// Linear interpolation over an ascending table, clamped at both ends.
pub(crate) fn lerp(pts: &[(f32, f32)], x: f32) -> f32 {
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
    lerp(if full { G444 } else { G420 }, q)
}

/// When to spend cycles on restoration.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Routing {
    /// Restore whenever the near-pristine gate allows it.
    Always,
    /// Never restore; decode and return. Useful for A/B and for kill switches.
    Never,
    /// Restore when the content-aware estimate reaches `min_gain` SSIM2 and
    /// the near-pristine gate permits it. Larger thresholds save work and
    /// suppress some regressions, but also discard real gains.
    Auto { min_gain: f32 },
}

impl Default for Routing {
    /// Explicit legacy [`Routing::Auto`] at 0.25 SSIM2. Omitting a request's
    /// routing override selects the default request policy instead.
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
            routing: None,
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
    routing: Option<Routing>,
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

    /// Override the default policy. With default features, Fidelity requests
    /// use the frozen decoded-feature router on native analysis windows.
    /// Explicit overrides retain their existing gain-threshold and gate rules.
    /// Conservative intent, unsupported analysis inputs and builds without
    /// `chooser` use calibrated Auto at 0.25. See the reserved routing report.
    #[must_use]
    pub fn routing(mut self, r: Routing) -> Self {
        self.routing = Some(r);
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

        // Routing decides before restoration inference. The learned default
        // also pays for plain decoded-pixel features; explicit overrides keep
        // their existing coefficient/quality routing behavior.
        let mut probe_for_routing =
            zenjpeg::detect::probe(self.jpeg).map_err(|e| RestoreError::Probe(format!("{e:?}")))?;
        let _recorded = crate::record::apply_record(&mut probe_for_routing);
        // Content class selects the fitted curve using the file's own luma
        // coefficients. It costs a coefficient parse even on files that then
        // skip, so compute it only when the decision can depend on it. Measured
        // quality and cost depend on the corpus and implementation. `None` falls back to
        // the pooled curve, so an unparseable file routes exactly as before.
        let routing = self.routing.unwrap_or_default();
        let content = match routing {
            Routing::Auto { .. } => classify_content(self.jpeg),
            _ => None,
        };
        let predicted = estimate_gain_for(&probe_for_routing, content);
        let skip = match routing {
            Routing::Always => false,
            Routing::Never => true,
            Routing::Auto { min_gain } => predicted < min_gain,
        };
        let learned = false;
        #[cfg(feature = "chooser")]
        let (predicted, skip, learned) =
            if self.routing.is_none() && self.intent == Intent::Fidelity {
                match crate::decoded_router::for_jpeg(
                    self.jpeg,
                    &probe_for_routing,
                    predicted,
                    _recorded,
                )? {
                    Some(decision) => (decision.score() as f32, !decision.restore(), true),
                    None => (predicted, skip, learned),
                }
            } else {
                (predicted, skip, learned)
            };

        // Slack, gate thresholds and deblock policy are derived from the probe
        // and the caller's declarations — never surfaced.
        let mut cfg = RestoreConfig::default()
            .with_threads(self.threads)
            .with_high_q_identity(!learned)
            .with_projection(Projection::Auto);

        // Conservative gates earlier, trading restoration coverage for fewer
        // observed regressions on the calibration grid. It is not a per-file
        // quality guarantee; alternative routing policies are evaluated separately.
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
        let geometry_unavailable = crate::chroma_full_of(&probe_for_routing).is_none();
        let mut r = crate::restore_jpeg_routed(
            self.jpeg,
            model,
            &cfg,
            skip || (self.require_consistency && geometry_unavailable),
        )?;
        let consistency_unavailable = geometry_unavailable || r.report.projection.len() != 3;
        if self.require_consistency
            && consistency_unavailable
            && !skip
            && !r.report.skipped_model_high_q
            && !geometry_unavailable
        {
            // A partial projection cannot satisfy the request. Return the
            // decode, never restored pixels labeled Unchanged.
            r = crate::restore_jpeg_routed(self.jpeg, model, &cfg, true)?;
        }
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
        let consistency = if consistency_unavailable {
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

    #[cfg(feature = "chooser")]
    #[test]
    fn default_decoded_router_matches_pipeline_and_preserves_overrides() {
        let weights = zensr_micro::decode_all_f16(include_bytes!(
            "../../../models/adopted/dejpeg_rt24i_canon4/weights_f16.raw"
        ));
        let model =
            || zensr_micro::adopted::AdoptedModel::load_compact(&weights, 24, 8, 1).unwrap();
        let reference = model();
        let restorer = Restorer::new().with_tier(Budget::Realtime, model());
        // Keep debug inference bounded; release also exercises the full
        // study geometry and a larger, odd-sized image.
        let sizes = if cfg!(debug_assertions) {
            [(127, 17), (512, 32), (513, 17)]
        } else {
            [(127, 17), (512, 384), (1025, 65)]
        };
        for (width, height) in sizes {
            let rgb = synth(width, height);
            for q in [10., 50., 98.] {
                for ss in [
                    zenjpeg::encoder::ChromaSubsampling::Quarter,
                    zenjpeg::encoder::ChromaSubsampling::None,
                ] {
                    let jpg = zenjpeg::encoder::EncoderConfig::ycbcr(q, ss)
                        .encode(
                            bytemuck::cast_slice::<_, rgb::Rgb<u8>>(&rgb),
                            width as u32,
                            height as u32,
                        )
                        .unwrap();
                    let mut probe = zenjpeg::detect::probe(&jpg).unwrap();
                    let recorded = crate::record::apply_record(&mut probe);
                    let gain = estimate_gain_for(&probe, classify_content(&jpg));
                    let decision = crate::decoded_router::for_jpeg(&jpg, &probe, gain, recorded)
                        .unwrap()
                        .unwrap();
                    let decoded = zenjpeg::decoder::Decoder::new()
                        .deblock(zenjpeg::decoder::DeblockMode::Off)
                        .decode(&jpg, enough::Unstoppable)
                        .unwrap();
                    let default = restorer.restore(&jpg).threads(1).run().unwrap();
                    assert_eq!(
                        matches!(default, Outcome::Restored { .. }),
                        decision.restore()
                    );
                    let expected = if decision.restore() {
                        crate::restore_jpeg(
                            &jpg,
                            &reference,
                            &RestoreConfig::default()
                                .with_threads(1)
                                .with_high_q_identity(false),
                        )
                        .unwrap()
                        .to_rgb8()
                    } else {
                        decoded.pixels_u8().unwrap().to_vec()
                    };
                    assert_eq!(default.pixels().to_rgb8(), expected);
                    assert_eq!(
                        (default.pixels().width, default.pixels().height),
                        (width, height)
                    );
                    for routing in [
                        Routing::Never,
                        Routing::Auto {
                            min_gain: f32::INFINITY,
                        },
                    ] {
                        let result = restorer
                            .restore(&jpg)
                            .routing(routing)
                            .threads(1)
                            .run()
                            .unwrap();
                        assert!(matches!(result, Outcome::Unchanged { .. }));
                        assert_eq!(result.pixels().to_rgb8(), decoded.pixels_u8().unwrap());
                    }
                }
            }
        }
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

    #[test]
    fn routed_out_pixels_are_the_plain_decode() {
        let weights = zensr_micro::decode_all_f16(include_bytes!(
            "../../../models/adopted/dejpeg_rt24i_canon4/weights_f16.raw"
        ));
        let model = zensr_micro::adopted::AdoptedModel::load_compact(&weights, 24, 8, 1)
            .expect("candidate model");
        let restorer = Restorer::new().with_tier(Budget::Quality, model);
        let jpg = jpeg_at(55.0, zenjpeg::encoder::ChromaSubsampling::Quarter);
        let decoded = zenjpeg::decoder::Decoder::new()
            .deblock(zenjpeg::decoder::DeblockMode::Off)
            .decode(&jpg, enough::Unstoppable)
            .unwrap();
        for routing in [
            Routing::Never,
            Routing::Auto {
                min_gain: f32::INFINITY,
            },
        ] {
            let outcome = restorer
                .restore(&jpg)
                .budget(Budget::Quality)
                .routing(routing)
                .run()
                .unwrap();
            assert!(matches!(outcome, Outcome::Unchanged { .. }));
            assert_eq!(outcome.pixels().to_rgb8(), decoded.pixels_u8().unwrap());
        }
    }

    /// Exercise the real request path at the gain boundary, not a copy of its
    /// comparison. A rejected request must preserve decoded pixels exactly.
    #[test]
    fn auto_gain_boundary_controls_actual_model_execution() {
        let weights = zensr_micro::decode_all_f16(include_bytes!(
            "../../../models/adopted/dejpeg_rt24i_canon4/weights_f16.raw"
        ));
        let model = zensr_micro::adopted::AdoptedModel::load_compact(&weights, 24, 8, 1).unwrap();
        let restorer = Restorer::new().with_tier(Budget::Quality, model);
        let jpg = jpeg_at(55.0, zenjpeg::encoder::ChromaSubsampling::Quarter);
        let probe = zenjpeg::detect::probe(&jpg).unwrap();
        assert!(!crate::policy_high_q_identity(&probe));
        let run = |min_gain| {
            restorer
                .restore(&jpg)
                .budget(Budget::Quality)
                .routing(Routing::Auto { min_gain })
                .run()
                .unwrap()
        };
        let skipped = run(f32::INFINITY);
        let predicted = match skipped {
            Outcome::Unchanged {
                why: SkipReason::NotWorthIt { estimated_gain },
                ..
            } => estimated_gain,
            _ => panic!("infinite threshold must skip"),
        };
        assert!(predicted.is_finite());
        assert!(matches!(run(predicted), Outcome::Restored { .. }));
        let rejected = run(predicted + 0.001);
        assert!(matches!(
            rejected,
            Outcome::Unchanged {
                why: SkipReason::NotWorthIt { .. },
                ..
            }
        ));
        let decoded = zenjpeg::decoder::Decoder::new()
            .deblock(zenjpeg::decoder::DeblockMode::Off)
            .decode(&jpg, enough::Unstoppable)
            .unwrap();
        assert_eq!(rejected.pixels().to_rgb8(), decoded.pixels_u8().unwrap());
    }

    #[test]
    fn unsupported_chroma_cannot_claim_full_consistency() {
        let weights = zensr_micro::decode_all_f16(include_bytes!(
            "../../../models/adopted/dejpeg_rt24i_canon4/weights_f16.raw"
        ));
        let model = zensr_micro::adopted::AdoptedModel::load_compact(&weights, 24, 8, 1).unwrap();
        let restorer = Restorer::new().with_tier(Budget::Quality, model);
        for ss in [
            zenjpeg::encoder::ChromaSubsampling::HalfHorizontal,
            zenjpeg::encoder::ChromaSubsampling::HalfVertical,
        ] {
            let jpg = jpeg_at(55.0, ss);
            let decoded = zenjpeg::decoder::Decoder::new()
                .deblock(zenjpeg::decoder::DeblockMode::Off)
                .decode(&jpg, enough::Unstoppable)
                .unwrap();
            let strict = restorer
                .restore(&jpg)
                .budget(Budget::Quality)
                .routing(Routing::Always)
                .require_consistency(true)
                .run()
                .unwrap();
            assert!(matches!(
                strict,
                Outcome::Unchanged {
                    why: SkipReason::ConsistencyUnavailable(NotCertified::ChromaGeometry),
                    ..
                }
            ));
            assert_eq!(strict.pixels().to_rgb8(), decoded.pixels_u8().unwrap());
            let loose = restorer
                .restore(&jpg)
                .budget(Budget::Quality)
                .routing(Routing::Always)
                .require_consistency(false)
                .run()
                .unwrap();
            assert!(matches!(
                loose,
                Outcome::Restored {
                    consistency: Consistency::NotEnforced(NotCertified::ChromaGeometry),
                    ..
                }
            ));
        }
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

    /// Canon4 calibration is positive at 420 q94 and negative at q98/100.
    /// The previous model's negative-q94 assumption no longer applies.
    #[test]
    fn gain_curve_matches_canon4_high_quality_crossover() {
        assert!(gain_at(false, 94.0) > 0.0, "canon4 benefits at 420 q94");
        for full in [false, true] {
            for q in [98.0f32, 100.0] {
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

    /// The measured ordering holds at q55 and above; at q15/q35 canon4
    /// has slightly larger pooled gains at 444, so do not impose it there.
    #[test]
    fn full_chroma_gains_less_in_the_high_quality_band() {
        let mut qi = 55.0f32;
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

    /// Graphic-class gain dominates at distance >=1.3. Canon4's measured
    /// coefficient classes cross nearer pristine input; preserve that evidence.
    #[test]
    fn graphic_distance_curve_dominates_in_damaged_band() {
        for full in [false, true] {
            let mut d = 1.3f32;
            while d <= 16.0 {
                let (g, p) = (
                    gain_at_distance(full, d, Some(Content::Graphic)),
                    gain_at_distance(full, d, Some(Content::Photographic)),
                );
                assert!(
                    g > p,
                    "full_chroma={full} distance {d}: graphic {g:+.3} <= photographic {p:+.3}"
                );
                d += 0.1;
            }
        }
    }

    /// Every distance curve rises with distance, content-split included.
    #[test]
    fn all_distance_curves_never_decrease_with_distance() {
        for content in [None, Some(Content::Graphic), Some(Content::Photographic)] {
            for full in [false, true] {
                let mut prev = f32::NEG_INFINITY;
                let mut d = 0.0f32;
                while d <= 16.0 {
                    let g = gain_at_distance(full, d, content);
                    assert!(
                        g >= prev - 1e-4,
                        "{content:?} full={full} distance {d}: fell to {g:+.3} from {prev:+.3}"
                    );
                    prev = g;
                    d += 0.1;
                }
            }
        }
    }

    /// Even graphic content loses at distance 0 — a pristine file has nothing
    /// to recover regardless of what it depicts. This is the guard that stops a
    /// content split from reintroducing the sign error the pooled curve had.
    #[test]
    fn no_content_class_predicts_gain_on_a_pristine_file() {
        for full in [false, true] {
            for content in [None, Some(Content::Graphic), Some(Content::Photographic)] {
                let g = gain_at_distance(full, 0.0, content);
                assert!(
                    g < 0.0,
                    "{content:?} full={full} distance 0 predicts {g:+.3}"
                );
            }
        }
    }

    #[test]
    fn distance_curve_matches_measured_crossover_band() {
        assert!(gain_at_distance(false, 0.5, None) < 0.0);
        assert!(gain_at_distance(false, 0.7, None) > 0.0);
        assert!(gain_at_distance(true, 0.2, None) < 0.0);
        assert!(gain_at_distance(true, 0.3, None) > 0.0);
    }

    // The monotonicity check that lived here is superseded by
    // `all_distance_curves_never_decrease_with_distance`, which covers the
    // pooled curve and both content-split curves rather than the pooled one
    // alone.

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
                gain_at_distance(full, p.quality.value, None),
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
                    lerp(if full { GRAPHIC444 } else { GRAPHIC420 }, q),
                    lerp(if full { PHOTO444 } else { PHOTO420 }, q),
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
        for pts in [GRAPHIC420, GRAPHIC444, PHOTO420, PHOTO444] {
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
                let mut p = zenjpeg::detect::probe(&jpg).expect("probe");
                crate::record::apply_record(&mut p);
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
