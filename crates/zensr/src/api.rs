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

/// Whether coefficient projection was applied to the output.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Consistency {
    /// Projection was applied to all three YCbCr components using the
    /// family-calibrated slack. This is not a bit-exact re-encoding guarantee:
    /// slack expands the quantisation intervals, and RGB conversion and
    /// rounding can move coefficients after projection.
    ///
    /// This records application of the projection, not a final RGB8 bound
    /// check or a guarantee of improved perceptual quality.
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
    /// A lightweight outcome record without retaining a second image buffer.
    pub fn summary(&self) -> Summary {
        Summary {
            applied: matches!(self, Self::Restored { .. }),
            reason: match self {
                Self::Restored { .. } => "Restoration and coefficient projection applied".into(),
                Self::Unchanged { why, .. } => format!("Restoration bypassed: {why:?}"),
            },
            report: self.report().clone(),
        }
    }
    /// Consume the result without cloning its pixel planes.
    pub fn into_pixels(self) -> Pixels {
        match self {
            Self::Restored { pixels, .. } | Self::Unchanged { pixels, .. } => pixels,
        }
    }
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

/// Codec-visible restoration status, separate from decoded sample storage.
#[derive(Clone, Debug)]
pub struct Summary {
    pub applied: bool,
    pub reason: String,
    pub report: Report,
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
    /// Consume unquantized planes into interleaved f32 RGB.
    pub fn into_rgb_f32(self) -> Vec<f32> {
        let plane = self.width * self.height;
        let mut output = vec![0.; self.planes.len()];
        for i in 0..plane {
            for c in 0..3 {
                output[i * 3 + c] = self.planes[c * plane + i];
            }
        }
        output
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
pub fn estimate_gain(probe: &crate::source::JpegInfo) -> f32 {
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

fn classify_coefficients(coeffs: &crate::source::Coefficients<'_>) -> Option<Content> {
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
pub(crate) fn estimate_gain_for(probe: &crate::source::JpegInfo, content: Option<Content>) -> f32 {
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
    pub fn restore<'a>(&'a self, source: &'a mut dyn crate::source::Source) -> Request<'a> {
        Request {
            restorer: self,
            source,
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
    source: &'a mut dyn crate::source::Source,
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

    /// Return [`Outcome::Unchanged`] if coefficient projection cannot be applied
    /// to all three YCbCr components. This requires projection availability;
    /// it does not verify the bounds again after RGB conversion and rounding.
    /// See [`Consistency::Certified`] for the scope of that report.
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
        let prepared = self.source;
        let mut probe_for_routing = prepared.info().clone();
        let _recorded = crate::record::apply_record(&mut probe_for_routing);
        let routing = self.routing.unwrap_or_default();
        let content = match routing {
            Routing::Auto { .. } => {
                prepared.prepare(crate::source::Needs {
                    pixels: None,
                    coefficients: true,
                })?;
                prepared
                    .coefficients()
                    .ok()
                    .as_ref()
                    .and_then(classify_coefficients)
            }
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
                match crate::decoded_router::for_source_prepared(
                    &probe_for_routing,
                    predicted,
                    _recorded,
                    prepared,
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
        let mut r = crate::restore_prepared(
            model,
            &cfg,
            skip || (self.require_consistency && geometry_unavailable),
            prepared,
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
            r = crate::restore_prepared(model, &cfg, true, prepared)?;
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
