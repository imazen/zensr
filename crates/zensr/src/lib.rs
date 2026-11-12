//! zenjpeg integration for zensr: the production restoration pipeline.
//!
//! One call: JPEG bytes -> probe -> deblock policy -> zenjpeg decode ->
//! guarded x1 model -> quantization-consistency projection -> RGB planes.
//!
//! Policy (measured, docs/research/SYSTEMS.md "S6-v2"): `DeblockMode::Auto` (Knusperli)
//! only for Annex-K-family files at probe-estimated q <= 9.5 — the regime where
//! coefficient-domain correction beats pixel-exact decode; AQ-family
//! (Cjpegli*/zenjpeg) never. Routing can skip model execution. Projection
//! constrains coefficients using family-calibrated slack; it guarantees neither
//! exact coefficient re-encoding nor improved perceptual quality.

#![forbid(unsafe_code)]

pub mod api;
mod calibration;
#[cfg(feature = "chooser")]
pub mod chooser;
#[cfg(feature = "chooser")]
mod decoded_router;
#[cfg(feature = "chooser")]
mod decoded_router_features;
#[cfg(all(feature = "chooser", test))]
mod decoded_router_model;
pub mod jpegli_params;
pub mod qtables;
mod qtables_data;
mod record;
#[cfg(feature = "chooser")]
mod router_coefficients;
pub mod source;

use zensr_micro::consist::{
    CoeffOrder, CoeffView, project_chroma_420, project_plane, rgb_to_ycbcr_planes,
    ycbcr_to_rgb_planes,
};
use zensr_micro::guards::guarded_merge;

// Narrow, deliberate re-exports — the full micro modules are NOT part of
// this crate's contract.
pub use zensr_micro::adopted::{AdoptedModel, ModelSpace};
pub use zensr_micro::consist::{ProjectionConfig, ProjectionReport};
pub use zensr_micro::decode_all_f16;
pub use zensr_micro::guards::{GuardConfig, GuardReport};

/// Measured deployment rule for zenjpeg's deblocker under the model.
/// Inputs are exact at the qualities where it matters (probe q5/8/12 verified
/// error-free for turbo + mozjpeg on the eval corpus).
///
/// Family/scale matching goes through the `Debug` strings deliberately: the
/// probe enums are `#[non_exhaustive]` upstream and this crate spans zenjpeg
/// ">=0.8.4, <0.10" — string prefixes stay stable across that range where a
/// `match` would not.
pub fn policy_wants_auto(probe: &crate::source::JpegInfo) -> bool {
    let fam = format!("{:?}", probe.encoder);
    let scale = format!("{:?}", probe.quality.scale);
    !fam.starts_with("Cjpegli")
        && (scale == "IjgQuality" || scale == "MozjpegQuality")
        && probe.quality.value <= 9.5
}

/// Family-conditional projection slack, calibrated 2026-07-25 on 1M luma
/// coefficients per (encoder, q) cell (benchmarks/slack_calibration_*.tsv):
/// round-to-nearest (turbo/IJG) p99 excess <=0.07Q; mozjpeg trellis p99
/// <=0.23Q (max ~15Q on zeroed runs); jpegli/zenjpeg AQ p99 <=0.41Q (stored
/// DQT understates per-block quantization). Slack covers p99 + margin; the
/// trellis-zero tail is a documented approximation (<=~4% of coefficients,
/// where the box may exclude the truth and non-expansiveness doesn't hold —
/// net effect measured in the eval).
///
/// **Unrecognised encoders get the WIDEST slack, not the narrowest** (fixed
/// 2026-08-02). The `else` arm previously returned 0.15 — the round-to-nearest
/// value — for both "known IJG-family" and "no idea what made this". That is
/// backwards: 0.15 is the tightest box we have, so the case where we know
/// least about the quantiser was getting the least room, and a trellising or
/// adaptively-quantising encoder we failed to identify would have its
/// coefficients clamped out of a box that never contained them.
///
/// This matters on real traffic, not hypothetically: a survey of 17,739
/// corpus JPEGs found 18.2% unrecognised, and the sampled e-commerce CDN files
/// probe as `Unknown` with a table no known preset produces
/// (`benchmarks/dqt_survey_2026-08-02.md`).
pub fn slack_for(probe: &crate::source::JpegInfo) -> f32 {
    let fam = format!("{:?}", probe.encoder);
    if fam.starts_with("Cjpegli") {
        0.45
    } else if fam == "Mozjpeg" {
        0.35
    } else if is_round_to_nearest_family(&fam) {
        0.15
    } else {
        // Unknown / unrecognised: assume the worst-behaved encoder we have
        // calibrated, so the box is wide enough to contain the truth.
        0.45
    }
}

/// Families measured to quantise by plain round-to-nearest, for which the
/// tight 0.15 box is justified. Anything NOT on this list is treated as
/// unknown by `slack_for`, which is the conservative direction.
fn is_round_to_nearest_family(fam: &str) -> bool {
    matches!(
        fam,
        "LibjpegTurbo" | "IjgFamily" | "ImageMagick" | "WindowsImaging" | "Photoshop"
    )
}

/// Family-conditional ABSOLUTE slack (coefficient units), calibrated
/// 2026-07-26 per-quantizer-value at q88..q98 (1M coeffs/cell,
/// benchmarks/slack_calibration_highq_*.tsv): encoders that quantize YCbCr
/// samples to u8 before their FDCT (libjpeg-turbo, mozjpeg) carry a bounded
/// absolute DCT noise — turbo Q=1 p99 1.32 / p99.9 2.15 / max 3.7 — which a
/// purely relative slack cannot cover at Q=1..3 (the q96 projection
/// regression). Cjpegli-family (float sample pipeline) measured far cleaner
/// (Q=1 p99 <= 0.24): small allowance only, so valid high-q projection is
/// not needlessly weakened. NOTE (falsified alternative): restricting the
/// clamp to nonzero-coded bands does NOT rescue trellis/AQ families —
/// measured violations sit on coded coefficients too (mozjpeg-nz p99 up to
/// 1.7, max 15Q); their tail stays a documented approximation.
pub fn slack_abs_for(probe: &crate::source::JpegInfo) -> f32 {
    let fam = format!("{:?}", probe.encoder);
    if fam.starts_with("Cjpegli") { 0.5 } else { 1.5 }
}

/// Identity gate calibrated for canon4 on an image-separated split.
/// A tail cell is harmful when at least half its calibration observations
/// lose more than 0.1 SSIM2 after projection. Only a contiguous harmful tail
/// sets a gate. Family distinctions use the probe after record correction;
/// unknown quality-family encoders use the IJG fallback. Unmeasured chroma
/// geometries retain the subsampled threshold. Evidence and held-out results:
/// `benchmarks/ship_rt24i_canon4_2026-11-04.md`.
pub fn policy_high_q_identity(probe: &crate::source::JpegInfo) -> bool {
    policy_high_q_identity_with_margin(probe, 0.0)
}

/// Luma-relative chroma geometry from the probe: `Some(true)` for 4:4:4,
/// `Some(false)` for 4:2:0, `None` for 4:2:2, 4:4:0, grayscale and anything
/// else. Separated out so the mapping is testable without model weights.
pub(crate) fn chroma_full_of(probe: &crate::source::JpegInfo) -> Option<bool> {
    match format!("{:?}", probe.subsampling).as_str() {
        "S444" => Some(true),
        "S420" => Some(false),
        _ => None,
    }
}

/// As [`policy_high_q_identity`], but shifted `margin` points earlier.
///
/// The two scales run in opposite directions — IJG quality rises with fidelity
/// while Butteraugli distance falls — so a positive margin lowers the quality
/// threshold and raises the distance one. Getting that backwards would make a
/// "conservative" caller restore MORE, which is why it is one function rather
/// than two call sites.
pub fn policy_high_q_identity_with_margin(probe: &crate::source::JpegInfo, margin: f32) -> bool {
    let mut corrected = probe.clone();
    record::apply_record(&mut corrected);
    let probe = &corrected;
    // Debug-string match for the same reason as `policy_wants_auto`: the probe
    // enums are `#[non_exhaustive]` upstream and `zenjpeg::types` is not a
    // public path, so the name is the stable handle across the version range.
    let full_chroma = format!("{:?}", probe.subsampling) == "S444";
    let scale = format!("{:?}", probe.quality.scale);
    match scale.as_str() {
        "IjgQuality" | "MozjpegQuality" => {
            let mozjpeg = format!("{:?}", probe.encoder) == "Mozjpeg";
            let threshold = match (mozjpeg, full_chroma) {
                (false, false) => calibration::GATE_IJG_420,
                (false, true) => calibration::GATE_IJG_444,
                (true, false) => calibration::GATE_MOZJPEG_420,
                (true, true) => calibration::GATE_MOZJPEG_444,
            };
            probe.quality.value >= threshold - margin
        }
        "ButteraugliDistance" => {
            probe.quality.value
                <= (if full_chroma {
                    calibration::GATE_DISTANCE_444
                } else {
                    calibration::GATE_DISTANCE_420
                }) + margin
        }
        _ => false,
    }
}

#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Projection {
    /// No projection (pixel pipeline only).
    Off,
    /// Project luma always; 4:4:4 chroma via the direct box clamp, 4:2:0
    /// chroma via exact one-pass back-projection on the half-res lattice.
    /// Slack comes from `slack_for(probe)` (family-calibrated).
    /// (4:2:2 / 4:4:0 chroma is left unprojected for now.)
    Auto,
    /// As Auto but with an explicit slack override.
    Fixed(ProjectionConfig),
}

#[non_exhaustive]
pub struct RestoreConfig {
    pub guard: GuardConfig,
    pub projection: Projection,
    /// Apply the measured deblock policy (Auto at Annex-K q<=9). When false,
    /// decode is always pixel-exact (DeblockMode::Off).
    pub deblock_policy: bool,
    /// Skip the model on the calibrated harmful high-quality tail —
    /// measured: the model loses to identity there (`policy_high_q_identity`).
    pub high_q_identity: bool,
    /// Shifts the near-pristine gate earlier, in quality points (or, on the
    /// Butteraugli scale, in distance). Exists so a conservative caller can
    /// trade gain for fewer regressions: gating earlier is the only mechanism
    /// measured to reduce per-file harm.
    pub high_q_margin: f32,
    pub threads: usize,
    /// Tile size for the model runner; 0 = auto.
    pub tile: usize,
}

impl RestoreConfig {
    /// Builder-style helpers (the struct is `#[non_exhaustive]`; construct via
    /// `RestoreConfig::default()` and adjust).
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }
    pub fn with_projection(mut self, p: Projection) -> Self {
        self.projection = p;
        self
    }
    pub fn with_deblock_policy(mut self, on: bool) -> Self {
        self.deblock_policy = on;
        self
    }
    pub fn with_high_q_identity(mut self, on: bool) -> Self {
        self.high_q_identity = on;
        self
    }
    pub fn with_high_q_margin(mut self, margin: f32) -> Self {
        self.high_q_margin = margin;
        self
    }
}

impl Default for RestoreConfig {
    fn default() -> Self {
        RestoreConfig {
            guard: GuardConfig::default(),
            projection: Projection::Auto,
            deblock_policy: true,
            high_q_identity: true,
            high_q_margin: 0.0,
            threads: 1,
            tile: 0,
        }
    }
}

#[derive(Debug, Default, Clone)]
#[non_exhaustive]
pub struct RestoreReport {
    pub encoder_family: String,
    pub est_quality: f32,
    pub quality_scale: String,
    pub used_deblock_auto: bool,
    /// Luma-relative chroma geometry: `Some(true)` for 4:4:4, `Some(false)`
    /// for 4:2:0, `None` for anything else. Recorded because the near-pristine
    /// threshold depends on it — 4:4:4 at q90 decodes as cleanly as 4:2:0 at
    /// q94, so a quality-only policy is unsafe (see docs/DEFAULT_ROUTING.md).
    pub chroma_full: Option<bool>,
    /// True when the high-q identity gate skipped the model (output is the
    /// plain decode).
    pub skipped_model_high_q: bool,
    pub guard: GuardReport,
    /// Per-projected-plane reports (Y, then Cb/Cr when projected).
    pub projection: Vec<ProjectionReport>,
}

/// Planar RGB f32 output ([3, h, w], values in [0,1]) + provenance report.
#[non_exhaustive]
pub struct Restored {
    pub planes: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub report: RestoreReport,
}

impl Restored {
    /// Interleaved RGB8 convenience view.
    pub fn to_rgb8(&self) -> Vec<u8> {
        let plane = self.width * self.height;
        let mut out = vec![0u8; 3 * plane];
        for i in 0..plane {
            for c in 0..3 {
                out[i * 3 + c] = (self.planes[c * plane + i] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
        out
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum RestoreError {
    Probe(String),
    Decode(String),
    UnsupportedPixels(&'static str),
    UnsupportedModel(&'static str),
    /// The requested [`api::Budget`] tier has no weights loaded.
    TierNotLoaded,
}

impl core::fmt::Display for RestoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RestoreError::Probe(e) => write!(f, "probe: {e}"),
            RestoreError::Decode(e) => write!(f, "decode: {e}"),
            RestoreError::UnsupportedPixels(e) => write!(f, "unsupported pixels: {e}"),
            RestoreError::UnsupportedModel(e) => write!(f, "unsupported model: {e}"),
            RestoreError::TierNotLoaded => {
                write!(f, "no weights loaded for the requested budget tier")
            }
        }
    }
}
impl std::error::Error for RestoreError {}

/// How a coefficient plane's block grid relates to the full-res image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlaneGeometry {
    /// Covers the image at full resolution (luma, or 4:4:4 chroma).
    Full,
    /// Half resolution in BOTH axes (4:2:0 chroma).
    HalfBoth,
    /// Anything else (4:2:2 / 4:4:0 half-one-axis, exotic factors).
    Other,
}

fn plane_geometry(
    h_samp: u8,
    v_samp: u8,
    coefficients: &crate::source::Coefficients<'_>,
) -> PlaneGeometry {
    let max_h = coefficients
        .components
        .iter()
        .map(|c| c.h_samp)
        .max()
        .unwrap_or(0);
    let max_v = coefficients
        .components
        .iter()
        .map(|c| c.v_samp)
        .max()
        .unwrap_or(0);
    if max_h == 0 || max_v == 0 || h_samp == 0 || v_samp == 0 {
        return PlaneGeometry::Other;
    }
    // Block dimensions include MCU padding and cannot identify sampling on
    // tiny images. SOF sampling factors express the actual component lattice.
    if h_samp == max_h && v_samp == max_v {
        PlaneGeometry::Full
    } else if u16::from(h_samp) * 2 == u16::from(max_h) && u16::from(v_samp) * 2 == u16::from(max_v)
    {
        PlaneGeometry::HalfBoth
    } else {
        PlaneGeometry::Other
    }
}

pub(crate) fn restore_prepared(
    model: &AdoptedModel,
    cfg: &RestoreConfig,
    skip_model: bool,
    prepared: &mut dyn source::Source,
) -> Result<Restored, RestoreError> {
    if model.scale != 1 {
        return Err(RestoreError::UnsupportedModel(
            "restore_jpeg is the x1 pipeline",
        ));
    }
    let mut report = RestoreReport::default();

    // 1. fingerprint + deblock policy
    let mut probe = prepared.info().clone();
    // A verified encode record overrides the fingerprint (record.rs); the
    // report says so, since every policy below then runs on exact inputs.
    let recorded = record::apply_record(&mut probe);
    report.encoder_family = if recorded {
        format!("{:?} (recorded)", probe.encoder)
    } else {
        format!("{:?}", probe.encoder)
    };
    report.est_quality = probe.quality.value;
    report.quality_scale = format!("{:?}", probe.quality.scale);
    // Debug-string match for the reason documented on `policy_wants_auto`:
    // the probe enums are non_exhaustive upstream and zenjpeg::types is not a
    // public path.
    report.chroma_full = chroma_full_of(&probe);
    let want_auto = !skip_model && cfg.deblock_policy && policy_wants_auto(&probe);
    report.used_deblock_auto = want_auto;
    let mode = if want_auto {
        source::DecodeMode::Deblocked
    } else {
        source::DecodeMode::Plain
    };
    prepared.prepare(source::Needs {
        pixels: Some(mode),
        coefficients: false,
    })?;
    let dec = prepared.pixels(mode)?;
    let (w32, h32) = dec.dimensions();
    let (w, h) = (w32 as usize, h32 as usize);
    if w == 0 || h == 0 {
        return Err(RestoreError::UnsupportedPixels("empty RGB geometry"));
    }
    let px = dec.data;
    let required = (h - 1)
        .checked_mul(dec.stride)
        .and_then(|n| n.checked_add(w * 3));
    if dec.stride < w * 3 || required.is_none_or(|n| n > px.len()) {
        return Err(RestoreError::UnsupportedPixels("expected strided RGB8"));
    }
    let plane = w * h;
    let mut planes = vec![0.0f32; 3 * plane];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                planes[c * plane + y * w + x] = px[y * dec.stride + x * 3 + c] as f32 / 255.0;
            }
        }
    }

    // 2b. high-q identity gate: near-pristine input — the model can only
    // do harm here (measured; see policy_high_q_identity). The decode is
    // already consistent, so projection is a no-op too: return it.
    if skip_model
        || (cfg.high_q_identity && policy_high_q_identity_with_margin(&probe, cfg.high_q_margin))
    {
        report.skipped_model_high_q = !skip_model;
        return Ok(Restored {
            planes,
            width: w,
            height: h,
            report,
        });
    }

    // 3. model space: YCbCr-native models run directly in the space where
    // quantization happened (S5b); RGB models keep the legacy path.
    let ycbcr_native = model.space() == ModelSpace::Ycbcr;
    let model_in = if ycbcr_native {
        let (mut y, mut cb, mut cr) = (vec![0.0; plane], vec![0.0; plane], vec![0.0; plane]);
        rgb_to_ycbcr_planes(&planes, plane, &mut y, &mut cb, &mut cr);
        let mut p = y;
        p.extend_from_slice(&cb);
        p.extend_from_slice(&cr);
        p
    } else {
        planes
    };

    // 4. guarded model (guard anchors in the model's own space)
    let mut sr = model.upscale_tiled(&model_in, h, w, cfg.threads, cfg.tile);
    report.guard = guarded_merge(&mut sr, &model_in, h, w, 1, &cfg.guard);

    // 5. quantization-consistency projection (S10): luma via direct box clamp,
    // full-res chroma (4:4:4) likewise, subsampled chroma (4:2:0) via exact
    // one-pass back-projection on the half-res lattice.
    let pcfg = match cfg.projection {
        Projection::Off => None,
        Projection::Auto => Some(
            ProjectionConfig::with_slack_q(slack_for(&probe)).with_slack_abs(slack_abs_for(&probe)),
        ),
        Projection::Fixed(c) => Some(c),
    };
    let mut ycc = if ycbcr_native {
        sr
    } else {
        let (mut y, mut cb, mut cr) = (vec![0.0; plane], vec![0.0; plane], vec![0.0; plane]);
        rgb_to_ycbcr_planes(&sr, plane, &mut y, &mut cb, &mut cr);
        let mut p = y;
        p.extend_from_slice(&cb);
        p.extend_from_slice(&cr);
        p
    };
    if let Some(pcfg) = pcfg {
        prepared.prepare(source::Needs {
            pixels: None,
            coefficients: true,
        })?;
        let coeffs = prepared.coefficients()?;
        // Only geometries whose color model we know: 1 (grayscale luma) or 3
        // (YCbCr). 4-component (Adobe CMYK/YCCK) coefficients do NOT live in
        // the YCbCr space we reconstruct here — skip projection entirely.
        let ncomp = coeffs.components.len();
        let projectable = ncomp == 1 || ncomp == 3;
        for (ci, comp) in coeffs
            .components
            .iter()
            .enumerate()
            .take(if projectable { 3 } else { 0 })
        {
            let Some(qt) = coeffs.quant_tables[comp.quant_table_idx as usize] else {
                continue;
            };
            let cv = CoeffView {
                coeffs: comp.coeffs,
                blocks_wide: comp.blocks_wide,
                blocks_high: comp.blocks_high,
                order: CoeffOrder::Zigzag,
                quant: &qt,
            };
            let target = &mut ycc[ci * plane..(ci + 1) * plane];
            match plane_geometry(comp.h_samp, comp.v_samp, &coeffs) {
                PlaneGeometry::Full => {
                    report
                        .projection
                        .push(project_plane(target, w, h, &cv, &pcfg));
                }
                PlaneGeometry::HalfBoth => {
                    report
                        .projection
                        .push(project_chroma_420(target, w, h, &cv, &pcfg));
                }
                // 4:2:2 / 4:4:0 (half in ONE axis only): the 2x2 box
                // back-projection would be wrong — leave unprojected.
                PlaneGeometry::Other => {}
            }
        }
    }
    let mut out = vec![0.0f32; 3 * plane];
    ycbcr_to_rgb_planes(
        &ycc[..plane],
        &ycc[plane..2 * plane],
        &ycc[2 * plane..],
        &mut out,
        plane,
    );

    Ok(Restored {
        planes: out,
        width: w,
        height: h,
        report,
    })
}
