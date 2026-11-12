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
    CoeffOrder, CoeffView, project_chroma_420_with_scratch, project_plane,
    rgb_to_ycbcr_planes_inplace, ycbcr_to_rgb_planes_inplace,
};
use zensr_micro::guards::guarded_merge_with_scratch;

// Narrow, deliberate re-exports — the full micro modules are NOT part of
// this crate's contract.
pub use crate::api::{PixelsRef, RepairWorkspace, WorkspaceOutcome};
pub use crate::source::{CoefficientsView, JpegDctBlocks};
pub use zensr_micro::adopted::{AdoptedModel, ModelSpace, ModelWorkspace};
pub use zensr_micro::consist::{ChromaScratch, ProjectionConfig, ProjectionReport};
pub use zensr_micro::decode_all_f16;
pub use zensr_micro::guards::{GuardConfig, GuardReport, GuardScratch};

/// Measured deployment rule for zenjpeg's deblocker under the model.
/// Inputs are exact at the qualities where it matters (probe q5/8/12 verified
/// error-free for turbo + mozjpeg on the eval corpus).
///
/// Family/scale matching goes through the `Debug` strings deliberately: the
/// probe enums are `#[non_exhaustive]` upstream and this crate spans zenjpeg
/// ">=0.8.4, <0.10" — string prefixes stay stable across that range where a
/// `match` would not.
pub fn policy_wants_auto(probe: &crate::source::JpegInfo) -> bool {
    let fam = probe.encoder.as_str();
    let scale = probe.quality.scale.as_str();
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
    let fam = probe.encoder.as_str();
    if fam.starts_with("Cjpegli") {
        0.45
    } else if fam == "Mozjpeg" {
        0.35
    } else if is_round_to_nearest_family(fam) {
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
    let fam = probe.encoder.as_str();
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
    match probe.subsampling.as_str() {
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
    let _ = record::is_recorded(probe);
    // String-spelling match for the same reason as `policy_wants_auto`: the
    // probe enums are `#[non_exhaustive]` upstream and `zenjpeg::types` is not a
    // public path, so the name is the stable handle across the version range.
    let full_chroma = probe.subsampling.as_str() == "S444";
    match probe.quality.scale.as_str() {
        "IjgQuality" | "MozjpegQuality" => {
            let mozjpeg = probe.encoder.as_str() == "Mozjpeg";
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
    coefficients: &crate::source::CoefficientsView<'_>,
) -> PlaneGeometry {
    let max_h = coefficients.iter().map(|c| c.h_samp).max().unwrap_or(0);
    let max_v = coefficients.iter().map(|c| c.v_samp).max().unwrap_or(0);
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

pub(crate) fn restore_prepared_into(
    model: &AdoptedModel,
    cfg: &RestoreConfig,
    skip_model: bool,
    prepared: &mut dyn source::Source,
    ws: &mut crate::api::RepairWorkspace,
) -> Result<(), RestoreError> {
    if model.scale != 1 {
        return Err(RestoreError::UnsupportedModel(
            "restore_jpeg is the x1 pipeline",
        ));
    }

    // Reset reusable RestoreReport fields in place without reallocating capacity.
    ws.restore_report.encoder_family.clear();
    ws.restore_report.quality_scale.clear();
    ws.restore_report.used_deblock_auto = false;
    ws.restore_report.chroma_full = None;
    ws.restore_report.skipped_model_high_q = false;
    ws.restore_report.guard = GuardReport::default();
    ws.restore_report.projection.clear();

    // 1. fingerprint + deblock policy (read directly from `&JpegInfo` without cloning Strings)
    let (want_auto, skip_high_q, pcfg) = {
        let probe = prepared.info();
        let recorded = record::is_recorded(probe);
        ws.restore_report
            .encoder_family
            .push_str(probe.encoder.as_str());
        if recorded {
            ws.restore_report.encoder_family.push_str(" (recorded)");
        }
        ws.restore_report.est_quality = probe.quality.value;
        ws.restore_report
            .quality_scale
            .push_str(probe.quality.scale.as_str());
        ws.restore_report.chroma_full = chroma_full_of(probe);
        let want_auto = !skip_model && cfg.deblock_policy && policy_wants_auto(probe);
        ws.restore_report.used_deblock_auto = want_auto;
        let skip_high_q = skip_model
            || (cfg.high_q_identity
                && policy_high_q_identity_with_margin(probe, cfg.high_q_margin));
        let pcfg = match cfg.projection {
            Projection::Off => None,
            Projection::Auto => Some(
                ProjectionConfig::with_slack_q(slack_for(probe))
                    .with_slack_abs(slack_abs_for(probe)),
            ),
            Projection::Fixed(c) => Some(c),
        };
        (want_auto, skip_high_q, pcfg)
    };

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
    let total = 3 * plane;
    ws.buf_a.resize(total, 0.0);
    for y in 0..h {
        let row = &px[y * dec.stride..y * dec.stride + w * 3];
        for x in 0..w {
            ws.buf_a[y * w + x] = row[x * 3] as f32 / 255.0;
            ws.buf_a[plane + y * w + x] = row[x * 3 + 1] as f32 / 255.0;
            ws.buf_a[2 * plane + y * w + x] = row[x * 3 + 2] as f32 / 255.0;
        }
    }
    ws.width = w;
    ws.height = h;

    // 2b. high-q identity gate: near-pristine input — the model can only
    // do harm here (measured; see policy_high_q_identity). The decode is
    // already consistent, so projection is a no-op too: return it in `ws.buf_a`.
    if skip_high_q {
        ws.restore_report.skipped_model_high_q = !skip_model;
        return Ok(());
    }

    // 3. model space: YCbCr-native models run directly in the space where
    // quantization happened (S5b); RGB models keep the legacy path.
    let ycbcr_native = model.space() == ModelSpace::Ycbcr;
    if ycbcr_native {
        rgb_to_ycbcr_planes_inplace(&mut ws.buf_a[..total], plane);
    }

    // 4. guarded model (guard anchors in the model's own space)
    ws.buf_b.resize(total, 0.0);
    model.upscale_tiled_into(
        &ws.buf_a[..total],
        h,
        w,
        cfg.threads,
        cfg.tile,
        &mut ws.buf_b[..total],
        &mut ws.model_ws,
    );
    ws.restore_report.guard = guarded_merge_with_scratch(
        &mut ws.buf_b[..total],
        &ws.buf_a[..total],
        h,
        w,
        1,
        &cfg.guard,
        &mut ws.guard_scratch,
    );

    // 5. quantization-consistency projection (S10): luma via direct box clamp,
    // full-res chroma (4:4:4) likewise, subsampled chroma (4:2:0) via exact
    // one-pass back-projection on the half-res lattice.
    if !ycbcr_native {
        rgb_to_ycbcr_planes_inplace(&mut ws.buf_b[..total], plane);
    }
    if let Some(pcfg) = pcfg {
        prepared.prepare(source::Needs {
            pixels: None,
            coefficients: true,
        })?;
        let coeffs = prepared.coefficients_view()?;
        // Only geometries whose color model we know: 1 (grayscale luma) or 3
        // (YCbCr). 4-component (Adobe CMYK/YCCK) coefficients do NOT live in
        // the YCbCr space we reconstruct here — skip projection entirely.
        let ncomp = coeffs.total_components();
        let projectable = ncomp == 1 || ncomp == 3;
        for (ci, comp) in coeffs
            .iter()
            .enumerate()
            .take(if projectable { 3 } else { 0 })
        {
            let Some(qt) = coeffs.quant_table_natural(comp.quant_table_idx) else {
                continue;
            };
            let cv = CoeffView {
                coeffs: comp.coeffs,
                blocks_wide: comp.blocks_wide,
                blocks_high: comp.blocks_high,
                order: CoeffOrder::Zigzag,
                quant: qt,
            };
            let target = &mut ws.buf_b[ci * plane..(ci + 1) * plane];
            match plane_geometry(comp.h_samp, comp.v_samp, &coeffs) {
                PlaneGeometry::Full => {
                    ws.restore_report
                        .projection
                        .push(project_plane(target, w, h, &cv, &pcfg));
                }
                PlaneGeometry::HalfBoth => {
                    ws.restore_report
                        .projection
                        .push(project_chroma_420_with_scratch(
                            target,
                            w,
                            h,
                            &cv,
                            &pcfg,
                            &mut ws.chroma_scratch,
                        ));
                }
                // 4:2:2 / 4:4:0 (half in ONE axis only): the 2x2 box
                // back-projection would be wrong — leave unprojected.
                PlaneGeometry::Other => {}
            }
        }
    }
    ycbcr_to_rgb_planes_inplace(&mut ws.buf_b[..total], plane);
    core::mem::swap(&mut ws.buf_a, &mut ws.buf_b);
    Ok(())
}
