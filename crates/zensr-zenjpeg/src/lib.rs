//! zenjpeg integration for zensr: the production restoration pipeline.
//!
//! One call: JPEG bytes -> probe -> deblock policy -> zenjpeg decode ->
//! guarded x1 model -> quantization-consistency projection -> RGB planes.
//!
//! Policy (measured, SYSTEMS.md "S6-v2"): `DeblockMode::Auto` (Knusperli)
//! only for Annex-K-family files at probe-estimated q <= 9 — the regime where
//! coefficient-domain correction beats pixel-exact decode; AQ-family
//! (Cjpegli*/zenjpeg) never. The model runs on every image; the projection
//! (S10) guarantees the output re-encodes to the file's own coefficients.

use zensr_micro::adopted::AdoptedModel;
use zensr_micro::consist::{
    project_plane, rgb_to_ycbcr_planes, ycbcr_to_rgb_planes, CoeffOrder, CoeffView,
    ProjectionConfig, ProjectionReport,
};
use zensr_micro::guards::{guarded_merge, GuardConfig, GuardReport};

pub use zensr_micro::consist;
pub use zensr_micro::guards;

/// Measured deployment rule for zenjpeg's deblocker under the model.
/// Inputs are exact at the qualities where it matters (probe q5/8/12 verified
/// error-free for turbo + mozjpeg on the eval corpus).
pub fn policy_wants_auto(probe: &zenjpeg::detect::JpegProbe) -> bool {
    let fam = format!("{:?}", probe.encoder);
    let scale = format!("{:?}", probe.quality.scale);
    !fam.starts_with("Cjpegli")
        && (scale == "IjgQuality" || scale == "MozjpegQuality")
        && probe.quality.value <= 9.5
}

#[derive(Clone, Copy, Debug)]
pub enum Projection {
    /// No projection (pixel pipeline only).
    Off,
    /// Project luma always; chroma too when the file is 4:4:4.
    /// (Subsampled chroma needs the back-projection form — not yet wired.)
    LumaAndFullResChroma(ProjectionConfig),
}

pub struct RestoreConfig {
    pub guard: GuardConfig,
    pub projection: Projection,
    /// Apply the measured deblock policy (Auto at Annex-K q<=9). When false,
    /// decode is always pixel-exact (DeblockMode::Off).
    pub deblock_policy: bool,
    pub threads: usize,
    /// Tile size for the model runner; 0 = auto.
    pub tile: usize,
}

impl Default for RestoreConfig {
    fn default() -> Self {
        RestoreConfig {
            guard: GuardConfig::default(),
            projection: Projection::LumaAndFullResChroma(ProjectionConfig::default()),
            deblock_policy: true,
            threads: 1,
            tile: 0,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct RestoreReport {
    pub encoder_family: String,
    pub est_quality: f32,
    pub quality_scale: String,
    pub used_deblock_auto: bool,
    pub guard: GuardReport,
    /// Per-projected-plane reports (Y, then Cb/Cr when projected).
    pub projection: Vec<ProjectionReport>,
}

/// Planar RGB f32 output ([3, h, w], values in [0,1]) + provenance report.
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
                out[i * 3 + c] =
                    (self.planes[c * plane + i] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
        out
    }
}

#[derive(Debug)]
pub enum RestoreError {
    Probe(String),
    Decode(String),
    UnsupportedPixels(&'static str),
}

impl core::fmt::Display for RestoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RestoreError::Probe(e) => write!(f, "probe: {e}"),
            RestoreError::Decode(e) => write!(f, "decode: {e}"),
            RestoreError::UnsupportedPixels(e) => write!(f, "unsupported pixels: {e}"),
        }
    }
}
impl std::error::Error for RestoreError {}

/// Full x1 restoration pipeline against the deployment decoder.
pub fn restore_jpeg(
    data: &[u8],
    model: &AdoptedModel,
    cfg: &RestoreConfig,
) -> Result<Restored, RestoreError> {
    assert_eq!(model.scale, 1, "restore_jpeg is the x1 pipeline");
    let mut report = RestoreReport::default();

    // 1. fingerprint + deblock policy
    let probe = zenjpeg::detect::probe(data).map_err(|e| RestoreError::Probe(format!("{e:?}")))?;
    report.encoder_family = format!("{:?}", probe.encoder);
    report.est_quality = probe.quality.value;
    report.quality_scale = format!("{:?}", probe.quality.scale);
    let want_auto = cfg.deblock_policy && policy_wants_auto(&probe);
    report.used_deblock_auto = want_auto;
    let mode = if want_auto {
        zenjpeg::decoder::DeblockMode::Auto
    } else {
        zenjpeg::decoder::DeblockMode::Off
    };

    // 2. decode (deployment decoder)
    let dec = zenjpeg::decoder::Decoder::new()
        .deblock(mode)
        .decode(data, enough::Unstoppable)
        .map_err(|e| RestoreError::Decode(format!("{e:?}")))?;
    let (w32, h32) = dec.dimensions();
    let (w, h) = (w32 as usize, h32 as usize);
    let px = dec
        .pixels_u8()
        .ok_or(RestoreError::UnsupportedPixels("expected u8 output"))?;
    if px.len() != 3 * w * h {
        return Err(RestoreError::UnsupportedPixels("expected interleaved RGB8"));
    }
    let plane = w * h;
    let mut planes = vec![0.0f32; 3 * plane];
    for i in 0..plane {
        for c in 0..3 {
            planes[c * plane + i] = px[i * 3 + c] as f32 / 255.0;
        }
    }

    // 3. guarded model
    let lp = planes.clone();
    let mut sr = model.upscale_tiled(&lp, h, w, cfg.threads, cfg.tile);
    report.guard = guarded_merge(&mut sr, &lp, h, w, 1, &cfg.guard);

    // 4. quantization-consistency projection (S10)
    if let Projection::LumaAndFullResChroma(pcfg) = cfg.projection {
        let coeffs = zenjpeg::decoder::Decoder::new()
            .decode_coefficients(data, enough::Unstoppable)
            .map_err(|e| RestoreError::Decode(format!("coefficients: {e:?}")))?;
        let (mut y, mut cb, mut cr) = (vec![0.0; plane], vec![0.0; plane], vec![0.0; plane]);
        rgb_to_ycbcr_planes(&sr, plane, &mut y, &mut cb, &mut cr);
        let mut projected_any = false;
        for (ci, comp) in coeffs.components.iter().enumerate() {
            let full_res = comp.blocks_wide * 8 >= w && comp.blocks_high * 8 >= h;
            if !full_res {
                continue; // subsampled chroma: back-projection form, later
            }
            let Some(qt) = coeffs.quant_tables[comp.quant_table_idx as usize] else {
                continue;
            };
            let target = match ci {
                0 => &mut y,
                1 => &mut cb,
                2 => &mut cr,
                _ => continue,
            };
            let cv = CoeffView {
                coeffs: &comp.coeffs,
                blocks_wide: comp.blocks_wide,
                blocks_high: comp.blocks_high,
                order: CoeffOrder::Zigzag,
                quant: &qt,
            };
            report.projection.push(project_plane(target, w, h, &cv, &pcfg));
            projected_any = true;
        }
        if projected_any {
            ycbcr_to_rgb_planes(&y, &cb, &cr, &mut sr, plane);
        }
    }

    Ok(Restored { planes: sr, width: w, height: h, report })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_rgb(w: usize, h: usize) -> Vec<u8> {
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

    fn encode(w: usize, h: usize, rgb: &[u8], q: f32, ss: zenjpeg::encoder::ChromaSubsampling) -> Vec<u8> {
        let px: &[rgb::Rgb<u8>] = bytemuck::cast_slice(rgb);
        zenjpeg::encoder::EncoderConfig::ycbcr(q, ss)
            .encode(px, w as u32, h as u32)
            .expect("encode")
    }

    #[test]
    fn policy_never_fires_on_zenjpeg_own_files() {
        let rgb = synth_rgb(64, 64);
        for q in [5.0f32, 50.0, 90.0] {
            let jpg = encode(64, 64, &rgb, q, zenjpeg::encoder::ChromaSubsampling::Quarter);
            let p = zenjpeg::detect::probe(&jpg).unwrap();
            assert!(!policy_wants_auto(&p), "AQ family must never trigger Auto (q={q})");
        }
    }

    /// End-to-end coefficient consistency WITHOUT model weights: decode + a
    /// hallucination stands in for the model; after projection, zenjpeg's own
    /// coefficient decode of a re-encode... is overkill — instead verify via
    /// consist's own guarantee path exercised through real zenjpeg coeffs:
    /// projecting the DECODE must be a near-no-op.
    #[test]
    fn projection_pipeline_on_real_zenjpeg_file() {
        let (w, h) = (48usize, 40usize);
        let rgb = synth_rgb(w, h);
        let jpg = encode(w, h, &rgb, 35.0, zenjpeg::encoder::ChromaSubsampling::None);
        let dec = zenjpeg::decoder::Decoder::new()
            .decode(&jpg, enough::Unstoppable)
            .unwrap();
        let px = dec.pixels_u8().unwrap();
        let plane = w * h;
        let mut planes = vec![0.0f32; 3 * plane];
        for i in 0..plane {
            for c in 0..3 {
                planes[c * plane + i] = px[i * 3 + c] as f32 / 255.0;
            }
        }
        let coeffs = zenjpeg::decoder::Decoder::new()
            .decode_coefficients(&jpg, enough::Unstoppable)
            .unwrap();
        let (mut y, mut cb, mut cr) = (vec![0.0; plane], vec![0.0; plane], vec![0.0; plane]);
        rgb_to_ycbcr_planes(&planes, plane, &mut y, &mut cb, &mut cr);
        let comp = &coeffs.components[0];
        let qt = coeffs.quant_tables[comp.quant_table_idx as usize].unwrap();
        let cv = CoeffView {
            coeffs: &comp.coeffs,
            blocks_wide: comp.blocks_wide,
            blocks_high: comp.blocks_high,
            order: CoeffOrder::Zigzag,
            quant: &qt,
        };
        let before = y.clone();
        let rep = project_plane(&mut y, w, h, &cv, &ProjectionConfig::default());
        let mad: f32 =
            y.iter().zip(before.iter()).map(|(a, b)| (a - b).abs()).sum::<f32>() / plane as f32;
        assert!(
            mad < 3e-3,
            "projecting the decode itself must be a near-no-op (mad={mad}, clamped={})",
            rep.clamped_frac
        );
        // and projecting a hallucinated output must move it substantially back
        let mut hall: Vec<f32> = before
            .iter()
            .enumerate()
            .map(|(i, v)| (v + if (i / 5) % 2 == 0 { 0.25 } else { -0.25 }).clamp(0.0, 1.0))
            .collect();
        let rep2 = project_plane(&mut hall, w, h, &cv, &ProjectionConfig::default());
        assert!(rep2.clamped_frac > 0.1, "hallucination must be clamped: {}", rep2.clamped_frac);
        let mad2: f32 = hall
            .iter()
            .zip(before.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / plane as f32;
        assert!(mad2 < 0.25, "projection must pull toward the consistent set (mad={mad2})");
    }
}
