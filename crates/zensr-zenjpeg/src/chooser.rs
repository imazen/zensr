//! Content-class chooser: routes decoded JPEGs to the photo or graphics
//! specialist model (feature `chooser`).
//!
//! A 21-feature logistic rule over zenanalyze features, fit 2026-07-26 on
//! compressed-then-decoded 512px crops (turbo 420, q35/75/92) of the
//! imazen-26 training files, validated on the pinned eval split
//! (benchmarks/chooser_fit_2026-07-26.txt). Deliberately PRECISION-biased
//! toward Photo: misrouting a photo into the aggressive graphics model is
//! the harmful direction; graphics falling back to the photo model is
//! merely conservative. At the default threshold the pinned-eval numbers
//! are: documents 24/24 routed, screen 21/24, maps 12/24, art-scans 6/24
//! (scans look photographic to these features and intentionally fall
//! through), false-positives 2 files of 96 (one an actual screenshot
//! stored in the photo class, one a 3D render).
//!
//! Runtime contract: features are computed on the CENTER 512x512 crop of
//! the decoded image (whole image when smaller) — the same geometry the
//! rule was calibrated on. Size-dependent and near-constant features were
//! excluded from the fit.

use zenanalyze::analyze_features_rgb8;
use zenanalyze::feature::{AnalysisQuery, FeatureSet};

/// Which specialist family a decoded image should be restored with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ContentClass {
    /// Natural-image path (default; also the safe fallback).
    Photo,
    /// Document / text / lineart / clipart / infographic path — the
    /// specialist model may correct more aggressively.
    Graphics,
}

/// Chooser decision + the probability behind it.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct ChooserReport {
    pub class: ContentClass,
    /// Logistic p(graphics) in [0,1].
    pub p_graphics: f32,
}

// baked 2026-07-26 from chooser_features.tsv (extracted on lianli),
// sklearn L1-select + L2-refit; (name, train-median, train-IQR, weight).
const CHOOSER_FEATURES: &[(&str, f32, f32, f32)] = &[
    ("variance", 2558.88, 3411.01, 0.68966),
    ("colourfulness", 15.3235, 26.3934, -1.28115),
    ("laplacian_variance", 0.926529, 2.57378, -2.0077),
    ("variance_spread", 1.06317, 0.548007, -0.96721),
    ("distinct_color_bins", 659.0, 1833.0, 0.244856),
    ("palette_density", 0.020111, 0.0559385, 0.244804),
    ("luma_histogram_entropy", 2.17591, 1.70967, -2.15949),
    ("luma_kurtosis", 10.7992, 19.2993, 0.351327),
    ("skin_tone_fraction", 0.005394, 0.154432, 0.228648),
    ("chroma_luma_covariance_cb", -0.164535, 0.515691, -0.601356),
    ("chroma_luma_covariance_cr", 0.034303, 0.380783, 0.229066),
    ("cr_peak_sharpness", 0.0, 2.0, -0.176541),
    ("orientation_energy_ratio", 1.14894, 0.0731275, -0.381569),
    ("aq_map_p90", 5.11083, 1.26966, 0.515373),
    ("noise_floor_y_p25", 0.0, 0.118233, -0.526114),
    ("noise_floor_y_p90", 1.0, 0.004217, 0.033027),
    ("noise_floor_uv_p50", 0.0, 0.029115, 0.029444),
    ("noise_floor_uv_p75", 0.026731, 0.087091, 0.214859),
    ("noise_floor_uv_p90", 0.082923, 0.233621, 0.204648),
    ("quant_survival_y", 0.071506, 0.088433, 1.29063),
    ("xyb444_color_loss", 0.692348, 0.030016, -0.398488),
];
const CHOOSER_BIAS: f32 = 2.6208;
/// p(graphics) gate. Precision-biased; revisit once the measured cost of
/// misrouting a photo through the graphics model is known.
pub const CHOOSER_THRESHOLD: f32 = 0.85;

/// Classify a decoded RGB8 image. Analyzes the center 512x512 crop (whole
/// image when smaller) to match the calibration geometry.
pub fn classify_rgb8(rgb: &[u8], w: usize, h: usize) -> ChooserReport {
    assert_eq!(rgb.len(), 3 * w * h);
    let (cw, ch) = (w.min(512), h.min(512));
    let (x0, y0) = ((w - cw) / 2, (h - ch) / 2);
    let mut crop = Vec::with_capacity(3 * cw * ch);
    for y in 0..ch {
        let row = &rgb[((y0 + y) * w + x0) * 3..][..cw * 3];
        crop.extend_from_slice(row);
    }
    let res = analyze_features_rgb8(&crop, cw as u32, ch as u32, &AnalysisQuery::new(FeatureSet::SUPPORTED));
    let mut s = CHOOSER_BIAS;
    for &(name, med, iqr, wgt) in CHOOSER_FEATURES {
        let v = FeatureSet::SUPPORTED
            .iter()
            .find(|f| f.name() == name)
            .and_then(|f| res.get(f))
            .map(|v| v.to_f32())
            .unwrap_or(med); // absent feature contributes z=0 (neutral)
        let z = ((v - med) / iqr).clamp(-8.0, 8.0);
        s += wgt * z;
    }
    let p = 1.0 / (1.0 + (-s).exp());
    ChooserReport {
        class: if p > CHOOSER_THRESHOLD { ContentClass::Graphics } else { ContentClass::Photo },
        p_graphics: p,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_photo_and_bounded() {
        // dense random noise: max entropy + huge laplacian -> firmly photo
        let (w, h) = (256usize, 256usize);
        let mut s = 0x2545F4914F6CDD1Du64;
        let px: Vec<u8> = (0..3 * w * h)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 33) as u8
            })
            .collect();
        let r = classify_rgb8(&px, w, h);
        assert!((0.0..=1.0).contains(&r.p_graphics));
        assert_eq!(r.class, ContentClass::Photo, "noise p={}", r.p_graphics);
    }

    #[test]
    fn flat_panels_score_more_graphics_than_noise() {
        let (w, h) = (256usize, 256usize);
        // two flat panels: low entropy, zero noise floor, low colourfulness
        let mut px = vec![235u8; 3 * w * h];
        for y in h / 2..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                px[i] = 40;
                px[i + 1] = 40;
                px[i + 2] = 48;
            }
        }
        let flat = classify_rgb8(&px, w, h);
        let mut s = 1u64;
        let noise: Vec<u8> = (0..3 * w * h)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                (s >> 33) as u8
            })
            .collect();
        let n = classify_rgb8(&noise, w, h);
        assert!(
            flat.p_graphics > n.p_graphics,
            "flat {} vs noise {}",
            flat.p_graphics,
            n.p_graphics
        );
    }
}
