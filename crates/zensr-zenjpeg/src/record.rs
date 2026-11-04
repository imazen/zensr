//! Authoritative encode parameters from zenjpeg's APP10 `ZENJ` record
//! (zenjpeg#189 / PR #253), folded into the probe the policies consume.
//!
//! Everything downstream of `probe()` — deblock policy, projection slack, the
//! identity gate, `estimate_gain` — keys off `probe.encoder` and
//! `probe.quality`. Those are fingerprint *estimates*, and on zenjpeg's own
//! Robidoux-table output they are wrong in the way that matters: family
//! `Unknown`, IJG quality 10-15 points low (zensr
//! `benchmarks/zenjpeg_knobs_2026-10-30.md`). When the file carries a record
//! whose newest entry's DQT CRC matches the tables, [`apply_record`] rewrites
//! the probe's family and quality from the dial the encoder was actually
//! given, so no policy needs to know the record exists.
//!
//! The one knob the record carries that the tables cannot show and that moves
//! restoration above the metric floor is adaptive quantization:
//! [`aq_off_gain_delta`] is the measured extra model gain on AQ-off encodes
//! at equal dial (63 clean-reference files, paired per file, rt24g).
//!
//! Everything here is a no-op on files without a verified record.

use zenjpeg::detect::record::{DialScale, EncodeEntry, TableSource};
use zenjpeg::detect::{Confidence, EncoderFamily, JpegProbe, QualityEstimate, QualityScale};
use zenjpeg::encoder::Quality;

/// The record entry that describes this file's coefficients, if any.
pub(crate) fn current_entry(probe: &JpegProbe) -> Option<&EncodeEntry> {
    probe.record.as_ref().and_then(|r| r.current())
}

/// Replace the fingerprint estimate with the recorded dial where the record
/// is verified and the table family tells us which scale the policies' curves
/// expect. Returns whether anything changed.
///
/// - Robidoux tables: the dial becomes the libjpeg quality that generated
///   them (`Quality::for_mozjpeg_tables`), family `Mozjpeg` — zenjpeg's
///   mozjpeg presets also trellis, so the mozjpeg slack is the right box.
/// - jpegli-lineage tables (`Jpegli`, `JpegliSharedChroma`): the dial becomes
///   the butteraugli distance the encoder targeted (`Quality::to_distance`),
///   family left as the fingerprint says (`Cjpegli*`, which it already does).
/// - `Custom`, `PiecewiseV4`, `GlassaLowBpp`, or a dial on the zensim scale:
///   the tables themselves remain the best estimate; untouched.
pub(crate) fn apply_record(probe: &mut JpegProbe) -> bool {
    let Some(e) = current_entry(probe).cloned() else {
        return false;
    };
    let dial = match e.quality_scale {
        DialScale::Jpegli => Quality::ApproxJpegli(e.quality),
        DialScale::Mozjpeg => Quality::ApproxMozjpeg(e.quality.round().clamp(0.0, 100.0) as u8),
        DialScale::Ssim2 => Quality::ApproxSsim2(e.quality),
        DialScale::Butteraugli => Quality::ApproxButteraugli(e.quality),
        _ => return false,
    };
    match e.table_source {
        TableSource::MozjpegRobidoux => {
            probe.encoder = EncoderFamily::Mozjpeg;
            probe.quality = QualityEstimate {
                value: dial.for_mozjpeg_tables() as f32,
                scale: QualityScale::MozjpegQuality,
                confidence: Confidence::Exact,
            };
            true
        }
        TableSource::Jpegli | TableSource::JpegliSharedChroma => {
            probe.quality = QualityEstimate {
                value: dial.to_distance(),
                scale: QualityScale::ButteraugliDistance,
                confidence: Confidence::Exact,
            };
            true
        }
        _ => false,
    }
}

/// Measured 2026-10-30 (`benchmarks/zenjpeg_knobs_2026-10-30.md`, rt24g, 63
/// files x 2 subsamplings, paired per file): median extra model gain, in
/// ssim2, when the same dial was encoded WITHOUT adaptive quantization,
/// indexed by the probe's butteraugli distance for that dial. AQ spends bits
/// perceptually rather than on SSIM, so an AQ-off file is both more damaged
/// by the metric and more restorable. Negative at the top: near-pristine
/// AQ-off files gain slightly less.
const AQ_OFF_DELTA_420: [(f32, f32); 5] = [
    (1.3, -0.03),
    (3.0, 0.13),
    (5.2, 0.25),
    (7.2, 0.35),
    (14.2, 0.56),
];
const AQ_OFF_DELTA_444: [(f32, f32); 5] = [
    (1.1, -0.10),
    (2.3, 0.04),
    (4.1, 0.17),
    (6.0, 0.29),
    (11.8, 0.36),
];

/// Adjustment to a predicted gain for a jpegli-lineage encode whose record
/// says adaptive quantization was off. Zero without a verified record, when
/// AQ was on, or for table families where AQ never applied.
pub(crate) fn aq_off_gain_delta(probe: &JpegProbe) -> f32 {
    let Some(e) = current_entry(probe) else {
        return 0.0;
    };
    if e.flags.has(zenjpeg::detect::record::EncodeFlags::AQ)
        || !matches!(
            e.table_source,
            TableSource::Jpegli | TableSource::JpegliSharedChroma
        )
        || !matches!(probe.quality.scale, QualityScale::ButteraugliDistance)
    {
        return 0.0;
    }
    let pts: &[(f32, f32)] = if crate::chroma_full_of(probe) == Some(true) {
        &AQ_OFF_DELTA_444
    } else {
        &AQ_OFF_DELTA_420
    };
    crate::api::lerp(pts, probe.quality.value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use enough::Unstoppable;
    use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, OptimizationPreset, PixelLayout};

    fn encode(cfg: &EncoderConfig) -> Vec<u8> {
        let (w, h) = (64u32, 48u32);
        let rgb: Vec<u8> = (0..w * h)
            .flat_map(|i| [(i * 7 % 251) as u8, (i * 13 % 241) as u8, (i % 239) as u8])
            .collect();
        let mut enc = cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
        enc.push_packed(&rgb, Unstoppable).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn robidoux_record_corrects_family_and_quality() {
        let jpg = encode(
            &EncoderConfig::ycbcr(55.0, ChromaSubsampling::Quarter)
                .optimization(OptimizationPreset::MozjpegBaseline),
        );
        let mut p = zenjpeg::detect::probe(&jpg).unwrap();
        let before = (format!("{:?}", p.encoder), p.quality.value);
        assert!(apply_record(&mut p));
        assert!(matches!(p.encoder, EncoderFamily::Mozjpeg));
        assert!(matches!(p.quality.scale, QualityScale::MozjpegQuality));
        assert_eq!(
            p.quality.value,
            Quality::ApproxJpegli(55.0).for_mozjpeg_tables() as f32
        );
        assert_ne!(before, (format!("{:?}", p.encoder), p.quality.value));
        assert_eq!(
            aq_off_gain_delta(&p),
            0.0,
            "AQ never applies to Robidoux tables"
        );
    }

    #[test]
    fn jpegli_record_sets_distance_and_aq_delta() {
        let jpg = encode(&EncoderConfig::ycbcr(35.0, ChromaSubsampling::Quarter).aq_enabled(false));
        let mut p = zenjpeg::detect::probe(&jpg).unwrap();
        assert!(apply_record(&mut p));
        assert!(matches!(p.quality.scale, QualityScale::ButteraugliDistance));
        assert!((p.quality.value - Quality::ApproxJpegli(35.0).to_distance()).abs() < 1e-6);
        let d = aq_off_gain_delta(&p);
        assert!((0.25..=0.45).contains(&d), "{d}");
        let on = encode(&EncoderConfig::ycbcr(35.0, ChromaSubsampling::Quarter));
        let mut p = zenjpeg::detect::probe(&on).unwrap();
        apply_record(&mut p);
        assert_eq!(aq_off_gain_delta(&p), 0.0);
    }

    #[test]
    fn no_record_is_a_no_op() {
        let jpg =
            encode(&EncoderConfig::ycbcr(70.0, ChromaSubsampling::Quarter).encode_record(false));
        let mut p = zenjpeg::detect::probe(&jpg).unwrap();
        let before = format!("{:?}", p.quality);
        assert!(!apply_record(&mut p));
        assert_eq!(before, format!("{:?}", p.quality));
        assert_eq!(aq_off_gain_delta(&p), 0.0);
    }
}
