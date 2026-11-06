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
//! Adaptive quantization is a record-carried knob whose restoration effect
//! cannot be inferred from the quantization tables alone:
//! [`aq_off_gain_delta`] is the measured extra model gain on AQ-off encodes
//! at equal dial (canon4, paired per file on the 33 calibration images).
//!
//! Everything here is a no-op on files without a verified record.

mod probe;
pub(crate) use probe::{apply_record, current_entry};
use zenjpeg::detect::record::TableSource;
use zenjpeg::detect::{JpegProbe, QualityScale};

// Calibrated from paired canon4 projected gains on calibration images only.
// Source: benchmarks/ship_rt24i_canon4_2026-11-04.md.
use crate::calibration::{AQ_OFF_DELTA_420, AQ_OFF_DELTA_444};

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
        AQ_OFF_DELTA_444
    } else {
        AQ_OFF_DELTA_420
    };
    crate::api::lerp(pts, probe.quality.value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use enough::Unstoppable;
    use zenjpeg::detect::EncoderFamily;
    use zenjpeg::encoder::Quality;
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
