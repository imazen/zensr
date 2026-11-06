//! Verified encode-record correction shared with diagnostic tooling.
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
