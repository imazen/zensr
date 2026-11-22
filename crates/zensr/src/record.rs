//! The codec authenticates metadata; zensr retains the calibrated policy.
use crate::source::JpegInfo;
pub(crate) fn is_recorded(probe: &JpegInfo) -> bool {
    probe.recorded
}
pub(crate) fn aq_off_gain_delta(probe: &JpegInfo) -> f32 {
    if !probe.jpegli_aq_off {
        return 0.0;
    }
    let points = if crate::chroma_full_of(probe) == Some(true) {
        crate::calibration::AQ_OFF_DELTA_444
    } else {
        crate::calibration::AQ_OFF_DELTA_420
    };
    crate::api::lerp(points, probe.quality.value)
}
