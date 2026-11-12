//! Metadata-independent linear routing, selected by the owner on 2026-11-06.
//! Native analysis windows preserve JPEG artifact scale at arbitrary sizes.
//! The historical baseline remains reproducible in tests. Larger images require
//! all sampled windows to accept restoration. Explicit caller overrides stay intact.
use crate::RestoreError;
use crate::decoded_router_features as definitions;
#[cfg(test)]
use crate::decoded_router_model as fitted;
use std::sync::OnceLock;

#[cfg(test)]
#[path = "decoded_router/spatial.rs"]
mod spatial;

#[path = "decoded_router_candidate.rs"]
mod candidate;

struct Heads {
    mean: &'static [f64; 37],
    scale: &'static [f64; 37],
    gain: &'static [f64; 47],
    risk: &'static [f64; 47],
    gain_bias: f64,
    risk_bias: f64,
    categories: [&'static [&'static str]; 3],
}

#[cfg(test)]
const FROZEN: Heads = Heads {
    mean: &fitted::MEAN,
    scale: &fitted::SCALE,
    gain: &fitted::GAIN,
    risk: &fitted::RISK,
    gain_bias: fitted::GAIN_BIAS,
    risk_bias: fitted::RISK_BIAS,
    categories: fitted::CATEGORIES,
};

const CANDIDATE: Heads = Heads {
    mean: &candidate::MEAN,
    scale: &candidate::SCALE,
    gain: &candidate::GAIN,
    risk: &candidate::RISK,
    gain_bias: candidate::GAIN_BIAS,
    risk_bias: candidate::RISK_BIAS,
    categories: candidate::CATEGORIES,
};

fn predict_candidate(numeric: &[f64; 37], categories: &[&str; 3]) -> Option<Prediction> {
    predict_heads(numeric, categories, &CANDIDATE)
}

pub(crate) struct Prediction {
    gain: f64,
    risk: f64,
}

impl Prediction {
    pub(crate) fn score(&self) -> f64 {
        self.gain - candidate::PENALTY * self.risk
    }

    pub(crate) fn restore(&self) -> bool {
        self.score() >= candidate::THRESHOLD
    }
}

#[cfg(test)]
fn predict(numeric: &[f64; 37], categories: &[&str; 3]) -> Option<Prediction> {
    predict_heads(numeric, categories, &FROZEN)
}

fn predict_heads(numeric: &[f64; 37], categories: &[&str; 3], heads: &Heads) -> Option<Prediction> {
    if !numeric.iter().all(|v| v.is_finite()) {
        return None;
    }
    let mut gain = heads.gain_bias;
    let mut risk = heads.risk_bias;
    for (i, value) in numeric.iter().enumerate() {
        let z = (value - heads.mean[i]) / heads.scale[i];
        gain += z * heads.gain[i];
        risk += z * heads.risk[i];
    }
    let mut offset = numeric.len();
    for (value, known) in categories.iter().zip(heads.categories) {
        for (i, category) in known.iter().enumerate() {
            if value == category {
                gain += heads.gain[offset + i];
                risk += heads.risk[offset + i];
            }
        }
        offset += known.len();
    }
    // The same stable logistic link as sklearn; avoid overflowing exp(-x).
    let risk = if risk >= 0.0 {
        1.0 / (1.0 + (-risk).exp())
    } else {
        let e = risk.exp();
        e / (1.0 + e)
    };
    (gain.is_finite() && risk.is_finite()).then_some(Prediction { gain, risk })
}

// The frozen tables serialize numeric inputs to twelve decimal places.
fn recorded_number(value: f64) -> f64 {
    struct StackFmt {
        buf: [u8; 64],
        len: usize,
    }
    impl core::fmt::Write for StackFmt {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            let bytes = s.as_bytes();
            let dst = self
                .buf
                .get_mut(self.len..self.len + bytes.len())
                .ok_or(core::fmt::Error)?;
            dst.copy_from_slice(bytes);
            self.len += bytes.len();
            Ok(())
        }
    }
    // Match fixed-decimal serialization, including halfway ties, without heap
    // allocation for moderate magnitudes. Multiplying then rounding can
    // introduce a new tie or round it in the other direction.
    let mut stack = StackFmt {
        buf: [0u8; 64],
        len: 0,
    };
    if core::fmt::Write::write_fmt(&mut stack, format_args!("{value:.12}")).is_ok()
        && let Ok(s) = core::str::from_utf8(&stack.buf[..stack.len])
    {
        return s.parse().expect("formatted study scalar");
    }
    format!("{value:.12}")
        .parse()
        .expect("formatted study scalar")
}

fn definitions_match() -> bool {
    static MATCH: OnceLock<bool> = OnceLock::new();
    *MATCH.get_or_init(|| {
        let actual = zenanalyze::versioning::feature_qualified_names();
        definitions::PIXELS
            .iter()
            .all(|(name, qualified)| actual.iter().any(|(n, q)| n == name && q == qualified))
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Window {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

#[derive(Clone, Copy, Debug)]
struct WindowList {
    buf: [Window; 5],
    len: usize,
}

impl core::ops::Deref for WindowList {
    type Target = [Window];
    fn deref(&self) -> &[Window] {
        &self.buf[..self.len]
    }
}

fn windows(width: usize, height: usize) -> WindowList {
    let (w, h) = (width.min(512), height.min(512));
    // Align with both 420 MCUs and luma DCT blocks; never resample pixels.
    let (last_x, last_y) = (((width - w) / 16) * 16, ((height - h) / 16) * 16);
    let (mid_x, mid_y) = (((width - w) / 32) * 16, ((height - h) / 32) * 16);
    let mut out = WindowList {
        buf: [Window::default(); 5],
        len: 0,
    };
    for (x, y) in [
        (mid_x, mid_y),
        (0, 0),
        (last_x, 0),
        (0, last_y),
        (last_x, last_y),
    ] {
        let window = Window {
            x,
            y,
            width: w,
            height: h,
        };
        if !out[..out.len].contains(&window) {
            out.buf[out.len] = window;
            out.len += 1;
        }
    }
    out
}

#[derive(Debug, Default)]
pub(crate) struct DecodedRouterScratch {
    crop: Vec<u8>,
    selected_coeffs: Vec<i16>,
    coeff_scratch: crate::router_coefficients::CoeffRouterScratch,
    family: String,
}

#[cfg(test)]
fn crop_rgb(
    rgb: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    window: Window,
) -> Result<Vec<u8>, RestoreError> {
    let mut out = Vec::new();
    crop_rgb_into(rgb, width, height, stride, window, &mut out)?;
    Ok(out)
}

fn crop_rgb_into(
    rgb: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    window: Window,
    out: &mut Vec<u8>,
) -> Result<(), RestoreError> {
    let row_bytes = width
        .checked_mul(3)
        .ok_or(RestoreError::UnsupportedPixels("router row size"))?;
    let required = height
        .checked_sub(1)
        .and_then(|h| h.checked_mul(stride))
        .and_then(|n| n.checked_add(row_bytes));
    if stride < row_bytes
        || required.is_none_or(|n| n > rgb.len())
        || window.x + window.width > width
        || window.y + window.height > height
    {
        return Err(RestoreError::UnsupportedPixels("router pixel geometry"));
    }
    out.clear();
    out.reserve(window.width * window.height * 3);
    for y in window.y..window.y + window.height {
        let start = y * stride + window.x * 3;
        out.extend_from_slice(&rgb[start..start + window.width * 3]);
    }
    Ok(())
}

#[cfg(test)]
fn coefficient_window(
    coeffs: &[i16],
    blocks_wide: usize,
    width: usize,
    height: usize,
    window: Window,
) -> Result<crate::router_coefficients::Feats, RestoreError> {
    let mut selected = Vec::new();
    let mut scratch = crate::router_coefficients::CoeffRouterScratch::default();
    coefficient_window_with_scratch(
        coeffs,
        blocks_wide,
        width,
        height,
        window,
        &mut selected,
        &mut scratch,
    )
}

fn coefficient_window_with_scratch(
    coeffs: &[i16],
    blocks_wide: usize,
    width: usize,
    height: usize,
    window: Window,
    selected: &mut Vec<i16>,
    scratch: &mut crate::router_coefficients::CoeffRouterScratch,
) -> Result<crate::router_coefficients::Feats, RestoreError> {
    let error = || RestoreError::UnsupportedPixels("router coefficient geometry");
    if blocks_wide == 0
        || !coeffs.len().is_multiple_of(64)
        || !(coeffs.len() / 64).is_multiple_of(blocks_wide)
    {
        return Err(error());
    }
    let blocks_high = coeffs.len() / 64 / blocks_wide;
    // Preserve MCU padding exactly when the window spans a complete axis.
    let bw = if window.width == width {
        blocks_wide
    } else {
        window.width.div_ceil(8)
    };
    let bh = if window.height == height {
        blocks_high
    } else {
        window.height.div_ceil(8)
    };
    let (x, y) = (window.x / 8, window.y / 8);
    if x + bw > blocks_wide || y + bh > blocks_high {
        return Err(error());
    }
    if x == 0 && y == 0 && bw == blocks_wide && bh == blocks_high {
        return crate::router_coefficients::features_with_scratch(
            coeffs,
            coeffs.len() / 64,
            blocks_wide,
            scratch,
        )
        .ok_or_else(error);
    }
    selected.clear();
    selected.reserve(bw * bh * 64);
    for row in y..y + bh {
        let start = (row * blocks_wide + x) * 64;
        selected.extend_from_slice(&coeffs[start..start + bw * 64]);
    }
    crate::router_coefficients::features_with_scratch(selected, bw * bh, bw, scratch)
        .ok_or_else(error)
}

pub(crate) fn for_source_prepared_with_scratch(
    legacy_gain: f32,
    recorded: bool,
    prepared: &mut dyn crate::source::Source,
    scratch: &mut DecodedRouterScratch,
) -> Result<Option<Prediction>, RestoreError> {
    for_source_prepared_windows_with_scratch(
        legacy_gain,
        recorded,
        prepared,
        None,
        predict_candidate,
        scratch,
    )
}

fn for_source_prepared_windows_with_scratch(
    legacy_gain: f32,
    recorded: bool,
    prepared: &mut dyn crate::source::Source,
    selected: Option<&[Window]>,
    classify: impl Fn(&[f64; 37], &[&str; 3]) -> Option<Prediction>,
    scratch: &mut DecodedRouterScratch,
) -> Result<Option<Prediction>, RestoreError> {
    {
        let probe = prepared.info();
        if probe.dimensions.width.min(probe.dimensions.height) < 8
            || probe.num_components != 3
            || crate::chroma_full_of(probe).is_none()
        {
            return Ok(None);
        }
    }
    if !definitions_match() {
        return Err(RestoreError::UnsupportedModel(
            "decoded router feature definitions changed",
        ));
    }
    prepared.prepare(crate::source::Needs {
        pixels: Some(crate::source::DecodeMode::Plain),
        coefficients: true,
    })?;
    let probe = prepared.info();
    let decoded = prepared.pixels(crate::source::DecodeMode::Plain)?;
    let (w, h) = decoded.dimensions();
    let rgb = decoded.data;
    let coefficients = prepared.coefficients_view()?;
    let plane = coefficients
        .first()
        .ok_or(RestoreError::UnsupportedPixels("missing luma"))?;
    let (width, height) = (w as usize, h as usize);
    let stride = decoded.stride;
    let mut weakest: Option<Prediction> = None;
    let default_windows;
    let regions = if let Some(selected) = selected {
        selected
    } else {
        default_windows = windows(width, height);
        &default_windows
    };
    for &window in regions {
        let offer_slice: &[u8] = if window.x == 0
            && window.y == 0
            && window.width == width
            && window.height == height
            && stride == width * 3
        {
            let needed = width
                .checked_mul(height)
                .and_then(|n| n.checked_mul(3))
                .ok_or(RestoreError::UnsupportedPixels("router row size"))?;
            if rgb.len() < needed {
                return Err(RestoreError::UnsupportedPixels("router pixel geometry"));
            }
            &rgb[..needed]
        } else {
            crop_rgb_into(rgb, width, height, stride, window, &mut scratch.crop)?;
            &scratch.crop
        };
        let offer = zenanalyze::offer_for_request(
            offer_slice,
            window.width as u32,
            window.height as u32,
            &crate::chooser::chooser_request(),
        )
        .map_err(|e| RestoreError::Decode(format!("router features: {e:?}")))?;
        let c = coefficient_window_with_scratch(
            plane.coeffs,
            plane.blocks_wide,
            width,
            height,
            window,
            &mut scratch.selected_coeffs,
            &mut scratch.coeff_scratch,
        )?;
        let window_gain = if window.width == width && window.height == height {
            legacy_gain
        } else {
            let content =
                if c.zero_ac_blocks as f32 >= crate::calibration::GRAPHIC_ZERO_AC_THRESHOLD {
                    crate::api::Content::Graphic
                } else {
                    crate::api::Content::Photographic
                };
            crate::api::estimate_gain_for(probe, Some(content))
        };
        let mut numeric = [0.; 37];
        numeric[..13].copy_from_slice(&[
            window.width as f64,
            window.height as f64,
            f64::from(probe.quality.value),
            f64::from(window_gain),
            probe.encoded_len as f64 * 8. / (f64::from(w) * f64::from(h)),
            c.zero_ac_blocks,
            c.zero_ac_coefs,
            c.hf_survival,
            c.mean_abs_ac,
            c.dc_spread,
            c.ac_count_spread,
            c.gutted_block_frac,
            c.gutted_clustering,
        ]);
        for (target, (name, _)) in numeric[13..].iter_mut().zip(definitions::PIXELS) {
            *target = f64::from(
                offer
                    .get(name)
                    .ok_or(RestoreError::UnsupportedModel("missing router feature"))?
                    .float(),
            );
        }
        numeric
            .iter_mut()
            .for_each(|value| *value = recorded_number(*value));
        let family: &str = if recorded {
            scratch.family.clear();
            scratch.family.push_str(probe.encoder.as_str());
            scratch.family.push_str(" (recorded)");
            &scratch.family
        } else {
            probe.encoder.as_str()
        };
        let scale = probe.quality.scale.as_str();
        let ss = if crate::chroma_full_of(probe) == Some(true) {
            "444"
        } else {
            "420"
        };
        let prediction = classify(&numeric, &[family, scale, ss])
            .ok_or(RestoreError::UnsupportedModel("non-finite router input"))?;
        if weakest
            .as_ref()
            .is_none_or(|previous| prediction.score() < previous.score())
        {
            weakest = Some(prediction);
        }
    }
    Ok(weakest)
}
#[cfg(test)]
mod tests {
    use super::*;

    fn replay(text: &str) -> usize {
        replay_using(text, predict)
    }

    fn replay_using(
        text: &str,
        classifier: impl Fn(&[f64; 37], &[&str; 3]) -> Option<Prediction>,
    ) -> usize {
        let mut count = 0;
        for line in text.lines().skip(1) {
            let fields: Vec<_> = line.split('\t').collect();
            assert_eq!(fields.len(), 43);
            let numeric = core::array::from_fn(|i| fields[i].parse::<f64>().unwrap());
            let p = classifier(&numeric, &[fields[37], fields[38], fields[39]]).unwrap();
            let gain: f64 = fields[40].parse().unwrap();
            let risk: f64 = fields[41].parse().unwrap();
            assert!((p.gain - gain).abs() < 1e-9, "gain mismatch at row {count}");
            assert!(
                (p.risk - risk).abs() < 1e-11,
                "risk mismatch at row {count}"
            );
            assert_eq!(
                p.restore(),
                fields[42] == "1",
                "decision mismatch at row {count}"
            );
            count += 1;
        }
        count
    }

    #[test]
    fn sklearn_golden_predictions_and_decisions() {
        assert_eq!(definitions::PIXELS, fitted::PIXELS);
        assert_eq!(
            (candidate::PENALTY, candidate::THRESHOLD),
            (fitted::PENALTY, fitted::THRESHOLD)
        );
        assert_eq!(replay(include_str!("data/decoded_router_golden.tsv")), 16);
    }

    #[test]
    fn candidate_sklearn_golden_predictions_and_decisions() {
        assert_eq!(
            replay_using(
                include_str!("data/decoded_router_candidate_golden.tsv"),
                predict_candidate
            ),
            16
        );
    }

    #[test]
    fn rejects_nonfinite_inputs() {
        let mut numeric = [0.; 37];
        numeric[4] = f64::NAN;
        assert!(predict(&numeric, &["Unknown", "IjgQuality", "420"]).is_none());
    }

    #[test]
    fn study_decimal_serialization_matches_halfway_values() {
        assert_eq!(recorded_number(0.0001220703125), 0.000122070312);
        assert_eq!(recorded_number(0.0003662109375), 0.000366210938);
    }

    #[test]
    fn native_windows_are_bounded_aligned_and_deduplicated() {
        for (w, h) in [
            (8, 8),
            (31, 17),
            (256, 192),
            (512, 384),
            (513, 385),
            (1024, 768),
            (4096, 128),
            (128, 4096),
        ] {
            let regions = windows(w, h);
            assert!(!regions.is_empty() && regions.len() <= 5);
            for (i, r) in regions.iter().enumerate() {
                assert!(
                    r.width <= 512 && r.height <= 512 && r.x + r.width <= w && r.y + r.height <= h
                );
                assert_eq!((r.x % 16, r.y % 16), (0, 0));
                assert!(!regions[..i].contains(r));
            }
            if w <= 512 && h <= 512 {
                assert_eq!(
                    &*regions,
                    &[Window {
                        x: 0,
                        y: 0,
                        width: w,
                        height: h
                    }]
                );
            }
        }
    }

    #[test]
    fn crop_handles_strided_rows_and_coefficient_windows_preserve_padding() {
        let rgb = [1, 2, 3, 4, 5, 6, 99, 99, 7, 8, 9, 10, 11, 12];
        let window = Window {
            x: 1,
            y: 0,
            width: 1,
            height: 2,
        };
        assert_eq!(
            crop_rgb(&rgb, 2, 2, 8, window).unwrap(),
            vec![4, 5, 6, 10, 11, 12]
        );
        let mut coeffs = vec![0i16; 4 * 4 * 64];
        for (i, b) in coeffs.chunks_mut(64).enumerate() {
            b[0] = i as i16;
            b[1] = i as i16;
        }
        let whole = coefficient_window(
            &coeffs,
            4,
            17,
            17,
            Window {
                x: 0,
                y: 0,
                width: 17,
                height: 17,
            },
        )
        .unwrap();
        let reference = crate::router_coefficients::features(&coeffs, 16, 4).unwrap();
        assert_eq!(whole.mean_abs_ac, reference.mean_abs_ac);
        assert_eq!(whole.zero_ac_blocks, reference.zero_ac_blocks);
        let local = coefficient_window(
            &coeffs,
            4,
            17,
            17,
            Window {
                x: 8,
                y: 8,
                width: 8,
                height: 8,
            },
        )
        .unwrap();
        assert_eq!(local.mean_abs_ac, 5.);
        assert_eq!(local.dc_spread, 0.);
    }
}
