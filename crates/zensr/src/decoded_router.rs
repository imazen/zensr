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
    // Match fixed-decimal serialization, including halfway ties. Multiplying
    // then rounding can introduce a new tie or round it in the other direction.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Window {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

fn windows(width: usize, height: usize) -> Vec<Window> {
    let (w, h) = (width.min(512), height.min(512));
    // Align with both 420 MCUs and luma DCT blocks; never resample pixels.
    let (last_x, last_y) = (((width - w) / 16) * 16, ((height - h) / 16) * 16);
    let (mid_x, mid_y) = (((width - w) / 32) * 16, ((height - h) / 32) * 16);
    let mut out = Vec::with_capacity(5);
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
        if !out.contains(&window) {
            out.push(window);
        }
    }
    out
}

fn crop_rgb(
    rgb: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    window: Window,
) -> Result<Vec<u8>, RestoreError> {
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
    let mut out = Vec::with_capacity(window.width * window.height * 3);
    for y in window.y..window.y + window.height {
        let start = y * stride + window.x * 3;
        out.extend_from_slice(&rgb[start..start + window.width * 3]);
    }
    Ok(out)
}

fn coefficient_window(
    coeffs: &[i16],
    blocks_wide: usize,
    width: usize,
    height: usize,
    window: Window,
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
        return crate::router_coefficients::features(coeffs, coeffs.len() / 64, blocks_wide)
            .ok_or_else(error);
    }
    let mut selected = Vec::with_capacity(bw * bh * 64);
    for row in y..y + bh {
        let start = (row * blocks_wide + x) * 64;
        selected.extend_from_slice(&coeffs[start..start + bw * 64]);
    }
    crate::router_coefficients::features(&selected, bw * bh, bw).ok_or_else(error)
}

pub(crate) fn for_source_prepared(
    probe: &crate::source::JpegInfo,
    legacy_gain: f32,
    recorded: bool,
    prepared: &mut dyn crate::source::Source,
) -> Result<Option<Prediction>, RestoreError> {
    for_source_prepared_windows(
        probe,
        legacy_gain,
        recorded,
        prepared,
        None,
        predict_candidate,
    )
}

fn for_source_prepared_windows(
    probe: &crate::source::JpegInfo,
    legacy_gain: f32,
    recorded: bool,
    prepared: &mut dyn crate::source::Source,
    selected: Option<&[Window]>,
    classify: impl Fn(&[f64; 37], &[&str; 3]) -> Option<Prediction>,
) -> Result<Option<Prediction>, RestoreError> {
    if probe.dimensions.width.min(probe.dimensions.height) < 8
        || probe.num_components != 3
        || crate::chroma_full_of(probe).is_none()
    {
        return Ok(None);
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
    let decoded = prepared.pixels(crate::source::DecodeMode::Plain)?;
    let (w, h) = decoded.dimensions();
    let rgb = decoded.data;
    let coefficients = prepared.coefficients()?;
    let plane = coefficients
        .components
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
        let crop = crop_rgb(rgb, width, height, stride, window)?;
        let offer = zenanalyze::offer_for_request(
            &crop,
            window.width as u32,
            window.height as u32,
            &crate::chooser::chooser_request(),
        )
        .map_err(|e| RestoreError::Decode(format!("router features: {e:?}")))?;
        let c = coefficient_window(&plane.coeffs, plane.blocks_wide, width, height, window)?;
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
        let family = if recorded {
            format!("{:?} (recorded)", probe.encoder)
        } else {
            format!("{:?}", probe.encoder)
        };
        let scale = format!("{:?}", probe.quality.scale);
        let ss = if crate::chroma_full_of(probe) == Some(true) {
            "444"
        } else {
            "420"
        };
        let prediction = classify(&numeric, &[&family, &scale, ss])
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
