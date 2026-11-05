//! Frozen decoded-linear-1 routing, selected by the owner on 2026-11-05.
//! Native analysis windows preserve JPEG artifact scale at arbitrary sizes.
//! The 512-pixel measured path remains unchanged; larger images require all
//! sampled windows to accept restoration. Explicit caller overrides stay intact.
use crate::decoded_router_model as fitted;
use crate::RestoreError;
use std::sync::OnceLock;

pub(crate) struct Prediction {
    gain: f64,
    risk: f64,
}

impl Prediction {
    pub(crate) fn score(&self) -> f64 {
        self.gain - fitted::PENALTY * self.risk
    }

    pub(crate) fn restore(&self) -> bool {
        self.score() >= fitted::THRESHOLD
    }
}

fn predict(numeric: &[f64; 37], categories: &[&str; 3]) -> Option<Prediction> {
    if !numeric.iter().all(|v| v.is_finite()) {
        return None;
    }
    let mut gain = fitted::GAIN_BIAS;
    let mut risk = fitted::RISK_BIAS;
    for (i, value) in numeric.iter().enumerate() {
        let z = (value - fitted::MEAN[i]) / fitted::SCALE[i];
        gain += z * fitted::GAIN[i];
        risk += z * fitted::RISK[i];
    }
    let mut offset = numeric.len();
    for (value, known) in categories.iter().zip(fitted::CATEGORIES) {
        for (i, category) in known.iter().enumerate() {
            if value == category {
                gain += fitted::GAIN[offset + i];
                risk += fitted::RISK[offset + i];
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
        fitted::PIXELS
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

pub(crate) fn for_jpeg(
    data: &[u8],
    probe: &zenjpeg::detect::JpegProbe,
    legacy_gain: f32,
    recorded: bool,
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
    let decoded = zenjpeg::decoder::Decoder::new()
        .deblock(zenjpeg::decoder::DeblockMode::Off)
        .decode(data, enough::Unstoppable)
        .map_err(|e| RestoreError::Decode(format!("{e:?}")))?;
    let (w, h) = decoded.dimensions();
    let rgb = decoded
        .pixels_u8()
        .ok_or(RestoreError::UnsupportedPixels("router requires RGB8"))?;
    let coefficients = zenjpeg::decoder::Decoder::new()
        .decode_coefficients(data, enough::Unstoppable)
        .map_err(|e| RestoreError::Decode(format!("{e:?}")))?;
    let plane = coefficients
        .components
        .first()
        .ok_or(RestoreError::UnsupportedPixels("missing luma"))?;
    let (width, height) = (w as usize, h as usize);
    let stride = width
        .checked_mul(3)
        .ok_or(RestoreError::UnsupportedPixels("router row size"))?;
    let mut weakest: Option<Prediction> = None;
    for window in windows(width, height) {
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
            data.len() as f64 * 8. / (f64::from(w) * f64::from(h)),
            c.zero_ac_blocks,
            c.zero_ac_coefs,
            c.hf_survival,
            c.mean_abs_ac,
            c.dc_spread,
            c.ac_count_spread,
            c.gutted_block_frac,
            c.gutted_clustering,
        ]);
        for (target, (name, _)) in numeric[13..].iter_mut().zip(fitted::PIXELS) {
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
        let prediction = predict(&numeric, &[&family, &scale, ss])
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
        let mut count = 0;
        for line in text.lines().skip(1) {
            let fields: Vec<_> = line.split('\t').collect();
            assert_eq!(fields.len(), 43);
            let numeric = core::array::from_fn(|i| fields[i].parse::<f64>().unwrap());
            let p = predict(&numeric, &[fields[37], fields[38], fields[39]]).unwrap();
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
        assert_eq!(
            replay(include_str!("../tests/data/decoded_router_golden.tsv")),
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
                    regions,
                    vec![Window {
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

    #[test]
    fn jpeg_routing_supports_native_sizes_and_extreme_aspects() {
        for (w, h) in [
            (8, 8),
            (31, 17),
            (128, 96),
            (511, 129),
            (513, 65),
            (1024, 64),
            (2048, 32),
            (32, 2048),
            (4096, 2160),
        ] {
            let pixels: Vec<rgb::Rgb<u8>> = (0..w * h)
                .map(|i| {
                    rgb::Rgb::new(
                        (i % 251) as u8,
                        ((i / w) * 13 % 256) as u8,
                        ((i * 7) % 256) as u8,
                    )
                })
                .collect();
            let jpg = zenjpeg::encoder::EncoderConfig::ycbcr(
                55.,
                zenjpeg::encoder::ChromaSubsampling::Quarter,
            )
            .encode(&pixels, w as u32, h as u32)
            .unwrap();
            let mut probe = zenjpeg::detect::probe(&jpg).unwrap();
            let recorded = crate::record::apply_record(&mut probe);
            let legacy = crate::api::estimate_gain_for(&probe, crate::api::classify_content(&jpg));
            let result = for_jpeg(&jpg, &probe, legacy, recorded)
                .unwrap()
                .expect("native size must use learned router");
            assert!(result.score().is_finite());
        }
    }

    #[test]
    fn valid_flat_jpegs_remain_supported() {
        for (w, h) in [(8, 8), (64, 64), (512, 384), (1024, 768)] {
            let pixels = vec![rgb::Rgb::new(128u8, 128, 128); w * h];
            let jpg = zenjpeg::encoder::EncoderConfig::ycbcr(
                55.,
                zenjpeg::encoder::ChromaSubsampling::Quarter,
            )
            .encode(&pixels, w as u32, h as u32)
            .unwrap();
            let mut probe = zenjpeg::detect::probe(&jpg).unwrap();
            let recorded = crate::record::apply_record(&mut probe);
            let legacy = crate::api::estimate_gain_for(&probe, crate::api::classify_content(&jpg));
            let p = for_jpeg(&jpg, &probe, legacy, recorded).unwrap().unwrap();
            assert!(p.score().is_finite());
        }
    }

    #[test]
    #[ignore = "explicit retained score-table replay; just decoded-router-replay supplies input"]
    fn publication_replay() {
        let path =
            std::env::var("ZENSR_ROUTER_REPLAY").expect("caller must supply ZENSR_ROUTER_REPLAY");
        assert_eq!(replay(&std::fs::read_to_string(path).unwrap()), 140940);
    }

    #[test]
    #[ignore = "explicit retained JPEG feature replay; caller supplies artifact paths"]
    fn publication_jpeg_features() {
        let path =
            std::env::var("ZENSR_ROUTER_JPEG_REPLAY").expect("caller must supply JPEG replay");
        let text = std::fs::read_to_string(path).unwrap();
        let mut count = 0;
        let weights = zensr_micro::decode_all_f16(include_bytes!(
            "../../../models/adopted/dejpeg_rt24i_canon4/weights_f16.raw"
        ));
        let model =
            || zensr_micro::adopted::AdoptedModel::load_compact(&weights, 24, 8, 1).unwrap();
        let reference = model();
        let restorer = crate::api::Restorer::new().with_tier(crate::api::Budget::Realtime, model());
        let (mut restored_checked, mut skipped_checked) = (false, false);
        for line in text.lines().skip(1) {
            let f: Vec<_> = line.split('\t').collect();
            let bytes = std::fs::read(f[43]).unwrap();
            let mut probe = zenjpeg::detect::probe(&bytes).unwrap();
            let recorded = crate::record::apply_record(&mut probe);
            let legacy =
                crate::api::estimate_gain_for(&probe, crate::api::classify_content(&bytes));
            let p = for_jpeg(&bytes, &probe, legacy, recorded).unwrap().unwrap();
            assert!(
                (p.gain - f[40].parse::<f64>().unwrap()).abs() < 1e-8,
                "gain at row {count}: {} vs {}",
                p.gain,
                f[40]
            );
            assert!(
                (p.risk - f[41].parse::<f64>().unwrap()).abs() < 1e-10,
                "risk at row {count}"
            );
            assert_eq!(p.restore(), f[42] == "1", "decision at row {count}");
            if (p.restore() && !restored_checked) || (!p.restore() && !skipped_checked) {
                let actual = restorer.restore(&bytes).threads(1).run().unwrap();
                assert_eq!(
                    matches!(actual, crate::api::Outcome::Restored { .. }),
                    f[42] == "1"
                );
                let expected = if f[42] == "1" {
                    restored_checked = true;
                    crate::restore_jpeg(
                        &bytes,
                        &reference,
                        &crate::RestoreConfig::default()
                            .with_threads(1)
                            .with_high_q_identity(false),
                    )
                    .unwrap()
                    .to_rgb8()
                } else {
                    skipped_checked = true;
                    zenjpeg::decoder::Decoder::new()
                        .deblock(zenjpeg::decoder::DeblockMode::Off)
                        .decode(&bytes, enough::Unstoppable)
                        .unwrap()
                        .pixels_u8()
                        .unwrap()
                        .to_vec()
                };
                assert_eq!(
                    actual.pixels().to_rgb8(),
                    expected,
                    "pixel mismatch at row {count}"
                );
            }
            count += 1;
        }
        assert_eq!(count, 16);
        assert!(
            restored_checked && skipped_checked,
            "retained vectors must exercise both default outcomes"
        );
    }
}
