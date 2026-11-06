//! Frozen decoded-linear-1 routing, selected by the owner on 2026-11-05.
//! Native analysis windows preserve JPEG artifact scale at arbitrary sizes.
//! The 512-pixel measured path remains unchanged; larger images require all
//! sampled windows to accept restoration. Explicit caller overrides stay intact.
use crate::decoded_router_model as fitted;
use crate::RestoreError;
use std::sync::OnceLock;

#[cfg(test)]
#[path = "decoded_router/spatial.rs"]
mod spatial;

#[cfg(test)]
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

const FROZEN: Heads = Heads {
    mean: &fitted::MEAN,
    scale: &fitted::SCALE,
    gain: &fitted::GAIN,
    risk: &fitted::RISK,
    gain_bias: fitted::GAIN_BIAS,
    risk_bias: fitted::RISK_BIAS,
    categories: fitted::CATEGORIES,
};

#[cfg(test)]
const CANDIDATE: Heads = Heads {
    mean: &candidate::MEAN,
    scale: &candidate::SCALE,
    gain: &candidate::GAIN,
    risk: &candidate::RISK,
    gain_bias: candidate::GAIN_BIAS,
    risk_bias: candidate::RISK_BIAS,
    categories: candidate::CATEGORIES,
};

#[cfg(test)]
fn predict_candidate(numeric: &[f64; 37], categories: &[&str; 3]) -> Option<Prediction> {
    predict_heads(numeric, categories, &CANDIDATE)
}

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

#[cfg(test)]
pub(crate) fn for_jpeg(
    data: &[u8],
    probe: &zenjpeg::detect::JpegProbe,
    legacy_gain: f32,
    recorded: bool,
) -> Result<Option<Prediction>, RestoreError> {
    for_jpeg_prepared(
        data,
        probe,
        legacy_gain,
        recorded,
        &mut crate::prepared::Prepared::default(),
    )
}

pub(crate) fn for_jpeg_prepared(
    data: &[u8],
    probe: &zenjpeg::detect::JpegProbe,
    legacy_gain: f32,
    recorded: bool,
    prepared: &mut crate::prepared::Prepared,
) -> Result<Option<Prediction>, RestoreError> {
    for_jpeg_prepared_windows(data, probe, legacy_gain, recorded, prepared, None, predict)
}

fn for_jpeg_prepared_windows(
    data: &[u8],
    probe: &zenjpeg::detect::JpegProbe,
    legacy_gain: f32,
    recorded: bool,
    prepared: &mut crate::prepared::Prepared,
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
    prepared.plain(data)?;
    prepared.coefficients(data)?;
    let decoded = prepared.plain.as_ref().unwrap();
    let (w, h) = decoded.dimensions();
    let rgb = decoded
        .pixels_u8()
        .ok_or(RestoreError::UnsupportedPixels("router requires RGB8"))?;
    let coefficients = prepared.coefficients.as_ref().unwrap();
    let plane = coefficients
        .components
        .first()
        .ok_or(RestoreError::UnsupportedPixels("missing luma"))?;
    let (width, height) = (w as usize, h as usize);
    let stride = width
        .checked_mul(3)
        .ok_or(RestoreError::UnsupportedPixels("router row size"))?;
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
        assert_eq!(
            replay(include_str!("../tests/data/decoded_router_golden.tsv")),
            16
        );
    }

    #[test]
    fn candidate_sklearn_golden_predictions_and_decisions() {
        assert_eq!(
            replay_using(
                include_str!("../tests/data/decoded_router_candidate_golden.tsv"),
                predict_candidate
            ),
            16
        );
    }

    #[test]
    #[ignore = "explicit independent candidate replay; caller provides vectors"]
    fn candidate_publication_replay() {
        let path = std::env::var("ZENSR_CANDIDATE_REPLAY").expect("candidate vectors required");
        assert_eq!(
            replay_using(&std::fs::read_to_string(path).unwrap(), predict_candidate),
            140940
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
    #[ignore = "explicit native-window diagnostic; caller supplies retained development JPEGs and report"]
    fn publication_spatial_windows() {
        use std::io::Write;
        let manifest =
            std::env::var("ZENSR_SPATIAL_INPUTS").expect("development manifest required");
        let output = std::env::var("ZENSR_SPATIAL_REPORT").expect("report path required");
        let mut report = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(output)
            .unwrap();
        writeln!(report, "id\tpolicy\tscore\trestore\tx\ty\twidth\theight").unwrap();
        let text = std::fs::read_to_string(manifest).unwrap();
        assert_eq!(text.lines().next(), Some("id\tjpeg"));
        let mut count = 0;
        for line in text.lines().skip(1) {
            let (id, path) = line.split_once('\t').unwrap();
            let bytes = std::fs::read(path).unwrap();
            let mut probe = zenjpeg::detect::probe(&bytes).unwrap();
            let recorded = crate::record::apply_record(&mut probe);
            let legacy =
                crate::api::estimate_gain_for(&probe, crate::api::classify_content(&bytes));
            let mut prepared = crate::prepared::Prepared::default();
            let coefficients = prepared.coefficients(&bytes).unwrap();
            let luma = &coefficients.components[0];
            let (w, h) = (
                probe.dimensions.width as usize,
                probe.dimensions.height as usize,
            );
            assert!(w.max(h) > 512, "native development input must exceed 512");
            let candidates = spatial::select(w, h, &luma.coeffs, luma.blocks_wide);
            for (policy, regions) in [("fixed", windows(w, h)), ("spatial", candidates)] {
                let prediction = for_jpeg_prepared_windows(
                    &bytes,
                    &probe,
                    legacy,
                    recorded,
                    &mut prepared,
                    Some(&regions),
                    predict,
                )
                .unwrap()
                .unwrap();
                for window in regions {
                    writeln!(
                        report,
                        "{id}\t{policy}\t{:.12}\t{}\t{}\t{}\t{}\t{}",
                        prediction.score(),
                        prediction.restore(),
                        window.x,
                        window.y,
                        window.width,
                        window.height
                    )
                    .unwrap();
                }
            }
            report.flush().unwrap();
            count += 1;
        }
        assert_eq!(
            count, 8,
            "bounded eight-cell development diagnostic required"
        );
    }

    #[test]
    #[ignore = "explicit metadata sensitivity diagnostic; caller supplies retained JPEGs and report"]
    fn publication_metadata_sensitivity() {
        use std::io::Write;
        let path =
            std::env::var("ZENSR_ROUTER_JPEG_REPLAY").expect("retained JPEG vectors required");
        let output = std::env::var("ZENSR_METADATA_REPORT").expect("report output required");
        let mut report = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(output)
            .unwrap();
        writeln!(report, "row\tpayload_bytes\tfile_bytes_added\tbaseline_gain\tbaseline_risk\tmodified_gain\tmodified_risk\tbaseline_restore\tmodified_restore").unwrap();
        let text = std::fs::read_to_string(path).unwrap();
        let mut count = 0;
        for (i, line) in text.lines().skip(1).enumerate() {
            let fields: Vec<_> = line.split('\t').collect();
            let bytes = std::fs::read(fields[43]).unwrap();
            let mut probe = zenjpeg::detect::probe(&bytes).unwrap();
            let recorded = crate::record::apply_record(&mut probe);
            let legacy =
                crate::api::estimate_gain_for(&probe, crate::api::classify_content(&bytes));
            let baseline = for_jpeg(&bytes, &probe, legacy, recorded).unwrap().unwrap();
            let mut prepared = crate::prepared::Prepared::default();
            prepared.plain(&bytes).unwrap();
            prepared.coefficients(&bytes).unwrap();
            let candidate_before = for_jpeg_prepared_windows(
                &bytes,
                &probe,
                legacy,
                recorded,
                &mut prepared,
                None,
                predict_candidate,
            )
            .unwrap()
            .unwrap();
            for payload in [4096, 16384, 65536, 262144] {
                // APP15 is an opaque diagnostic payload, not ICC/EXIF or an
                // encode record. Segment lengths include their own two bytes.
                let mut modified = Vec::with_capacity(bytes.len() + payload + 32);
                modified.extend_from_slice(&bytes[..2]);
                let mut left = payload;
                while left > 0 {
                    let n = left.min(65533);
                    modified.extend_from_slice(&[0xff, 0xef]);
                    modified.extend_from_slice(&((n + 2) as u16).to_be_bytes());
                    modified.resize(modified.len() + n, 0);
                    left -= n;
                }
                modified.extend_from_slice(&bytes[2..]);
                let mut changed_probe = zenjpeg::detect::probe(&modified).unwrap();
                let changed_recorded = crate::record::apply_record(&mut changed_probe);
                assert_eq!(
                    format!("{probe:?}"),
                    format!("{changed_probe:?}"),
                    "metadata changed probe"
                );
                assert_eq!(recorded, changed_recorded);
                let mut changed = crate::prepared::Prepared::default();
                changed.plain(&modified).unwrap();
                changed.coefficients(&modified).unwrap();
                assert_eq!(
                    prepared.plain.as_ref().unwrap().pixels_u8(),
                    changed.plain.as_ref().unwrap().pixels_u8()
                );
                let original_components = &prepared.coefficients.as_ref().unwrap().components;
                let changed_components = &changed.coefficients.as_ref().unwrap().components;
                assert_eq!(original_components.len(), changed_components.len());
                for (a, b) in original_components.iter().zip(changed_components) {
                    assert_eq!(a.coeffs, b.coeffs);
                }
                let prediction =
                    for_jpeg_prepared(&modified, &changed_probe, legacy, recorded, &mut changed)
                        .unwrap()
                        .unwrap();
                let candidate_after = for_jpeg_prepared_windows(
                    &modified,
                    &changed_probe,
                    legacy,
                    recorded,
                    &mut changed,
                    None,
                    predict_candidate,
                )
                .unwrap()
                .unwrap();
                assert_eq!(
                    candidate_before.gain.to_bits(),
                    candidate_after.gain.to_bits()
                );
                assert_eq!(
                    candidate_before.risk.to_bits(),
                    candidate_after.risk.to_bits()
                );
                assert_eq!(candidate_before.restore(), candidate_after.restore());
                writeln!(
                    report,
                    "{i}\t{payload}\t{}\t{:.12}\t{:.12}\t{:.12}\t{:.12}\t{}\t{}",
                    modified.len() - bytes.len(),
                    baseline.gain,
                    baseline.risk,
                    prediction.gain,
                    prediction.risk,
                    baseline.restore(),
                    prediction.restore()
                )
                .unwrap();
                count += 1;
            }
            report.flush().unwrap();
        }
        assert_eq!(
            count, 64,
            "all sixteen retained JPEGs and four padding sizes required"
        );
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
