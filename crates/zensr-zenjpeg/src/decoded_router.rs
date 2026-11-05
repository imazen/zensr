//! Frozen decoded-linear-1 routing, selected by the owner on 2026-11-05.
//! The measured geometry is a 512-pixel longest edge. Explicit caller routing
//! and other geometries continue to use the calibrated legacy policy.
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

pub(crate) fn for_jpeg(
    data: &[u8],
    probe: &zenjpeg::detect::JpegProbe,
    legacy_gain: f32,
    recorded: bool,
) -> Result<Option<Prediction>, RestoreError> {
    if probe.dimensions.width.max(probe.dimensions.height) != 512
        || probe.dimensions.width.min(probe.dimensions.height) < 8
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
    let (crop, cw, ch) = crate::chooser::center_crop_rgb8(rgb, w as usize, h as usize);
    // Keep the same analysis request/tier activation as the frozen study.
    let offer = zenanalyze::offer_for_request(&crop, cw, ch, &crate::chooser::chooser_request())
        .map_err(|e| RestoreError::Decode(format!("router features: {e:?}")))?;
    let coefficients = zenjpeg::decoder::Decoder::new()
        .decode_coefficients(data, enough::Unstoppable)
        .map_err(|e| RestoreError::Decode(format!("{e:?}")))?;
    let plane = coefficients
        .components
        .first()
        .ok_or(RestoreError::UnsupportedPixels("missing luma"))?;
    let c = crate::router_coefficients::features(
        &plane.coeffs,
        plane.coeffs.len() / 64,
        plane.blocks_wide,
    )
    .ok_or(RestoreError::UnsupportedPixels("empty luma coefficients"))?;
    let mut numeric = [0.; 37];
    numeric[..13].copy_from_slice(&[
        f64::from(w),
        f64::from(h),
        f64::from(probe.quality.value),
        f64::from(legacy_gain),
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
    predict(&numeric, &[&family, &scale, ss])
        .map(Some)
        .ok_or(RestoreError::UnsupportedModel("non-finite router input"))
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
