use super::*;
use crate::api::{Intent, Outcome, Routing};
use crate::calibration as c;
mod candidate {
    include!("decoded_router_candidate.rs");
}

fn string(out: &mut Vec<u8>, value: &str) {
    out.extend((value.len() as u16).to_le_bytes());
    out.extend(value.as_bytes());
}
fn f32s(out: &mut Vec<u8>, values: &[f32]) {
    for v in values {
        out.extend(v.to_le_bytes());
    }
}
fn f64s(out: &mut Vec<u8>, values: &[f64]) {
    for v in values {
        out.extend(v.to_le_bytes());
    }
}
fn seal(metadata: &[u8], policy: &[u8], weights: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::from(*MAGIC);
    bytes.extend(1u32.to_le_bytes());
    bytes.extend((metadata.len() as u32).to_le_bytes());
    bytes.extend((policy.len() as u64).to_le_bytes());
    bytes.extend((weights.len() as u64).to_le_bytes());
    bytes.extend([0; 32]);
    bytes.extend(metadata);
    bytes.extend(policy);
    bytes.extend(weights);
    checksum(&mut bytes);
    bytes
}
fn checksum(bytes: &mut [u8]) {
    let mut hash = Sha256::new();
    hash.update(&bytes[..32]);
    hash.update(&bytes[64..]);
    bytes[32..64].copy_from_slice(&hash.finalize());
}

pub(crate) fn canon4_bytes(mode: u8) -> Vec<u8> {
    let weights = include_bytes!("../../../models/adopted/dejpeg_rt24i_canon4/weights_f16.raw");
    let mut meta = Vec::new();
    string(&mut meta, "dejpeg_rt24i_canon4");
    meta.extend([1, 0, 1]);
    meta.extend(24u16.to_le_bytes());
    meta.extend(8u16.to_le_bytes());
    meta.extend([0, 1]);
    let mut policy = vec![mode];
    policy.extend(1u32.to_le_bytes());
    policy.extend(0u32.to_le_bytes());
    f32s(&mut policy, &[9.5, 0.25]);
    policy.push(1);
    f32s(
        &mut policy,
        &[
            0.35, 0.10, 0.06, 0.15, 0.35, 0.45, 0.45, 0.5, 1.5, 0.25, 6.0,
        ],
    );
    let mut identity = Sha256::new();
    identity.update(&meta);
    identity.update(weights);
    policy.extend_from_slice(&identity.finalize());
    if mode != 0 {
        f32s(
            &mut policy,
            &[
                c::GRAPHIC_ZERO_AC_THRESHOLD,
                c::GATE_IJG_420,
                c::GATE_IJG_444,
                c::GATE_MOZJPEG_420,
                c::GATE_MOZJPEG_444,
                c::GATE_DISTANCE_420,
                c::GATE_DISTANCE_444,
            ],
        );
        for curve in [
            c::G420,
            c::GRAPHIC420,
            c::PHOTO420,
            c::G444,
            c::GRAPHIC444,
            c::PHOTO444,
            c::DIST420,
            c::GRAPHIC_DIST420,
            c::PHOTO_DIST420,
            c::DIST444,
            c::GRAPHIC_DIST444,
            c::PHOTO_DIST444,
            c::AQ_OFF_DELTA_420,
            c::AQ_OFF_DELTA_444,
        ] {
            policy.extend((curve.len() as u16).to_le_bytes());
            for &(x, y) in curve {
                f32s(&mut policy, &[x, y]);
            }
        }
    }
    if mode == 1 {
        policy.extend_from_slice(&Sha256::digest(include_bytes!(
            "decoded_router_features.rs"
        )));
        f64s(&mut policy, &candidate::MEAN);
        f64s(&mut policy, &candidate::SCALE);
        f64s(&mut policy, &candidate::GAIN);
        f64s(&mut policy, &candidate::RISK);
        f64s(
            &mut policy,
            &[
                candidate::GAIN_BIAS,
                candidate::RISK_BIAS,
                candidate::PENALTY,
                candidate::THRESHOLD,
            ],
        );
        for group in candidate::CATEGORIES {
            policy.push(group.len() as u8);
            for value in group {
                string(&mut policy, value);
            }
        }
    }
    seal(&meta, &policy, weights)
}
fn error(bytes: &[u8]) -> String {
    ModelBundle::from_bytes(bytes).err().expect("must reject")
}
fn policy_start(bytes: &[u8]) -> usize {
    64 + u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize
}

#[test]
fn corruption_and_model_identity_are_rejected() {
    let original = canon4_bytes(2);
    for n in [0, 1, 8, 31, 32, 63, 64, original.len() - 1] {
        assert!(ModelBundle::from_bytes(&original[..n]).is_err());
    }
    let mut bytes = original.clone();
    bytes.push(0);
    assert!(error(&bytes).contains("length"));
    let mut bytes = original.clone();
    bytes[8] = 99;
    assert!(error(&bytes).contains("version"));
    let mut bytes = original.clone();
    *bytes.last_mut().unwrap() ^= 1;
    assert!(error(&bytes).contains("checksum"));
    checksum(&mut bytes);
    assert!(error(&bytes).contains("calibration belongs"));
    let mut bytes = original.clone();
    bytes[66] ^= 1;
    checksum(&mut bytes);
    assert!(error(&bytes).contains("calibration belongs"));
    let mut bytes = original.clone();
    let p = policy_start(&bytes);
    bytes[p + 9..p + 13].copy_from_slice(&f32::NAN.to_le_bytes());
    checksum(&mut bytes);
    assert!(error(&bytes).contains("non-finite"));
    let mut bytes = original;
    let p = policy_start(&bytes);
    bytes[p + 1] = 99;
    checksum(&mut bytes);
    assert!(error(&bytes).contains("pipeline"));
}

#[test]
fn manual_policy_requires_an_explicit_choice() {
    let mut bytes = canon4_bytes(0);
    // Turn a final f16 value into a NaN, repairing the outer checksum.
    let end = bytes.len();
    bytes[end - 2..].copy_from_slice(&0x7e00u16.to_le_bytes());
    checksum(&mut bytes);
    assert!(error(&bytes).contains("non-finite model weights"));
    let bundle = ModelBundle::from_bytes(&canon4_bytes(0)).unwrap();
    assert_eq!(bundle.name(), "dejpeg_rt24i_canon4");
    assert_eq!(bundle.model().scale, 1);
    let restorer = bundle.into_restorer().unwrap();
    assert!(restorer
        .restore(&[])
        .run()
        .unwrap_err()
        .to_string()
        .contains("explicit Routing"));
    assert!(restorer
        .restore(&[])
        .routing(Routing::Auto { min_gain: 0.0 })
        .run()
        .unwrap_err()
        .to_string()
        .contains("explicit Routing"));
}

#[test]
fn malformed_calibrated_payloads_are_rejected() {
    let original = canon4_bytes(2);
    let p = policy_start(&original);
    let curves = p + 94 + 28;
    // Count and sortedness must be validated even with a valid file checksum.
    let mut bytes = original.clone();
    bytes[curves..curves + 2].copy_from_slice(&0u16.to_le_bytes());
    checksum(&mut bytes);
    assert!(error(&bytes).contains("curve size"));
    let mut bytes = original;
    let first_x = bytes[curves + 2..curves + 6].to_vec();
    bytes[curves + 10..curves + 14].copy_from_slice(&first_x);
    checksum(&mut bytes);
    assert!(error(&bytes).contains("unsorted"));
}

fn jpeg(w: usize, h: usize, quality: f32, full: bool) -> Vec<u8> {
    let pixels: Vec<_> = (0..w * h)
        .map(|i| {
            rgb::Rgb::new(
                (i % 251) as u8,
                ((i / w) * 13 % 256) as u8,
                ((i * 7) % 256) as u8,
            )
        })
        .collect();
    zenjpeg::encoder::EncoderConfig::ycbcr(
        quality,
        if full {
            zenjpeg::encoder::ChromaSubsampling::None
        } else {
            zenjpeg::encoder::ChromaSubsampling::Quarter
        },
    )
    .encode(&pixels, w as u32, h as u32)
    .unwrap()
}
fn same(actual: &Outcome, expected: &Outcome) {
    assert_eq!(
        std::mem::discriminant(actual),
        std::mem::discriminant(expected)
    );
    assert_eq!(actual.pixels().planes(), expected.pixels().planes());
    assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
}
#[test]
fn coefficient_container_preserves_existing_pixels_and_policy() {
    let restored = ModelBundle::from_bytes(&canon4_bytes(2))
        .unwrap()
        .into_restorer()
        .unwrap();
    let raw = zensr_micro::decode_all_f16(include_bytes!(
        "../../../models/adopted/dejpeg_rt24i_canon4/weights_f16.raw"
    ));
    let legacy = Restorer::new().with_tier(
        Budget::Realtime,
        AdoptedModel::load_compact(&raw, 24, 8, 1).unwrap(),
    );
    for (w, h, q, full) in [
        (8, 8, 5., false),
        (17, 15, 55., true),
        (33, 19, 98., false),
        (65, 31, 88., false),
    ] {
        let data = jpeg(w, h, q, full);
        for routing in [Routing::Always, Routing::Never, Routing::default()] {
            for intent in [Intent::Fidelity, Intent::Conservative] {
                same(
                    &restored
                        .restore(&data)
                        .routing(routing)
                        .intent(intent)
                        .threads(1)
                        .run()
                        .unwrap(),
                    &legacy
                        .restore(&data)
                        .routing(routing)
                        .intent(intent)
                        .threads(1)
                        .run()
                        .unwrap(),
                );
            }
        }
    }
}
#[cfg(feature = "chooser")]
#[test]
fn decoded_container_preserves_default_decisions_and_pixels() {
    let restored = ModelBundle::from_bytes(&canon4_bytes(1))
        .unwrap()
        .into_restorer()
        .unwrap();
    let raw = zensr_micro::decode_all_f16(include_bytes!(
        "../../../models/adopted/dejpeg_rt24i_canon4/weights_f16.raw"
    ));
    let legacy = Restorer::new().with_tier(
        Budget::Realtime,
        AdoptedModel::load_compact(&raw, 24, 8, 1).unwrap(),
    );
    // The wide case exercises more than one native router window.
    for (w, h, q, full) in [
        (8, 8, 5., false),
        (33, 19, 55., true),
        (513, 8, 85., false),
        (17, 15, 98., true),
    ] {
        let data = jpeg(w, h, q, full);
        same(
            &restored.restore(&data).threads(1).run().unwrap(),
            &legacy.restore(&data).threads(1).run().unwrap(),
        );
    }
}
#[cfg(not(feature = "chooser"))]
#[test]
fn decoded_container_requires_feature_support() {
    assert!(error(&canon4_bytes(1)).contains("chooser support"));
}

#[test]
fn f32_colour_and_sr_metadata_follow_the_model() {
    let bytes = canon4_bytes(0);
    let p = policy_start(&bytes);
    let policy = &bytes[p..p + 94];
    for (arch, nf, nc, scale, space) in [
        (1, 4, 0, 1, 0),
        (1, 4, 0, 2, 1),
        (1, 4, 1, 4, 0),
        (2, 48, 6, 2, 0),
        (2, 48, 6, 4, 0),
    ] {
        let mut meta = Vec::new();
        string(&mut meta, "synthetic-layout-control");
        meta.extend([arch, space, scale as u8]);
        meta.extend((nf as u16).to_le_bytes());
        meta.extend((nc as u16).to_le_bytes());
        meta.extend([1, 2]);
        let count = if arch == 1 {
            29 * nf + nc * (9 * nf * nf + 2 * nf) + 27 * nf * scale * scale + 3 * scale * scale
        } else {
            28 * nf
                + 19 * (9 * nf * nf + nf)
                + 4 * nf * nf
                + nf
                + 27 * nf * scale * scale
                + 3 * scale * scale
        };
        let weights = vec![0.0f32; count];
        let mut raw = Vec::new();
        f32s(&mut raw, &weights);
        let bundle = ModelBundle::from_bytes(&seal(&meta, policy, &raw)).unwrap();
        assert_eq!(bundle.model().scale, scale);
        assert_eq!(
            bundle.model().space(),
            if space == 0 {
                ModelSpace::Rgb
            } else {
                ModelSpace::Ycbcr
            }
        );
        let legacy = if arch == 1 {
            AdoptedModel::load_compact(&weights, nf, nc, scale)
        } else {
            AdoptedModel::load_span48(&weights, scale)
        }
        .unwrap();
        let input = vec![0.2f32; 3 * 8 * 8];
        assert_eq!(
            bundle.model().upscale_tiled(&input, 8, 8, 1, 0),
            legacy.upscale_tiled(&input, 8, 8, 1, 0)
        );
        if scale != 1 {
            assert!(bundle.into_restorer().err().unwrap().contains("x1 model"));
        } else {
            // Stored Quality tier becomes the request default.
            let restorer = bundle.into_restorer().unwrap();
            assert!(restorer
                .restore(&jpeg(8, 8, 55., false))
                .routing(Routing::Never)
                .run()
                .is_ok());
        }
    }
}

#[cfg(feature = "chooser")]
#[test]
fn model_router_and_pipeline_values_control_the_request() {
    let data = jpeg(33, 19, 55., false);
    for (threshold, should_restore) in [(-1e6, true), (1e6, false)] {
        let mut bundle = ModelBundle::from_bytes(&canon4_bytes(1)).unwrap();
        let policy = Arc::get_mut(&mut bundle.policy).unwrap();
        policy.router.as_mut().unwrap().threshold = threshold;
        let result = bundle
            .into_restorer()
            .unwrap()
            .restore(&data)
            .threads(1)
            .run()
            .unwrap();
        assert_eq!(matches!(result, Outcome::Restored { .. }), should_restore);
    }
    let mut bundle = ModelBundle::from_bytes(&canon4_bytes(2)).unwrap();
    let policy = Arc::get_mut(&mut bundle.policy).unwrap();
    policy.min_gain = 1e6;
    assert!(matches!(
        bundle
            .into_restorer()
            .unwrap()
            .restore(&data)
            .run()
            .unwrap(),
        Outcome::Unchanged { .. }
    ));
    let mut bundle = ModelBundle::from_bytes(&canon4_bytes(2)).unwrap();
    let policy = Arc::get_mut(&mut bundle.policy).unwrap();
    policy.gates = [0.0, 0.0, 0.0, 0.0, 1e6, 1e6];
    assert!(matches!(
        bundle
            .into_restorer()
            .unwrap()
            .restore(&data)
            .routing(Routing::Always)
            .run()
            .unwrap(),
        Outcome::Unchanged {
            why: crate::api::SkipReason::NearPristine,
            ..
        }
    ));
}

#[cfg(feature = "chooser")]
#[test]
fn bad_router_schema_scales_and_categories_fail_loading() {
    let original = canon4_bytes(1);
    let schema = Sha256::digest(include_bytes!("decoded_router_features.rs"));
    let offset = original
        .windows(32)
        .position(|w| w == schema.as_slice())
        .unwrap();
    let mut bytes = original.clone();
    bytes[offset] ^= 1;
    checksum(&mut bytes);
    assert!(error(&bytes).contains("feature schema"));
    let mut bytes = original.clone();
    let scale = offset + 32 + 37 * 8;
    bytes[scale..scale + 8].copy_from_slice(&0f64.to_le_bytes());
    checksum(&mut bytes);
    assert!(error(&bytes).contains("dimensions or scale"));
    let mut bytes = original.clone();
    let risk = offset + 32 + (37 + 37 + 47) * 8;
    bytes[risk..risk + 8].copy_from_slice(&f64::NAN.to_le_bytes());
    checksum(&mut bytes);
    assert!(error(&bytes).contains("non-finite router"));
    let mut bytes = original;
    let cats = offset + 32 + (37 + 37 + 47 + 47 + 4) * 8;
    bytes[cats] = 0;
    checksum(&mut bytes);
    assert!(error(&bytes).contains("category count"));
}
