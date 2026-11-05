//! Descriptive cost arms on retained JPEGs. Model loading and input I/O are
//! outside timing; output allocation and destruction are inside. Learned
//! classifier heads are not included in the feature-only arms.
use std::fs;
use std::sync::Arc;
use std::time::Duration;
use zenbench::prelude::*;
use zensr_zenjpeg::api::{Budget, Intent, Restorer, Routing};

#[path = "coef_features/metrics.rs"]
mod coefficients;

fn decode(data: &[u8]) -> (Vec<u8>, usize, usize) {
    let image = zenjpeg::decoder::Decoder::new()
        .deblock(zenjpeg::decoder::DeblockMode::Off)
        .decode(data, enough::Unstoppable)
        .unwrap();
    let (w, h) = image.dimensions();
    (image.pixels_u8().unwrap().to_vec(), w as usize, h as usize)
}

fn coefficient_features(data: &[u8]) -> [f64; 8] {
    let image = zenjpeg::decoder::Decoder::new()
        .decode_coefficients(data, enough::Unstoppable)
        .unwrap();
    let plane = &image.components[0];
    let c =
        coefficients::features(&plane.coeffs, plane.coeffs.len() / 64, plane.blocks_wide).unwrap();
    [
        c.zero_ac_blocks,
        c.zero_ac_coefs,
        c.hf_survival,
        c.mean_abs_ac,
        c.dc_spread,
        c.ac_count_spread,
        c.gutted_block_frac,
        c.gutted_clustering,
    ]
}

fn costs(suite: &mut Suite) {
    // Large-image restoration can exceed the default group wall budget before
    // enough rounds finish. Keep the default, but let the recorded study plan
    // provide a larger budget; confirmation still requires every planned round.
    let wall_seconds = std::env::var_os("ZENSR_COST_MAX_WALL_SECONDS")
        .map(|v| {
            v.to_string_lossy()
                .parse::<u64>()
                .expect("ZENSR_COST_MAX_WALL_SECONDS must be a positive integer")
        })
        .unwrap_or(120);
    assert!(wall_seconds > 0, "cost wall budget must be positive");
    let manifest = fs::read_to_string(
        std::env::var("ZENSR_COST_MANIFEST").expect("ZENSR_COST_MANIFEST required"),
    )
    .unwrap();
    let bytes = fs::read("models/adopted/dejpeg_rt24i_canon4/weights_f16.raw").unwrap();
    let weights = zensr_micro::decode_all_f16(&bytes);
    let model =
        Arc::new(zensr_micro::adopted::AdoptedModel::load_compact(&weights, 24, 8, 1).unwrap());
    let restorer = Arc::new(Restorer::new().with_tier(
        Budget::Quality,
        zensr_micro::adopted::AdoptedModel::load_compact(&weights, 24, 8, 1).unwrap(),
    ));
    let mut count = 0;
    for line in manifest.lines().skip(1) {
        let (label, path) = line.split_once('\t').expect("label and JPEG path required");
        let data = Arc::new(fs::read(path).unwrap());
        let (_, w, h) = decode(&data);
        count += 1;
        suite.group(label, |g| {
            g.config().max_wall_time(Duration::from_secs(wall_seconds));
            g.throughput(Throughput::Elements((w * h) as u64));
            let d = Arc::clone(&data);
            g.bench("plain-decode", move |b| b.iter(|| decode(&d)));
            let d = Arc::clone(&data);
            g.bench("coefficient-parse-and-eight-features", move |b| {
                b.iter(|| coefficient_features(&d))
            });
            let d = Arc::clone(&data);
            g.bench("decode-and-all-image-features", move |b| {
                b.iter(|| {
                    let (rgb, w, h) = decode(&d);
                    let (crop, cw, ch) = zensr_zenjpeg::chooser::center_crop_rgb8(&rgb, w, h);
                    let offer = zenanalyze::offer_for_request(
                        &crop,
                        cw,
                        ch,
                        &zensr_zenjpeg::chooser::chooser_request(),
                    )
                    .unwrap();
                    let chooser = zensr_zenjpeg::chooser::classify_from_owned_offer(&offer);
                    (offer, chooser)
                })
            });
            let d = Arc::clone(&data);
            let m = Arc::clone(&model);
            g.bench("ungated-restoration-with-guards-and-projection", move |b| {
                b.iter(|| {
                    zensr_zenjpeg::restore_jpeg(
                        &d,
                        &m,
                        &zensr_zenjpeg::RestoreConfig::default()
                            .with_threads(1)
                            .with_high_q_identity(false),
                    )
                    .unwrap()
                    .to_rgb8()
                })
            });
            for (name, intent) in [
                ("current-fidelity", Intent::Fidelity),
                ("current-conservative", Intent::Conservative),
            ] {
                let d = Arc::clone(&data);
                let r = Arc::clone(&restorer);
                g.bench(name, move |b| {
                    b.iter(|| {
                        r.restore(&d)
                            .budget(Budget::Quality)
                            .intent(intent)
                            .routing(Routing::Auto { min_gain: 0.25 })
                            .threads(1)
                            .run()
                            .unwrap()
                    })
                });
            }
        });
    }
    assert!(count > 0, "empty cost manifest");
}

zenbench::main!(costs);
