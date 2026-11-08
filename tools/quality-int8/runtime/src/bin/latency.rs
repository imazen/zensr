#![forbid(unsafe_code)]
#[path = "../engine.rs"]
mod engine;
use engine::*;
use std::{path::Path, sync::Arc, time::Duration};
use zenbench::prelude::*;
fn costs(suite: &mut Suite) {
    let root = std::env::var("ZENSR_QUALITY_EXPORTS").expect("ZENSR_QUALITY_EXPORTS");
    let sizes = std::env::var("ZENSR_QUALITY_SIZES").unwrap_or_else(|_| "128,256,512,1024".into());
    for s in sizes.split(',') {
        let n: usize = s.parse().unwrap();
        let input: Vec<f32> = (0..3 * n * n)
            .map(|i| ((i * 13 + i / 17) % 251) as f32 / 255.)
            .collect();
        let x = Arc::new(tensor(&input, n, n));
        let models: Vec<_> = ARMS
            .iter()
            .map(|arm| Arc::new(Engine::load(Path::new(&root), arm, n, n).unwrap()))
            .collect();
        suite.group(format!("quality-{n}x{n}-single-thread"), |g| {
            g.config()
                .max_rounds(15)
                .auto_rounds(false)
                .warmup_time(Duration::from_millis(100))
                .max_wall_time(Duration::from_secs(1800))
                .sample_target_ns(1000000)
                .min_sample_ns(0);
            g.throughput(Throughput::Elements((n * n) as u64));
            for (arm, model) in ARMS.iter().zip(&models) {
                let m = Arc::clone(model);
                let input = Arc::clone(&x);
                g.bench(*arm, move |b| {
                    b.iter(|| m.run(&input, n, n, false).unwrap())
                });
            }
            let m = Arc::clone(&models[0]);
            let input = Arc::clone(&x);
            g.bench("native-serving-tiled", move |b| {
                b.iter(|| m.run(&input, n, n, true).unwrap())
            });
            g.baseline("native");
        });
    }
}
zenbench::main!(costs);
