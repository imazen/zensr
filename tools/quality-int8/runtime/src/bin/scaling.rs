#![forbid(unsafe_code)]
#[path = "../engine.rs"]
mod engine;
use engine::*;
use std::{path::Path, sync::Arc, time::Duration};
use tract_onnx::prelude::{TValue, TractResult};
use zenbench::prelude::*;
fn costs(suite: &mut Suite) {
    let root = std::env::var("ZENSR_QUALITY_EXPORTS").expect("ZENSR_QUALITY_EXPORTS");
    let _reference = Engine::load(Path::new(&root), "native", 17, 13).unwrap();
    let threads: usize = std::env::var("ZENSR_QUALITY_THREADS")
        .unwrap_or_else(|_| "4".into())
        .parse()
        .unwrap();
    for n in [512usize, 1024] {
        let input: Vec<f32> = (0..3 * n * n)
            .map(|i| ((i * 13 + i / 17) % 251) as f32 / 255.)
            .collect();
        let x = Arc::new(tensor(&input, n, n));
        let models: Vec<_> = ARMS
            .iter()
            .map(|arm| {
                Arc::new(Engine::load_threads(Path::new(&root), arm, n, n, threads).unwrap())
            })
            .collect();
        for (arm, model) in ARMS.iter().zip(&models) {
            let single = Engine::load(Path::new(&root), arm, n, n).unwrap();
            let expected = single.run_threads(&x, n, n, 1).unwrap();
            let actual = model.run_threads(&x, n, n, threads).unwrap();
            assert_eq!(
                actual.len(),
                expected.len(),
                "threaded shape differs: {arm}"
            );
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.is_finite() && a.to_bits() == b.to_bits()),
                "threaded pixels differ: {arm}, {n}x{n}, {threads} threads"
            );
            eprintln!("thread-parity passed: {arm}, {n}x{n}, {threads} threads");
        }
        suite.group(format!("quality-{n}x{n}-{threads}-threads"), |g| {
            g.config()
                .max_rounds(5)
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
                    b.iter(|| m.run_threads(&input, n, n, threads).unwrap())
                });
            }
            g.baseline("native");
        });
    }
}
zenbench::main!(costs);

impl Engine {
    pub(crate) fn run_threads(
        &self,
        input: &TValue,
        h: usize,
        w: usize,
        threads: usize,
    ) -> TractResult<Vec<f32>> {
        match self {
            Self::Native(m) => Ok(m.upscale_tiled(
                input.try_as_plain_ram()?.as_slice::<f32>()?,
                h,
                w,
                threads,
                0,
            )),
            Self::Tract(_) => self.run(input, h, w, false),
        }
    }
}
