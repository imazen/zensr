#![forbid(unsafe_code)]
#[path = "../engine.rs"]
mod engine;
use engine::*;
use std::{path::Path, sync::Arc, time::Duration};
use tract_onnx::prelude::{TValue, TractResult};
use zenbench::prelude::*;
// Zenbench 0.1.11 runs children sequentially and stops if a child fails.
// The optional mode checks every model/size in child zero before any timings;
// later children retain the same frozen binary, models and CPU boundary.
fn checks_in_context(
    mode: Option<&str>,
    subprocess: Option<&str>,
    processes: Option<&str>,
    result: Option<&Path>,
    args: &[String],
) -> bool {
    match mode {
        None | Some("each-process") => true,
        Some("first-process") => {
            if subprocess.is_none() {
                // Modern Confirm/Evidence registers callbacks before dispatching.
                assert!(
                    args.iter()
                        .any(|v| v == "--mode=confirm" || v == "--mode=evidence")
                        && args.iter().any(|v| v == "--processes=3")
                        && !args.iter().any(|v| v == "--dry-run"),
                    "first-process parity requires three-process confirmation context"
                );
                return false;
            }
            assert_eq!(subprocess, Some("1"));
            assert_eq!(processes, Some("3"));
            let path = result.expect("result path");
            let name = path.file_name().unwrap().to_str().unwrap();
            assert!(matches!(name, "0.json.gz" | "1.json.gz" | "2.json.gz"));
            assert!(
                name == "0.json.gz" || path.with_file_name("0.json.gz").is_file(),
                "first subprocess must finish before parity is reused"
            );
            name == "0.json.gz"
        }
        other => panic!("unsupported explicit parity mode: {other:?}"),
    }
}
fn check_threads_in_this_process() -> bool {
    let mode = std::env::var("ZENSR_THREAD_PARITY").ok();
    let subprocess = std::env::var("ZENBENCH_SUBPROCESS").ok();
    let processes = std::env::var("ZENBENCH_PROCESSES").ok();
    let result = std::env::var("ZENBENCH_RESULT_PATH")
        .ok()
        .map(std::path::PathBuf::from);
    checks_in_context(
        mode.as_deref(),
        subprocess.as_deref(),
        processes.as_deref(),
        result.as_deref(),
        &std::env::args().collect::<Vec<_>>(),
    )
}
#[test]
fn confirmation_registration_defers_checks_to_the_child() {
    let args = vec!["--mode=confirm".to_owned(), "--processes=3".to_owned()];
    assert!(!checks_in_context(
        Some("first-process"),
        None,
        None,
        None,
        &args
    ));
}
#[test]
fn first_child_always_performs_exact_output_checks() {
    assert!(checks_in_context(
        Some("first-process"),
        Some("1"),
        Some("3"),
        Some(Path::new("0.json.gz")),
        &[]
    ));
}

fn costs(suite: &mut Suite) {
    let check_threads = check_threads_in_this_process();
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
        if check_threads {
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
