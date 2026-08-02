//! Speed table for the five systems: tiled throughput at 1 and 12 threads.
//! Usage: systems_bench [reps=5]

use std::time::Instant;
use zensr_micro::adopted::AdoptedModel;
use zensr_micro::guards::{guarded_merge, GuardConfig};

fn read_f32_file(p: &str) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn load(dir: &str, arch: &str, nf: usize, nc: usize, scale: usize) -> Option<AdoptedModel> {
    let path = format!("models/adopted/{dir}/weights.raw");
    if !std::path::Path::new(&path).exists() {
        return None;
    }
    let raw = read_f32_file(&path);
    match arch {
        "compact" => AdoptedModel::load_compact(&raw, nf, nc, scale).ok(),
        _ => AdoptedModel::load_span48(&raw, scale).ok(),
    }
}

fn main() {
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    // (label, dir, arch, nf, nc, scale, input h=w)
    let mut cases: Vec<(&str, Option<AdoptedModel>, usize)> = vec![
        (
            "S-A x2 span",
            load("nomosuni_span_2x", "span48", 48, 6, 2),
            512,
        ),
        (
            "S-A x4 span",
            load("nomosuni_span_4x", "span48", 48, 6, 4),
            256,
        ),
        (
            "S-B x4 general",
            load("general_x4v3", "compact", 64, 32, 4),
            256,
        ),
        (
            "S-C x1 compact2x",
            load("nomosuni_compact_2x", "compact", 64, 16, 2),
            512,
        ),
        (
            "S-D x4 anime",
            load("animevideo_x4v3", "compact", 64, 16, 4),
            256,
        ),
        ("S-E x2 rt", load("rt_distill_2x", "compact", 24, 8, 2), 512),
        (
            "S-E2 x2 rt32",
            load("rt32_distill_2x", "compact", 32, 12, 2),
            512,
        ),
    ];
    println!("system\tin\tout_mp\tthreads\tmin_ms\tmp_out_per_s\tguard_ms");
    for (name, m, n) in cases.iter_mut() {
        let Some(m) = m else {
            println!("{name}\t-\t-\t-\tNOT AVAILABLE");
            continue;
        };
        let input: Vec<f32> = (0..3 * *n * *n).map(|i| (i % 251) as f32 / 251.0).collect();
        let out_mp = (*n * *n * m.scale * m.scale) as f64 / 1e6;
        for threads in [1usize, 12] {
            let mut best = f64::INFINITY;
            let mut gbest = f64::INFINITY;
            for _ in 0..reps {
                let t = Instant::now();
                let mut sr = m.upscale_tiled(&input, *n, *n, threads, 0);
                let ms = t.elapsed().as_secs_f64() * 1e3;
                best = best.min(ms);
                let t2 = Instant::now();
                guarded_merge(&mut sr, &input, *n, *n, m.scale, &GuardConfig::default());
                gbest = gbest.min(t2.elapsed().as_secs_f64() * 1e3);
                std::hint::black_box(&sr);
            }
            println!(
                "{name}\t{n}x{n}\t{out_mp:.2}\t{threads}\t{best:.1}\t{:.2}\t{gbest:.1}",
                out_mp / (best / 1e3)
            );
        }
    }
}
