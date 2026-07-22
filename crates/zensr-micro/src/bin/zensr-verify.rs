//! Golden verification + timing for zensr-micro vs the PyTorch reference.
//! Usage: zensr-verify <models-dir>   (expects spanf_weights.raw, spanf_in_64.raw, spanf_gold_256.raw)
//! Exits nonzero when max|diff| > 1e-3.

use std::time::Instant;
use zensr_micro::{SpanfWeights, spanf_x4};

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: zensr-verify <models-dir>");
    let dir = std::path::Path::new(&dir);
    let wbuf = read_f32(&dir.join("spanf_weights.raw"));
    let input = read_f32(&dir.join("spanf_in_64.raw"));
    let gold = read_f32(&dir.join("spanf_gold_256.raw"));
    let w = SpanfWeights::parse(&wbuf).expect("weights parse");

    let out = spanf_x4(&input, 64, 64, &w);
    assert_eq!(out.len(), gold.len(), "output length mismatch");
    let (mut max_abs, mut sum_sq) = (0.0f32, 0.0f64);
    for (a, b) in out.iter().zip(gold.iter()) {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        sum_sq += (d as f64) * (d as f64);
    }
    let rmse = (sum_sq / out.len() as f64).sqrt();
    println!("golden 64x64->256x256: max_abs_diff={max_abs:.3e} rmse={rmse:.3e}");

    let mut times = Vec::new();
    for _ in 0..8 {
        let t = Instant::now();
        std::hint::black_box(spanf_x4(&input, 64, 64, &w));
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("micro 64x64 x4: min={:.2}ms p50={:.2}ms", times[0], times[times.len() / 2]);

    // Like-for-like size vs the tract bench grid (synthetic input, no golden).
    let input128: Vec<f32> = (0..3 * 128 * 128).map(|i| (i % 251) as f32 / 251.0).collect();
    let mut times = Vec::new();
    for _ in 0..6 {
        let t = Instant::now();
        std::hint::black_box(spanf_x4(&input128, 128, 128, &w));
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = times[times.len() / 2];
    println!(
        "micro 128x128 x4: min={:.2}ms p50={:.2}ms out_mp_s={:.3}",
        times[0],
        p50,
        (512.0 * 512.0 / 1e6) / (p50 / 1e3)
    );

    if max_abs > 1e-3 {
        eprintln!("FAIL: max_abs_diff {max_abs:.3e} > 1e-3");
        std::process::exit(1);
    }
    println!("PASS");
}
