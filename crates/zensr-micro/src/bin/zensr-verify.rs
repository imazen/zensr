//! Golden verification + timing for zensr-micro vs the PyTorch reference.
//! Usage: zensr-verify <models-dir>   (expects spanf_weights.raw, spanf_in_64.raw, spanf_gold_256.raw)
//! Exits nonzero when max|diff| > 1e-3 on either path.

use std::time::Instant;
use zensr_micro::{spanf_x4, spanf_x4_simd, SpanfWeights};

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn diff_stats(a: &[f32], b: &[f32]) -> (f32, f64) {
    let mut max_abs = 0.0f32;
    let mut sum_sq = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        sum_sq += (d as f64) * (d as f64);
    }
    (max_abs, (sum_sq / a.len() as f64).sqrt())
}

fn time_runs<F: FnMut() -> Vec<f32>>(mut f: F, iters: usize) -> (f64, f64) {
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        std::hint::black_box(f());
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[0], times[times.len() / 2])
}

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 251) as f32 / 251.0).collect()
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: zensr-verify <models-dir>");
    let dir = std::path::Path::new(&dir);
    let wbuf = read_f32(&dir.join("spanf_weights.raw"));
    let input = read_f32(&dir.join("spanf_in_64.raw"));
    let gold = read_f32(&dir.join("spanf_gold_256.raw"));
    let w = SpanfWeights::parse(&wbuf).expect("weights parse");

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        use archmage::prelude::*;
        println!(
            "tiers: v4x={} v4={} v3={}",
            X64V4xToken::summon().is_some(),
            X64V4Token::summon().is_some(),
            X64V3Token::summon().is_some()
        );
    }

    let out_ref = spanf_x4(&input, 64, 64, &w);
    let (max_ref, rmse_ref) = diff_stats(&out_ref, &gold);
    println!("scalar vs golden: max_abs={max_ref:.3e} rmse={rmse_ref:.3e}");

    let out_simd = spanf_x4_simd(&input, 64, 64, &w);
    let (max_simd, rmse_simd) = diff_stats(&out_simd, &gold);
    println!("simd   vs golden: max_abs={max_simd:.3e} rmse={rmse_simd:.3e}");
    let nans: Vec<usize> = out_simd
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_finite())
        .map(|(i, _)| i)
        .collect();
    if !nans.is_empty() {
        let (h4, w4) = (256usize, 256usize);
        let i = nans[0];
        let (c, rem) = (i / (h4 * w4), i % (h4 * w4));
        println!(
            "simd non-finite: count={} first at idx {i} (c={c}, y={}, x={}) val={}",
            nans.len(),
            rem / w4,
            rem % w4,
            out_simd[i]
        );
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    for (name, out) in [
        ("v4x", zensr_micro::simd::spanf_x4_simd_force_v4x(&input, 64, 64, &w)),
        #[cfg(feature = "tier_v4")]
        ("v4", zensr_micro::simd::spanf_x4_simd_force_v4(&input, 64, 64, &w)),
        ("v3", zensr_micro::simd::spanf_x4_simd_force_v3(&input, 64, 64, &w)),
    ] {
        match out {
            Some(o) => {
                let nan = o.iter().filter(|v| !v.is_finite()).count();
                let (mx, _) = diff_stats(&o, &gold);
                println!("tier {name}: nonfinite={nan} max_abs_vs_gold={mx:.3e}");
            }
            None => println!("tier {name}: not available"),
        }
    }

    for (h, wd) in [(64usize, 64usize), (128, 128), (256, 256)] {
        let inp = ramp(3 * h * wd);
        let iters = if h >= 256 { 5 } else { 8 };
        let (min_s, p50_s) = time_runs(|| spanf_x4_simd(&inp, h, wd, &w), iters);
        let mp = (h * wd * 16) as f64 / 1e6;
        println!(
            "micro-simd {h}x{wd} x4: min={min_s:.2}ms p50={p50_s:.2}ms out_mp_s={:.3}",
            mp / (p50_s / 1e3)
        );
    }
    let inp128 = ramp(3 * 128 * 128);
    let (min_sc, p50_sc) = time_runs(|| spanf_x4(&inp128, 128, 128, &w), 5);
    println!(
        "micro-scalar-ref 128x128 x4: min={min_sc:.2}ms p50={p50_sc:.2}ms out_mp_s={:.3}",
        (128.0 * 128.0 * 16.0 / 1e6) / (p50_sc / 1e3)
    );

    // Compressed-weight paths: decode -> run -> PSNR vs the fp32-weight golden.
    let psnr = |a: &[f32], b: &[f32]| {
        let mse: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = (x.clamp(0.0, 1.0) - y.clamp(0.0, 1.0)) as f64;
                d * d
            })
            .sum::<f64>()
            / a.len() as f64;
        -10.0 * mse.log10()
    };
    let f16b = std::fs::read(dir.join("spanf_weights_f16.raw")).expect("f16 weights");
    let wf16buf = zensr_micro::decode_f16_weights(&f16b).expect("f16 decode");
    let wf16 = SpanfWeights::parse(&wf16buf).unwrap();
    let out_f16 = spanf_x4_simd(&input, 64, 64, &wf16);
    let (mx16, _) = diff_stats(&out_f16, &gold);
    println!(
        "f16 weights ({} B file): max_abs_vs_gold={mx16:.3e} psnr={:.2} dB",
        f16b.len(),
        psnr(&out_f16, &gold)
    );
    if mx16 > 0.05 {
        eprintln!("FAIL: f16 path exceeds tolerance");
        std::process::exit(1);
    }

    let i8b = std::fs::read(dir.join("spanf_weights_int8pc.raw")).expect("int8 weights");
    let wi8buf = zensr_micro::decode_int8pc_weights(&i8b).expect("int8 decode");
    let wi8 = SpanfWeights::parse(&wi8buf).unwrap();
    let out_i8 = spanf_x4_simd(&input, 64, 64, &wi8);
    println!(
        "int8pc weights ({} B file): psnr={:.2} dB (accuracy study: NOT viable for SPANF)",
        i8b.len(),
        psnr(&out_i8, &gold)
    );

    if max_ref > 1e-3 || max_simd > 1e-3 {
        eprintln!("FAIL: golden diff too large (ref {max_ref:.3e}, simd {max_simd:.3e})");
        std::process::exit(1);
    }
    println!("PASS");
}
