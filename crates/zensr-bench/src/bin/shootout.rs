//! Interleaved paired bench: tract vs zensr-micro on the same SPANF x4 model.
//! Contenders alternate round-by-round so they see identical machine
//! conditions (zenbench-style pairing); report min + median per contender.
//!
//! Usage: shootout <models-dir> [rounds]

use std::time::Instant;
use tract_onnx::prelude::*;
use zensr_micro::{spanf_x4_simd, SpanfWeights};

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn stats(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[0], v[v.len() / 2])
}

fn main() -> TractResult<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: shootout <models-dir> [rounds]");
    let dir = std::path::Path::new(&dir);
    let rounds: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(10);

    let wbuf = read_f32(&dir.join("spanf_weights.raw"));
    let w = SpanfWeights::parse(&wbuf).expect("weights");

    println!(
        "# interleaved rounds={rounds}; load {}",
        std::fs::read_to_string("/proc/loadavg")
            .unwrap_or_default()
            .trim()
    );

    for (h, wd) in [(64usize, 64usize), (128, 128), (256, 256)] {
        if rounds == 0 {
            break;
        }
        let onnx = dir.join(format!("SPANF_x4_{h}x{wd}.onnx"));
        let plan = tract_onnx::onnx()
            .model_for_path(&onnx)?
            .with_input_fact(0, f32::fact([1, 3, h, wd]).into())?
            .into_optimized()?
            .into_runnable()?;
        let input: Vec<f32> = (0..3 * h * wd).map(|i| (i % 251) as f32 / 251.0).collect();
        let t_in = Tensor::from_shape(&[1, 3, h, wd], &input)?;

        // warmup both
        for _ in 0..2 {
            let _ = plan.run(tvec!(t_in.clone().into()))?;
            let _ = spanf_x4_simd(&input, h, wd, &w);
        }
        let (mut tt, mut tm) = (Vec::new(), Vec::new());
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        let mut tv3 = Vec::new();
        for _ in 0..rounds {
            let t = Instant::now();
            let r = plan.run(tvec!(t_in.clone().into()))?;
            std::hint::black_box(&r);
            tt.push(t.elapsed().as_secs_f64() * 1e3);

            let t = Instant::now();
            std::hint::black_box(spanf_x4_simd(&input, h, wd, &w));
            tm.push(t.elapsed().as_secs_f64() * 1e3);

            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            {
                let t = Instant::now();
                std::hint::black_box(zensr_micro::simd::spanf_x4_simd_force_v3(&input, h, wd, &w));
                tv3.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        let mp = (h * wd * 16) as f64 / 1e6;
        let (tmin, tmed) = stats(tt);
        let (mmin, mmed) = stats(tm);
        println!(
            "{h}x{wd}\ttract min={tmin:.2} med={tmed:.2} ({:.3} MP/s)\tmicro-v4x min={mmin:.2} med={mmed:.2} ({:.3} MP/s)\tratio(min) micro/tract={:.2}x",
            mp / (tmin / 1e3),
            mp / (mmin / 1e3),
            mmin / tmin
        );
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        {
            let (vmin, vmed) = stats(tv3);
            println!(
                "\tmicro-v3(AVX2) min={vmin:.2} med={vmed:.2} ({:.3} MP/s)\tv4x-vs-v3 speedup={:.2}x",
                mp / (vmin / 1e3),
                vmin / stats_min_helper(&[mmin])
            );
        }
    }
    // Multithreaded tiled scaling: 512^2 input -> 2048^2 out.
    let (h, wd) = (512usize, 512usize);
    let input: Vec<f32> = (0..3 * h * wd).map(|i| (i % 251) as f32 / 251.0).collect();
    let model =
        zensr_micro::SpanfModel::new(read_f32(&dir.join("spanf_weights.raw"))).expect("model");
    let mp = (h * wd * 16) as f64 / 1e6;
    println!(
        "# tiled scaling {h}x{wd} -> {}x{}; grid = ceil(512/tile)^2 tiles",
        4 * h,
        4 * wd
    );
    for tile in [112usize, 128, 160, 224] {
        let grid = wd.div_ceil(tile);
        for threads in [1usize, 4, 8, 16, 24] {
            let mut ts = Vec::new();
            let reps = if threads >= 8 { 5 } else { 3 };
            for _ in 0..reps {
                let t = Instant::now();
                std::hint::black_box(zensr_micro::spanf_x4_tiled(
                    &model, &input, h, wd, threads, tile,
                ));
                ts.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let (mn, _md) = stats(ts);
            println!(
                "tile={tile} ({grid}x{grid}={} tiles)	threads={threads}	min={mn:.1}ms	{:.2} MP-out/s",
                grid * grid,
                mp / (mn / 1e3)
            );
        }
    }
    Ok(())
}

fn stats_min_helper(v: &[f64]) -> f64 {
    v[0]
}
