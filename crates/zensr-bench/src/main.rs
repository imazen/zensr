//! CPU inference bench for fixed-shape SR ONNX exports via tract.
//!
//! Usage: zensr-bench <model.onnx> <H> <W> [iters]
//! Prints TSV: model, input, load_ms, opt_ms, min_ms, p50_ms, out_mp_per_s (median).
//! tract runs single-threaded per call (no rayon feature enabled) — the number is
//! per-core; tile-level parallelism multiplies it by usable cores.

use std::time::Instant;
use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: zensr-bench model.onnx H W [iters]");
    let h: usize = args.next().expect("H").parse().unwrap();
    let w: usize = args.next().expect("W").parse().unwrap();
    let iters: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(12);

    let t0 = Instant::now();
    let model = tract_onnx::onnx()
        .model_for_path(&path)?
        .with_input_fact(0, f32::fact([1, 3, h, w]).into())?;
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = Instant::now();
    let plan = model.into_optimized()?.into_runnable()?;
    let opt_ms = t1.elapsed().as_secs_f64() * 1e3;

    // Deterministic non-constant input in [0,1).
    let input = Tensor::from_shape(
        &[1, 3, h, w],
        &(0..3 * h * w)
            .map(|i| (i % 251) as f32 / 251.0)
            .collect::<Vec<f32>>(),
    )?;

    let mut out_pixels = 0usize;
    for _ in 0..3 {
        let r = plan.run(tvec!(input.clone().into()))?;
        out_pixels = r[0].shape()[2] * r[0].shape()[3];
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let r = plan.run(tvec!(input.clone().into()))?;
        std::hint::black_box(&r);
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = times[0];
    let p50 = times[times.len() / 2];
    let mp_s = (out_pixels as f64 / 1e6) / (p50 / 1e3);

    let name = std::path::Path::new(&path)
        .file_stem()
        .unwrap()
        .to_string_lossy();
    println!(
        "{name}\t{h}x{w}\tload_ms={load_ms:.1}\topt_ms={opt_ms:.1}\tmin_ms={min:.2}\tp50_ms={p50:.2}\tout_mp_s={mp_s:.3}"
    );
    Ok(())
}
