//! Isolated conv3x3 loop for hardware-counter profiling.
//!
//! `perf stat -e cycles,instructions,fp_ret_sse_avx_ops.all ... conv_probe`
//! The bench harness interleaves two arms; this runs one kernel only, so the
//! counters describe the kernel rather than the harness.
use std::time::Instant;

fn ramp(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

fn main() {
    const CIN: usize = 32;
    const COUT: usize = 32;
    const H: usize = 128;
    const WD: usize = 128;
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let inp = ramp(CIN * H * WD, 7);
    let wts = ramp(COUT * CIN * 9, 11);
    let bias = ramp(COUT, 13);
    let mut out = vec![0f32; COUT * H * WD];
    // warm
    for _ in 0..20 {
        zensr_micro::simd::conv3x3_dispatch(&inp, CIN, &wts, &bias, &mut out, COUT, H, WD);
    }
    let t = Instant::now();
    for _ in 0..iters {
        zensr_micro::simd::conv3x3_dispatch(&inp, CIN, &wts, &bias, &mut out, COUT, H, WD);
    }
    let el = t.elapsed();
    let flops = 2.0 * (CIN * COUT * 9 * H * WD) as f64 * iters as f64;
    eprintln!(
        "{iters} iters, {:.3} ms/iter, {:.1} GFLOP/s, checksum {:.3}",
        el.as_secs_f64() * 1e3 / iters as f64,
        flops / el.as_secs_f64() / 1e9,
        out.iter().take(64).sum::<f32>()
    );
}
