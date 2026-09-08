//! Isolated conv3x3 loop for hardware-counter profiling.
//!
//! `perf stat -e cycles,instructions,fp_ret_sse_avx_ops.all ... conv_probe`
//! The bench harness interleaves two arms; this runs one kernel only, so the
//! counters describe the kernel rather than the harness.
//!
//! Args: `conv_probe [iters] [side] [channels] [tier]`. `tier` is `v4x` (the
//! default on a machine that has it), `v3` (disables the AVX-512 token so the
//! ladder falls to AVX2), or `scalar`. Forcing the tier matters because a
//! formulation can win on one and lose on another — twelve accumulator chains
//! were +4.2% on AVX-512 and -12% on AVX2, and we ship ONE binary that picks
//! the tier at runtime, so a change has to be measured on both before it can
//! be called a win.
use std::time::Instant;

/// Disable the tokens above `tier` process-wide so `incant!` falls through to
/// it. Returns the tier actually in force. Requires archmage's
/// `testable_dispatch`, which the dev-dependency enables.
#[cfg(target_arch = "x86_64")]
fn force_tier(tier: &str) -> &str {
    match tier {
        "v3" => {
            let _ = archmage::X64V4xToken::dangerously_disable_token_process_wide(true);
            let _ = archmage::X64V4Token::dangerously_disable_token_process_wide(true);
            "v3"
        }
        "scalar" => {
            let _ = archmage::X64V4xToken::dangerously_disable_token_process_wide(true);
            let _ = archmage::X64V4Token::dangerously_disable_token_process_wide(true);
            let _ = archmage::X64V3Token::dangerously_disable_token_process_wide(true);
            "scalar"
        }
        _ => "default",
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn force_tier(_tier: &str) -> &str {
    "default"
}

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
    // Third arg: channel count. cin*3 rows are touched per tile, each on its own
    // page at large sizes, so this is the knob that decides TLB pressure.
    let ch: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let (cin, cout) = (ch, ch);
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    // Second arg picks the plane size. 128 keeps the whole working set in L2;
    // larger sizes are where input-row locality across output quads can matter.
    let side: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    // Fourth arg forces the tier; see force_tier.
    let tier = std::env::args().nth(4).unwrap_or_else(|| "default".into());
    let tier = force_tier(&tier);
    let (h, wd) = (side, side);
    let inp = ramp(cin * h * wd, 7);
    let wts = ramp(cout * cin * 9, 11);
    let bias = ramp(cout, 13);
    let mut out = vec![0f32; cout * h * wd];
    // warm
    for _ in 0..20 {
        zensr_micro::simd::conv3x3_dispatch(&inp, cin, &wts, &bias, &mut out, cout, h, wd);
    }
    let t = Instant::now();
    for _ in 0..iters {
        zensr_micro::simd::conv3x3_dispatch(&inp, cin, &wts, &bias, &mut out, cout, h, wd);
    }
    let el = t.elapsed();
    let flops = 2.0 * (cin * cout * 9 * h * wd) as f64 * iters as f64;
    eprintln!(
        "{iters} iters @ {side}px c{ch} tier={tier}, {:.3} ms/iter, {:.1} GFLOP/s, checksum {:.3}",
        el.as_secs_f64() * 1e3 / iters as f64,
        flops / el.as_secs_f64() / 1e9,
        out.iter().take(64).sum::<f32>()
    );
}
