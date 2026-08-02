//! Per-kernel NEON-vs-forced-scalar for zensr-micro's super-resolution kernels.
//!
//! This crate had NO benchmark of any kind despite being 25 dispatch sites of
//! pure SIMD (conv3x3, winograd, prelu/silu, the spanf forward). An end-to-end
//! number cannot reveal a kernel slower than its own scalar fallback, and that
//! failure mode was real in eight other zen crates during the 2026-07 aarch64
//! sweep.
//!
//! NEON is BASELINE on aarch64, so the "scalar" arm is autovectorized too.
//! Ratios here are also biased AGAINST the dispatched arm: the forced-scalar
//! path can inline into this loop while the `#[arcane]` arm carries a
//! target_feature boundary and cannot (measured at ~8% in zenresize). So
//! ~1.00x is a PASS; below ~0.95x is the finding.
//!
//! Run: `cargo bench -p zensr-micro --bench kernel_tiers`

use zenbench::prelude::*;

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") { "neon" } else { "v3(avx2)" };

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(enabled: bool) -> bool {
    TierToken::dangerously_disable_token_process_wide(!enabled).is_ok()
}
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_e: bool) -> bool { false }

fn ramp(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

fn bench_kernels(suite: &mut Suite) {
    if !set_simd(true) || !set_simd(false) {
        eprintln!("[kernel_tiers] SIMD tier not toggleable here. Skipping.");
        return;
    }
    set_simd(true);
    eprintln!("[kernel_tiers] comparing {TIER_NAME} vs forced scalar");

    // SiLU — the pointwise activation, run after every conv.
    const N: usize = 1 << 20;
    suite.compare("silu_dispatch/1M", |g| {
        g.throughput(Throughput::Bytes((N * 4) as u64));
        for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
            g.bench(arm, move |b| {
                b.with_input(move || {
                    set_simd(simd);
                    ramp(N, 3)
                })
                .run(move |mut d| {
                    zensr_micro::simd::silu_dispatch(&mut d);
                    d
                })
            });
        }
    });

    // conv3x3 — the dominant cost of the network: 32 in / 32 out channels on a
    // 128x128 plane, the shape the adopted graph actually runs.
    const CIN: usize = 32;
    const COUT: usize = 32;
    const H: usize = 128;
    const WD: usize = 128;
    let inp: &'static [f32] = Box::leak(ramp(CIN * H * WD, 7).into_boxed_slice());
    let wts: &'static [f32] = Box::leak(ramp(COUT * CIN * 9, 11).into_boxed_slice());
    let bias: &'static [f32] = Box::leak(ramp(COUT, 13).into_boxed_slice());
    suite.compare("conv3x3_dispatch/32x32x128x128", move |g| {
        g.throughput(Throughput::Elements((COUT * H * WD) as u64));
        for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
            g.bench(arm, move |b| {
                b.with_input(move || {
                    set_simd(simd);
                    vec![0f32; COUT * H * WD]
                })
                .run(move |mut out| {
                    zensr_micro::simd::conv3x3_dispatch(
                        inp, CIN, wts, bias, &mut out, COUT, H, WD,
                    );
                    out
                })
            });
        }
    });

    set_simd(true);
}

zenbench::main!(bench_kernels);
