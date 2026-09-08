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

// Only the non-x86 path still needs a single alias; x86 disables every tier
// explicitly (see set_simd).
#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;

/// The tier the dispatcher will ACTUALLY pick here, probed at runtime.
///
/// This used to be a hardcoded `"v3(avx2)"` on every x86_64 host. It is wrong on
/// any CPU with AVX-512: the `incant!` ladder is `[v4x(cfg(avx512)), v3, ...]`,
/// so v4x is tried first and wins — measured on a 7950X (avx512f/bw/cd/dq/vl,
/// both tokens summon, 153 zmm instructions in the bench binary), where the arm
/// labelled "v3(avx2)" was running AVX-512. A benchmark that names the arm it
/// did not measure is worse than no benchmark; naming is the one thing a
/// measurement cannot be wrong about.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn tier_name() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    {
        "neon"
    }
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken as _;
        // Ladder order, strongest first — must match the `incant!` lists in
        // src/simd.rs, or this label lies in the other direction.
        #[cfg(feature = "avx512")]
        if archmage::X64V4xToken::summon().is_some() {
            return "v4x(avx512)";
        }
        if archmage::X64V3Token::summon().is_some() {
            return "v3(avx2)";
        }
        "scalar(no simd token)"
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(enabled: bool) -> bool {
    // Disable every x86 tier, not just V3. V4x is a superset, so disabling V3
    // alone could leave AVX-512 live and make the "scalar" arm not scalar —
    // the ratio would then be v4x-vs-v4x and read as ~1.00x, i.e. a silent PASS.
    #[cfg(target_arch = "x86_64")]
    {
        let a = archmage::X64V3Token::dangerously_disable_token_process_wide(!enabled).is_ok();
        #[cfg(feature = "avx512")]
        let b = archmage::X64V4xToken::dangerously_disable_token_process_wide(!enabled).is_ok();
        #[cfg(not(feature = "avx512"))]
        let b = true;
        a && b
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        TierToken::dangerously_disable_token_process_wide(!enabled).is_ok()
    }
}
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_e: bool) -> bool {
    false
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

fn bench_kernels(suite: &mut Suite) {
    if !set_simd(true) || !set_simd(false) {
        eprintln!("[kernel_tiers] SIMD tier not toggleable here. Skipping.");
        return;
    }
    set_simd(true);
    let tier = tier_name();
    eprintln!("[kernel_tiers] comparing {tier} vs forced scalar");

    // SiLU — the pointwise activation, run after every conv.
    const N: usize = 1 << 20;
    suite.compare("silu_dispatch/1M", |g| {
        g.throughput(Throughput::Bytes((N * 4) as u64));
        for (arm, simd) in [(tier, true), ("scalar", false)] {
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

    // conv3x3 — the dominant cost of the network, SWEPT OVER PLANE SIZE.
    //
    // One size is not enough and 128px is the worst possible choice. The kernel
    // is compute-bound while the working set fits cache, and TLB-bound above it:
    // in the planar layout a tile touches cin*3 rows that are `cs` floats apart,
    // so at 1024px they occupy 96 pages against a 64-entry L1 dTLB. Measured
    // 2026-09-08, the loop-order + channel-blocking fixes were worth **+237% at
    // 1024px and ~0% at 128px** — this bench, at 128px only, called them noise.
    // See benchmarks/realtime_kernels_x86_2026-09-08.md.
    //
    // 64 / 128 / 512 spans tiny, cache-resident and TLB-pressured. 1024px is
    // more representative still but costs 256 MB of buffers per arm.
    const CIN: usize = 32;
    const COUT: usize = 32;
    for &side in &[64usize, 128, 512] {
        let (h, wd) = (side, side);
        let inp: &'static [f32] = Box::leak(ramp(CIN * h * wd, 7).into_boxed_slice());
        let wts: &'static [f32] = Box::leak(ramp(COUT * CIN * 9, 11).into_boxed_slice());
        let bias: &'static [f32] = Box::leak(ramp(COUT, 13).into_boxed_slice());
        let name: &'static str = Box::leak(
            format!("conv3x3_dispatch/{CIN}x{COUT}x{side}x{side}").into_boxed_str(),
        );
        suite.compare(name, move |g| {
            g.throughput(Throughput::Elements((COUT * h * wd) as u64));
            for (arm, simd) in [(tier, true), ("scalar", false)] {
                g.bench(arm, move |b| {
                    b.with_input(move || {
                        set_simd(simd);
                        vec![0f32; COUT * h * wd]
                    })
                    .run(move |mut out| {
                        zensr_micro::simd::conv3x3_dispatch(
                            inp, CIN, wts, bias, &mut out, COUT, h, wd,
                        );
                        out
                    })
                });
            }
        });
    }
}

zenbench::main!(bench_kernels);
