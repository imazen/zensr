//! Does synthesising the shifted conv taps beat loading them?
//!
//! The 3x3 kernel needs three W-lane vectors at x-1, x, x+1. Two of the three
//! are 4-byte-misaligned by construction, so on AVX-512 a 64-byte load crosses a
//! cache line every time. The alternative is two ALIGNED loads (x-1 and
//! x-1+W) plus two `valignd` funnel shifts. This prices both, in isolation, with
//! identical FMA work.
//!
//! Requires `--features internals,unsafe-experiments`. TEMPORARY — see the
//! safety policy in lib.rs.
#![cfg(all(target_arch = "x86_64", feature = "unsafe-experiments"))]
use std::time::Instant;

#[allow(unsafe_code)]
mod probe {
    use core::arch::x86_64::*;

    /// Current shape: three unaligned loads per tap.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn loads(rows: &[*const f32], w: &[f32], n_tap: usize, x: usize) -> __m512 {
        // EIGHT chains, as the shipped kernel has: (l,m) share one per output,
        // r gets its own. With two the loop is latency-bound and cannot show a
        // load-path difference at all.
        let mut lm = [_mm512_setzero_ps(); 4];
        let mut rr = [_mm512_setzero_ps(); 4];
        for t in 0..n_tap {
            let p = rows[t].add(x);
            let l = _mm512_loadu_ps(p);
            let m = _mm512_loadu_ps(p.add(1));
            let r = _mm512_loadu_ps(p.add(2));
            let wp = w.as_ptr().add(t * 12);
            for ob in 0..4 {
                lm[ob] = _mm512_fmadd_ps(l, _mm512_set1_ps(*wp.add(ob * 3)), lm[ob]);
                lm[ob] = _mm512_fmadd_ps(m, _mm512_set1_ps(*wp.add(ob * 3 + 1)), lm[ob]);
                rr[ob] = _mm512_fmadd_ps(r, _mm512_set1_ps(*wp.add(ob * 3 + 2)), rr[ob]);
            }
        }
        let mut acc = _mm512_setzero_ps();
        for ob in 0..4 {
            acc = _mm512_add_ps(acc, _mm512_add_ps(lm[ob], rr[ob]));
        }
        acc
    }

    /// Two UNALIGNED loads plus two VALIGND funnel shifts. If this matches the
    /// aligned form, the kernel needs no alignment guarantee at all.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn unaligned_shift(
        rows: &[*const f32],
        w: &[f32],
        n_tap: usize,
        x: usize,
    ) -> __m512 {
        let mut lm = [_mm512_setzero_ps(); 4];
        let mut rr = [_mm512_setzero_ps(); 4];
        for t in 0..n_tap {
            let p = rows[t].add(x);
            let lo = _mm512_loadu_ps(p);
            let hi = _mm512_loadu_ps(p.add(16));
            let l = lo;
            let m = _mm512_castsi512_ps(_mm512_alignr_epi32(
                _mm512_castps_si512(hi),
                _mm512_castps_si512(lo),
                1,
            ));
            let r = _mm512_castsi512_ps(_mm512_alignr_epi32(
                _mm512_castps_si512(hi),
                _mm512_castps_si512(lo),
                2,
            ));
            let wp = w.as_ptr().add(t * 12);
            for ob in 0..4 {
                lm[ob] = _mm512_fmadd_ps(l, _mm512_set1_ps(*wp.add(ob * 3)), lm[ob]);
                lm[ob] = _mm512_fmadd_ps(m, _mm512_set1_ps(*wp.add(ob * 3 + 1)), lm[ob]);
                rr[ob] = _mm512_fmadd_ps(r, _mm512_set1_ps(*wp.add(ob * 3 + 2)), rr[ob]);
            }
        }
        let mut acc = _mm512_setzero_ps();
        for ob in 0..4 {
            acc = _mm512_add_ps(acc, _mm512_add_ps(lm[ob], rr[ob]));
        }
        acc
    }

    /// Two aligned loads plus two VALIGND funnel shifts.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn aligned_shift(rows: &[*const f32], w: &[f32], n_tap: usize, x: usize) -> __m512 {
        // EIGHT chains, as the shipped kernel has: (l,m) share one per output,
        // r gets its own. With two the loop is latency-bound and cannot show a
        // load-path difference at all.
        let mut lm = [_mm512_setzero_ps(); 4];
        let mut rr = [_mm512_setzero_ps(); 4];
        for t in 0..n_tap {
            let p = rows[t].add(x);
            let lo = _mm512_load_ps(p);
            let hi = _mm512_load_ps(p.add(16));
            let l = lo;
            // valignd: concatenate (hi:lo) and shift right by k 32-bit elements.
            let m = _mm512_castsi512_ps(_mm512_alignr_epi32(
                _mm512_castps_si512(hi),
                _mm512_castps_si512(lo),
                1,
            ));
            let r = _mm512_castsi512_ps(_mm512_alignr_epi32(
                _mm512_castps_si512(hi),
                _mm512_castps_si512(lo),
                2,
            ));
            let wp = w.as_ptr().add(t * 12);
            for ob in 0..4 {
                lm[ob] = _mm512_fmadd_ps(l, _mm512_set1_ps(*wp.add(ob * 3)), lm[ob]);
                lm[ob] = _mm512_fmadd_ps(m, _mm512_set1_ps(*wp.add(ob * 3 + 1)), lm[ob]);
                rr[ob] = _mm512_fmadd_ps(r, _mm512_set1_ps(*wp.add(ob * 3 + 2)), rr[ob]);
            }
        }
        let mut acc = _mm512_setzero_ps();
        for ob in 0..4 {
            acc = _mm512_add_ps(acc, _mm512_add_ps(lm[ob], rr[ob]));
        }
        acc
    }
}

/// Fold a vector to a scalar so the loop cannot be optimised away.
#[allow(unsafe_code)]
unsafe fn reduce(v: core::arch::x86_64::__m512) -> f32 {
    let mut out = [0f32; 16];
    core::arch::x86_64::_mm512_storeu_ps(out.as_mut_ptr(), v);
    out[0]
}

#[allow(unsafe_code)]
fn main() {
    const NTAP: usize = 96; // cin 32 x 3 vertical taps
    const ROWLEN: usize = 4096;
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000);
    // 64-byte-aligned rows so `_mm512_load_ps` is legal.
    let mut store: Vec<f32> = vec![0.0; NTAP * ROWLEN + 32];
    let mut s = 12345u32;
    for v in store.iter_mut() {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *v = (s >> 8) as f32 / 16_777_216.0;
    }
    let base = store.as_ptr();
    let off = (64 - (base as usize % 64)) / 4 % 16;
    let rows: Vec<*const f32> = (0..NTAP)
        .map(|t| unsafe { base.add(off + t * ROWLEN) })
        .collect();
    let w: Vec<f32> = (0..NTAP * 12).map(|i| (i % 17) as f32 * 0.01).collect();

    for (name, f) in [
        ("3 unaligned loads", probe::loads as unsafe fn(&[*const f32], &[f32], usize, usize) -> _),
        ("2 aligned + 2 valignd", probe::aligned_shift),
        ("2 UNaligned + 2 valignd", probe::unaligned_shift),
    ] {
        let mut acc = 0.0f32;
        // warm
        for _ in 0..1000 {
            acc += unsafe { reduce(f(&rows, &w, NTAP, 0)) };
        }
        let t = Instant::now();
        for i in 0..iters {
            acc += unsafe { reduce(f(&rows, &w, NTAP, (i % 64) * 16)) };
        }
        let el = t.elapsed().as_secs_f64();
        // 96 taps x 12 FMA x 16 lanes x 2 flop
        let flops = NTAP as f64 * 12.0 * 16.0 * 2.0 * iters as f64;
        println!(
            "  {name:<24} {:>8.2} ns/tap-loop  {:>7.1} GFLOP/s  (guard {acc:.1})",
            el * 1e9 / iters as f64,
            flops / el / 1e9
        );
    }
}
