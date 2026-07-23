//! Bounded-downside guards for blind en-masse usage.
//!
//! Three independent mechanisms, all anchored to a cheap bilinear upsample of
//! the input ("base"):
//! 1. **Residual clamp** — out = base + clamp(sr - base, ±tau). The maximum
//!    deviation from the linear baseline is bounded BY CONSTRUCTION; no model
//!    failure mode can exceed it.
//! 2. **Texture gate** — per-cell alpha in [alpha_min, 1] shrinks the SR
//!    residual where the input looks like stochastic texture (high local
//!    high-pass energy), where SR models measurably underperform linear
//!    filters. Alpha is bilinearly interpolated between cells (no blocking).
//! 3. **Round-trip fallback** — box-downscale the SR output back to input
//!    resolution; if the mean abs error vs the input exceeds a threshold the
//!    model has gone off-distribution and the whole image blends toward base.
//!
//! All passes are O(pixels) and memory-bound. The implementation is separable
//! and row-fused: horizontal bilinear rows are computed once per LR row (not
//! per output row), vertical lerp / alpha / clamp run as contiguous row loops
//! that LLVM auto-vectorizes — no explicit SIMD needed (and none wanted; see
//! the 26x MIR-inline incident in the README).
#![allow(clippy::too_many_arguments)]

#[derive(Clone, Copy, Debug)]
pub struct GuardConfig {
    /// Max |out - base| in [0,1] units. INFINITY disables. Default 0.25.
    pub residual_clamp: f32,
    /// Enable stochastic-texture gating. Default true.
    pub texture_gate: bool,
    /// Minimum SR-residual weight in textured cells. Default 0.35.
    pub texture_alpha_min: f32,
    /// High-pass energy (mean |laplacian| of luma) at which alpha hits min.
    /// Default 0.10 (calibrated on imazen-26 textures subcorpus).
    pub texture_k: f32,
    /// Round-trip mean-abs-error threshold; above it, blend toward base.
    /// Default 0.06. INFINITY disables.
    pub roundtrip_mae_max: f32,
}

impl Default for GuardConfig {
    fn default() -> Self {
        GuardConfig {
            residual_clamp: 0.25,
            texture_gate: true,
            texture_alpha_min: 0.35,
            texture_k: 0.10,
            roundtrip_mae_max: 0.06,
        }
    }
}

/// Diagnostics from a guarded merge (for logging / eval).
#[derive(Clone, Copy, Debug, Default)]
pub struct GuardReport {
    /// Fraction of samples whose residual hit the clamp.
    pub clamped_frac: f32,
    /// Mean texture alpha applied (1.0 = no gating anywhere).
    pub mean_alpha: f32,
    /// Round-trip MAE measured on the (guarded) SR output.
    pub roundtrip_mae: f32,
    /// Blend weight applied toward base by the round-trip fallback (0 = none).
    pub fallback_blend: f32,
}

/// Center-aligned source-coordinate table: for each output x, the two source
/// columns and the interpolation weight.
struct XTab {
    x0: Vec<u32>,
    x1: Vec<u32>,
    tx: Vec<f32>,
}

fn xtab(src: usize, dst: usize, s: usize) -> XTab {
    let fs = s as f32;
    let mut x0 = Vec::with_capacity(dst);
    let mut x1 = Vec::with_capacity(dst);
    let mut tx = Vec::with_capacity(dst);
    for ox in 0..dst {
        let fx = ((ox as f32 + 0.5) / fs - 0.5).max(0.0);
        let a = (fx as usize).min(src - 1);
        x0.push(a as u32);
        x1.push(((a + 1).min(src - 1)) as u32);
        tx.push(fx - a as f32);
    }
    XTab { x0, x1, tx }
}

/// Vertical source rows for each output row (y0, y1, ty).
fn ytab(src: usize, dst: usize, s: usize) -> Vec<(usize, usize, f32)> {
    let fs = s as f32;
    (0..dst)
        .map(|oy| {
            let fy = ((oy as f32 + 0.5) / fs - 0.5).max(0.0);
            let a = (fy as usize).min(src - 1);
            (a, (a + 1).min(src - 1), fy - a as f32)
        })
        .collect()
}

/// Horizontal bilinear interpolation of one LR row into width-`ow` `dst`.
fn hinterp_row(ip: &[f32], xt: &XTab, dst: &mut [f32]) {
    for i in 0..dst.len() {
        let t = xt.tx[i];
        dst[i] = ip[xt.x0[i] as usize] * (1.0 - t) + ip[xt.x1[i] as usize] * t;
    }
}

/// Bilinear upsample [3,h,w] -> [3,s*h,s*w] (edge-clamped, center-aligned).
/// Separable: h-interp once per LR row, then a contiguous vertical lerp.
pub fn bilinear_up(lr: &[f32], h: usize, w: usize, s: usize) -> Vec<f32> {
    let (oh, ow) = (h * s, w * s);
    let mut out = vec![0.0f32; 3 * oh * ow];
    let xt = xtab(w, ow, s);
    let yt = ytab(h, oh, s);
    let mut h0 = vec![0.0f32; ow];
    let mut h1 = vec![0.0f32; ow];
    for c in 0..3 {
        let ip = &lr[c * h * w..(c + 1) * h * w];
        let op = &mut out[c * oh * ow..(c + 1) * oh * ow];
        let mut cached: (usize, usize) = (usize::MAX, usize::MAX);
        for (oy, &(y0, y1, ty)) in yt.iter().enumerate() {
            if cached != (y0, y1) {
                hinterp_row(&ip[y0 * w..y0 * w + w], &xt, &mut h0);
                hinterp_row(&ip[y1 * w..y1 * w + w], &xt, &mut h1);
                cached = (y0, y1);
            }
            let dst = &mut op[oy * ow..oy * ow + ow];
            for i in 0..ow {
                dst[i] = h0[i] * (1.0 - ty) + h1[i] * ty;
            }
        }
    }
    out
}

/// Per-cell (16x16 LR) mean |laplacian| of luma — stochastic-texture proxy.
fn texture_map(lr: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
    const CELL: usize = 16;
    let (ch, cw) = (h.div_ceil(CELL), w.div_ceil(CELL));
    let plane = h * w;
    let mut acc = vec![0.0f32; ch * cw];
    let mut cnt = vec![0u32; ch * cw];
    let mut luma = vec![0.0f32; plane];
    for i in 0..plane {
        luma[i] = 0.299 * lr[i] + 0.587 * lr[plane + i] + 0.114 * lr[2 * plane + i];
    }
    for y in 1..h.saturating_sub(1) {
        let ci_row = (y / CELL) * cw;
        for x in 1..w.saturating_sub(1) {
            let i = y * w + x;
            let lap = 4.0 * luma[i] - luma[i - w] - luma[i + w] - luma[i - 1] - luma[i + 1];
            let ci = ci_row + x / CELL;
            acc[ci] += lap.abs();
            cnt[ci] += 1;
        }
    }
    for i in 0..acc.len() {
        if cnt[i] > 0 {
            acc[i] /= cnt[i] as f32;
        }
    }
    (acc, ch, cw)
}

/// Apply guards in place on `sr` ([3, s*h, s*w]); `lr` is the [3,h,w] input.
pub fn guarded_merge(
    sr: &mut [f32],
    lr: &[f32],
    h: usize,
    w: usize,
    s: usize,
    cfg: &GuardConfig,
) -> GuardReport {
    let (oh, ow) = (h * s, w * s);
    let oplane = oh * ow;
    let mut report = GuardReport {
        mean_alpha: 1.0,
        ..Default::default()
    };

    // texture gate cell map (LR-resolution cells)
    let (tmap, tch, tcw) = if cfg.texture_gate {
        texture_map(lr, h, w)
    } else {
        (Vec::new(), 0, 0)
    };

    // Separable tables for base (LR -> output) and alpha (cell map -> output).
    let xt = xtab(w, ow, s);
    let yt = ytab(h, oh, s);
    // cell-space coordinate of output pixel: ((o+0.5)/s - 0.5)/16 - 0.5, clamped
    let (cxt, cyt) = if cfg.texture_gate {
        let fs = s as f32;
        let mk = |dst: usize, cells: usize| -> (Vec<u32>, Vec<u32>, Vec<f32>) {
            let mut i0 = Vec::with_capacity(dst);
            let mut i1 = Vec::with_capacity(dst);
            let mut tt = Vec::with_capacity(dst);
            for o in 0..dst {
                let f = ((((o as f32 + 0.5) / fs - 0.5) / 16.0) - 0.5).max(0.0);
                let a = (f as usize).min(cells - 1);
                i0.push(a as u32);
                i1.push(((a + 1).min(cells - 1)) as u32);
                tt.push(f - a as f32);
            }
            (i0, i1, tt)
        };
        (mk(ow, tcw), mk(oh, tch))
    } else {
        ((Vec::new(), Vec::new(), Vec::new()), (Vec::new(), Vec::new(), Vec::new()))
    };

    let tau = cfg.residual_clamp;
    let mut clamped = 0usize;
    let mut alpha_sum = 0.0f64;
    let mut h0 = vec![0.0f32; ow];
    let mut h1 = vec![0.0f32; ow];
    let mut base_row = vec![0.0f32; ow];
    let mut alpha_row = vec![1.0f32; ow];
    // horizontal-interped tmap rows (tiny: tcw cells wide -> ow)
    let mut arow0 = vec![0.0f32; ow];
    let mut arow1 = vec![0.0f32; ow];

    for c in 0..3 {
        let ip = &lr[c * h * w..(c + 1) * h * w];
        let sp = &mut sr[c * oplane..(c + 1) * oplane];
        let mut cached: (usize, usize) = (usize::MAX, usize::MAX);
        let mut acached: (usize, usize) = (usize::MAX, usize::MAX);
        for (oy, &(y0, y1, ty)) in yt.iter().enumerate() {
            if cached != (y0, y1) {
                hinterp_row(&ip[y0 * w..y0 * w + w], &xt, &mut h0);
                hinterp_row(&ip[y1 * w..y1 * w + w], &xt, &mut h1);
                cached = (y0, y1);
            }
            for i in 0..ow {
                base_row[i] = h0[i] * (1.0 - ty) + h1[i] * ty;
            }
            if cfg.texture_gate {
                let cy0 = cyt.0[oy] as usize;
                let cy1 = cyt.1[oy] as usize;
                let cty = cyt.2[oy];
                if acached != (cy0, cy1) {
                    for i in 0..ow {
                        let t = cxt.2[i];
                        arow0[i] = tmap[cy0 * tcw + cxt.0[i] as usize] * (1.0 - t)
                            + tmap[cy0 * tcw + cxt.1[i] as usize] * t;
                        arow1[i] = tmap[cy1 * tcw + cxt.0[i] as usize] * (1.0 - t)
                            + tmap[cy1 * tcw + cxt.1[i] as usize] * t;
                    }
                    acached = (cy0, cy1);
                }
                for i in 0..ow {
                    let t = arow0[i] * (1.0 - cty) + arow1[i] * cty;
                    alpha_row[i] = (1.2 - t / cfg.texture_k).clamp(cfg.texture_alpha_min, 1.0);
                }
            }
            if c == 0 {
                let mut asum = 0.0f32;
                for i in 0..ow {
                    asum += alpha_row[i];
                }
                alpha_sum += asum as f64;
            }
            let dst = &mut sp[oy * ow..oy * ow + ow];
            let mut hit = 0usize;
            for i in 0..ow {
                let b = base_row[i];
                let r = (dst[i] - b) * alpha_row[i];
                let rc = r.clamp(-tau, tau);
                hit += usize::from(rc != r);
                dst[i] = (b + rc).clamp(0.0, 1.0);
            }
            clamped += hit;
        }
    }
    report.clamped_frac = clamped as f32 / (3 * oplane) as f32;
    report.mean_alpha = (alpha_sum / oplane as f64) as f32;

    // 3. round-trip fallback: box-downscale sr and compare to lr
    if cfg.roundtrip_mae_max.is_finite() {
        let mut mae = 0.0f64;
        let inv = 1.0 / (s * s) as f32;
        let plane = h * w;
        let mut rowacc = vec![0.0f32; w];
        for c in 0..3 {
            let sp = &sr[c * oplane..(c + 1) * oplane];
            for y in 0..h {
                rowacc.fill(0.0);
                for dy in 0..s {
                    let row = &sp[(y * s + dy) * ow..(y * s + dy) * ow + ow];
                    for x in 0..w {
                        let mut sum = 0.0f32;
                        for dx in 0..s {
                            sum += row[x * s + dx];
                        }
                        rowacc[x] += sum;
                    }
                }
                let lrow = &lr[c * plane + y * w..c * plane + y * w + w];
                let mut m = 0.0f32;
                for x in 0..w {
                    m += (rowacc[x] * inv - lrow[x]).abs();
                }
                mae += m as f64;
            }
        }
        let mae = (mae / (3 * plane) as f64) as f32;
        report.roundtrip_mae = mae;
        if mae > cfg.roundtrip_mae_max {
            // blend toward base proportionally to the excess (full fallback at 2x threshold)
            let t = ((mae - cfg.roundtrip_mae_max) / cfg.roundtrip_mae_max).min(1.0);
            report.fallback_blend = t;
            let base = bilinear_up(lr, h, w, s);
            for i in 0..sr.len() {
                sr[i] = sr[i] * (1.0 - t) + base[i] * t;
            }
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_bounds_worst_case_by_construction() {
        let (h, w, s) = (12usize, 10usize, 2usize);
        let lr: Vec<f32> = (0..3 * h * w).map(|i| (i % 97) as f32 / 97.0).collect();
        // adversarial "model": outputs garbage
        let mut sr = vec![7.0f32; 3 * h * w * s * s];
        let cfg = GuardConfig {
            texture_gate: false,
            roundtrip_mae_max: f32::INFINITY,
            ..Default::default()
        };
        let rep = guarded_merge(&mut sr, &lr, h, w, s, &cfg);
        let base = bilinear_up(&lr, h, w, s);
        for (o, b) in sr.iter().zip(base.iter()) {
            assert!((o - b).abs() <= cfg.residual_clamp + 1e-6);
            assert!(*o >= 0.0 && *o <= 1.0);
        }
        assert!(rep.clamped_frac > 0.99);
    }

    #[test]
    fn identity_when_sr_equals_base() {
        let (h, w, s) = (9usize, 11usize, 3usize);
        let lr: Vec<f32> = (0..3 * h * w).map(|i| (i % 53) as f32 / 53.0).collect();
        let base = bilinear_up(&lr, h, w, s);
        let mut sr = base.clone();
        let rep = guarded_merge(&mut sr, &lr, h, w, s, &GuardConfig::default());
        for (o, b) in sr.iter().zip(base.iter()) {
            assert!((o - b).abs() < 1e-5);
        }
        assert_eq!(rep.fallback_blend, 0.0);
    }

    #[test]
    fn roundtrip_fallback_fires_on_garbage() {
        let (h, w, s) = (16usize, 16usize, 2usize);
        let lr = vec![0.5f32; 3 * h * w];
        let mut sr = vec![0.9f32; 3 * h * w * s * s]; // consistent garbage within clamp
        let cfg = GuardConfig {
            residual_clamp: f32::INFINITY,
            texture_gate: false,
            roundtrip_mae_max: 0.06,
            ..Default::default()
        };
        let rep = guarded_merge(&mut sr, &lr, h, w, s, &cfg);
        assert!(rep.fallback_blend > 0.9, "blend {}", rep.fallback_blend);
        // fully blended back to base = flat 0.5
        assert!((sr[10] - 0.5).abs() < 0.05);
    }

    /// Old-vs-new equivalence: the separable implementation must match the
    /// direct per-pixel formulation (same math, different association) tightly.
    #[test]
    fn separable_matches_direct_reference() {
        let (h, w, s) = (23usize, 17usize, 4usize);
        let lr: Vec<f32> = (0..3 * h * w).map(|i| ((i * 37) % 251) as f32 / 251.0).collect();
        let got = bilinear_up(&lr, h, w, s);
        let (oh, ow) = (h * s, w * s);
        let fs = s as f32;
        for c in 0..3 {
            let ip = &lr[c * h * w..(c + 1) * h * w];
            for oy in 0..oh {
                let fy = ((oy as f32 + 0.5) / fs - 0.5).max(0.0);
                let y0 = (fy as usize).min(h - 1);
                let y1 = (y0 + 1).min(h - 1);
                let ty = fy - y0 as f32;
                for ox in 0..ow {
                    let fx = ((ox as f32 + 0.5) / fs - 0.5).max(0.0);
                    let x0 = (fx as usize).min(w - 1);
                    let x1 = (x0 + 1).min(w - 1);
                    let tx = fx - x0 as f32;
                    let a = ip[y0 * w + x0] * (1.0 - tx) + ip[y0 * w + x1] * tx;
                    let b = ip[y1 * w + x0] * (1.0 - tx) + ip[y1 * w + x1] * tx;
                    let want = a * (1.0 - ty) + b * ty;
                    let g = got[c * oh * ow + oy * ow + ox];
                    assert!((g - want).abs() < 1e-6, "({c},{oy},{ox}): {g} vs {want}");
                }
            }
        }
    }
}
