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
//! All passes are O(pixels) and memory-bound; scalar implementations.
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

/// Bilinear upsample [3,h,w] -> [3,s*h,s*w] (edge-clamped, center-aligned).
pub fn bilinear_up(lr: &[f32], h: usize, w: usize, s: usize) -> Vec<f32> {
    let (oh, ow) = (h * s, w * s);
    let mut out = vec![0.0f32; 3 * oh * ow];
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
                out[c * oh * ow + oy * ow + ox] = a * (1.0 - ty) + b * ty;
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
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let luma = |yy: usize, xx: usize| -> f32 {
                let i = yy * w + xx;
                0.299 * lr[i] + 0.587 * lr[plane + i] + 0.114 * lr[2 * plane + i]
            };
            let lap = 4.0 * luma(y, x)
                - luma(y - 1, x)
                - luma(y + 1, x)
                - luma(y, x - 1)
                - luma(y, x + 1);
            let ci = (y / CELL) * cw + (x / CELL);
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
    let base = bilinear_up(lr, h, w, s);
    let mut report = GuardReport {
        mean_alpha: 1.0,
        ..Default::default()
    };

    // 2. texture gate (computed on LR, applied to the residual)
    let (tmap, tch, tcw) = if cfg.texture_gate {
        texture_map(lr, h, w)
    } else {
        (Vec::new(), 0, 0)
    };
    let alpha_at = |oy: usize, ox: usize| -> f32 {
        if !cfg.texture_gate || tmap.is_empty() {
            return 1.0;
        }
        // sample cell map bilinearly at LR coords
        let fy = ((oy as f32 + 0.5) / s as f32 - 0.5) / 16.0 - 0.5;
        let fx = ((ox as f32 + 0.5) / s as f32 - 0.5) / 16.0 - 0.5;
        let fy = fy.max(0.0);
        let fx = fx.max(0.0);
        let y0 = (fy as usize).min(tch - 1);
        let x0 = (fx as usize).min(tcw - 1);
        let y1 = (y0 + 1).min(tch - 1);
        let x1 = (x0 + 1).min(tcw - 1);
        let (ty, tx) = (fy - y0 as f32, fx - x0 as f32);
        let t = tmap[y0 * tcw + x0] * (1.0 - ty) * (1.0 - tx)
            + tmap[y0 * tcw + x1] * (1.0 - ty) * tx
            + tmap[y1 * tcw + x0] * ty * (1.0 - tx)
            + tmap[y1 * tcw + x1] * ty * tx;
        (1.2 - t / cfg.texture_k).clamp(cfg.texture_alpha_min, 1.0)
    };

    // 1.+2. clamp + gate in one pass
    let tau = cfg.residual_clamp;
    let mut clamped = 0usize;
    let mut alpha_sum = 0.0f64;
    for oy in 0..oh {
        for ox in 0..ow {
            let a = alpha_at(oy, ox);
            alpha_sum += a as f64;
            for c in 0..3 {
                let i = c * oplane + oy * ow + ox;
                let mut r = (sr[i] - base[i]) * a;
                if r > tau {
                    r = tau;
                    clamped += 1;
                } else if r < -tau {
                    r = -tau;
                    clamped += 1;
                }
                sr[i] = (base[i] + r).clamp(0.0, 1.0);
            }
        }
    }
    report.clamped_frac = clamped as f32 / (3 * oplane) as f32;
    report.mean_alpha = (alpha_sum / oplane as f64) as f32;

    // 3. round-trip fallback: box-downscale sr and compare to lr
    if cfg.roundtrip_mae_max.is_finite() {
        let mut mae = 0.0f64;
        let inv = 1.0 / (s * s) as f32;
        let plane = h * w;
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let mut sum = 0.0f32;
                    for dy in 0..s {
                        for dx in 0..s {
                            sum += sr[c * oplane + (y * s + dy) * ow + (x * s + dx)];
                        }
                    }
                    mae += (sum * inv - lr[c * plane + y * w + x]).abs() as f64;
                }
            }
        }
        let mae = (mae / (3 * plane) as f64) as f32;
        report.roundtrip_mae = mae;
        if mae > cfg.roundtrip_mae_max {
            // blend toward base proportionally to the excess (full fallback at 2x threshold)
            let t = ((mae - cfg.roundtrip_mae_max) / cfg.roundtrip_mae_max).min(1.0);
            report.fallback_blend = t;
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
}
