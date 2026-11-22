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
#[non_exhaustive]
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
#[derive(Default, Debug)]
struct XTab {
    x0: Vec<u32>,
    x1: Vec<u32>,
    tx: Vec<f32>,
}

fn fill_xtab(xt: &mut XTab, src: usize, dst: usize, s: usize) {
    let fs = s as f32;
    xt.x0.clear();
    xt.x1.clear();
    xt.tx.clear();
    xt.x0.reserve(dst);
    xt.x1.reserve(dst);
    xt.tx.reserve(dst);
    for ox in 0..dst {
        let fx = ((ox as f32 + 0.5) / fs - 0.5).max(0.0);
        let a = (fx as usize).min(src - 1);
        xt.x0.push(a as u32);
        xt.x1.push(((a + 1).min(src - 1)) as u32);
        xt.tx.push(fx - a as f32);
    }
}

/// Vertical source rows for each output row (y0, y1, ty).
fn fill_ytab(yt: &mut Vec<(usize, usize, f32)>, src: usize, dst: usize, s: usize) {
    let fs = s as f32;
    yt.clear();
    yt.reserve(dst);
    for oy in 0..dst {
        let fy = ((oy as f32 + 0.5) / fs - 0.5).max(0.0);
        let a = (fy as usize).min(src - 1);
        yt.push((a, (a + 1).min(src - 1), fy - a as f32));
    }
}

#[derive(Default, Debug)]
struct CellTab {
    i0: Vec<u32>,
    i1: Vec<u32>,
    tt: Vec<f32>,
}

fn fill_cell_tab(ct: &mut CellTab, dst: usize, cells: usize, s: usize) {
    let fs = s as f32;
    ct.i0.clear();
    ct.i1.clear();
    ct.tt.clear();
    ct.i0.reserve(dst);
    ct.i1.reserve(dst);
    ct.tt.reserve(dst);
    for o in 0..dst {
        let f = ((((o as f32 + 0.5) / fs - 0.5) / 16.0) - 0.5).max(0.0);
        let a = (f as usize).min(cells - 1);
        ct.i0.push(a as u32);
        ct.i1.push(((a + 1).min(cells - 1)) as u32);
        ct.tt.push(f - a as f32);
    }
}

/// Reusable scratch buffers for zero-allocation [`guarded_merge_with_scratch`].
#[derive(Default, Debug)]
pub struct GuardScratch {
    tmap: Vec<f32>,
    cnt: Vec<u32>,
    luma: Vec<f32>,
    xt: XTab,
    yt: Vec<(usize, usize, f32)>,
    cxt: CellTab,
    cyt: CellTab,
    h0: Vec<f32>,
    h1: Vec<f32>,
    base_row: Vec<f32>,
    alpha_row: Vec<f32>,
    arow0: Vec<f32>,
    arow1: Vec<f32>,
    rowacc: Vec<f32>,
}

impl GuardScratch {
    /// Creates an empty scratch workspace that allocates lazily on first use.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-allocates scratch capacity for `x1` inputs up to `max_h × max_w`.
    #[must_use]
    pub fn with_capacity(max_h: usize, max_w: usize) -> Self {
        Self::with_scale_capacity(max_h, max_w, 1)
    }

    /// Pre-allocates scratch capacity for inputs up to `max_h × max_w` at
    /// upscale factor `max_scale`.
    #[must_use]
    pub fn with_scale_capacity(max_h: usize, max_w: usize, max_scale: usize) -> Self {
        let mut sc = Self::default();
        let cells = max_h.div_ceil(16).saturating_mul(max_w.div_ceil(16));
        let plane = max_h.saturating_mul(max_w);
        let oh = max_h.saturating_mul(max_scale.max(1));
        let ow = max_w.saturating_mul(max_scale.max(1));
        sc.tmap.reserve(cells);
        sc.cnt.reserve(cells);
        sc.luma.reserve(plane);
        sc.xt.x0.reserve(ow);
        sc.xt.x1.reserve(ow);
        sc.xt.tx.reserve(ow);
        sc.yt.reserve(oh);
        sc.cxt.i0.reserve(ow);
        sc.cxt.i1.reserve(ow);
        sc.cxt.tt.reserve(ow);
        sc.cyt.i0.reserve(oh);
        sc.cyt.i1.reserve(oh);
        sc.cyt.tt.reserve(oh);
        sc.h0.reserve(ow);
        sc.h1.reserve(ow);
        sc.base_row.reserve(ow);
        sc.alpha_row.reserve(ow);
        sc.arow0.reserve(ow);
        sc.arow1.reserve(ow);
        sc.rowacc.reserve(ow);
        sc
    }
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
    if oh == 0 || ow == 0 {
        return out;
    }
    let mut xt = XTab::default();
    let mut yt = Vec::new();
    fill_xtab(&mut xt, w, ow, s);
    fill_ytab(&mut yt, h, oh, s);
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
fn fill_texture_map(
    lr: &[f32],
    h: usize,
    w: usize,
    acc_vec: &mut Vec<f32>,
    cnt_vec: &mut Vec<u32>,
    luma_vec: &mut Vec<f32>,
) -> (usize, usize) {
    const CELL: usize = 16;
    let (ch, cw) = (h.div_ceil(CELL), w.div_ceil(CELL));
    let cells = ch * cw;
    let plane = h * w;
    if acc_vec.len() < cells {
        acc_vec.resize(cells, 0.0);
    }
    if cnt_vec.len() < cells {
        cnt_vec.resize(cells, 0);
    }
    if luma_vec.len() < plane {
        luma_vec.resize(plane, 0.0);
    }
    let acc = &mut acc_vec[..cells];
    let cnt = &mut cnt_vec[..cells];
    let luma = &mut luma_vec[..plane];
    acc.fill(0.0);
    cnt.fill(0);
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
    (ch, cw)
}

#[inline]
fn ensure_f32_slice(buf: &mut Vec<f32>, len: usize) -> &mut [f32] {
    if buf.len() < len {
        buf.resize(len, 0.0);
    }
    &mut buf[..len]
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
    let mut scratch = GuardScratch::new();
    guarded_merge_with_scratch(sr, lr, h, w, s, cfg, &mut scratch)
}

/// Apply guards in place on `sr` ([3, s*h, s*w]) reusing `scratch` buffers.
pub fn guarded_merge_with_scratch(
    sr: &mut [f32],
    lr: &[f32],
    h: usize,
    w: usize,
    s: usize,
    cfg: &GuardConfig,
    scratch: &mut GuardScratch,
) -> GuardReport {
    let (oh, ow) = (h * s, w * s);
    let oplane = oh * ow;
    let mut report = GuardReport {
        mean_alpha: 1.0,
        ..Default::default()
    };
    if oplane == 0 {
        return report;
    }

    let GuardScratch {
        tmap: tmap_vec,
        cnt: cnt_vec,
        luma: luma_vec,
        xt,
        yt,
        cxt,
        cyt,
        h0: h0_vec,
        h1: h1_vec,
        base_row: base_row_vec,
        alpha_row: alpha_row_vec,
        arow0: arow0_vec,
        arow1: arow1_vec,
        rowacc: rowacc_vec,
    } = scratch;

    // texture gate cell map (LR-resolution cells)
    let (tch, tcw) = if cfg.texture_gate {
        fill_texture_map(lr, h, w, tmap_vec, cnt_vec, luma_vec)
    } else {
        (0, 0)
    };
    let tmap = if cfg.texture_gate {
        &tmap_vec[..tch * tcw]
    } else {
        &[]
    };

    // Separable tables for base (LR -> output) and alpha (cell map -> output).
    fill_xtab(xt, w, ow, s);
    fill_ytab(yt, h, oh, s);
    if cfg.texture_gate {
        fill_cell_tab(cxt, ow, tcw, s);
        fill_cell_tab(cyt, oh, tch, s);
    }

    let tau = cfg.residual_clamp;
    let mut clamped = 0usize;
    let mut alpha_sum = 0.0f64;
    let h0 = ensure_f32_slice(h0_vec, ow);
    let h1 = ensure_f32_slice(h1_vec, ow);
    let base_row = ensure_f32_slice(base_row_vec, ow);
    let alpha_row = ensure_f32_slice(alpha_row_vec, ow);
    alpha_row.fill(1.0);
    let arow0 = ensure_f32_slice(arow0_vec, ow);
    let arow1 = ensure_f32_slice(arow1_vec, ow);

    for c in 0..3 {
        let ip = &lr[c * h * w..(c + 1) * h * w];
        let sp = &mut sr[c * oplane..(c + 1) * oplane];
        let mut cached: (usize, usize) = (usize::MAX, usize::MAX);
        let mut acached: (usize, usize) = (usize::MAX, usize::MAX);
        for (oy, &(y0, y1, ty)) in yt.iter().enumerate() {
            if cached != (y0, y1) {
                hinterp_row(&ip[y0 * w..y0 * w + w], xt, h0);
                hinterp_row(&ip[y1 * w..y1 * w + w], xt, h1);
                cached = (y0, y1);
            }
            for i in 0..ow {
                base_row[i] = h0[i] * (1.0 - ty) + h1[i] * ty;
            }
            if cfg.texture_gate {
                let cy0 = cyt.i0[oy] as usize;
                let cy1 = cyt.i1[oy] as usize;
                let cty = cyt.tt[oy];
                if acached != (cy0, cy1) {
                    for i in 0..ow {
                        let t = cxt.tt[i];
                        arow0[i] = tmap[cy0 * tcw + cxt.i0[i] as usize] * (1.0 - t)
                            + tmap[cy0 * tcw + cxt.i1[i] as usize] * t;
                        arow1[i] = tmap[cy1 * tcw + cxt.i0[i] as usize] * (1.0 - t)
                            + tmap[cy1 * tcw + cxt.i1[i] as usize] * t;
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
        let rowacc = ensure_f32_slice(rowacc_vec, w);
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
            for c in 0..3 {
                let ip = &lr[c * h * w..(c + 1) * h * w];
                let sp = &mut sr[c * oplane..(c + 1) * oplane];
                let mut cached: (usize, usize) = (usize::MAX, usize::MAX);
                for (oy, &(y0, y1, ty)) in yt.iter().enumerate() {
                    if cached != (y0, y1) {
                        hinterp_row(&ip[y0 * w..y0 * w + w], xt, h0);
                        hinterp_row(&ip[y1 * w..y1 * w + w], xt, h1);
                        cached = (y0, y1);
                    }
                    for i in 0..ow {
                        base_row[i] = h0[i] * (1.0 - ty) + h1[i] * ty;
                    }
                    let dst = &mut sp[oy * ow..oy * ow + ow];
                    for i in 0..ow {
                        dst[i] = dst[i] * (1.0 - t) + base_row[i] * t;
                    }
                }
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
        let lr: Vec<f32> = (0..3 * h * w)
            .map(|i| ((i * 37) % 251) as f32 / 251.0)
            .collect();
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
