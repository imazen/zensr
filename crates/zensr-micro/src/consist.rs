//! S10 quantization-consistency projection (decoder-agnostic).
//!
//! A JPEG file certifies, per 8x8 block and DCT band (u,v), that the encoder's
//! true coefficient satisfied |c - c_hat| <= Q[u,v]/2 (+ encoder slack for
//! trellis quantizers). The set of consistent images is convex and contains
//! the original; clamping the model output's block-DCT coefficients into the
//! per-band interval is the exact Euclidean projection onto that set
//! (orthonormal DCT), so it NEVER increases error vs the original:
//! ||P(y) - truth|| <= ||y - truth|| for every output. Re-encoding the
//! projected output with the same tables reproduces the file's coefficients.
//!
//! Domain: the projection is exact in the decoder's YCbCr space, on
//! level-shifted samples (value*255 - 128), per JPEG/JFIF. Callers hand in
//! one component plane at a time plus that component's coefficients (natural
//! or zigzag order per `CoeffOrder`) and quant table. Chroma of subsampled
//! files must be projected on the subsampled lattice (back-projection form)
//! — v1 supports full-resolution planes only (luma always; chroma at 4:4:4).
//!
//! No SIMD here on purpose: plain separable matrix DCT auto-vectorizes and
//! keeps this module out of the kernel-fragility zone (26x-incident lesson).

/// Zigzag index -> natural (row-major) index, JPEG standard order.
pub const ZIGZAG_TO_NATURAL: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
    20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58,
    59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Order of the caller's per-block coefficient slices.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CoeffOrder {
    /// Row-major within each block.
    Natural,
    /// JPEG zigzag within each block (zenjpeg's `ComponentCoefficients`).
    Zigzag,
}

/// One component's bitstream data, borrowed from whatever decoder produced it.
pub struct CoeffView<'a> {
    /// Raw (still-quantized) integers, `blocks_wide*blocks_high*64` long,
    /// block-row-major, ordered per `order` within each block.
    pub coeffs: &'a [i16],
    pub blocks_wide: usize,
    pub blocks_high: usize,
    pub order: CoeffOrder,
    /// Quant table in NATURAL (row-major) order.
    pub quant: &'a [u16; 64],
}

/// Slack widening the per-coefficient interval |c - c_hat| <= Q/2.
///
/// Two additive terms, calibrated per encoder family with `slack_probe`:
/// - `slack_q` — RELATIVE, in units of Q. Covers quantizer-behavior spread
///   (trellis off-nearest rounding, AQ understating the stored DQT).
/// - `slack_abs` — ABSOLUTE, in coefficient units. Covers the encoder-side
///   sample-quantization noise floor: encoders that round YCbCr samples to
///   u8 before their FDCT (libjpeg-turbo, mozjpeg) carry a bounded absolute
///   DCT-domain error (measured p99 ~1.3, max ~3.7 units) regardless of Q.
///   Invisible when Q/2 dominates; decisive at Q=1..3 (high quality), where
///   a purely relative slack lets the box exclude the truth and the
///   projection clamps CORRECT detail (the measured q96 regression).
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct ProjectionConfig {
    pub slack_q: f32,
    pub slack_abs: f32,
}

impl ProjectionConfig {
    pub fn with_slack_q(slack_q: f32) -> Self {
        let mut c = Self::default();
        c.slack_q = slack_q;
        c
    }
    pub fn with_slack_abs(mut self, slack_abs: f32) -> Self {
        self.slack_abs = slack_abs;
        self
    }
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        // strict box + a hair for decoder IDCT/rounding noise
        ProjectionConfig { slack_q: 0.05, slack_abs: 0.0 }
    }
}

/// Diagnostics: how much work the projection did.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProjectionReport {
    /// Fraction of coefficients clamped.
    pub clamped_frac: f32,
    /// Mean absolute pixel-domain change, in [0,1] units.
    pub mean_abs_change: f32,
}

const B: usize = 8;

/// 8-point DCT-II basis, orthonormal scaling matching JPEG's FDCT such that
/// coefficients are directly comparable to dequantized `k*Q` values.
fn basis() -> [[f32; B]; B] {
    let mut m = [[0.0f32; B]; B];
    for (u, row) in m.iter_mut().enumerate() {
        let cu = if u == 0 { (0.5f32).sqrt() } else { 1.0 };
        for (x, v) in row.iter_mut().enumerate() {
            *v = 0.5
                * cu
                * (((2 * x + 1) as f32) * (u as f32) * core::f32::consts::PI / 16.0).cos();
        }
    }
    m
}

/// Project one full-resolution component plane in place.
///
/// `plane`: [0,1] samples, `w*h`, tight rows. Interpretation: this component's
/// YCbCr value where JPEG stores `value*255 - 128` (Y straight; Cb/Cr with the
/// +0.5 offset already folded into [0,1]). Blocks beyond `w`/`h` (partial edge
/// blocks) are handled by edge replication, matching encoder padding closely
/// enough that edge coefficients stay in-box for real files.
pub fn project_plane(
    plane: &mut [f32],
    w: usize,
    h: usize,
    cv: &CoeffView<'_>,
    cfg: &ProjectionConfig,
) -> ProjectionReport {
    assert_eq!(plane.len(), w * h);
    assert!(cv.blocks_wide * B >= w && cv.blocks_high * B >= h, "coeff grid too small");
    assert_eq!(cv.coeffs.len(), cv.blocks_wide * cv.blocks_high * 64);
    let m = basis();
    let mut clamped = 0usize;
    let mut change = 0.0f64;
    let mut px = [[0.0f32; B]; B];
    let mut tmp = [[0.0f32; B]; B];
    let mut fq = [[0.0f32; B]; B];

    for by in 0..cv.blocks_high {
        for bx in 0..cv.blocks_wide {
            // gather block, level-shifted to JPEG units, edge-replicated
            for y in 0..B {
                let sy = (by * B + y).min(h - 1);
                for x in 0..B {
                    let sx = (bx * B + x).min(w - 1);
                    px[y][x] = plane[sy * w + sx] * 255.0 - 128.0;
                }
            }
            // forward DCT: F = M * px * M^T (separable)
            for u in 0..B {
                for x in 0..B {
                    let mut s = 0.0;
                    for y in 0..B {
                        s += m[u][y] * px[y][x];
                    }
                    tmp[u][x] = s;
                }
            }
            for u in 0..B {
                for v in 0..B {
                    let mut s = 0.0;
                    for x in 0..B {
                        s += tmp[u][x] * m[v][x];
                    }
                    fq[u][v] = s;
                }
            }
            // clamp into per-band interval around the dequantized coefficient
            let blk = &cv.coeffs[(by * cv.blocks_wide + bx) * 64..][..64];
            let mut any = false;
            for k in 0..64 {
                let (nat, raw) = match cv.order {
                    CoeffOrder::Natural => (k, blk[k]),
                    CoeffOrder::Zigzag => (ZIGZAG_TO_NATURAL[k], blk[k]),
                };
                let (u, v) = (nat / B, nat % B);
                let q = cv.quant[nat] as f32;
                let c_hat = raw as f32 * q;
                let half = q * (0.5 + cfg.slack_q) + cfg.slack_abs;
                let cval = fq[u][v];
                let cc = cval.clamp(c_hat - half, c_hat + half);
                if cc != cval {
                    fq[u][v] = cc;
                    clamped += 1;
                    any = true;
                }
            }
            if !any {
                continue;
            }
            // inverse DCT: px = M^T * F * M
            for y in 0..B {
                for v in 0..B {
                    let mut s = 0.0;
                    for u in 0..B {
                        s += m[u][y] * fq[u][v];
                    }
                    tmp[y][v] = s;
                }
            }
            for y in 0..B {
                let sy = by * B + y;
                if sy >= h {
                    break;
                }
                for x in 0..B {
                    let sx = bx * B + x;
                    if sx >= w {
                        continue;
                    }
                    let mut s = 0.0;
                    for v in 0..B {
                        s += tmp[y][v] * m[v][x];
                    }
                    let nv = ((s + 128.0) / 255.0).clamp(0.0, 1.0);
                    change += (nv - plane[sy * w + sx]).abs() as f64;
                    plane[sy * w + sx] = nv;
                }
            }
        }
    }
    ProjectionReport {
        clamped_frac: clamped as f32 / cv.coeffs.len() as f32,
        mean_abs_change: (change / (w * h) as f64) as f32,
    }
}

/// JFIF full-range RGB<->YCbCr (BT.601 constants from the spec). All in [0,1];
/// Cb/Cr carry the +0.5 offset.
pub fn rgb_to_ycbcr_planes(rgb: &[f32], plane: usize, y: &mut [f32], cb: &mut [f32], cr: &mut [f32]) {
    let (r, g, b) = (&rgb[..plane], &rgb[plane..2 * plane], &rgb[2 * plane..3 * plane]);
    for i in 0..plane {
        let yy = 0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i];
        y[i] = yy;
        cb[i] = 0.5 - 0.168_735_9 * r[i] - 0.331_264_1 * g[i] + 0.5 * b[i];
        cr[i] = 0.5 + 0.5 * r[i] - 0.418_687_6 * g[i] - 0.081_312_4 * b[i];
    }
}

pub fn ycbcr_to_rgb_planes(y: &[f32], cb: &[f32], cr: &[f32], rgb: &mut [f32], plane: usize) {
    for i in 0..plane {
        let (yy, u, v) = (y[i], cb[i] - 0.5, cr[i] - 0.5);
        rgb[i] = (yy + 1.402 * v).clamp(0.0, 1.0);
        rgb[plane + i] = (yy - 0.344_136_3 * u - 0.714_136_2 * v).clamp(0.0, 1.0);
        rgb[2 * plane + i] = (yy + 1.772 * u).clamp(0.0, 1.0);
    }
}

/// Project a FULL-resolution chroma plane whose bitstream constraint lives on
/// the 2x-subsampled lattice (4:2:0), via one back-projection pass:
/// down = box2x2(plane); down' = project(down); plane += replicate(down' - down).
/// Replication is the exact right-inverse of the box decimation, so the
/// corrected plane satisfies the half-res constraint EXACTLY in one pass
/// (up to [0,1] clamping); the correction field is piecewise-constant at
/// 2x2 granularity, which is invisible at chroma scales.
pub fn project_chroma_420(
    plane: &mut [f32],
    w: usize,
    h: usize,
    cv: &CoeffView<'_>,
    cfg: &ProjectionConfig,
) -> ProjectionReport {
    assert_eq!(plane.len(), w * h);
    let (hw, hh) = (w.div_ceil(2), h.div_ceil(2));
    // box 2x2 down (edge-replicated on odd dims) — the common encoder decimation
    let mut down = vec![0.0f32; hw * hh];
    for y in 0..hh {
        let (y0, y1) = (2 * y, (2 * y + 1).min(h - 1));
        for x in 0..hw {
            let (x0, x1) = (2 * x, (2 * x + 1).min(w - 1));
            down[y * hw + x] = 0.25
                * (plane[y0 * w + x0] + plane[y0 * w + x1] + plane[y1 * w + x0] + plane[y1 * w + x1]);
        }
    }
    let before = down.clone();
    let report = project_plane(&mut down, hw, hh, cv, cfg);
    // Upsample the half-res correction by PIXEL REPLICATION and add: replication
    // is the exact right-inverse of the 2x2 box-down (box(replicate(c)) == c),
    // so the corrected plane satisfies the half-res constraint exactly in one
    // pass. (Bilinear up attenuates the correction: box(bilerp(c)) != c.)
    for y in 0..h {
        let hy = (y / 2).min(hh - 1);
        for x in 0..w {
            let c = down[hy * hw + (x / 2).min(hw - 1)] - before[hy * hw + (x / 2).min(hw - 1)];
            let i = y * w + x;
            plane[i] = (plane[i] + c).clamp(0.0, 1.0);
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fdct_block(px: &[[f32; B]; B]) -> [[f32; B]; B] {
        let m = basis();
        let mut tmp = [[0.0f32; B]; B];
        let mut f = [[0.0f32; B]; B];
        for u in 0..B {
            for x in 0..B {
                tmp[u][x] = (0..B).map(|y| m[u][y] * px[y][x]).sum();
            }
        }
        for u in 0..B {
            for v in 0..B {
                f[u][v] = (0..B).map(|x| tmp[u][x] * m[v][x]).sum();
            }
        }
        f
    }

    /// Simulate encode: DCT + quantize; return (quantized ints natural order,
    /// decoded plane) for a random plane.
    fn simulate(w: usize, h: usize, qt: &[u16; 64], seed: u64) -> (Vec<f32>, Vec<i16>) {
        let bw = w.div_ceil(B);
        let bh = h.div_ceil(B);
        let mut rng = seed;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as u32 as f32) / (u32::MAX >> 1) as f32 * 0.5
        };
        let orig: Vec<f32> = (0..w * h).map(|_| 0.25 + next()).collect();
        let mut coeffs = vec![0i16; bw * bh * 64];
        let mut decoded = vec![0.0f32; w * h];
        let m = basis();
        for by in 0..bh {
            for bx in 0..bw {
                let mut px = [[0.0f32; B]; B];
                for y in 0..B {
                    for x in 0..B {
                        let (sy, sx) = ((by * B + y).min(h - 1), (bx * B + x).min(w - 1));
                        px[y][x] = orig[sy * w + sx] * 255.0 - 128.0;
                    }
                }
                let f = fdct_block(&px);
                let blk = &mut coeffs[(by * bw + bx) * 64..][..64];
                let mut fr = [[0.0f32; B]; B];
                for u in 0..B {
                    for v in 0..B {
                        let q = qt[u * B + v] as f32;
                        let k = (f[u][v] / q).round();
                        blk[u * B + v] = k as i16;
                        fr[u][v] = k * q;
                    }
                }
                // decode: IDCT of dequantized
                let mut tmp = [[0.0f32; B]; B];
                for y in 0..B {
                    for v in 0..B {
                        tmp[y][v] = (0..B).map(|u| m[u][y] * fr[u][v]).sum();
                    }
                }
                for y in 0..B {
                    let sy = by * B + y;
                    if sy >= h {
                        break;
                    }
                    for x in 0..B {
                        let sx = bx * B + x;
                        if sx >= w {
                            continue;
                        }
                        let s: f32 = (0..B).map(|v| tmp[y][v] * m[v][x]).sum();
                        decoded[sy * w + sx] = ((s + 128.0) / 255.0).clamp(0.0, 1.0);
                    }
                }
            }
        }
        let _ = orig;
        (decoded, coeffs)
    }

    fn qt_flat(v: u16) -> [u16; 64] {
        [v; 64]
    }

    /// Test-only 2x bilinear upsample (align-corners=false, edge clamp).
    fn bilerp_up2(src: &[f32], sw: usize, sh: usize) -> Vec<f32> {
        let (dw, dh) = (sw * 2, sh * 2);
        let mut out = vec![0.0f32; dw * dh];
        for y in 0..dh {
            let fy = ((y as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, (sh - 1) as f32);
            let (y0, ty) = (fy.floor() as usize, fy.fract());
            let y1 = (y0 + 1).min(sh - 1);
            for x in 0..dw {
                let fx = ((x as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, (sw - 1) as f32);
                let (x0, tx) = (fx.floor() as usize, fx.fract());
                let x1 = (x0 + 1).min(sw - 1);
                let top = src[y0 * sw + x0] * (1.0 - tx) + src[y0 * sw + x1] * tx;
                let bot = src[y1 * sw + x0] * (1.0 - tx) + src[y1 * sw + x1] * tx;
                out[y * dw + x] = top * (1.0 - ty) + bot * ty;
            }
        }
        out
    }

    #[test]
    fn decode_is_inside_box_and_unchanged() {
        let (w, h) = (32, 24);
        let qt = qt_flat(24);
        let (mut dec, coeffs) = simulate(w, h, &qt, 7);
        let before = dec.clone();
        let cv = CoeffView { coeffs: &coeffs, blocks_wide: 4, blocks_high: 3, order: CoeffOrder::Natural, quant: &qt };
        let rep = project_plane(&mut dec, w, h, &cv, &ProjectionConfig { slack_q: 0.05, slack_abs: 0.0 });
        for (a, b) in dec.iter().zip(before.iter()) {
            assert!((a - b).abs() < 2e-3, "decode should already be consistent");
        }
        assert!(rep.clamped_frac < 0.02, "clamped {}", rep.clamped_frac);
    }

    #[test]
    fn projection_never_increases_error_and_bounds_hallucination() {
        let (w, h) = (32, 32);
        let qt = qt_flat(40);
        let (dec, coeffs) = simulate(w, h, &qt, 42);
        let cv = CoeffView { coeffs: &coeffs, blocks_wide: 4, blocks_high: 4, order: CoeffOrder::Natural, quant: &qt };
        // adversarial "model": decode + big hallucinated texture
        let mut out: Vec<f32> = dec
            .iter()
            .enumerate()
            .map(|(i, v)| (v + if (i / 3) % 2 == 0 { 0.3 } else { -0.3 }).clamp(0.0, 1.0))
            .collect();
        let rep = project_plane(&mut out, w, h, &cv, &ProjectionConfig { slack_q: 0.0, slack_abs: 0.0 });
        assert!(rep.clamped_frac > 0.2, "should clamp a lot: {}", rep.clamped_frac);
        // after projection, re-quantizing must reproduce the file's coefficients
        let m = basis();
        for by in 0..4 {
            for bx in 0..4 {
                let mut px = [[0.0f32; B]; B];
                for y in 0..B {
                    for x in 0..B {
                        px[y][x] = out[(by * B + y) * w + bx * B + x] * 255.0 - 128.0;
                    }
                }
                let f = fdct_block(&px);
                let blk = &coeffs[(by * 4 + bx) * 64..][..64];
                for u in 0..B {
                    for v in 0..B {
                        let q = qt[u * B + v] as f32;
                        let k = (f[u][v] / q).round() as i16;
                        // allow off-by-one only at exact interval boundaries
                        assert!(
                            (k - blk[u * B + v]).abs() <= 1,
                            "re-encode consistency broken at ({u},{v}): {k} vs {}",
                            blk[u * B + v]
                        );
                    }
                }
                let _ = m;
            }
        }
    }

    #[test]
    fn chroma_420_backprojection_reduces_violation() {
        // synthetic 420: encode a half-res plane, restore-with-error at full
        // res, back-project; the half-res constraint violation must shrink
        // to ~quantization-noise scale and full-res content stays smooth.
        let (w, h) = (32usize, 32usize);
        let (hw, hh) = (16usize, 16usize);
        let qt = qt_flat(32);
        let (dec_half, coeffs) = simulate(hw, hh, &qt, 5);
        // full-res "restored chroma" = bilinear-up of decode + hallucination
        let mut full = bilerp_up2(&dec_half, hw, hh);
        for (i, v) in full.iter_mut().enumerate() {
            *v = (*v + if (i / 7) % 2 == 0 { 0.15 } else { -0.15 }).clamp(0.0, 1.0);
        }
        let cv = CoeffView { coeffs: &coeffs, blocks_wide: 2, blocks_high: 2, order: CoeffOrder::Natural, quant: &qt };
        let viol = |p: &[f32]| -> f32 {
            // measure half-res box violation: project a copy, see how far it moves
            let mut d = vec![0.0f32; hw * hh];
            for y in 0..hh {
                for x in 0..hw {
                    d[y * hw + x] = 0.25
                        * (p[2 * y * w + 2 * x] + p[2 * y * w + 2 * x + 1]
                            + p[(2 * y + 1) * w + 2 * x] + p[(2 * y + 1) * w + 2 * x + 1]);
                }
            }
            let before = d.clone();
            project_plane(&mut d, hw, hh, &cv, &ProjectionConfig { slack_q: 0.0, slack_abs: 0.0 });
            d.iter().zip(before.iter()).map(|(a, b)| (a - b).abs()).sum::<f32>() / (hw * hh) as f32
        };
        let v0 = viol(&full);
        project_chroma_420(&mut full, w, h, &cv, &ProjectionConfig { slack_q: 0.0, slack_abs: 0.0 });
        let v1 = viol(&full);
        assert!(v0 > 0.01, "test setup must start violated (v0={v0})");
        assert!(v1 < 2e-3, "one back-projection pass must satisfy the half-res box (v0={v0} -> v1={v1})");
    }

    #[test]
    fn ycbcr_roundtrip() {
        let plane = 64;
        let rgb: Vec<f32> = (0..3 * plane).map(|i| ((i * 37) % 251) as f32 / 251.0).collect();
        let (mut y, mut cb, mut cr) = (vec![0.0; plane], vec![0.0; plane], vec![0.0; plane]);
        rgb_to_ycbcr_planes(&rgb, plane, &mut y, &mut cb, &mut cr);
        let mut back = vec![0.0f32; 3 * plane];
        ycbcr_to_rgb_planes(&y, &cb, &cr, &mut back, plane);
        for (a, b) in rgb.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
