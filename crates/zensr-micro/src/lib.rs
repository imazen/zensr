//! Hand-rolled inference for SPANF (NTIRE25 team24, feature_channels=32, x4).
//!
//! Feasibility probe for a sub-300KB SR runtime: the entire op vocabulary is
//! grouped/plain conv3x3, conv1x1, SiLU, a sigmoid gate, channel concat, and
//! PixelShuffle. Scalar-but-vectorizable safe Rust; magetypes SIMD comes later
//! if the size/speed answer justifies it.
#![forbid(unsafe_code)]

pub mod simd;
pub use simd::spanf_x4_simd;

pub const FC: usize = 32; // feature channels
pub const S2: usize = 16; // upscale^2
pub const NEAR_CH: usize = 3 * S2; // 48
pub const CAT_CH: usize = NEAR_CH + 2 * FC; // 112
pub const TOTAL_FLOATS: usize = 148_288;

/// Weight views into one flat buffer, in tools/dump_spanf_weights.py order.
pub struct SpanfWeights<'a> {
    pub conv_near: &'a [f32],                    // [48,1,3,3]
    pub blocks: [[(&'a [f32], &'a [f32]); 3]; 5], // (w,b) per conv, per block
    pub conv_cat_w: &'a [f32],                   // [32,112]
    pub conv_cat_b: &'a [f32],                   // [32]
    pub conv2_w: &'a [f32],                      // [48,32,3,3]
    pub conv2_b: &'a [f32],                      // [48]
}

impl<'a> SpanfWeights<'a> {
    pub fn parse(buf: &'a [f32]) -> Result<Self, String> {
        if buf.len() != TOTAL_FLOATS {
            return Err(format!("expected {TOTAL_FLOATS} floats, got {}", buf.len()));
        }
        let mut off = 0usize;
        let mut take = |n: usize| {
            let s = &buf[off..off + n];
            off += n;
            s
        };
        let conv_near = take(NEAR_CH * 9);
        let mut blocks: [[(&[f32], &[f32]); 3]; 5] = [[(&[], &[]); 3]; 5];
        for (bi, block) in blocks.iter_mut().enumerate() {
            let cin = if bi == 0 { 3 } else { FC };
            for (ci, conv) in block.iter_mut().enumerate() {
                let ic = if ci == 0 { cin } else { FC };
                *conv = (take(FC * ic * 9), take(FC));
            }
        }
        let conv_cat_w = take(FC * CAT_CH);
        let conv_cat_b = take(FC);
        let conv2_w = take(NEAR_CH * FC * 9);
        let conv2_b = take(NEAR_CH);
        debug_assert_eq!(off, TOTAL_FLOATS);
        Ok(SpanfWeights { conv_near, blocks, conv_cat_w, conv_cat_b, conv2_w, conv2_b })
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Accumulate one 3x3-conv input-channel plane into `out` (zero padding),
/// weights w[0..9] row-major. Interior columns run branch-free.
fn conv3x3_acc_plane(inp: &[f32], out: &mut [f32], w: &[f32], h: usize, wd: usize) {
    for oy in 0..h {
        let orow = &mut out[oy * wd..(oy + 1) * wd];
        for ky in 0..3usize {
            let iy = oy + ky;
            if iy < 1 || iy > h {
                continue;
            }
            let irow = &inp[(iy - 1) * wd..iy * wd];
            let wk = &w[ky * 3..ky * 3 + 3];
            // kx=0 shifts left, kx=1 aligned, kx=2 shifts right.
            let (w0, w1, w2) = (wk[0], wk[1], wk[2]);
            // aligned
            for x in 0..wd {
                orow[x] += w1 * irow[x];
            }
            // left neighbor (ix = x-1)
            for x in 1..wd {
                orow[x] += w0 * irow[x - 1];
            }
            // right neighbor (ix = x+1)
            for x in 0..wd - 1 {
                orow[x] += w2 * irow[x + 1];
            }
        }
    }
}

/// Plain conv3x3, NCHW planes, bias, cin->cout.
fn conv3x3(inp: &[f32], out: &mut [f32], w: &[f32], b: &[f32], cin: usize, cout: usize, h: usize, wd: usize) {
    let plane = h * wd;
    for oc in 0..cout {
        let orow = &mut out[oc * plane..(oc + 1) * plane];
        orow.fill(b[oc]);
        for ic in 0..cin {
            conv3x3_acc_plane(
                &inp[ic * plane..(ic + 1) * plane],
                orow,
                &w[(oc * cin + ic) * 9..(oc * cin + ic) * 9 + 9],
                h,
                wd,
            );
        }
    }
}

/// conv_near: groups=3, 1 in-channel per group, 16 out per group, no bias.
fn conv3x3_grouped_near(inp: &[f32], out: &mut [f32], w: &[f32], h: usize, wd: usize) {
    let plane = h * wd;
    for oc in 0..NEAR_CH {
        let ic = oc / S2;
        let orow = &mut out[oc * plane..(oc + 1) * plane];
        orow.fill(0.0);
        conv3x3_acc_plane(&inp[ic * plane..(ic + 1) * plane], orow, &w[oc * 9..oc * 9 + 9], h, wd);
    }
}

fn conv1x1(inp: &[f32], out: &mut [f32], w: &[f32], b: &[f32], cin: usize, cout: usize, plane: usize) {
    for oc in 0..cout {
        let orow = &mut out[oc * plane..(oc + 1) * plane];
        orow.fill(b[oc]);
        for ic in 0..cin {
            let wv = w[oc * cin + ic];
            let irow = &inp[ic * plane..(ic + 1) * plane];
            for x in 0..plane {
                orow[x] += wv * irow[x];
            }
        }
    }
}

fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v *= sigmoid(*v);
    }
}

/// out = (a + skip) * (sigmoid(a) - 0.5), elementwise.
fn gate(a: &[f32], skip: &[f32], out: &mut [f32]) {
    for i in 0..out.len() {
        out[i] = (a[i] + skip[i]) * (sigmoid(a[i]) - 0.5);
    }
}

fn pixel_shuffle4(inp: &[f32], out: &mut [f32], h: usize, wd: usize) {
    // [48,h,w] -> [3,4h,4w]; input channel c*16 + ry*4 + rx feeds (y*4+ry, x*4+rx) of out c.
    let plane = h * wd;
    let (oh, ow) = (h * 4, wd * 4);
    for c in 0..3 {
        for ry in 0..4 {
            for rx in 0..4 {
                let ip = &inp[(c * S2 + ry * 4 + rx) * plane..][..plane];
                for y in 0..h {
                    let obase = c * oh * ow + (y * 4 + ry) * ow + rx;
                    let irow = &ip[y * wd..(y + 1) * wd];
                    for x in 0..wd {
                        out[obase + x * 4] = irow[x];
                    }
                }
            }
        }
    }
}

/// One SPAB block. `cin`==3 for block 1 (no gate), FC otherwise (gated).
fn spab(inp: &[f32], out: &mut [f32], tmp: &mut [f32], convs: &[(&[f32], &[f32]); 3], cin: usize, h: usize, wd: usize) {
    let plane = h * wd;
    conv3x3(inp, out, convs[0].0, convs[0].1, cin, FC, h, wd);
    silu_inplace(&mut out[..FC * plane]);
    conv3x3(&out[..FC * plane], tmp, convs[1].0, convs[1].1, FC, FC, h, wd);
    silu_inplace(&mut tmp[..FC * plane]);
    conv3x3(&tmp[..FC * plane], out, convs[2].0, convs[2].1, FC, FC, h, wd);
    if cin == FC {
        // gated: out = (out3 + inp) * (sigmoid(out3) - 0.5); reuse tmp as scratch
        tmp[..FC * plane].copy_from_slice(&out[..FC * plane]);
        gate(&tmp[..FC * plane], &inp[..FC * plane], &mut out[..FC * plane]);
    }
}

pub(crate) fn pixel_shuffle4_pub(inp: &[f32], out: &mut [f32], h: usize, wd: usize) {
    pixel_shuffle4(inp, out, h, wd)
}

/// Stage-level non-finite reporting, active only with the `nan-debug` feature.
#[cfg(feature = "nan-debug")]
pub(crate) fn nan_debug(stage: &str, data: &[f32]) {
    let bad = data.iter().filter(|v| !v.is_finite()).count();
    let first = data.iter().position(|v| !v.is_finite());
    let mx = data.iter().cloned().fold(0.0f32, |a, v| if v.is_finite() { a.max(v.abs()) } else { a });
    eprintln!(
        "nan-debug {stage}: nonfinite={bad}/{} first={first:?} max_abs={mx:.3e}",
        data.len()
    );
}
#[cfg(not(feature = "nan-debug"))]
#[inline(always)]
pub(crate) fn nan_debug(_stage: &str, _data: &[f32]) {}

/// Full SPANF x4 forward. `input` is [3,h,w] NCHW f32; returns [3,4h,4w].
pub fn spanf_x4(input: &[f32], h: usize, wd: usize, w: &SpanfWeights) -> Vec<f32> {
    let plane = h * wd;
    assert_eq!(input.len(), 3 * plane);
    let mut near = vec![0.0f32; NEAR_CH * plane];
    conv3x3_grouped_near(input, &mut near, w.conv_near, h, wd);

    let mut b_out = vec![0.0f32; FC * plane];
    let mut tmp = vec![0.0f32; FC * plane];
    let mut b1 = vec![0.0f32; FC * plane];
    spab(input, &mut b1, &mut tmp, &w.blocks[0], 3, h, wd);
    b_out.copy_from_slice(&b1);
    let mut cur = vec![0.0f32; FC * plane];
    for bi in 1..5 {
        spab(&b_out, &mut cur, &mut tmp, &w.blocks[bi], FC, h, wd);
        core::mem::swap(&mut b_out, &mut cur);
    }

    // concat [near(48), b5(32), b1(32)] -> conv1x1 -> conv2 -> pixelshuffle
    let mut cat = vec![0.0f32; CAT_CH * plane];
    cat[..NEAR_CH * plane].copy_from_slice(&near);
    cat[NEAR_CH * plane..(NEAR_CH + FC) * plane].copy_from_slice(&b_out);
    cat[(NEAR_CH + FC) * plane..].copy_from_slice(&b1);

    let mut catd = vec![0.0f32; FC * plane];
    conv1x1(&cat, &mut catd, w.conv_cat_w, w.conv_cat_b, CAT_CH, FC, plane);
    let mut pre = vec![0.0f32; NEAR_CH * plane];
    conv3x3(&catd, &mut pre, w.conv2_w, w.conv2_b, FC, NEAR_CH, h, wd);

    let mut out = vec![0.0f32; 3 * plane * 16];
    pixel_shuffle4(&pre, &mut out, h, wd);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG values in [-0.5, 0.5).
    fn lcg(n: usize, seed: u32, scale: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * scale
            })
            .collect()
    }

    fn assert_close_finite(a: &[f32], b: &[f32], tol: f32, what: &str) {
        assert_eq!(a.len(), b.len());
        let mut max = 0.0f32;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                x.is_finite() && y.is_finite(),
                "{what}: non-finite at {i}: {x} vs {y}"
            );
            max = max.max((x - y).abs());
        }
        assert!(max < tol, "{what}: max diff {max} > {tol}");
    }

    #[test]
    fn silu_dispatch_matches_scalar() {
        let mut a = lcg(1000, 7, 20.0); // range ±10, covers both sigmoid tails
        let mut b = a.clone();
        for v in b.iter_mut() {
            *v *= 1.0 / (1.0 + (-*v).exp());
        }
        simd::silu_dispatch(&mut a);
        assert_close_finite(&a, &b, 2e-4, "silu");
    }

    #[test]
    fn conv3x3_dispatch_matches_scalar() {
        let (h, wd, cin, cout) = (9usize, 37usize, 5usize, 8usize);
        let inp = lcg(cin * h * wd, 3, 2.0);
        let wts = lcg(cout * cin * 9, 5, 0.5);
        let bias = lcg(cout, 11, 0.5);
        // scalar reference via the lib's own conv3x3
        let mut want = vec![0.0f32; cout * h * wd];
        conv3x3(&inp, &mut want, &wts, &bias, cin, cout, h, wd);
        let mut got = vec![0.0f32; cout * h * wd];
        simd::conv3x3_dispatch(&inp, cin, &wts, &bias, &mut got, cout, h, wd);
        assert_close_finite(&got, &want, 1e-4, "conv3x3");
    }

    #[test]
    fn simd_matches_scalar_reference() {
        // Small weights keep activations bounded through 5 blocks.
        let wbuf = lcg(TOTAL_FLOATS, 12345, 0.12);
        let w = SpanfWeights::parse(&wbuf).unwrap();
        let (h, wd) = (10usize, 24usize); // exercises vector interior + scalar edges
        let input = lcg(3 * h * wd, 999, 1.0);

        let a = spanf_x4(&input, h, wd, &w);
        let b = spanf_x4_simd(&input, h, wd, &w);
        // exp_midp vs libm exp: tiny per-element error, bounded through the net.
        assert_close_finite(&b, &a, 5e-4, "full forward simd vs scalar");
    }
}
