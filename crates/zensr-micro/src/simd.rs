//! SIMD SPANF forward: one `#[magetypes]` family (f32x8: v4/v3/neon/wasm128/scalar)
//! plus a hand `#[arcane]` `_v4x` variant on native 512-bit f32x16.
//!
//! Kernels are macro-instantiated per width so both variants share one body.
//! Interior columns run vector FMA (loads at x-1/x/x+1 are guarded by
//! `x + W + 1 <= wd`); the x edges and non-multiple tails run scalar.

use crate::{SpanfWeights, CAT_CH, FC, NEAR_CH, S2};
use archmage::prelude::*;

/// Repack a [cout][cin][3][3] conv weight into quad-major iteration order:
/// per output-channel-quad q, per (ic, ky): 12 floats [ob0 t0,t1,t2, ob1 t0,..].
/// The hot loop then reads one contiguous 12-float chunk per (ic, ky).
fn pack_conv3x3(wts: &[f32], cin: usize, cout: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(cout * cin * 9);
    for q in 0..cout / 4 {
        for ic in 0..cin {
            for ky in 0..3 {
                for ob in 0..4 {
                    let base = ((q * 4 + ob) * cin + ic) * 9 + ky * 3;
                    out.extend_from_slice(&wts[base..base + 3]);
                }
            }
        }
    }
    out
}

/// Repack a [cout][cin] 1x1 conv weight into quad-major [wi][4] order.
fn pack_conv1x1(wts: &[f32], cin_total: usize, cout: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(cout * cin_total);
    for q in 0..cout / 4 {
        for wi in 0..cin_total {
            for ob in 0..4 {
                out.push(wts[(q * 4 + ob) * cin_total + wi]);
            }
        }
    }
    out
}

/// Per-inference packed weights (built once per call; ~150K floats, trivial).
struct Packed {
    blocks: [[Vec<f32>; 3]; 5],
    conv_cat: Vec<f32>,
    conv2: Vec<f32>,
}

impl Packed {
    fn build(w: &SpanfWeights) -> Self {
        let blocks = core::array::from_fn(|bi| {
            let cin0 = if bi == 0 { 3 } else { FC };
            core::array::from_fn(|ci| {
                let cin = if ci == 0 { cin0 } else { FC };
                pack_conv3x3(w.blocks[bi][ci].0, cin, FC)
            })
        });
        Packed {
            blocks,
            conv_cat: pack_conv1x1(w.conv_cat_w, CAT_CH, FC),
            conv2: pack_conv3x3(w.conv2_w, FC, NEAR_CH),
        }
    }
}

macro_rules! define_kernels {
    ($modname:ident, $vec:ident, $bnd:ident, $w:expr) => {
        pub(crate) mod $modname {
            use super::*;
            use magetypes::simd::backends::$bnd as Backend;
            use magetypes::simd::generic::$vec as V;

            /// 3x3 conv for one output row `oy`, four output channels
            /// `oc0..oc0+4`. Quad accumulators live in REGISTERS across the
            /// whole (ic, ky) reduction per x-tile; one store per tile.
            #[inline(always)]
            #[allow(clippy::too_many_arguments)]
            pub(crate) fn conv3x3_row4<T: Backend>(
                token: T,
                inp: &[f32],
                cin: usize,
                wts: &[f32], // PACKED quad-major (pack_conv3x3)
                bias: &[f32],
                out: &mut [f32],
                h: usize,
                wd: usize,
                oy: usize,
                oc0: usize,
            ) {
                const W: usize = $w;
                let plane = h * wd;
                // Valid vertical taps for this output row (zero padding).
                let ky_lo = if oy == 0 { 1usize } else { 0 };
                let ky_hi = if oy + 1 == h { 1usize } else { 2 };

                // Row-slice table for this output row: rowtab[ic*3+ky].
                // Built once per (oy); empty slice marks an invalid ky.
                let mut rowtab: [&[f32]; 96] = [&[]; 96];
                for ic in 0..cin {
                    for ky in ky_lo..=ky_hi {
                        rowtab[ic * 3 + ky] = &inp[ic * plane + (oy + ky - 1) * wd..][..wd];
                    }
                }
                let q = oc0 / 4;
                let qbase = q * cin * 36; // cin * 3ky * 12

                let mut x = 1usize;
                while x + W + 1 <= wd {
                    let mut acc = [
                        V::<T>::splat(token, bias[oc0]),
                        V::<T>::splat(token, bias[oc0 + 1]),
                        V::<T>::splat(token, bias[oc0 + 2]),
                        V::<T>::splat(token, bias[oc0 + 3]),
                    ];
                    for ic in 0..cin {
                        for ky in ky_lo..=ky_hi {
                            let irow = rowtab[ic * 3 + ky];
                            let o = qbase + (ic * 3 + ky) * 12;
                            let w12: &[f32; 12] = (&wts[o..o + 12]).try_into().unwrap();
                            let l = V::<T>::from_slice(token, &irow[x - 1..]);
                            let m = V::<T>::from_slice(token, &irow[x..]);
                            let r = V::<T>::from_slice(token, &irow[x + 1..]);
                            for ob in 0..4 {
                                acc[ob] = l.mul_add(V::<T>::splat(token, w12[ob * 3]), acc[ob]);
                                acc[ob] = m.mul_add(V::<T>::splat(token, w12[ob * 3 + 1]), acc[ob]);
                                acc[ob] = r.mul_add(V::<T>::splat(token, w12[ob * 3 + 2]), acc[ob]);
                            }
                        }
                    }
                    for ob in 0..4 {
                        let dst: &mut [f32; W] = (&mut out
                            [(oc0 + ob) * plane + oy * wd + x..(oc0 + ob) * plane + oy * wd + x + W])
                            .try_into()
                            .unwrap();
                        acc[ob].store(dst);
                    }
                    x += W;
                }
                // Scalar edges: x = 0 and the right tail.
                for xx in core::iter::once(0).chain(x..wd) {
                    for ob in 0..4 {
                        let mut s = bias[oc0 + ob];
                        for ic in 0..cin {
                            for ky in ky_lo..=ky_hi {
                                let irow = rowtab[ic * 3 + ky];
                                let o = qbase + (ic * 3 + ky) * 12 + ob * 3;
                                if xx >= 1 {
                                    s += wts[o] * irow[xx - 1];
                                }
                                s += wts[o + 1] * irow[xx];
                                if xx + 1 < wd {
                                    s += wts[o + 2] * irow[xx + 1];
                                }
                            }
                        }
                        out[(oc0 + ob) * plane + oy * wd + xx] = s;
                    }
                }
            }

            /// Grouped conv_near: 48 out channels, ic = oc/16, no bias.
            /// Register accumulator per x-tile over the <=3 vertical taps.
            #[inline(always)]
            pub(crate) fn conv_near<T: Backend>(
                token: T,
                inp: &[f32],
                wts: &[f32], // [48][9]
                out: &mut [f32],
                h: usize,
                wd: usize,
            ) {
                const W: usize = $w;
                let plane = h * wd;
                for oc in 0..NEAR_CH {
                    let ic = oc / S2;
                    let ip = &inp[ic * plane..(ic + 1) * plane];
                    let w9 = &wts[oc * 9..oc * 9 + 9];
                    for oy in 0..h {
                        let ky_lo = if oy == 0 { 1usize } else { 0 };
                        let ky_hi = if oy + 1 == h { 1usize } else { 2 };
                        let mut x = 1usize;
                        while x + W + 1 <= wd {
                            let mut acc = V::<T>::zero(token);
                            for ky in ky_lo..=ky_hi {
                                let irow = &ip[(oy + ky - 1) * wd..][..wd];
                                let l = V::<T>::from_slice(token, &irow[x - 1..]);
                                let m = V::<T>::from_slice(token, &irow[x..]);
                                let r = V::<T>::from_slice(token, &irow[x + 1..]);
                                acc = l.mul_add(V::<T>::splat(token, w9[ky * 3]), acc);
                                acc = m.mul_add(V::<T>::splat(token, w9[ky * 3 + 1]), acc);
                                acc = r.mul_add(V::<T>::splat(token, w9[ky * 3 + 2]), acc);
                            }
                            let dst: &mut [f32; W] = (&mut out
                                [oc * plane + oy * wd + x..oc * plane + oy * wd + x + W])
                                .try_into()
                                .unwrap();
                            acc.store(dst);
                            x += W;
                        }
                        for xx in core::iter::once(0).chain(x..wd) {
                            let mut s = 0.0f32;
                            for ky in ky_lo..=ky_hi {
                                let irow = &ip[(oy + ky - 1) * wd..][..wd];
                                if xx >= 1 {
                                    s += w9[ky * 3] * irow[xx - 1];
                                }
                                s += w9[ky * 3 + 1] * irow[xx];
                                if xx + 1 < wd {
                                    s += w9[ky * 3 + 2] * irow[xx + 1];
                                }
                            }
                            out[oc * plane + oy * wd + xx] = s;
                        }
                    }
                }
            }

            /// conv1x1 over a virtual concat of sources (no materialized cat).
            /// Weights are PACKED quad-major [q][wi][4] (pack_conv1x1).
            #[inline(always)]
            #[allow(clippy::too_many_arguments)]
            pub(crate) fn conv1x1_multi<T: Backend>(
                token: T,
                srcs: &[(&[f32], usize)],
                wts: &[f32],
                bias: &[f32],
                out: &mut [f32],
                cout: usize,
                cin_total: usize,
                plane: usize,
            ) {
                const W: usize = $w;
                let n = plane / W * W;
                for oc0 in (0..cout).step_by(4) {
                    let qbase = (oc0 / 4) * cin_total * 4;
                    let mut x = 0usize;
                    while x < n {
                        let mut acc = [
                            V::<T>::splat(token, bias[oc0]),
                            V::<T>::splat(token, bias[oc0 + 1]),
                            V::<T>::splat(token, bias[oc0 + 2]),
                            V::<T>::splat(token, bias[oc0 + 3]),
                        ];
                        let mut wi = 0usize;
                        for (src, cin) in srcs.iter() {
                            for ic in 0..*cin {
                                let v = V::<T>::from_slice(token, &src[ic * plane + x..]);
                                let o = qbase + wi * 4;
                                let w4: &[f32; 4] = (&wts[o..o + 4]).try_into().unwrap();
                                for ob in 0..4 {
                                    acc[ob] = v.mul_add(V::<T>::splat(token, w4[ob]), acc[ob]);
                                }
                                wi += 1;
                            }
                        }
                        for ob in 0..4 {
                            let dst: &mut [f32; W] = (&mut out
                                [(oc0 + ob) * plane + x..(oc0 + ob) * plane + x + W])
                                .try_into()
                                .unwrap();
                            acc[ob].store(dst);
                        }
                        x += W;
                    }
                    for xx in n..plane {
                        for ob in 0..4 {
                            let mut s = bias[oc0 + ob];
                            let mut wi = 0usize;
                            for (src, cin) in srcs.iter() {
                                for ic in 0..*cin {
                                    s += wts[qbase + wi * 4 + ob] * src[ic * plane + xx];
                                    wi += 1;
                                }
                            }
                            out[(oc0 + ob) * plane + xx] = s;
                        }
                    }
                }
            }

            /// In-place SiLU: v * sigmoid(v).
            #[inline(always)]
            pub(crate) fn silu<T: Backend>(token: T, data: &mut [f32]) {
                const W: usize = $w;
                let one = V::<T>::splat(token, 1.0);
                let n = data.len() / W * W;
                let mut x = 0usize;
                while x < n {
                    let v = V::<T>::from_slice(token, &data[x..]);
                    let s = one / ((-v).exp_midp() + one);
                    let dst: &mut [f32; W] = (&mut data[x..x + W]).try_into().unwrap();
                    (v * s).store(dst);
                    x += W;
                }
                for v in &mut data[n..] {
                    *v *= 1.0 / (1.0 + (-*v).exp());
                }
            }

            /// out = (a + skip) * (sigmoid(a) - 0.5)
            #[inline(always)]
            pub(crate) fn gate<T: Backend>(
                token: T,
                a: &[f32],
                skip: &[f32],
                out: &mut [f32],
            ) {
                const W: usize = $w;
                let one = V::<T>::splat(token, 1.0);
                let half = V::<T>::splat(token, 0.5);
                let n = out.len() / W * W;
                let mut x = 0usize;
                while x < n {
                    let av = V::<T>::from_slice(token, &a[x..]);
                    let sk = V::<T>::from_slice(token, &skip[x..]);
                    let sig = one / ((-av).exp_midp() + one);
                    let dst: &mut [f32; W] = (&mut out[x..x + W]).try_into().unwrap();
                    ((av + sk) * (sig - half)).store(dst);
                    x += W;
                }
                for i in n..out.len() {
                    let s = 1.0 / (1.0 + (-a[i]).exp());
                    out[i] = (a[i] + skip[i]) * (s - 0.5);
                }
            }

            #[inline(always)]
            #[allow(clippy::too_many_arguments)]
            pub(crate) fn conv3x3<T: Backend>(
                token: T,
                inp: &[f32],
                cin: usize,
                wts: &[f32],
                bias: &[f32],
                out: &mut [f32],
                cout: usize,
                h: usize,
                wd: usize,
            ) {
                for oc0 in (0..cout).step_by(4) {
                    for oy in 0..h {
                        conv3x3_row4(token, inp, cin, wts, bias, out, h, wd, oy, oc0);
                    }
                }
            }

            #[inline(always)]
            #[allow(clippy::too_many_arguments)]
            pub(crate) fn spab<T: Backend>(
                token: T,
                inp: &[f32],
                convs: &[(&[f32], &[f32]); 3],
                cin: usize,
                out: &mut [f32],
                tmp: &mut [f32],
                h: usize,
                wd: usize,
            ) {
                let plane = h * wd;
                conv3x3(token, inp, cin, convs[0].0, convs[0].1, out, FC, h, wd);
                silu(token, &mut out[..FC * plane]);
                conv3x3(token, &out[..FC * plane], FC, convs[1].0, convs[1].1, tmp, FC, h, wd);
                silu(token, &mut tmp[..FC * plane]);
                conv3x3(token, &tmp[..FC * plane], FC, convs[2].0, convs[2].1, out, FC, h, wd);
                if cin == FC {
                    tmp[..FC * plane].copy_from_slice(&out[..FC * plane]);
                    gate(token, &tmp[..FC * plane], &inp[..FC * plane], &mut out[..FC * plane]);
                }
            }

            /// Whole SPANF forward into `out` ([3, 4h, 4w]).
            #[inline(always)]
            pub(crate) fn forward<T: Backend>(
                token: T,
                input: &[f32],
                h: usize,
                wd: usize,
                w: &SpanfWeights,
                out: &mut [f32],
            ) {
                let plane = h * wd;
                let mut near = vec![0.0f32; NEAR_CH * plane];
                conv_near(token, input, w.conv_near, &mut near, h, wd);
                crate::nan_debug("near", &near);

                let packed = crate::simd::Packed::build(w);
                let pb: [[(&[f32], &[f32]); 3]; 5] = core::array::from_fn(|bi| {
                    core::array::from_fn(|ci| {
                        (packed.blocks[bi][ci].as_slice(), w.blocks[bi][ci].1)
                    })
                });
                let mut b1 = vec![0.0f32; FC * plane];
                let mut tmp = vec![0.0f32; FC * plane];
                spab(token, input, &pb[0], 3, &mut b1, &mut tmp, h, wd);
                crate::nan_debug("b1", &b1);

                let mut b_out = vec![0.0f32; FC * plane];
                b_out.copy_from_slice(&b1);
                let mut cur = vec![0.0f32; FC * plane];
                for bi in 1..5 {
                    spab(token, &b_out, &pb[bi], FC, &mut cur, &mut tmp, h, wd);
                    core::mem::swap(&mut b_out, &mut cur);
                    crate::nan_debug("block", &b_out);
                }

                let mut catd = vec![0.0f32; FC * plane];
                conv1x1_multi(
                    token,
                    &[(&near[..], NEAR_CH), (&b_out[..], FC), (&b1[..], FC)],
                    &packed.conv_cat,
                    w.conv_cat_b,
                    &mut catd,
                    FC,
                    CAT_CH,
                    plane,
                );
                crate::nan_debug("catd", &catd);
                let mut pre = vec![0.0f32; NEAR_CH * plane];
                conv3x3(token, &catd, FC, &packed.conv2, w.conv2_b, &mut pre, NEAR_CH, h, wd);
                crate::nan_debug("pre", &pre);
                crate::pixel_shuffle4_pub(&pre, out, h, wd);
            }
        }
    };
}

define_kernels!(k8, f32x8, F32x8Convert, 8);
#[cfg(feature = "avx512")]
define_kernels!(k16, f32x16, F32x16Convert, 16);

// f32x8 family: v3 = AVX2+FMA native 256-bit; neon/wasm128 polyfill 2x128;
// scalar polyfills lane-wise. (v4/v4x are AVX-512 tiers — handled below on f32x16.)
#[magetypes(v3, neon, wasm128, scalar)]
fn spanf_forward_impl(
    token: Token,
    input: &[f32],
    h: usize,
    wd: usize,
    w: &SpanfWeights,
    out: &mut [f32],
) {
    k8::forward(token, input, h, wd, w, out);
}

#[cfg(all(target_arch = "x86_64", feature = "tier_v4"))]
#[arcane]
fn spanf_forward_impl_v4(
    token: X64V4Token,
    input: &[f32],
    h: usize,
    wd: usize,
    w: &SpanfWeights,
    out: &mut [f32],
) {
    k16::forward(token, input, h, wd, w, out);
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
fn spanf_forward_impl_v4x(
    token: X64V4xToken,
    input: &[f32],
    h: usize,
    wd: usize,
    w: &SpanfWeights,
    out: &mut [f32],
) {
    k16::forward(token, input, h, wd, w, out);
}

// --- kernel-level dispatch shims for parity tests -------------------------

#[magetypes(v3, neon, wasm128, scalar)]
fn silu_only_impl(token: Token, data: &mut [f32]) {
    k8::silu(token, data);
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
fn silu_only_impl_v4x(token: X64V4xToken, data: &mut [f32]) {
    k16::silu(token, data);
}
/// SiLU through the same dispatch the full forward uses (test/bisect entry).
pub fn silu_dispatch(data: &mut [f32]) {
    incant!(
        silu_only_impl(data),
        [v4x(cfg(avx512)), v3, neon, wasm128, scalar]
    );
}

#[magetypes(v3, neon, wasm128, scalar)]
fn conv3x3_only_impl(
    token: Token,
    inp: &[f32],
    cin: usize,
    wts: &[f32],
    bias: &[f32],
    out: &mut [f32],
    cout: usize,
    h: usize,
    wd: usize,
) {
    let packed = pack_conv3x3(wts, cin, cout);
    k8::conv3x3(token, inp, cin, &packed, bias, out, cout, h, wd);
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn conv3x3_only_impl_v4x(
    token: X64V4xToken,
    inp: &[f32],
    cin: usize,
    wts: &[f32],
    bias: &[f32],
    out: &mut [f32],
    cout: usize,
    h: usize,
    wd: usize,
) {
    let packed = pack_conv3x3(wts, cin, cout);
    k16::conv3x3(token, inp, cin, &packed, bias, out, cout, h, wd);
}
/// conv3x3 through full dispatch (test/bisect entry).
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_dispatch(
    inp: &[f32],
    cin: usize,
    wts: &[f32],
    bias: &[f32],
    out: &mut [f32],
    cout: usize,
    h: usize,
    wd: usize,
) {
    incant!(
        conv3x3_only_impl(inp, cin, wts, bias, out, cout, h, wd),
        [v4x(cfg(avx512)), v3, neon, wasm128, scalar]
    );
}

/// SIMD SPANF x4 forward with runtime dispatch (AVX-512x → AVX-512 → AVX2 →
/// NEON → WASM128 → scalar-polyfill).
pub fn spanf_x4_simd(input: &[f32], h: usize, wd: usize, w: &SpanfWeights) -> Vec<f32> {
    let plane = h * wd;
    assert_eq!(input.len(), 3 * plane);
    let mut out = vec![0.0f32; 3 * plane * 16];
    incant!(
        spanf_forward_impl(input, h, wd, w, &mut out),
        [v4x(cfg(avx512)), v4(cfg(tier_v4)), v3, neon, wasm128, scalar]
    );
    out
}

/// Tier-forced variants for debugging/bench isolation.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub fn spanf_x4_simd_force_v4x(input: &[f32], h: usize, wd: usize, w: &SpanfWeights) -> Option<Vec<f32>> {
    use archmage::prelude::*;
    let token = X64V4xToken::summon()?;
    let mut out = vec![0.0f32; 3 * h * wd * 16];
    spanf_forward_impl_v4x(token, input, h, wd, w, &mut out);
    Some(out)
}

#[cfg(all(target_arch = "x86_64", feature = "tier_v4"))]
pub fn spanf_x4_simd_force_v4(input: &[f32], h: usize, wd: usize, w: &SpanfWeights) -> Option<Vec<f32>> {
    use archmage::prelude::*;
    let token = X64V4Token::summon()?;
    let mut out = vec![0.0f32; 3 * h * wd * 16];
    spanf_forward_impl_v4(token, input, h, wd, w, &mut out);
    Some(out)
}

#[cfg(target_arch = "x86_64")]
pub fn spanf_x4_simd_force_v3(input: &[f32], h: usize, wd: usize, w: &SpanfWeights) -> Option<Vec<f32>> {
    let mut out = vec![0.0f32; 3 * h * wd * 16];
    incant!(
        spanf_forward_impl(input, h, wd, w, &mut out),
        [v3, scalar]
    );
    Some(out)
}
