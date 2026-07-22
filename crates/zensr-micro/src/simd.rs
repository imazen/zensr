//! SIMD SPANF forward: one `#[magetypes]` family (f32x8: v4/v3/neon/wasm128/scalar)
//! plus a hand `#[arcane]` `_v4x` variant on native 512-bit f32x16.
//!
//! Kernels are macro-instantiated per width so both variants share one body.
//! Interior columns run vector FMA (loads at x-1/x/x+1 are guarded by
//! `x + W + 1 <= wd`); the x edges and non-multiple tails run scalar.

use crate::{SpanfWeights, CAT_CH, FC, NEAR_CH, S2};
use archmage::prelude::*;

macro_rules! define_kernels {
    ($modname:ident, $vec:ident, $bnd:ident, $w:expr) => {
        pub(crate) mod $modname {
            use super::*;
            use magetypes::simd::backends::$bnd as Backend;
            use magetypes::simd::generic::$vec as V;

            /// 3x3 conv for one output row `oy`, four output channels
            /// `oc0..oc0+4`, accumulating over all `cin` input planes.
            #[inline(always)]
            #[allow(clippy::too_many_arguments)]
            pub(crate) fn conv3x3_row4<T: Backend>(
                token: T,
                inp: &[f32],
                cin: usize,
                wts: &[f32], // [cout][cin][9]
                bias: &[f32],
                out: &mut [f32],
                h: usize,
                wd: usize,
                oy: usize,
                oc0: usize,
            ) {
                const W: usize = $w;
                let plane = h * wd;
                for ob in 0..4 {
                    out[(oc0 + ob) * plane + oy * wd..][..wd].fill(bias[oc0 + ob]);
                }
                for ic in 0..cin {
                    let ip = &inp[ic * plane..(ic + 1) * plane];
                    for ky in 0..3usize {
                        let iy = oy + ky;
                        if iy < 1 || iy > h {
                            continue;
                        }
                        let irow = &ip[(iy - 1) * wd..iy * wd];
                        let mut wk = [[0.0f32; 3]; 4];
                        for ob in 0..4 {
                            let base = ((oc0 + ob) * cin + ic) * 9 + ky * 3;
                            wk[ob] = [wts[base], wts[base + 1], wts[base + 2]];
                        }

                        let mut x = 1usize;
                        while x + W + 1 <= wd {
                            let l = V::<T>::from_slice(token, &irow[x - 1..]);
                            let m = V::<T>::from_slice(token, &irow[x..]);
                            let r = V::<T>::from_slice(token, &irow[x + 1..]);
                            for ob in 0..4 {
                                let orow =
                                    &mut out[(oc0 + ob) * plane + oy * wd..][..wd];
                                let cur = V::<T>::from_slice(token, &orow[x..]);
                                let mut acc =
                                    l.mul_add(V::<T>::splat(token, wk[ob][0]), cur);
                                acc = m.mul_add(V::<T>::splat(token, wk[ob][1]), acc);
                                acc = r.mul_add(V::<T>::splat(token, wk[ob][2]), acc);
                                let dst: &mut [f32; W] =
                                    (&mut orow[x..x + W]).try_into().unwrap();
                                acc.store(dst);
                            }
                            x += W;
                        }
                        for xx in core::iter::once(0).chain(x..wd) {
                            for ob in 0..4 {
                                let mut s = 0.0f32;
                                if xx >= 1 {
                                    s += wk[ob][0] * irow[xx - 1];
                                }
                                s += wk[ob][1] * irow[xx];
                                if xx + 1 < wd {
                                    s += wk[ob][2] * irow[xx + 1];
                                }
                                out[(oc0 + ob) * plane + oy * wd + xx] += s;
                            }
                        }
                    }
                }
            }

            /// Grouped conv_near: 48 out channels, ic = oc/16, no bias.
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
                    out[oc * plane..(oc + 1) * plane].fill(0.0);
                    for oy in 0..h {
                        for ky in 0..3usize {
                            let iy = oy + ky;
                            if iy < 1 || iy > h {
                                continue;
                            }
                            let irow = &ip[(iy - 1) * wd..iy * wd];
                            let (w0, w1, w2) =
                                (w9[ky * 3], w9[ky * 3 + 1], w9[ky * 3 + 2]);
                            let mut x = 1usize;
                            while x + W + 1 <= wd {
                                let orow = &mut out[oc * plane + oy * wd..][..wd];
                                let l = V::<T>::from_slice(token, &irow[x - 1..]);
                                let m = V::<T>::from_slice(token, &irow[x..]);
                                let r = V::<T>::from_slice(token, &irow[x + 1..]);
                                let cur = V::<T>::from_slice(token, &orow[x..]);
                                let mut acc = l.mul_add(V::<T>::splat(token, w0), cur);
                                acc = m.mul_add(V::<T>::splat(token, w1), acc);
                                acc = r.mul_add(V::<T>::splat(token, w2), acc);
                                let dst: &mut [f32; W] =
                                    (&mut orow[x..x + W]).try_into().unwrap();
                                acc.store(dst);
                                x += W;
                            }
                            for xx in core::iter::once(0).chain(x..wd) {
                                let mut s = 0.0f32;
                                if xx >= 1 {
                                    s += w0 * irow[xx - 1];
                                }
                                s += w1 * irow[xx];
                                if xx + 1 < wd {
                                    s += w2 * irow[xx + 1];
                                }
                                out[oc * plane + oy * wd + xx] += s;
                            }
                        }
                    }
                }
            }

            /// conv1x1 over a virtual concat of sources (no materialized cat).
            #[inline(always)]
            #[allow(clippy::too_many_arguments)]
            pub(crate) fn conv1x1_multi<T: Backend>(
                token: T,
                srcs: &[(&[f32], usize)],
                wts: &[f32], // [cout][cin_total]
                bias: &[f32],
                out: &mut [f32],
                cout: usize,
                cin_total: usize,
                plane: usize,
            ) {
                const W: usize = $w;
                let n = plane / W * W;
                for oc in 0..cout {
                    let wrow = &wts[oc * cin_total..(oc + 1) * cin_total];
                    let op = &mut out[oc * plane..(oc + 1) * plane];
                    op.fill(bias[oc]);
                    let mut wi = 0usize;
                    for (src, cin) in srcs.iter() {
                        for ic in 0..*cin {
                            let wv = wrow[wi];
                            wi += 1;
                            let ip = &src[ic * plane..(ic + 1) * plane];
                            let wv_v = V::<T>::splat(token, wv);
                            let mut x = 0usize;
                            while x < n {
                                let cur = V::<T>::from_slice(token, &op[x..]);
                                let acc = V::<T>::from_slice(token, &ip[x..])
                                    .mul_add(wv_v, cur);
                                let dst: &mut [f32; W] =
                                    (&mut op[x..x + W]).try_into().unwrap();
                                acc.store(dst);
                                x += W;
                            }
                            for x in n..plane {
                                op[x] += wv * ip[x];
                            }
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
                crate::nan_debug("  spab.c1", &out[..FC * plane]);
                silu(token, &mut out[..FC * plane]);
                crate::nan_debug("  spab.silu1", &out[..FC * plane]);
                conv3x3(token, &out[..FC * plane], FC, convs[1].0, convs[1].1, tmp, FC, h, wd);
                crate::nan_debug("  spab.c2", &tmp[..FC * plane]);
                silu(token, &mut tmp[..FC * plane]);
                crate::nan_debug("  spab.silu2", &tmp[..FC * plane]);
                conv3x3(token, &tmp[..FC * plane], FC, convs[2].0, convs[2].1, out, FC, h, wd);
                crate::nan_debug("  spab.c3", &out[..FC * plane]);
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

                let mut b1 = vec![0.0f32; FC * plane];
                let mut tmp = vec![0.0f32; FC * plane];
                spab(token, input, &w.blocks[0], 3, &mut b1, &mut tmp, h, wd);
                crate::nan_debug("b1", &b1);

                let mut b_out = vec![0.0f32; FC * plane];
                b_out.copy_from_slice(&b1);
                let mut cur = vec![0.0f32; FC * plane];
                for bi in 1..5 {
                    spab(token, &b_out, &w.blocks[bi], FC, &mut cur, &mut tmp, h, wd);
                    core::mem::swap(&mut b_out, &mut cur);
                    crate::nan_debug("block", &b_out);
                }

                let mut catd = vec![0.0f32; FC * plane];
                conv1x1_multi(
                    token,
                    &[(&near[..], NEAR_CH), (&b_out[..], FC), (&b1[..], FC)],
                    w.conv_cat_w,
                    w.conv_cat_b,
                    &mut catd,
                    FC,
                    CAT_CH,
                    plane,
                );
                crate::nan_debug("catd", &catd);
                let mut pre = vec![0.0f32; NEAR_CH * plane];
                conv3x3(token, &catd, FC, w.conv2_w, w.conv2_b, &mut pre, NEAR_CH, h, wd);
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

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
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
    k8::conv3x3(token, inp, cin, wts, bias, out, cout, h, wd);
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
    k16::conv3x3(token, inp, cin, wts, bias, out, cout, h, wd);
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
        [v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar]
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

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
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
        [v3]
    );
    Some(out)
}
