//! Bit-exact path and tier consistency for the adopted-model kernels.
//!
//! Every output element must have ONE value, whichever code path computes it
//! (main tile loop, overlapped tail tile, narrow row, vector remainder) and
//! whichever dispatch tier runs. Before this was enforced, an image restored
//! with 2 threads differed from the same image with 1 thread, and AVX2 from
//! AVX-512, by one RGB8 level in final pixels (see CHANGELOG).
//!
//! The oracle is the same kernel on input ZERO-EXTENDED to a width or length
//! that is a multiple of 16 and at least 16, so that only the main vector loop
//! runs on every tier, cropped back. Zero extension is exactly what the
//! convolution's own border padding supplies, so every kept column sees the
//! same receptive field. On top of that, every tier must equal the scalar
//! tier bit for bit.
//!
//! Tiers are selected by calling the per-tier kernel functions with a
//! summoned token, never by disabling tokens process-wide, so the tier that
//! ran is known exactly and concurrent tests cannot change it.

use super::*;

#[derive(Clone, Copy, Debug)]
enum Tier {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    V3(archmage::X64V3Token),
    #[cfg(all(target_arch = "x86_64", feature = "tier_v4"))]
    V4(archmage::X64V4Token),
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    V4x(archmage::X64V4xToken),
    #[cfg(target_arch = "aarch64")]
    Neon(archmage::NeonToken),
    #[cfg(target_arch = "wasm32")]
    Wasm(archmage::Wasm128Token),
}

/// Every tier this build compiled and this CPU can run; scalar first.
fn tiers() -> Vec<Tier> {
    #[allow(unused_mut)]
    let mut v = vec![Tier::Scalar];
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = archmage::X64V3Token::summon() {
        v.push(Tier::V3(t));
    }
    #[cfg(all(target_arch = "x86_64", feature = "tier_v4"))]
    if let Some(t) = archmage::X64V4Token::summon() {
        v.push(Tier::V4(t));
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = archmage::X64V4xToken::summon() {
        v.push(Tier::V4x(t));
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = archmage::NeonToken::summon() {
        v.push(Tier::Neon(t));
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = archmage::Wasm128Token::summon() {
        v.push(Tier::Wasm(t));
    }
    v
}

#[allow(clippy::too_many_arguments)]
fn conv(
    t: Tier,
    inp: &[f32],
    cin: usize,
    p: &[f32],
    b: &[f32],
    out: &mut [f32],
    cout: usize,
    h: usize,
    wd: usize,
    cs: usize,
) {
    match t {
        Tier::Scalar => {
            conv3x3_lay_scalar(archmage::ScalarToken, inp, cin, p, b, out, cout, h, wd, cs)
        }
        #[cfg(target_arch = "x86_64")]
        Tier::V3(k) => conv3x3_lay_v3(k, inp, cin, p, b, out, cout, h, wd, cs),
        #[cfg(all(target_arch = "x86_64", feature = "tier_v4"))]
        Tier::V4(k) => conv3x3_lay_v4(k, inp, cin, p, b, out, cout, h, wd, cs),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Tier::V4x(k) => conv3x3_lay_v4x(k, inp, cin, p, b, out, cout, h, wd, cs),
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(k) => conv3x3_lay_neon(k, inp, cin, p, b, out, cout, h, wd, cs),
        #[cfg(target_arch = "wasm32")]
        Tier::Wasm(k) => conv3x3_lay_wasm128(k, inp, cin, p, b, out, cout, h, wd, cs),
    }
}

#[allow(clippy::too_many_arguments)]
fn conv1x1(
    t: Tier,
    srcs: &[(&[f32], usize)],
    w: &[f32],
    b: &[f32],
    out: &mut [f32],
    cout: usize,
    cin_total: usize,
    plane: usize,
    cs: usize,
) {
    match t {
        Tier::Scalar => conv1x1_gen_lay_scalar(
            archmage::ScalarToken,
            srcs,
            w,
            b,
            out,
            cout,
            cin_total,
            plane,
            cs,
        ),
        #[cfg(target_arch = "x86_64")]
        Tier::V3(k) => conv1x1_gen_lay_v3(k, srcs, w, b, out, cout, cin_total, plane, cs),
        #[cfg(all(target_arch = "x86_64", feature = "tier_v4"))]
        Tier::V4(k) => conv1x1_gen_lay_v4(k, srcs, w, b, out, cout, cin_total, plane, cs),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Tier::V4x(k) => conv1x1_gen_lay_v4x(k, srcs, w, b, out, cout, cin_total, plane, cs),
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(k) => conv1x1_gen_lay_neon(k, srcs, w, b, out, cout, cin_total, plane, cs),
        #[cfg(target_arch = "wasm32")]
        Tier::Wasm(k) => conv1x1_gen_lay_wasm128(k, srcs, w, b, out, cout, cin_total, plane, cs),
    }
}

fn prelu(t: Tier, data: &mut [f32], slopes: &[f32], channels: usize, plane: usize, cs: usize) {
    match t {
        Tier::Scalar => prelu_lay_scalar(archmage::ScalarToken, data, slopes, channels, plane, cs),
        #[cfg(target_arch = "x86_64")]
        Tier::V3(k) => prelu_lay_v3(k, data, slopes, channels, plane, cs),
        #[cfg(all(target_arch = "x86_64", feature = "tier_v4"))]
        Tier::V4(k) => prelu_lay_v4(k, data, slopes, channels, plane, cs),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Tier::V4x(k) => prelu_lay_v4x(k, data, slopes, channels, plane, cs),
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(k) => prelu_lay_neon(k, data, slopes, channels, plane, cs),
        #[cfg(target_arch = "wasm32")]
        Tier::Wasm(k) => prelu_lay_wasm128(k, data, slopes, channels, plane, cs),
    }
}

fn gate(t: Tier, a: &[f32], skip: &[f32], out: &mut [f32]) {
    match t {
        Tier::Scalar => gate_lay_scalar(archmage::ScalarToken, a, skip, out),
        #[cfg(target_arch = "x86_64")]
        Tier::V3(k) => gate_lay_v3(k, a, skip, out),
        #[cfg(all(target_arch = "x86_64", feature = "tier_v4"))]
        Tier::V4(k) => gate_lay_v4(k, a, skip, out),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Tier::V4x(k) => gate_lay_v4x(k, a, skip, out),
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(k) => gate_lay_neon(k, a, skip, out),
        #[cfg(target_arch = "wasm32")]
        Tier::Wasm(k) => gate_lay_wasm128(k, a, skip, out),
    }
}

/// SiLU has no `_v4` kernel: the production dispatch sends that tier to v3.
fn silu(t: Tier, data: &mut [f32]) {
    match t {
        Tier::Scalar => silu_only_impl_scalar(archmage::ScalarToken, data),
        #[cfg(target_arch = "x86_64")]
        Tier::V3(k) => silu_only_impl_v3(k, data),
        #[cfg(all(target_arch = "x86_64", feature = "tier_v4"))]
        Tier::V4(_) => silu_only_impl_v3(archmage::X64V3Token::summon().unwrap(), data),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Tier::V4x(k) => silu_only_impl_v4x(k, data),
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(k) => silu_only_impl_neon(k, data),
        #[cfg(target_arch = "wasm32")]
        Tier::Wasm(k) => silu_only_impl_wasm128(k, data),
    }
}

struct Lcg(u32);
impl Lcg {
    fn f(&mut self, scale: f32) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        ((self.0 >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * scale
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.f(scale)).collect()
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// A width that runs only the main tile loop on every tier.
fn main_loop_width(wd: usize) -> usize {
    wd.next_multiple_of(16).max(16)
}

/// conv3x3 of a planar [cin][h][wd] input, on tier `t`.
fn conv_tight(
    t: Tier,
    inp: &[f32],
    cin: usize,
    packed: &[f32],
    bias: &[f32],
    cout: usize,
    h: usize,
    wd: usize,
) -> Vec<f32> {
    let plane = h * wd;
    let mut out = vec![0.0f32; cout * plane];
    conv(t, inp, cin, packed, bias, &mut out, cout, h, wd, plane);
    out
}

/// The oracle: the same conv on the input zero-extended to `main_loop_width`
/// columns, cropped back to `wd`.
fn conv_oracle(
    t: Tier,
    inp: &[f32],
    cin: usize,
    packed: &[f32],
    bias: &[f32],
    cout: usize,
    h: usize,
    wd: usize,
) -> Vec<f32> {
    let wx = main_loop_width(wd);
    let mut ext = vec![0.0f32; cin * h * wx];
    for c in 0..cin {
        for y in 0..h {
            ext[(c * h + y) * wx..][..wd].copy_from_slice(&inp[(c * h + y) * wd..][..wd]);
        }
    }
    let wide = conv_tight(t, &ext, cin, packed, bias, cout, h, wx);
    let mut out = vec![0.0f32; cout * h * wd];
    for c in 0..cout {
        for y in 0..h {
            out[(c * h + y) * wd..][..wd].copy_from_slice(&wide[(c * h + y) * wx..][..wd]);
        }
    }
    out
}

#[test]
fn conv3x3_every_path_and_tier_matches_the_main_loop() {
    let ts = tiers();
    let mut g = Lcg(0x5eed_0001);
    let mut cases = 0usize;
    for cin in [3usize, 8, 16, 17, 24, 31, 33, 48, 64] {
        for cout in [4usize, 8, 24] {
            let raw = g.vec(cout * cin * 9, 0.5);
            let packed = crate::pack_conv3x3(&raw, cin, cout);
            let bias = g.vec(cout, 0.5);
            for h in [1usize, 3, 5] {
                for wd in [
                    1usize, 2, 3, 5, 7, 8, 9, 12, 15, 16, 17, 23, 24, 25, 31, 32, 33, 36, 47, 48,
                    49,
                ] {
                    let inp = g.vec(cin * h * wd, 2.0);
                    let reference = bits(&conv_oracle(
                        Tier::Scalar,
                        &inp,
                        cin,
                        &packed,
                        &bias,
                        cout,
                        h,
                        wd,
                    ));
                    for &t in &ts {
                        let oracle = bits(&conv_oracle(t, &inp, cin, &packed, &bias, cout, h, wd));
                        let got = bits(&conv_tight(t, &inp, cin, &packed, &bias, cout, h, wd));
                        let shape = format!("{t:?} cin={cin} cout={cout} h={h} wd={wd}");
                        assert_eq!(
                            oracle, reference,
                            "main loop differs from the scalar tier: {shape}"
                        );
                        if got != oracle {
                            let i = got.iter().zip(&oracle).position(|(a, b)| a != b).unwrap();
                            panic!(
                                "{shape}: element {i} (x={}) {:#010x} != main-loop {:#010x}",
                                i % wd,
                                got[i],
                                oracle[i]
                            );
                        }
                        cases += 1;
                    }
                }
            }
        }
    }
    assert!(cases > 0);
}

/// The counterexample that exposed the defect: one receptive field, three row
/// widths. Columns 0..=30 of 32- and 36-wide rows, and 0..=10 of a 12-wide
/// row, read identical inputs, so they must hold identical bits. Before the
/// fix, the 36-wide row's overlapped tail (one FMA chain over all taps) gave
/// columns 20..=30 (W=16) or 28..=30 (W=8) different bits whenever cin > 16,
/// and a 12-wide row on the AVX-512 tier went through an unfused scalar loop.
#[test]
fn conv3x3_same_receptive_field_same_bits_at_any_row_width() {
    for t in tiers() {
        for cin in [8usize, 16, 17, 24, 64] {
            let (cout, h) = (4usize, 3usize);
            let mut g = Lcg(4242 + cin as u32);
            let raw = g.vec(cout * cin * 9, 0.5);
            let bias = g.vec(cout, 0.5);
            let wide = g.vec(cin * h * 36, 2.0);
            let packed = crate::pack_conv3x3(&raw, cin, cout);
            let at = |wd: usize| -> Vec<f32> {
                let mut inp = vec![0.0f32; cin * h * wd];
                for ic in 0..cin {
                    for y in 0..h {
                        inp[(ic * h + y) * wd..][..wd]
                            .copy_from_slice(&wide[(ic * h + y) * 36..][..wd]);
                    }
                }
                conv_tight(t, &inp, cin, &packed, &bias, cout, h, wd)
            };
            let o36 = at(36);
            for (wd, upto) in [(32usize, 30usize), (12, 10)] {
                let o = at(wd);
                for c in 0..cout {
                    for y in 0..h {
                        for x in 0..=upto {
                            assert_eq!(
                                o[(c * h + y) * wd + x].to_bits(),
                                o36[(c * h + y) * 36 + x].to_bits(),
                                "{t:?} cin={cin} wd={wd} c={c} y={y} x={x}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn conv1x1_remainder_and_tier_match_full_tiles() {
    let ts = tiers();
    let mut g = Lcg(0x5eed_0002);
    for nsrc in [1usize, 2, 4] {
        for cin in [3usize, 16, 48] {
            let cout = 8usize;
            let cin_total = nsrc * cin;
            let raw = g.vec(cout * cin_total, 0.5);
            let w = crate::pack_conv1x1(&raw, cin_total, cout);
            let bias = g.vec(cout, 0.5);
            for plane in [1usize, 5, 7, 8, 9, 15, 16, 17, 31, 33, 50] {
                let srcs_data: Vec<Vec<f32>> = (0..nsrc).map(|_| g.vec(cin * plane, 2.0)).collect();
                let px = main_loop_width(plane);
                let ext: Vec<Vec<f32>> = srcs_data
                    .iter()
                    .map(|s| {
                        let mut e = vec![0.0f32; cin * px];
                        for c in 0..cin {
                            e[c * px..][..plane].copy_from_slice(&s[c * plane..][..plane]);
                        }
                        e
                    })
                    .collect();
                let mut reference = None;
                for &t in &ts {
                    let srcs: Vec<(&[f32], usize)> =
                        srcs_data.iter().map(|s| (&s[..], cin)).collect();
                    let mut got = vec![0.0f32; cout * plane];
                    conv1x1(t, &srcs, &w, &bias, &mut got, cout, cin_total, plane, plane);
                    let esrcs: Vec<(&[f32], usize)> = ext.iter().map(|s| (&s[..], cin)).collect();
                    let mut wide = vec![0.0f32; cout * px];
                    conv1x1(t, &esrcs, &w, &bias, &mut wide, cout, cin_total, px, px);
                    let oracle: Vec<f32> = (0..cout)
                        .flat_map(|c| wide[c * px..][..plane].to_vec())
                        .collect();
                    let shape = format!("{t:?} nsrc={nsrc} cin={cin} plane={plane}");
                    assert_eq!(
                        bits(&got),
                        bits(&oracle),
                        "remainder differs from full tiles: {shape}"
                    );
                    let r = reference.get_or_insert_with(|| bits(&oracle));
                    assert_eq!(&bits(&oracle), r, "tier differs from scalar: {shape}");
                }
            }
        }
    }
}

/// Finite inputs plus every special class. NaN payloads are not compared:
/// Rust and the backends do not specify them, only that the result is NaN.
fn special_values() -> Vec<f32> {
    let mut v = vec![
        0.0,
        -0.0,
        1.0e-45,
        -1.0e-45,
        1.0e-39,
        -1.0e-39,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1.0e-30,
        -1.0e-30,
        0.5,
        -0.5,
        3.0,
        -3.0,
        87.0,
        -87.0,
        89.0,
        -104.0,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let mut g = Lcg(0x5eed_0003);
    v.extend(g.vec(200, 40.0));
    v
}

fn same_value(a: f32, b: f32) -> bool {
    if a.is_nan() || b.is_nan() {
        a.is_nan() && b.is_nan()
    } else {
        a.to_bits() == b.to_bits()
    }
}

fn assert_same(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(
            same_value(*a, *b),
            "{what}: element {i}: {a:e} ({:#010x}) != {b:e} ({:#010x})",
            a.to_bits(),
            b.to_bits()
        );
    }
}

#[test]
fn elementwise_remainders_and_tiers_match_full_vectors() {
    let ts = tiers();
    let vals = special_values();
    for n in [
        0usize,
        1,
        3,
        7,
        8,
        9,
        15,
        16,
        17,
        31,
        33,
        47,
        100,
        vals.len(),
    ] {
        let data: Vec<f32> = (0..n).map(|i| vals[i % vals.len()]).collect();
        let skip: Vec<f32> = (0..n).map(|i| vals[(i * 7 + 3) % vals.len()]).collect();
        let nx = main_loop_width(n);
        let mut silu_ref = None;
        let mut gate_ref = None;
        let mut prelu_ref = None;
        for &t in &ts {
            // SiLU
            let mut got = data.clone();
            silu(t, &mut got);
            let mut wide = data.clone();
            wide.resize(nx, 0.0);
            silu(t, &mut wide);
            assert_same(
                &got,
                &wide[..n],
                &format!("silu {t:?} n={n} vs full vectors"),
            );
            assert_same(
                &got,
                silu_ref.get_or_insert_with(|| got.clone()),
                &format!("silu {t:?} n={n} vs scalar"),
            );
            // gate
            let mut got = vec![0.0f32; n];
            gate(t, &data, &skip, &mut got);
            let (mut wa, mut ws) = (data.clone(), skip.clone());
            wa.resize(nx, 0.0);
            ws.resize(nx, 0.0);
            let mut wide = vec![0.0f32; nx];
            gate(t, &wa, &ws, &mut wide);
            assert_same(
                &got,
                &wide[..n],
                &format!("gate {t:?} n={n} vs full vectors"),
            );
            assert_same(
                &got,
                gate_ref.get_or_insert_with(|| got.clone()),
                &format!("gate {t:?} n={n} vs scalar"),
            );
            // PReLU: two channels of `n` with a stride gap; slopes include ±0.
            for slopes in [[0.25f32, -1.5], [0.0, -0.0]] {
                let cs = n + 3;
                let mut got = vec![7.0f32; 2 * cs];
                got[..n].copy_from_slice(&data);
                got[cs..cs + n].copy_from_slice(&skip);
                let before_gap: Vec<f32> = got[n..cs].to_vec();
                prelu(t, &mut got, &slopes, 2, n, cs);
                assert_eq!(
                    bits(&got[n..cs]),
                    bits(&before_gap),
                    "prelu wrote outside the plane"
                );
                let mut wide = vec![0.0f32; 2 * nx];
                wide[..n].copy_from_slice(&data);
                wide[nx..nx + n].copy_from_slice(&skip);
                prelu(t, &mut wide, &slopes, 2, nx, nx);
                let g2: Vec<f32> = got[..n].iter().chain(&got[cs..cs + n]).copied().collect();
                let w2: Vec<f32> = wide[..n].iter().chain(&wide[nx..nx + n]).copied().collect();
                assert_same(
                    &g2,
                    &w2,
                    &format!("prelu {t:?} n={n} slopes={slopes:?} vs full vectors"),
                );
                let key = (slopes[0].to_bits(), n);
                let r = prelu_ref.get_or_insert_with(std::collections::HashMap::new);
                let want = r.entry(key).or_insert_with(|| g2.clone());
                assert_same(
                    &g2,
                    want,
                    &format!("prelu {t:?} n={n} slopes={slopes:?} vs scalar"),
                );
            }
        }
    }
}

/// `exp_midp_portable` must reproduce magetypes' `exp_midp` exactly on the
/// FMA tiers (so the repair does not move their outputs), and must agree
/// across all tiers including scalar.
#[test]
fn portable_exp_matches_exp_midp_on_fma_tiers_and_agrees_across_tiers() {
    #[magetypes(v3, neon, wasm128, scalar)]
    fn both(token: Token, xs: &[f32], portable: &mut [f32], midp: &mut [f32]) {
        for ((x, p), m) in xs
            .as_chunks::<8>()
            .0
            .iter()
            .zip(portable.as_chunks_mut::<8>().0.iter_mut())
            .zip(midp.as_chunks_mut::<8>().0.iter_mut())
        {
            let v = magetypes::simd::generic::f32x8::from_slice_t(token, x);
            k8::exp_midp_portable(token, v).store(p);
            v.exp_midp().store(m);
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[arcane]
    fn both_v4x(token: archmage::X64V4xToken, xs: &[f32], portable: &mut [f32], midp: &mut [f32]) {
        for ((x, p), m) in xs
            .as_chunks::<16>()
            .0
            .iter()
            .zip(portable.as_chunks_mut::<16>().0.iter_mut())
            .zip(midp.as_chunks_mut::<16>().0.iter_mut())
        {
            let v = magetypes::simd::generic::f32x16::from_slice_t(token, x);
            k16::exp_midp_portable(token, v).store(p);
            v.exp_midp().store(m);
        }
    }
    let mut xs = special_values();
    // A dense sweep over the range SiLU and the gate feed it.
    xs.extend((0..200_000).map(|i| -100.0 + i as f32 * 0.001));
    xs.resize(xs.len().next_multiple_of(16), 0.0);
    let n = xs.len();
    let run = |t: Tier| -> (Vec<f32>, Vec<f32>) {
        let (mut p, mut m) = (vec![0.0f32; n], vec![0.0f32; n]);
        match t {
            Tier::Scalar => both_scalar(archmage::ScalarToken, &xs, &mut p, &mut m),
            #[cfg(target_arch = "x86_64")]
            Tier::V3(k) => both_v3(k, &xs, &mut p, &mut m),
            #[cfg(all(target_arch = "x86_64", feature = "tier_v4"))]
            Tier::V4(_) => both_v3(archmage::X64V3Token::summon().unwrap(), &xs, &mut p, &mut m),
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            Tier::V4x(k) => both_v4x(k, &xs, &mut p, &mut m),
            #[cfg(target_arch = "aarch64")]
            Tier::Neon(k) => both_neon(k, &xs, &mut p, &mut m),
            #[cfg(target_arch = "wasm32")]
            Tier::Wasm(k) => both_wasm128(k, &xs, &mut p, &mut m),
        }
        (p, m)
    };
    let (scalar_portable, _) = run(Tier::Scalar);
    for t in tiers() {
        let (p, m) = run(t);
        let fma_tier = !matches!(t, Tier::Scalar);
        #[cfg(target_arch = "wasm32")]
        let fma_tier = fma_tier && !matches!(t, Tier::Wasm(_));
        // On x86, where the existing outputs come from, the portable exp must
        // equal magetypes' exp_midp for every input, NaN included. NEON's
        // magetypes max/min return NaN for a NaN operand, so its exp_midp(NaN)
        // is NaN while the portable exp keeps x86's clamp: there the comparison
        // covers the non-NaN inputs, and NaN is held to the scalar/x86 value by
        // the cross-tier check below.
        #[cfg(target_arch = "aarch64")]
        let nan_too = !matches!(t, Tier::Neon(_));
        #[cfg(not(target_arch = "aarch64"))]
        let nan_too = true;
        if fma_tier {
            let keep: Vec<usize> = (0..n).filter(|&i| nan_too || !xs[i].is_nan()).collect();
            let pk: Vec<f32> = keep.iter().map(|&i| p[i]).collect();
            let mk: Vec<f32> = keep.iter().map(|&i| m[i]).collect();
            assert_same(&pk, &mk, &format!("{t:?}: exp_midp_portable vs exp_midp"));
        }
        assert_same(
            &p,
            &scalar_portable,
            &format!("{t:?}: exp_midp_portable vs scalar tier"),
        );
    }
}

/// Procedural compact weights in `load_compact` order (never committed data).
fn compact_weights(nf: usize, nc: usize, s: usize, seed: u32) -> Vec<f32> {
    let mut g = Lcg(seed);
    let s2 = s * s;
    let mut w = Vec::new();
    let conv = |g: &mut Lcg, w: &mut Vec<f32>, cin: usize, cout: usize| {
        w.extend(g.vec(cout * cin * 9, 1.2 / ((9 * cin) as f32).sqrt()));
    };
    conv(&mut g, &mut w, 3, nf);
    w.extend(g.vec(nf, 0.1));
    w.extend(g.vec(nf, 0.6));
    for _ in 0..nc {
        conv(&mut g, &mut w, nf, nf);
        w.extend(g.vec(nf, 0.1));
        w.extend(g.vec(nf, 0.6));
    }
    conv(&mut g, &mut w, nf, 3 * s2);
    w.extend(g.vec(3 * s2, 0.1));
    w
}

fn span48_weights(s: usize, seed: u32) -> Vec<f32> {
    let mut g = Lcg(seed);
    let (fc, s2) = (48usize, s * s);
    let mut w = Vec::new();
    let mut conv = |cin: usize, cout: usize, k: usize| {
        w.extend(g.vec(cout * cin * k, 1.2 / ((k * cin) as f32).sqrt()));
        w.extend(g.vec(cout, 0.1));
    };
    conv(3, fc, 9);
    for _ in 0..18 {
        conv(fc, fc, 9);
    }
    conv(fc, fc, 9);
    conv(4 * fc, fc, 1);
    conv(fc, 3 * s2, 9);
    w
}

/// Whole-image forward and the tiled runner at several thread counts and tile
/// sizes must agree bit for bit: tiling is exact-halo, and every kernel now
/// gives each element one value whatever its position in a tile.
#[test]
fn adopted_models_tiled_equals_whole_bitwise() {
    use crate::adopted::AdoptedModel;
    let mut models: Vec<(String, AdoptedModel)> = Vec::new();
    for (i, &(nf, nc, s)) in [
        (4usize, 1usize, 1usize),
        (12, 2, 3),
        (24, 2, 1),
        (24, 1, 2),
        (64, 1, 4),
    ]
    .iter()
    .enumerate()
    {
        let raw = compact_weights(nf, nc, s, 900 + i as u32);
        models.push((
            format!("compact nf{nf} nc{nc} s{s}"),
            AdoptedModel::load_compact(&raw, nf, nc, s).unwrap(),
        ));
    }
    // span48 at scale 1 or 3 is not executable (upsampler cout 3 or 27 is
    // not a multiple of 4); `load_span48` handles that separately.
    for s in [2usize, 4] {
        models.push((
            format!("span48 s{s}"),
            AdoptedModel::load_span48(&span48_weights(s, 990 + s as u32), s).unwrap(),
        ));
    }
    let mut g = Lcg(0x5eed_0004);
    for (name, m) in &models {
        for (h, w) in [(5usize, 7usize), (37, 53)] {
            let input = g
                .vec(3 * h * w, 1.0)
                .iter()
                .map(|v| v + 0.5)
                .collect::<Vec<f32>>();
            let s = m.scale;
            let mut whole = vec![0.0f32; 3 * h * w * s * s];
            m.forward(&input, h, w, &mut whole);
            assert!(
                whole.iter().all(|v| v.is_finite()),
                "{name}: non-finite forward"
            );
            // Auto tiles (tile = 0) on images under 32 px are covered by the
            // small-image regression in `adopted.rs`; here every size gets
            // explicit tiles and the larger ones also get the auto ladder.
            let auto: &[(usize, usize)] = if h.max(w) >= 32 {
                &[(4, 0), (16, 0)]
            } else {
                &[]
            };
            for &(threads, tile) in [(1usize, 32usize), (3, 32), (2, 40), (1, 64)]
                .iter()
                .chain(auto)
            {
                let tiled = m.upscale_tiled(&input, h, w, threads, tile);
                if let Some(i) = tiled
                    .iter()
                    .zip(&whole)
                    .position(|(a, b)| a.to_bits() != b.to_bits())
                {
                    panic!("{name} {h}x{w} threads={threads} tile={tile}: element {i} {:e} != whole {:e}", tiled[i], whole[i]);
                }
            }
        }
    }
}
