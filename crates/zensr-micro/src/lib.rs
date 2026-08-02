//! Hand-rolled inference for SPANF (NTIRE25 team24, feature_channels=32, x4).
//!
//! Feasibility probe for a sub-300KB SR runtime: the entire op vocabulary is
//! grouped/plain conv3x3, conv1x1, SiLU, a sigmoid gate, channel concat, and
//! PixelShuffle. Scalar-but-vectorizable safe Rust; magetypes SIMD comes later
//! if the size/speed answer justifies it.
#![forbid(unsafe_code)]
// Three lints fight the kernel style rather than finding defects here:
//
// `too_many_arguments` — convolution kernels take their shape (in/out channels,
//   width, height, stride, padding, group count) as scalars precisely so the
//   optimiser sees constants at the call site. Bundling them into a struct is
//   what the lint wants and is exactly what loses that.
// `needless_range_loop` — indexed loops over fixed-size arrays are the pattern
//   LLVM turns into shuffles and bounds-check-free code; the iterator rewrite
//   the lint suggests has repeatedly failed to vectorise the same way.
// `identity_op` — `+ 0` keeps a column of offsets in the layout table aligned
//   and readable as a table.
//
// They are allowed crate-wide rather than at ~20 sites; every other clippy lint
// is denied in CI.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::identity_op)]

pub mod adopted;
pub mod consist;
pub mod guards;
mod wino;
// SPANF research kernels: crate-internal unless `internals` opts in.
#[cfg(all(feature = "px", feature = "internals"))]
pub mod px;
#[cfg(feature = "internals")]
pub mod simd;
#[cfg(not(feature = "internals"))]
#[allow(dead_code)] // SPANF-only paths idle when internals is off
pub(crate) mod simd;
#[cfg(feature = "internals")]
pub mod tiled;
#[cfg(not(feature = "internals"))]
#[allow(dead_code)]
pub(crate) mod tiled;
#[cfg(all(feature = "px", feature = "internals"))]
pub use px::{spanf_x4_px, PxError};
#[cfg(feature = "internals")]
pub use simd::spanf_x4_simd;
#[cfg(feature = "internals")]
pub use tiled::{spanf_x4_tiled, HALO};

#[doc(hidden)] // SPANF research surface — NOT part of the contract
pub const FC: usize = 32; // feature channels
#[doc(hidden)] // SPANF research surface — NOT part of the contract
pub const S2: usize = 16; // upscale^2
#[doc(hidden)] // SPANF research surface — NOT part of the contract
pub const NEAR_CH: usize = 3 * S2; // 48
#[doc(hidden)] // SPANF research surface — NOT part of the contract
pub const CAT_CH: usize = NEAR_CH + 2 * FC; // 112
#[doc(hidden)] // SPANF research surface — NOT part of the contract
pub const TOTAL_FLOATS: usize = 148_288;

/// Weight views into one flat buffer, in tools/dump_spanf_weights.py order.
#[doc(hidden)] // SPANF research surface — NOT part of the contract
pub struct SpanfWeights<'a> {
    pub conv_near: &'a [f32],                     // [48,1,3,3]
    pub blocks: [[(&'a [f32], &'a [f32]); 3]; 5], // (w,b) per conv, per block
    pub conv_cat_w: &'a [f32],                    // [32,112]
    pub conv_cat_b: &'a [f32],                    // [32]
    pub conv2_w: &'a [f32],                       // [48,32,3,3]
    pub conv2_b: &'a [f32],                       // [48]
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
        Ok(SpanfWeights {
            conv_near,
            blocks,
            conv_cat_w,
            conv_cat_b,
            conv2_w,
            conv2_b,
        })
    }
}

/// Tensor layout in dump order: (weight_floats, bias_floats) per conv.
#[doc(hidden)] // SPANF research surface — NOT part of the contract
pub const TENSOR_LAYOUT: [(usize, usize); 17] = [
    (NEAR_CH * 9, 0),      // conv_near (grouped, no bias)
    (3 * FC * 9, FC),      // b1.c1
    (FC * FC * 9, FC),     // b1.c2
    (FC * FC * 9, FC),     // b1.c3
    (FC * FC * 9, FC),     // b2.c1
    (FC * FC * 9, FC),     // b2.c2
    (FC * FC * 9, FC),     // b2.c3
    (FC * FC * 9, FC),     // b3.c1
    (FC * FC * 9, FC),     // b3.c2
    (FC * FC * 9, FC),     // b3.c3
    (FC * FC * 9, FC),     // b4.c1
    (FC * FC * 9, FC),     // b4.c2
    (FC * FC * 9, FC),     // b4.c3
    (FC * FC * 9, FC),     // b5.c1
    (FC * FC * 9, FC),     // b5.c2
    (FC * FC * 9, FC),     // b5.c3
    (CAT_CH * FC + 0, FC), // conv_cat (1x1); conv_2 handled below
];
// conv_2 is the 18th tensor: (NEAR_CH*FC*9 weights, NEAR_CH bias).

/// IEEE 754 half → f32 (safe, handles subnormals/inf/nan).
#[inline]
#[doc(hidden)] // SPANF research surface — NOT part of the contract
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let man = (bits & 0x3ff) as u32;
    let f = match (exp, man) {
        (0, 0) => sign << 31,
        (0, m) => {
            // subnormal: normalize
            let shift = m.leading_zeros() - 21;
            let m2 = (m << (shift + 1)) & 0x3ff;
            let e2 = 127 - 15 - shift;
            (sign << 31) | (e2 << 23) | (m2 << 13)
        }
        (0x1f, 0) => (sign << 31) | 0x7f80_0000,
        (0x1f, _) => (sign << 31) | 0x7fc0_0000,
        (e, m) => (sign << 31) | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(f)
}

fn tensor_dims() -> [(usize, usize); 18] {
    let mut dims = [(0usize, 0usize); 18];
    dims[..17].copy_from_slice(&TENSOR_LAYOUT);
    dims[17] = (NEAR_CH * FC * 9, NEAR_CH);
    dims
}

/// Decode an ALL-f16 weight file (every tensor f16) into f32 — the adopted-
/// model ship format (halves file size; f16 weights measured transparent).
pub fn decode_all_f16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Decode the f16 weight dump (weights f16, biases f32) into the canonical
/// TOTAL_FLOATS f32 buffer accepted by `SpanfWeights::parse`.
#[doc(hidden)] // SPANF research surface — NOT part of the contract
pub fn decode_f16_weights(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let mut out = Vec::with_capacity(TOTAL_FLOATS);
    let mut off = 0usize;
    for (wn, bn) in tensor_dims() {
        let wbytes = wn * 2;
        if off + wbytes > bytes.len() {
            return Err("f16 file truncated (weights)".into());
        }
        for c in bytes[off..off + wbytes].chunks_exact(2) {
            out.push(f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
        }
        off += wbytes;
        let bbytes = bn * 4;
        if off + bbytes > bytes.len() {
            return Err("f16 file truncated (bias)".into());
        }
        for c in bytes[off..off + bbytes].chunks_exact(4) {
            out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        off += bbytes;
    }
    if off != bytes.len() || out.len() != TOTAL_FLOATS {
        return Err(format!(
            "f16 file size mismatch: consumed {off}/{} -> {} floats",
            bytes.len(),
            out.len()
        ));
    }
    Ok(out)
}

/// Decode the int8-per-channel dump (per conv: cout f32 scales, then int8
/// weights; biases f32) into the canonical f32 buffer. NOTE: accuracy study
/// shows this is NOT production-viable for SPANF (~35 dB); kept for size demo.
#[doc(hidden)] // SPANF research surface — NOT part of the contract
pub fn decode_int8pc_weights(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let mut out = Vec::with_capacity(TOTAL_FLOATS);
    let mut off = 0usize;
    for (wn, bn) in tensor_dims() {
        // cout = weights / (cin*9) — recover from layout: bias count equals
        // cout except conv_near (48) — handle via explicit table instead.
        let cout = if bn > 0 { bn } else { NEAR_CH };
        let sbytes = cout * 4;
        if off + sbytes + wn > bytes.len() {
            return Err("int8 file truncated".into());
        }
        let scales: Vec<f32> = bytes[off..off + sbytes]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        off += sbytes;
        let per_oc = wn / cout;
        for (i, &b) in bytes[off..off + wn].iter().enumerate() {
            out.push((b as i8) as f32 * scales[i / per_oc]);
        }
        off += wn;
        for c in bytes[off..off + bn * 4].chunks_exact(4) {
            out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        off += bn * 4;
    }
    if off != bytes.len() || out.len() != TOTAL_FLOATS {
        return Err("int8 file size mismatch".into());
    }
    Ok(out)
}

/// Repack a [cout][cin][3][3] conv weight into quad-major iteration order:
/// per output-channel-quad q, per (ic, ky): 12 floats [ob0 t0,t1,t2, ob1 t0,..].
pub(crate) fn pack_conv3x3(wts: &[f32], cin: usize, cout: usize) -> Vec<f32> {
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
pub(crate) fn pack_conv1x1(wts: &[f32], cin_total: usize, cout: usize) -> Vec<f32> {
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

/// Owned, pre-packed model: build once, share across threads (`Sync`).
#[doc(hidden)] // SPANF research surface — NOT part of the contract
pub struct SpanfModel {
    raw: Vec<f32>,
    pub(crate) packed_blocks: [[Vec<f32>; 3]; 5],
    pub(crate) packed_cat: Vec<f32>,
    pub(crate) packed_conv2: Vec<f32>,
}

impl SpanfModel {
    pub fn new(raw: Vec<f32>) -> Result<Self, String> {
        let w = SpanfWeights::parse(&raw)?;
        let packed_blocks = core::array::from_fn(|bi| {
            let cin0 = if bi == 0 { 3 } else { FC };
            core::array::from_fn(|ci| {
                let cin = if ci == 0 { cin0 } else { FC };
                pack_conv3x3(w.blocks[bi][ci].0, cin, FC)
            })
        });
        let packed_cat = pack_conv1x1(w.conv_cat_w, CAT_CH, FC);
        let packed_conv2 = pack_conv3x3(w.conv2_w, FC, NEAR_CH);
        Ok(SpanfModel {
            raw,
            packed_blocks,
            packed_cat,
            packed_conv2,
        })
    }

    pub fn from_f16_bytes(bytes: &[u8]) -> Result<Self, String> {
        Self::new(decode_f16_weights(bytes)?)
    }

    pub(crate) fn weights(&self) -> SpanfWeights<'_> {
        SpanfWeights::parse(&self.raw).expect("validated at construction")
    }
}

/// Per-thread work buffers for one (h, w) tile shape.
#[doc(hidden)] // SPANF research surface — NOT part of the contract
pub struct Scratch {
    pub(crate) h: usize,
    pub(crate) w: usize,
    /// channel stride (identity for this probe)
    pub(crate) cs: usize,
    pub(crate) inp3: Vec<f32>,
    pub(crate) near: Vec<f32>,
    pub(crate) b1: Vec<f32>,
    pub(crate) tmp: Vec<f32>,
    pub(crate) b_out: Vec<f32>,
    pub(crate) cur: Vec<f32>,
    pub(crate) catd: Vec<f32>,
    pub(crate) pre: Vec<f32>,
}

impl Scratch {
    pub fn new(h: usize, w: usize) -> Self {
        let plane = h * w;
        // Channel stride. NOTE 2026-07-22: +16 padding at 4KiB-multiple planes
        // was MEASURED 2x SLOWER at 128^2/256^2 (137 vs 68 ms) — the L1-set
        // aliasing hypothesis is falsified for this access pattern. Identity
        // stride; the cs plumbing stays (tiling/layout flexibility).
        let cs = plane;
        Scratch {
            h,
            w,
            cs,
            inp3: vec![0.0; 3 * cs],
            near: vec![0.0; NEAR_CH * cs],
            b1: vec![0.0; FC * cs],
            tmp: vec![0.0; FC * cs],
            b_out: vec![0.0; FC * cs],
            cur: vec![0.0; FC * cs],
            catd: vec![0.0; FC * cs],
            pre: vec![0.0; NEAR_CH * cs],
        }
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
fn conv3x3(
    inp: &[f32],
    out: &mut [f32],
    w: &[f32],
    b: &[f32],
    cin: usize,
    cout: usize,
    h: usize,
    wd: usize,
) {
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
        conv3x3_acc_plane(
            &inp[ic * plane..(ic + 1) * plane],
            orow,
            &w[oc * 9..oc * 9 + 9],
            h,
            wd,
        );
    }
}

fn conv1x1(
    inp: &[f32],
    out: &mut [f32],
    w: &[f32],
    b: &[f32],
    cin: usize,
    cout: usize,
    plane: usize,
) {
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
fn spab(
    inp: &[f32],
    out: &mut [f32],
    tmp: &mut [f32],
    convs: &[(&[f32], &[f32]); 3],
    cin: usize,
    h: usize,
    wd: usize,
) {
    let plane = h * wd;
    conv3x3(inp, out, convs[0].0, convs[0].1, cin, FC, h, wd);
    silu_inplace(&mut out[..FC * plane]);
    conv3x3(
        &out[..FC * plane],
        tmp,
        convs[1].0,
        convs[1].1,
        FC,
        FC,
        h,
        wd,
    );
    silu_inplace(&mut tmp[..FC * plane]);
    conv3x3(
        &tmp[..FC * plane],
        out,
        convs[2].0,
        convs[2].1,
        FC,
        FC,
        h,
        wd,
    );
    if cin == FC {
        // gated: out = (out3 + inp) * (sigmoid(out3) - 0.5); reuse tmp as scratch
        tmp[..FC * plane].copy_from_slice(&out[..FC * plane]);
        gate(
            &tmp[..FC * plane],
            &inp[..FC * plane],
            &mut out[..FC * plane],
        );
    }
}

/// Scale-generic strided pixel shuffle: [3*s*s, h, w] (stride cs) -> [3, s*h, s*w].
pub(crate) fn pixel_shuffle_s_strided(
    inp: &[f32],
    cstride: usize,
    out: &mut [f32],
    h: usize,
    wd: usize,
    s: usize,
) {
    let plane = h * wd;
    let s2 = s * s;
    let (oh, ow) = (h * s, wd * s);
    for c in 0..3 {
        for ry in 0..s {
            for rx in 0..s {
                let ip = &inp[(c * s2 + ry * s + rx) * cstride..][..plane];
                for y in 0..h {
                    let obase = c * oh * ow + (y * s + ry) * ow + rx;
                    let irow = &ip[y * wd..(y + 1) * wd];
                    for x in 0..wd {
                        out[obase + x * s] = irow[x];
                    }
                }
            }
        }
    }
}

/// out ([3, s*h, s*w]) += nearest-neighbor upsample of base ([3, h, w] tight).
pub(crate) fn nearest_add(base: &[f32], out: &mut [f32], h: usize, wd: usize, s: usize) {
    let (oh, ow) = (h * s, wd * s);
    for c in 0..3 {
        for oy in 0..oh {
            let brow = &base[c * h * wd + (oy / s) * wd..][..wd];
            let orow = &mut out[c * oh * ow + oy * ow..][..ow];
            for ox in 0..ow {
                orow[ox] += brow[ox / s];
            }
        }
    }
}

pub(crate) fn pixel_shuffle4_strided(
    inp: &[f32],
    cstride: usize,
    out: &mut [f32],
    h: usize,
    wd: usize,
) {
    let plane = h * wd;
    let (oh, ow) = (h * 4, wd * 4);
    for c in 0..3 {
        for ry in 0..4 {
            for rx in 0..4 {
                let ip = &inp[(c * S2 + ry * 4 + rx) * cstride..][..plane];
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

/// Stage-level non-finite reporting, active only with the `nan-debug` feature.
#[cfg(feature = "nan-debug")]
pub(crate) fn nan_debug(stage: &str, data: &[f32]) {
    let bad = data.iter().filter(|v| !v.is_finite()).count();
    let first = data.iter().position(|v| !v.is_finite());
    let mx = data.iter().cloned().fold(
        0.0f32,
        |a, v| if v.is_finite() { a.max(v.abs()) } else { a },
    );
    eprintln!(
        "nan-debug {stage}: nonfinite={bad}/{} first={first:?} max_abs={mx:.3e}",
        data.len()
    );
}
#[cfg(not(feature = "nan-debug"))]
#[inline(always)]
pub(crate) fn nan_debug(_stage: &str, _data: &[f32]) {}

/// Full SPANF x4 forward. `input` is [3,h,w] NCHW f32; returns [3,4h,4w].
#[doc(hidden)] // SPANF research surface — NOT part of the contract
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
    conv1x1(
        &cat,
        &mut catd,
        w.conv_cat_w,
        w.conv_cat_b,
        CAT_CH,
        FC,
        plane,
    );
    let mut pre = vec![0.0f32; NEAR_CH * plane];
    conv3x3(&catd, &mut pre, w.conv2_w, w.conv2_b, FC, NEAR_CH, h, wd);

    let mut out = vec![0.0f32; 3 * plane * 16];
    pixel_shuffle4(&pre, &mut out, h, wd);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    // internal paths so unit tests build without the `internals` re-exports
    #[allow(unused_imports)]
    use crate::{simd::spanf_x4_simd, tiled::spanf_x4_tiled};

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
    fn arbitrary_dims_simd_vs_scalar() {
        // Adversarial shapes: below/at/above the f32x8 (W=8) and f32x16 (W=16)
        // vector thresholds, 1-px strips, primes, non-squares.
        let wbuf = lcg(TOTAL_FLOATS, 777, 0.12);
        let w = SpanfWeights::parse(&wbuf).unwrap();
        for &(h, wd) in &[
            (1usize, 1usize),
            (1, 9),
            (2, 3),
            (3, 17),
            (5, 18),
            (8, 19),
            (17, 2),
            (7, 33),
            (13, 13),
            (61, 47),
            (16, 16),
            (18, 18),
        ] {
            let input = lcg(3 * h * wd, (h * 131 + wd) as u32, 1.0);
            let a = spanf_x4(&input, h, wd, &w);
            let b = spanf_x4_simd(&input, h, wd, &w);
            assert_close_finite(&b, &a, 5e-4, &format!("dims {h}x{wd}"));
        }
    }

    #[test]
    fn tiled_arbitrary_dims_and_edges() {
        let wbuf = lcg(TOTAL_FLOATS, 555, 0.12);
        let model = SpanfModel::new(wbuf.clone()).unwrap();
        let w = SpanfWeights::parse(&wbuf).unwrap();
        // (h, w, tile): clamped edge tiles down to 1-px cores (65 % 32 = 1),
        // tile larger than the image, tiny images.
        for &(h, wd, tile) in &[
            (65usize, 33usize, 32usize), // 1-px core column+row at the edges
            (33, 65, 32),
            (40, 40, 64), // single tile > image
            (5, 90, 32),  // short strip, many tiles
            (90, 5, 32),  // narrow strip
            (17, 17, 32), // tiny single tile
            (1, 90, 32),  // 1-row image split across tiles
        ] {
            let input = lcg(3 * h * wd, (h * 977 + wd) as u32, 1.0);
            let whole = spanf_x4_simd(&input, h, wd, &w);
            for threads in [1usize, 3] {
                let tiled = spanf_x4_tiled(&model, &input, h, wd, threads, tile);
                assert_close_finite(
                    &tiled,
                    &whole,
                    1e-4,
                    &format!("tiled {h}x{wd} tile{tile} t{threads}"),
                );
            }
        }
    }

    #[cfg(feature = "px")]
    #[test]
    fn px_strided_matches_tight_and_raw() {
        use zenpixels::{PixelDescriptor, PixelSlice};
        let wbuf = lcg(TOTAL_FLOATS, 4321, 0.12);
        let model = SpanfModel::new(wbuf.clone()).unwrap();
        let wv = SpanfWeights::parse(&wbuf).unwrap();
        let (h, wd) = (23usize, 31usize);
        let inter = lcg(3 * h * wd, 99, 1.0);
        let mut planar = vec![0f32; 3 * h * wd];
        for y in 0..h {
            for x in 0..wd {
                for c in 0..3 {
                    planar[c * h * wd + y * wd + x] = inter[(y * wd + x) * 3 + c];
                }
            }
        }
        let want = spanf_x4_simd(&planar, h, wd, &wv);

        let bytes: &[u8] = bytemuck::cast_slice(&inter);
        let tight =
            PixelSlice::new(bytes, wd as u32, h as u32, wd * 12, PixelDescriptor::RGBF32).unwrap();
        let out_t = px::spanf_x4_px(&model, &tight, 2, 32).unwrap();

        // Same pixels behind a padded stride (5 extra px per row).
        let sw = wd + 5;
        let mut sdata = vec![0f32; 3 * sw * h];
        for y in 0..h {
            sdata[y * sw * 3..y * sw * 3 + wd * 3]
                .copy_from_slice(&inter[y * wd * 3..(y + 1) * wd * 3]);
        }
        let sbytes: &[u8] = bytemuck::cast_slice(&sdata);
        let strided = PixelSlice::new(
            sbytes,
            wd as u32,
            h as u32,
            sw * 12,
            PixelDescriptor::RGBF32,
        )
        .unwrap();
        let out_s = px::spanf_x4_px(&model, &strided, 2, 32).unwrap();

        let (oh, ow) = (4 * h, 4 * wd);
        let (vt, vs) = (out_t.as_slice(), out_s.as_slice());
        assert_eq!(vt.width() as usize, ow);
        assert_eq!(vt.rows() as usize, oh);
        let mut max = 0f32;
        for y in 0..oh {
            let rt: &[f32] = bytemuck::cast_slice(vt.row(y as u32));
            let rs: &[f32] = bytemuck::cast_slice(vs.row(y as u32));
            assert_eq!(rt, rs, "strided vs tight must be byte-identical (row {y})");
            for x in 0..ow {
                for c in 0..3 {
                    let d = (rt[3 * x + c] - want[c * oh * ow + y * ow + x]).abs();
                    assert!(d.is_finite());
                    max = max.max(d);
                }
            }
        }
        assert!(max < 1e-4, "px vs raw max {max}");
    }

    #[test]
    fn tiled_matches_whole_image() {
        let wbuf = lcg(TOTAL_FLOATS, 12345, 0.12);
        let model = SpanfModel::new(wbuf.clone()).unwrap();
        let w = SpanfWeights::parse(&wbuf).unwrap();
        let (h, wd) = (75usize, 90usize); // non-multiples: edge tiles + interior seams
        let input = lcg(3 * h * wd, 4242, 1.0);
        let whole = spanf_x4_simd(&input, h, wd, &w);
        for threads in [1usize, 3] {
            let tiled = spanf_x4_tiled(&model, &input, h, wd, threads, 40);
            assert_close_finite(&tiled, &whole, 1e-4, &format!("tiled t{threads} vs whole"));
        }
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
