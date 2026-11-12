//! Adopted-model support: SRVGGNetCompact ("compact") and official SPAN-48
//! ("span48") graphs running on the shared kernel set, with a generic
//! exact-halo tiled runner.
//!
//! Weight files come from tools/dump_adopted.py (fixed order, f32 LE);
//! Conv3XC branches are pre-merged and verified at dump time.
#![allow(clippy::too_many_arguments)]

use crate::simd::{
    conv1x1_gen_dispatch, conv3x3_packed_dispatch_with_scratch, conv3x3_wino_dispatch_with_scratch,
    gate_all_dispatch, prelu_dispatch, silu_all_dispatch,
};
use crate::{nearest_add, pack_conv1x1, pack_conv3x3, pixel_shuffle_s_strided};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Colorspace the model was trained in. Graph-identical either way; the
/// caller must feed planes in the matching space (see zensr-zenjpeg).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[non_exhaustive]
pub enum ModelSpace {
    #[default]
    Rgb,
    /// JFIF full-range YCbCr, [0,1] planes, Cb/Cr offset +0.5.
    Ycbcr,
}

pub enum Arch {
    /// [conv3x3+PReLU] x (nc+1) -> conv3x3(3*s^2) -> shuffle -> +nearest(x)
    Compact { nf: usize, nc: usize },
    /// conv_1 + 6 gated SPAB blocks + conv_2 + 4-way cat + conv1x1 + upsampler
    Span48,
}

pub struct AdoptedModel {
    pub arch: Arch,
    pub scale: usize,
    space: ModelSpace,
    /// packed conv weights in graph order
    packed: Vec<Vec<f32>>,
    biases: Vec<Vec<f32>>,
    /// per-channel PReLU slopes (compact only), in order
    slopes: Vec<Vec<f32>>,
    /// Winograd F(2x2,3x3) data for the nf->nf middle layers: (U transformed
    /// [16][nf][nf], raw [cout][cin][9] for scalar borders). Empty when
    /// disabled (ZENSR_WINOGRAD=0) or non-applicable.
    wino: Vec<Option<(Vec<f32>, Vec<f32>)>>,
}

const SPAN_FC: usize = 48;

fn take<'a>(buf: &mut &'a [f32], n: usize) -> &'a [f32] {
    let (h, t) = buf.split_at(n);
    *buf = t;
    h
}

impl AdoptedModel {
    pub fn space(&self) -> ModelSpace {
        self.space
    }

    pub fn set_space(&mut self, space: ModelSpace) {
        self.space = space;
    }

    /// Receptive-field radius in input pixels (number of chained 3x3 convs),
    /// as far as bit-exact tiling is concerned: a Winograd layer
    /// (`ZENSR_WINOGRAD=1`) computes each output from its tile's 4x4 input
    /// patch, so it reaches 2 pixels instead of 1.
    pub fn halo(&self) -> usize {
        match self.arch {
            Arch::Compact { nc, .. } => nc + 2 + self.wino.iter().filter(|w| w.is_some()).count(),
            Arch::Span48 => 21,
        }
    }

    pub fn load_compact(raw: &[f32], nf: usize, nc: usize, scale: usize) -> Result<Self, String> {
        let s2 = scale * scale;
        let expect =
            (3 * nf * 9 + nf + nf) + nc * (nf * nf * 9 + nf + nf) + (nf * 3 * s2 * 9 + 3 * s2);
        if raw.len() != expect {
            return Err(format!(
                "compact: expected {expect} floats, got {}",
                raw.len()
            ));
        }
        // The 3x3 kernel computes output channels in quads and holds at most 64
        // input channels in its row table (`cin * 3 <= 192`). Any other nf used to
        // load and then panic inside the first forward.
        if nf < 4 || !nf.is_multiple_of(4) || nf > 64 {
            return Err(format!(
                "compact: nf {nf} cannot run (needs a multiple of 4 in 4..=64)"
            ));
        }
        // final cout must be a multiple of 4 for the quad-blocked kernel; pad
        // with zero output channels (s=1: 3->4, s=3: 27->28). The pixel shuffle
        // reads only the first 3*s^2 channels, so pad rows are never consumed.
        let cout = 3 * s2;
        let cpad = cout.next_multiple_of(4);
        let mut buf = raw;
        let mut packed = Vec::new();
        let mut biases = Vec::new();
        let mut slopes = Vec::new();
        // first conv 3->nf + prelu
        packed.push(pack_conv3x3(take(&mut buf, 3 * nf * 9), 3, nf));
        biases.push(take(&mut buf, nf).to_vec());
        slopes.push(take(&mut buf, nf).to_vec());
        // Winograd stays OFF by default: it is measurably slower than the
        // magetypes direct kernel, and that is now a settled result rather than
        // a v1 artifact.
        //
        // History, because the obvious reading of this flag is wrong: v1 was
        // scalar transforms + an untiered GEMM and lost 1.8x on lianli, which is
        // why the flag exists. **v2/v3 (b714829) fixed exactly that** —
        // magetypes-tiered, deinterleaved vector transforms, quad GEMM,
        // row-wide U amortization, in simd.rs::conv3x3_wino_dispatch. Since
        // 2026-10-18 that kernel runs every shape (zero-padded lanes, tile grid
        // anchored to the image) and wino.rs keeps only the weight transform,
        // the direct border pixel and a test reference.
        //
        // RE-MEASURED 2026-09-08 against the TIERED v2/v3 on AVX-512 (7950X,
        // v4x, end-to-end prod_bench with dejpeg_rt24g, wd>=1024 so nt=511 >> W
        // and the vector path is the one running): 1.78x slower at 1 thread,
        // 1.73x at 12. Vectorizing the transforms did not close the gap — the
        // 2.25x multiply reduction simply does not pay for the transform and
        // scatter/gather overhead at this model's channel count (nf=24).
        // So this is not a "needs tiering" TODO; it is a negative result.
        // benchmarks/realtime_kernels_x86_2026-09-08.md
        let use_wino = std::env::var("ZENSR_WINOGRAD").as_deref() == Ok("1");
        let mut wino: Vec<Option<(Vec<f32>, Vec<f32>)>> = vec![None];
        for _ in 0..nc {
            let wraw = take(&mut buf, nf * nf * 9);
            packed.push(pack_conv3x3(wraw, nf, nf));
            if use_wino {
                wino.push(Some((
                    crate::wino::wino_weights(wraw, nf, nf),
                    wraw.to_vec(),
                )));
            } else {
                wino.push(None);
            }
            biases.push(take(&mut buf, nf).to_vec());
            slopes.push(take(&mut buf, nf).to_vec());
        }
        wino.push(None); // final conv stays direct
        let wfin = take(&mut buf, nf * cout * 9);
        if cpad == cout {
            packed.push(pack_conv3x3(wfin, nf, cout));
        } else {
            let mut wp = vec![0.0f32; cpad * nf * 9];
            wp[..wfin.len()].copy_from_slice(wfin);
            packed.push(pack_conv3x3(&wp, nf, cpad));
        }
        let mut bfin = take(&mut buf, cout).to_vec();
        bfin.resize(cpad, 0.0);
        biases.push(bfin);
        debug_assert!(buf.is_empty());
        Ok(AdoptedModel {
            arch: Arch::Compact { nf, nc },
            scale,
            space: ModelSpace::Rgb,
            packed,
            biases,
            slopes,
            wino,
        })
    }

    pub fn load_span48(raw: &[f32], scale: usize) -> Result<Self, String> {
        let fc = SPAN_FC;
        let s2 = scale * scale;
        let expect = (3 * fc * 9 + fc)
            + 18 * (fc * fc * 9 + fc)
            + (fc * fc * 9 + fc)
            + (4 * fc * fc + fc)
            + (fc * 3 * s2 * 9 + 3 * s2);
        if raw.len() != expect {
            return Err(format!(
                "span48: expected {expect} floats, got {}",
                raw.len()
            ));
        }
        // The upsampler's 3*s^2 output channels go through the quad-blocked 3x3
        // kernel unpadded, so a scale whose count is not a multiple of 4 (1, 3)
        // used to load and then panic inside the first forward.
        if !(3 * s2).is_multiple_of(4) {
            return Err(format!(
                "span48: scale {scale} gives {} output channels, which cannot run (needs a multiple of 4)",
                3 * s2
            ));
        }
        let mut buf = raw;
        let mut packed = Vec::new();
        let mut biases = Vec::new();
        packed.push(pack_conv3x3(take(&mut buf, 3 * fc * 9), 3, fc));
        biases.push(take(&mut buf, fc).to_vec());
        for _ in 0..18 {
            packed.push(pack_conv3x3(take(&mut buf, fc * fc * 9), fc, fc));
            biases.push(take(&mut buf, fc).to_vec());
        }
        packed.push(pack_conv3x3(take(&mut buf, fc * fc * 9), fc, fc)); // conv_2
        biases.push(take(&mut buf, fc).to_vec());
        packed.push(pack_conv1x1(take(&mut buf, 4 * fc * fc), 4 * fc, fc)); // conv_cat
        biases.push(take(&mut buf, fc).to_vec());
        packed.push(pack_conv3x3(take(&mut buf, fc * 3 * s2 * 9), fc, 3 * s2)); // upsampler
        biases.push(take(&mut buf, 3 * s2).to_vec());
        debug_assert!(buf.is_empty());
        Ok(AdoptedModel {
            arch: Arch::Span48,
            scale,
            space: ModelSpace::Rgb,
            packed,
            biases,
            slopes: Vec::new(),
            wino: Vec::new(),
        })
    }
}

#[derive(Default, Debug)]
struct TileScratch {
    pin_buf: Vec<f32>,
    wino_buf: Vec<f32>,
    inp3: Vec<f32>,
    cur: Vec<f32>,
    nxt: Vec<f32>,
    pre: Vec<f32>,
    feat: Vec<f32>,
    b_prev: Vec<f32>,
    b1_keep: Vec<f32>,
    b6_o1: Vec<f32>,
    catd: Vec<f32>,
    ext_in: Vec<f32>,
    tile_out: Vec<f32>,
}

impl TileScratch {
    fn reserve_for(&mut self, model: &AdoptedModel, eh: usize, ew: usize, tiled: bool) {
        let plane = eh * ew;
        if plane == 0 {
            return;
        }
        let s2 = model.scale * model.scale;
        let pwd = (ew + 18).max(32);
        let pcs = 3 * pwd;
        match model.arch {
            Arch::Compact { nf, .. } => {
                let cpad = (3 * s2).next_multiple_of(4);
                let max_cin = nf.max(3);
                ensure_len(&mut self.pin_buf, max_cin * pcs);
                if model.wino.iter().any(|w| w.is_some()) && eh >= 3 && ew >= 3 {
                    let nt = ew / 2 + 1;
                    let ntp = nt.div_ceil(16) * 16;
                    let half = ntp + 1;
                    let wino_need = nf * 8 * half + 32 * nf * ntp;
                    ensure_len(&mut self.wino_buf, wino_need);
                }
                ensure_len(&mut self.cur, nf * plane);
                ensure_len(&mut self.nxt, nf * plane);
                ensure_len(&mut self.pre, cpad * plane);
            }
            Arch::Span48 => {
                let fc = SPAN_FC;
                ensure_len(&mut self.pin_buf, fc * pcs);
                ensure_len(&mut self.inp3, 3 * plane);
                ensure_len(&mut self.feat, fc * plane);
                ensure_len(&mut self.b_prev, fc * plane);
                ensure_len(&mut self.b1_keep, fc * plane);
                ensure_len(&mut self.b6_o1, fc * plane);
                ensure_len(&mut self.cur, fc * plane);
                ensure_len(&mut self.nxt, fc * plane);
                ensure_len(&mut self.catd, fc * plane);
                ensure_len(&mut self.pre, 3 * s2 * plane);
            }
        }
        if tiled {
            ensure_len(&mut self.ext_in, 3 * plane);
            ensure_len(&mut self.tile_out, 3 * plane * s2);
        }
    }
}

#[inline]
fn ensure_len(buf: &mut Vec<f32>, len: usize) -> &mut [f32] {
    if buf.len() < len {
        buf.resize(len, 0.0);
    }
    &mut buf[..len]
}

/// Reusable scratch buffers for zero-allocation [`AdoptedModel`] inference.
///
/// Once warmed on a given tile or image extent (or pre-sized with
/// [`ModelWorkspace::with_tile_capacity`]), single-threaded calls to
/// [`AdoptedModel::forward_with_workspace`] and
/// [`AdoptedModel::upscale_tiled_into`] at equal or smaller tile sizes perform
/// zero heap allocations.
#[derive(Default, Debug)]
pub struct ModelWorkspace {
    primary: TileScratch,
    workers: Vec<TileScratch>,
}

impl ModelWorkspace {
    /// Creates an empty workspace that allocates lazily on first use.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-allocates single-thread scratch for tiles up to `max_tile_h × max_tile_w`
    /// (plus `2 * model.halo()` when `tiled` halo extraction may be used).
    #[must_use]
    pub fn with_tile_capacity(model: &AdoptedModel, max_tile_h: usize, max_tile_w: usize) -> Self {
        let mut ws = Self::new();
        let halo = model.halo();
        let eh = max_tile_h.saturating_add(2 * halo);
        let ew = max_tile_w.saturating_add(2 * halo);
        ws.primary.reserve_for(model, eh, ew, true);
        ws
    }
}

impl AdoptedModel {
    /// Whole-tile forward: input [3,h,w] tight -> out [3,s*h,s*w] tight.
    pub fn forward(&self, input: &[f32], h: usize, w: usize, out: &mut [f32]) {
        let mut ws = ModelWorkspace::new();
        self.forward_with_workspace(input, h, w, out, &mut ws);
    }

    /// Whole-tile forward reusing `ws` scratch buffers.
    pub fn forward_with_workspace(
        &self,
        input: &[f32],
        h: usize,
        w: usize,
        out: &mut [f32],
        ws: &mut ModelWorkspace,
    ) {
        self.forward_at_with_scratch(input, h, w, out, 0, 0, &mut ws.primary);
    }

    /// `forward` on a window whose column 0 / row 0 sit at image x / y parity
    /// `ox` / `oy`. Only the Winograd layers use it: they anchor their 2x2 tile
    /// grid to the image, so a tile window and the whole image compute every
    /// kept pixel the same way.
    fn forward_at_with_scratch(
        &self,
        input: &[f32],
        h: usize,
        w: usize,
        out: &mut [f32],
        ox: usize,
        oy: usize,
        sc: &mut TileScratch,
    ) {
        let plane = h * w;
        let cs = plane; // identity stride (padding measured 2x slower — see PLAN appendix)
        let s = self.scale;
        assert_eq!(input.len(), 3 * plane);
        assert_eq!(out.len(), 3 * plane * s * s);
        if plane == 0 {
            // No rows or no columns: the upscaled image is empty as well (`out`
            // is checked to be empty above). The kernels assume one row to fill.
            return;
        }
        match self.arch {
            Arch::Compact { nf, nc } => {
                let s2 = s * s;
                let span3 = 2 * cs + plane;
                let span_nf = (nf - 1) * cs + plane;
                let TileScratch {
                    pin_buf,
                    wino_buf,
                    cur: cur_vec,
                    nxt: nxt_vec,
                    pre: pre_vec,
                    ..
                } = sc;
                let mut cur = ensure_len(cur_vec, nf * cs);
                let mut nxt = ensure_len(nxt_vec, nf * cs);
                conv3x3_packed_dispatch_with_scratch(
                    &input[..span3],
                    3,
                    &self.packed[0],
                    &self.biases[0],
                    cur,
                    nf,
                    h,
                    w,
                    cs,
                    pin_buf,
                );
                prelu_dispatch(&mut cur[..span_nf], &self.slopes[0], nf, plane, cs);
                for i in 0..nc {
                    match self.wino.get(1 + i).and_then(|o| o.as_ref()) {
                        Some((u, raw)) => conv3x3_wino_dispatch_with_scratch(
                            &cur[..span_nf],
                            nf,
                            u,
                            raw,
                            &self.biases[1 + i],
                            nxt,
                            nf,
                            h,
                            w,
                            cs,
                            ox,
                            oy,
                            wino_buf,
                        ),
                        None => conv3x3_packed_dispatch_with_scratch(
                            &cur[..span_nf],
                            nf,
                            &self.packed[1 + i],
                            &self.biases[1 + i],
                            nxt,
                            nf,
                            h,
                            w,
                            cs,
                            pin_buf,
                        ),
                    }
                    prelu_dispatch(&mut nxt[..span_nf], &self.slopes[1 + i], nf, plane, cs);
                    core::mem::swap(&mut cur, &mut nxt);
                }
                let cpad = (3 * s2).next_multiple_of(4);
                let pre = ensure_len(pre_vec, cpad * cs);
                conv3x3_packed_dispatch_with_scratch(
                    &cur[..span_nf],
                    nf,
                    &self.packed[1 + nc],
                    &self.biases[1 + nc],
                    pre,
                    cpad,
                    h,
                    w,
                    cs,
                    pin_buf,
                );
                pixel_shuffle_s_strided(pre, cs, out, h, w, s);
                nearest_add(input, out, h, w, s);
            }
            Arch::Span48 => {
                let fc = SPAN_FC;
                let s2 = s * s;
                // Official SPAN input norm: (x - rgb_mean) * 255, applied BEFORE the
                // convs so zero-padding at image borders equals mean gray (folding the
                // norm into conv_1 gets borders wrong — measured 0.32; don't re-attempt).
                const SPAN_MEAN: [f32; 3] = [0.4488, 0.4371, 0.4040];
                const SPAN_IMG_RANGE: f32 = 255.0;
                let TileScratch {
                    pin_buf,
                    inp3: inp3_vec,
                    feat: feat_vec,
                    b_prev: b_prev_vec,
                    b1_keep: b1_keep_vec,
                    b6_o1: b6_o1_vec,
                    cur: t1_vec,
                    nxt: t2_vec,
                    catd: catd_vec,
                    pre: pre_vec,
                    ..
                } = sc;
                let inp3 = ensure_len(inp3_vec, 3 * cs);
                for c in 0..3 {
                    for (d, sv) in inp3[c * cs..c * cs + plane]
                        .iter_mut()
                        .zip(&input[c * plane..(c + 1) * plane])
                    {
                        *d = (sv - SPAN_MEAN[c]) * SPAN_IMG_RANGE;
                    }
                }
                let span3 = 2 * cs + plane;
                let span_fc = (fc - 1) * cs + plane;
                let feat = ensure_len(feat_vec, fc * cs);
                conv3x3_packed_dispatch_with_scratch(
                    &inp3[..span3],
                    3,
                    &self.packed[0],
                    &self.biases[0],
                    feat,
                    fc,
                    h,
                    w,
                    cs,
                    pin_buf,
                );

                let mut b_prev = ensure_len(b_prev_vec, fc * cs);
                b_prev[..span_fc].copy_from_slice(&feat[..span_fc]);
                let b1_keep = ensure_len(b1_keep_vec, fc * cs);
                let b6_o1 = ensure_len(b6_o1_vec, fc * cs);
                let t1 = ensure_len(t1_vec, fc * cs);
                let mut t2 = ensure_len(t2_vec, fc * cs);
                for blk in 0..6 {
                    let base = 1 + blk * 3;
                    conv3x3_packed_dispatch_with_scratch(
                        &b_prev[..span_fc],
                        fc,
                        &self.packed[base],
                        &self.biases[base],
                        t1,
                        fc,
                        h,
                        w,
                        cs,
                        pin_buf,
                    );
                    silu_all_dispatch(&mut t1[..span_fc]);
                    // official SPAB uses SiLU(inplace=True): the out1 returned to the
                    // final concat is the POST-activation tensor (keep for block 6)
                    if blk == 5 {
                        b6_o1[..span_fc].copy_from_slice(&t1[..span_fc]);
                    }
                    conv3x3_packed_dispatch_with_scratch(
                        &t1[..span_fc],
                        fc,
                        &self.packed[base + 1],
                        &self.biases[base + 1],
                        t2,
                        fc,
                        h,
                        w,
                        cs,
                        pin_buf,
                    );
                    silu_all_dispatch(&mut t2[..span_fc]);
                    conv3x3_packed_dispatch_with_scratch(
                        &t2[..span_fc],
                        fc,
                        &self.packed[base + 2],
                        &self.biases[base + 2],
                        t1,
                        fc,
                        h,
                        w,
                        cs,
                        pin_buf,
                    );
                    // gate: out = (o3 + x) * (sigmoid(o3) - 0.5)   (always, official SPAB)
                    gate_all_dispatch(&t1[..span_fc], &b_prev[..span_fc], &mut t2[..span_fc]);
                    core::mem::swap(&mut b_prev, &mut t2);
                    if blk == 0 {
                        b1_keep[..span_fc].copy_from_slice(&b_prev[..span_fc]);
                    }
                }
                // conv_2 on b6
                conv3x3_packed_dispatch_with_scratch(
                    &b_prev[..span_fc],
                    fc,
                    &self.packed[19],
                    &self.biases[19],
                    t1,
                    fc,
                    h,
                    w,
                    cs,
                    pin_buf,
                );
                // cat [feat, conv_2(b6), b1, b6_o1] -> conv1x1 -> upsampler
                let catd = ensure_len(catd_vec, fc * cs);
                conv1x1_gen_dispatch(
                    &[
                        (&feat[..span_fc], fc),
                        (&t1[..span_fc], fc),
                        (&b1_keep[..span_fc], fc),
                        (&b6_o1[..span_fc], fc),
                    ],
                    &self.packed[20],
                    &self.biases[20],
                    catd,
                    fc,
                    4 * fc,
                    plane,
                    cs,
                );
                let pre = ensure_len(pre_vec, 3 * s2 * cs);
                conv3x3_packed_dispatch_with_scratch(
                    &catd[..span_fc],
                    fc,
                    &self.packed[21],
                    &self.biases[21],
                    pre,
                    3 * s2,
                    h,
                    w,
                    cs,
                    pin_buf,
                );
                pixel_shuffle_s_strided(pre, cs, out, h, w, s);
            }
        }
    }

    fn choose_tile_size(&self, h: usize, w: usize, threads: usize, tile: usize) -> usize {
        let halo = self.halo();
        // tile == 0 asks for the default, which is chosen from the thread count.
        //
        // A fixed 128 was leaving 21-45% on the table. Every tile recomputes a
        // `halo` border it then discards, so small tiles waste work:
        // (tile + 2*halo)^2 / tile^2 is 1.34x for the realtime tier at 128 and
        // 1.64x for the quality tier (halo 10 and 18). Bigger tiles amortise
        // that — but they also produce fewer tiles, and once n_tiles drops near
        // the thread count the run loses parallelism, so the optimum falls as
        // threads rise. Measured 2026-09-08 at 1024px (median ms, vs tile=128):
        //
        //   realtime nf=24 halo=10   T=1 512:896/1212  T=2 512:476/610
        //                            T=4 512:247/312   T=8 256:145/185
        //                            T=12 128:118 (128 already optimal)
        //   quality  nf=64 halo=18   T=1 512:11249/20309   T=8 384:2094/3038
        //
        // The ladder below never regresses against the old fixed 128 on either
        // model at any measured thread count, and captures most of the gain.
        // Tiling is BIT-EXACT in the tile size, so this changes speed only.
        // (The 2026-09-08 checksum sweep that first said so did not reach the
        // overlapped-tail and narrow-row paths, which until 2026-10-18 made tiled
        // runs differ from whole ones in the last bit; enforced now by
        // `simd::exactness::adopted_models_tiled_equals_whole_bitwise`.)
        // benchmarks/realtime_kernels_x86_2026-09-08.md
        //
        // 2026-09-08: the ladder above picks a tile SIZE, which is the wrong
        // quantity. A size that does not divide the image leaves a RUNT tile,
        // and a tiled run is paced by its LARGEST tile, so the cost is the
        // ratio of their convolved areas. At 512px the T=3..4 rung picks 396,
        // giving tiles of 396 and 116 — (416/288)^2 = 2.09x — and the measured
        // penalty is 2.25x. Same defect at the T=5..8 rung.
        //
        // So: choose the tile COUNT first and let the size follow, which makes
        // every tile the same size by construction. Then raise the count until
        // there is at least one tile per thread — four tiles across eight
        // threads leaves half of them idle for the whole run, worth another
        // 1.85x at 512px.
        //
        // MEASURED 2026-09-08, 2 models x 3 sizes x 4 thread counts x 7 tiles
        // (benchmarks/tile_ladder_2026-09-08.tsv). Against the size-only
        // ladder this is NEVER worse in any measured cell, and at 512px:
        //   realtime T=4  396 -> 268   142.8 -> 67.4 ms   -53%
        //   realtime T=8  268 -> 172    68.8 -> 54.8 ms   -20%
        //   quality  T=4  396 -> 268  1584.3 -> 791.5 ms  -50%
        //   quality  T=8  268 -> 172   765.9 -> 677.0 ms  -12%
        // Mean penalty against the per-cell optimum falls 20.4% -> 6.8%, worst
        // case 125% -> 47%. The residual is the tile COUNT, which the measured
        // optimum picks differently for the two models at the same thread count
        // (halo 10 wants more, smaller tiles than halo 18) — a thread-count
        // ladder cannot express that, and every analytic rule I fitted for it
        // regressed at least one cell, so the count rule is left alone. See the
        // benchmark note for the rules tried and rejected.
        let tile = if tile == 0 {
            // Plan for the REQUESTED threads. Planning for a measured
            // machine-wide saturation instead was tried and FALSIFIED — see
            // crate::scaling, which keeps the probe as a diagnostic and the
            // numbers that rejected it.
            let t = threads.max(1);
            let base = match t {
                1..=2 => 512,
                3..=4 => 384,
                5..=8 => 256,
                _ => 128,
            };
            // Only RE-TILE when the ladder's tile is actually pathological.
            // Re-tiling for its own sake loses: at 1024px with 4 threads both
            // the ladder's 396 and the even 348 give three tiles per axis and
            // the same total convolved area, and 348 measured 6.7% (realtime)
            // and 11.9% (quality) SLOWER, 0/3 — the tile-size landscape is
            // bumpy in ways an area model does not see. Pathological means one
            // of two things, both of which cost far more than that:
            //   starved   — fewer tiles than threads, so threads sit idle
            //   lopsided  — the last tile is under half the others, and the run
            //               is paced by the largest, so the cost is the ratio
            //               of their convolved areas
            // A floor on the tile, so a small image at many threads cannot be
            // subdivided into nothing. It has to exist — the loop below stops
            // only when there are enough tiles, and on a 64px image at 12
            // threads that would run the tile to zero.
            //
            // MEASURED 2026-09-08, and it wants to be much lower than the halo
            // cost alone suggests. Halo cost is (1 + 2*halo/tile)^2, so 2*halo
            // pays 4x in wasted border — and paying it is still right, because
            // it buys threads:
            //   halo 10, 128px, 12T   tile 44 (4.4x halo) 3.8 ms   vs 76: 6.6
            //   halo 10, 128px, 28T   tile 44             3.8 ms   vs 140: 12.3
            //   halo 18, 128px, 12T   tile 44 (2.4x halo) 70.2 ms  vs 140: 156.0
            //   halo 21, 256px, 12T   tile 86 (4.1x halo) 154.0 ms vs 134: 204.4
            // A 6*halo floor (the first version of this) picks the right-hand
            // column: it is 74% slow at 128px on the realtime model and 2.2x
            // slow on the quality one. 2*halo lands within 8% of the measured
            // optimum in every cell swept, across halos 10/18/21 and 8/12/28
            // threads.
            let min_tile = (2 * halo).max(32);
            // Tile count on the LONGER axis; the size follows from it.
            let side = h.max(w);
            let mut n = side.div_ceil(base).max(1);
            while side.div_ceil(n + 1) >= min_tile {
                let cand = side.div_ceil(n);
                // The real tile count, not n^2 — a wide image tiles once on its
                // short axis and n^2 would over-count it into starvation.
                if w.div_ceil(cand) * h.div_ceil(cand) >= t {
                    break;
                }
                n += 1;
            }
            let even = side.div_ceil(n).max(min_tile.min(side));
            // Round UP so that the CONVOLVED width, tile + 2*halo, is a multiple
            // of the widest vector (16 f32). Otherwise the row kernel needs an
            // overlapping tail tile that recomputes (16 - width%16) columns on
            // every row. At the production shape that is tile 128 + 2*10 = 148,
            // a 4-column tail, and rounding to 140 (width 160) measured -3.5%.
            // Rounding up rather than down also shrinks the discarded halo
            // fraction, so it wins on both counts. It reintroduces a runt of at
            // most 15*(n-1) px, which is a fraction of a tile rather than the
            // 3.4x imbalance a size-first ladder produces.
            let align = |tile: usize| tile + (16 - (tile + 2 * halo) % 16) % 16;
            let ladder = align(base);
            // Is the ladder's own tile pathological on this image?
            let runt = |extent: usize| -> bool {
                extent > ladder && 2 * (extent - (extent.div_ceil(ladder) - 1) * ladder) < ladder
            };
            let n_tiles = w.div_ceil(ladder) * h.div_ceil(ladder);
            let starved = n_tiles < t;
            // Lopsidedness only costs when the run is PACED by one tile. On a
            // single thread the tiles are sequential, so only the total matters
            // and re-tiling is pure added halo: at 768x512 with one thread it
            // measured -15.2% (quality, 0/3). Once there are several tiles per
            // thread the runt is a small share of the work and averages out.
            let lopsided = t > 1 && n_tiles <= 2 * t && (runt(w) || runt(h));
            let chosen = if starved || lopsided {
                align(even)
            } else {
                ladder
            };
            // An image under 32 px on a side made `even` (and so `chosen`) smaller
            // than the 32 asserted below, panicking whenever threads >= 2. At
            // least 32 covers such an image with one tile.
            chosen.max(32)
        } else {
            tile
        };
        assert!(tile >= 32);
        tile
    }

    /// Multithreaded exact-halo tiled upscale (same guarantees as spanf_x4_tiled).
    pub fn upscale_tiled(
        &self,
        input: &[f32],
        h: usize,
        w: usize,
        threads: usize,
        tile: usize,
    ) -> Vec<f32> {
        let s = self.scale;
        let mut out = vec![0.0f32; 3 * (s * h) * (s * w)];
        let mut ws = ModelWorkspace::new();
        self.upscale_tiled_into(input, h, w, threads, tile, &mut out, &mut ws);
        out
    }

    /// Exact-halo tiled upscale writing into caller-supplied `out` (`[3, s*h, s*w]`)
    /// and reusing `ws` scratch buffers.
    ///
    /// When `threads <= 1` and `ws` is already warmed for the chosen tile size,
    /// this method performs zero heap allocations.
    pub fn upscale_tiled_into(
        &self,
        input: &[f32],
        h: usize,
        w: usize,
        threads: usize,
        tile: usize,
        out: &mut [f32],
        ws: &mut ModelWorkspace,
    ) {
        let s = self.scale;
        let halo = self.halo();
        let tile = self.choose_tile_size(h, w, threads, tile);
        assert_eq!(input.len(), 3 * h * w);
        let (oh, ow) = (s * h, s * w);
        assert_eq!(out.len(), 3 * oh * ow);
        if h == 0 || w == 0 {
            return;
        }
        let tiles_x = w.div_ceil(tile);
        let tiles_y = h.div_ceil(tile);
        let n_tiles = tiles_x * tiles_y;
        if n_tiles == 1 {
            self.forward_at_with_scratch(input, h, w, out, 0, 0, &mut ws.primary);
            return;
        }
        let threads = threads.max(1).min(n_tiles);
        let run_tile = |idx: usize,
                        sc: &mut TileScratch|
         -> (usize, usize, usize, usize, usize, usize, usize, usize) {
            let (tx, ty) = (idx % tiles_x, idx / tiles_x);
            let x0 = tx * tile;
            let y0 = ty * tile;
            let x1 = (x0 + tile).min(w);
            let y1 = (y0 + tile).min(h);
            let ex0 = x0.saturating_sub(halo);
            let ey0 = y0.saturating_sub(halo);
            let ex1 = (x1 + halo).min(w);
            let ey1 = (y1 + halo).min(h);
            let (ew, eh) = (ex1 - ex0, ey1 - ey0);
            let ext_len = 3 * eh * ew;
            if sc.ext_in.len() < ext_len {
                sc.ext_in.resize(ext_len, 0.0);
            }
            for c in 0..3 {
                for y in 0..eh {
                    let src = &input[c * h * w + (ey0 + y) * w + ex0..][..ew];
                    sc.ext_in[c * eh * ew + y * ew..][..ew].copy_from_slice(src);
                }
            }
            let tile_out_len = 3 * eh * ew * s * s;
            if sc.tile_out.len() < tile_out_len {
                sc.tile_out.resize(tile_out_len, 0.0);
            }
            let ext_in = core::mem::take(&mut sc.ext_in);
            let mut tile_out = core::mem::take(&mut sc.tile_out);
            self.forward_at_with_scratch(
                &ext_in[..ext_len],
                eh,
                ew,
                &mut tile_out[..tile_out_len],
                ex0 % 2,
                ey0 % 2,
                sc,
            );
            sc.ext_in = ext_in;
            sc.tile_out = tile_out;
            let (cx, cy) = ((x0 - ex0) * s, (y0 - ey0) * s);
            let (cw, ch) = ((x1 - x0) * s, (y1 - y0) * s);
            (x0, y0, ew, eh, cx, cy, cw, ch)
        };

        if threads == 1 {
            for idx in 0..n_tiles {
                let (x0, y0, ew, eh, cx, cy, cw, ch) = run_tile(idx, &mut ws.primary);
                for c in 0..3 {
                    for y in 0..ch {
                        let src = &ws.primary.tile_out
                            [c * (s * eh) * (s * ew) + (cy + y) * (s * ew) + cx..][..cw];
                        out[c * oh * ow + (y0 * s + y) * ow + x0 * s..][..cw].copy_from_slice(src);
                    }
                }
            }
        } else {
            let extra = threads - 1;
            if ws.workers.len() < extra {
                ws.workers.resize_with(extra, TileScratch::default);
            }
            let out_mu = Mutex::new(out);
            let next = AtomicUsize::new(0);
            let worker_loop = |sc: &mut TileScratch| loop {
                let idx = next.fetch_add(1, Ordering::Relaxed);
                if idx >= n_tiles {
                    break;
                }
                let (x0, y0, ew, eh, cx, cy, cw, ch) = run_tile(idx, sc);
                let mut guard = out_mu.lock().unwrap();
                for c in 0..3 {
                    for y in 0..ch {
                        let src = &sc.tile_out
                            [c * (s * eh) * (s * ew) + (cy + y) * (s * ew) + cx..][..cw];
                        guard[c * oh * ow + (y0 * s + y) * ow + x0 * s..][..cw]
                            .copy_from_slice(src);
                    }
                }
            };
            let primary = &mut ws.primary;
            let extras = &mut ws.workers[..extra];
            std::thread::scope(|sc_scope| {
                for worker_sc in extras.iter_mut() {
                    sc_scope.spawn(move || worker_loop(worker_sc));
                }
                worker_loop(primary);
            });
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn canon4_f16_matches_torch_goldens() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/adopted/dejpeg_rt24i_canon4");
        let weights = crate::decode_all_f16(&std::fs::read(root.join("weights_f16.raw")).unwrap());
        let model = super::AdoptedModel::load_compact(&weights, 24, 8, 1).unwrap();
        let read = |name: String| -> Vec<f32> {
            let bytes = std::fs::read(root.join(name)).unwrap();
            assert_eq!(bytes.len() % 4, 0);
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect()
        };
        for (h, w) in [(40, 36), (17, 13)] {
            let input = read(format!("gold_in_{h}x{w}.raw"));
            let golden = read(format!("gold_out_{h}x{w}.raw"));
            assert_eq!(input.len(), 3 * h * w);
            assert_eq!(golden.len(), input.len());
            let mut whole = vec![0.0; input.len()];
            model.forward(&input, h, w, &mut whole);
            let tiled = model.upscale_tiled(&input, h, w, 2, 32);
            assert_eq!(whole, tiled, "whole/tiled mismatch");
            for (&actual, &expected) in whole.iter().zip(&golden) {
                assert!(actual.is_finite() && expected.is_finite());
                assert!(
                    (actual - expected).abs() < 5e-4,
                    "{h}x{w}: {actual} vs {expected}"
                );
            }
        }
    }

    use super::*;

    fn lcg_vec(n: usize, seed: u32, scale: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * scale
            })
            .collect()
    }

    fn compact_len(nf: usize, nc: usize, s: usize) -> usize {
        (3 * nf * 9 + nf + nf) + nc * (nf * nf * 9 + nf + nf) + (nf * 3 * s * s * 9 + 3 * s * s)
    }

    fn span48_len(s: usize) -> usize {
        let fc = SPAN_FC;
        (3 * fc * 9 + fc)
            + 18 * (fc * fc * 9 + fc)
            + (fc * fc * 9 + fc)
            + (4 * fc * fc + fc)
            + (fc * 3 * s * s * 9 + 3 * s * s)
    }

    /// These nf loaded and then panicked in the first forward (quad-blocked
    /// output channels; at most 64 input channels); they are now an error. Every
    /// multiple of 4 in 4..=64 still loads and runs.
    #[test]
    fn compact_rejects_nf_the_kernel_cannot_run() {
        for nf in [0usize, 1, 2, 3, 5, 6, 7, 10, 62, 66, 68, 72] {
            let raw = vec![0.01f32; compact_len(nf, 1, 1)];
            let e = AdoptedModel::load_compact(&raw, nf, 1, 1).err();
            assert!(
                e.is_some_and(|e| e.contains("cannot run")),
                "nf={nf} must be rejected"
            );
        }
        for nf in (4..=64).step_by(4) {
            let raw = lcg_vec(compact_len(nf, 1, 2), nf as u32, 0.1);
            let m = AdoptedModel::load_compact(&raw, nf, 1, 2)
                .unwrap_or_else(|e| panic!("nf={nf}: {e}"));
            let mut out = vec![0.0f32; 3 * 5 * 7 * 4];
            m.forward(&lcg_vec(3 * 5 * 7, 7, 1.0), 5, 7, &mut out);
            assert!(out.iter().all(|v| v.is_finite()), "nf={nf}");
        }
    }

    /// The span48 upsampler runs 3*s^2 output channels unpadded, so scales 1 and
    /// 3 loaded and then panicked; they are now an error. 2 and 4 still run.
    #[test]
    fn span48_rejects_scales_the_kernel_cannot_run() {
        for s in [1usize, 3] {
            let e = AdoptedModel::load_span48(&vec![0.01f32; span48_len(s)], s).err();
            assert!(
                e.is_some_and(|e| e.contains("cannot run")),
                "scale={s} must be rejected"
            );
        }
        for s in [2usize, 4] {
            let m = AdoptedModel::load_span48(&lcg_vec(span48_len(s), s as u32, 0.02), s).unwrap();
            let mut out = vec![0.0f32; 3 * 6 * 5 * s * s];
            m.forward(&lcg_vec(3 * 6 * 5, 9, 1.0), 6, 5, &mut out);
            assert!(out.iter().all(|v| v.is_finite()), "scale={s}");
        }
    }

    /// A zero-sized image upscales to a zero-sized one instead of panicking in
    /// the kernels; input and output lengths are still checked first.
    #[test]
    fn zero_sized_images_give_empty_output() {
        let raw = lcg_vec(compact_len(8, 1, 2), 3, 0.1);
        let m = AdoptedModel::load_compact(&raw, 8, 1, 2).unwrap();
        for (h, w) in [(0usize, 0usize), (0, 5), (5, 0), (0, 40), (40, 0)] {
            let mut out: Vec<f32> = Vec::new();
            m.forward(&[], h, w, &mut out);
            for threads in [1usize, 4] {
                assert!(
                    m.upscale_tiled(&[], h, w, threads, 0).is_empty(),
                    "{h}x{w} t{threads}"
                );
                assert!(
                    m.upscale_tiled(&[], h, w, threads, 32).is_empty(),
                    "{h}x{w} t{threads}"
                );
            }
        }
    }

    /// Images under 32 px on a side with threads >= 2 and the automatic tile
    /// used to panic (`tile >= 32`); they now run as one tile and match the
    /// whole-image forward bit for bit.
    #[test]
    fn tiny_images_with_auto_tiles_match_whole() {
        let raw = lcg_vec(compact_len(24, 2, 1), 11, 0.1);
        let m = AdoptedModel::load_compact(&raw, 24, 2, 1).unwrap();
        for (h, w) in [(1usize, 1usize), (5, 7), (13, 9), (31, 31), (8, 30)] {
            let input = lcg_vec(3 * h * w, (h * 31 + w) as u32, 1.0);
            let mut whole = vec![0.0f32; 3 * h * w];
            m.forward(&input, h, w, &mut whole);
            for threads in [1usize, 2, 4, 16] {
                let tiled = m.upscale_tiled(&input, h, w, threads, 0);
                let same = tiled
                    .iter()
                    .zip(&whole)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(
                    same && tiled.len() == whole.len(),
                    "{h}x{w} threads={threads}"
                );
            }
        }
    }

    /// The opt-in Winograd layers (`ZENSR_WINOGRAD=1`) tile bit-exactly too:
    /// the whole-image forward and tiled runs at several thread counts and tile
    /// sizes agree bit for bit, because the tile grid is anchored to the image
    /// and `halo` covers Winograd's 2-pixel reach.
    #[test]
    fn winograd_models_tiled_equals_whole_bitwise() {
        for (nf, nc, s) in [(8usize, 2usize, 1usize), (24, 2, 2), (64, 1, 1)] {
            let raw = lcg_vec(compact_len(nf, nc, s), (nf * 7 + nc) as u32, 0.1);
            let mut m = AdoptedModel::load_compact(&raw, nf, nc, s).unwrap();
            // What load_compact does under ZENSR_WINOGRAD=1, without the env var.
            let mut off = 3 * nf * 9 + 2 * nf;
            for i in 0..nc {
                let wraw = &raw[off..off + nf * nf * 9];
                m.wino[1 + i] = Some((crate::wino::wino_weights(wraw, nf, nf), wraw.to_vec()));
                off += nf * nf * 9 + 2 * nf;
            }
            assert_eq!(m.halo(), 2 * nc + 2, "nf={nf} nc={nc}");
            for (h, w) in [(37usize, 53usize), (21, 40), (6, 9)] {
                let input = lcg_vec(3 * h * w, (h * 13 + w) as u32, 1.0);
                let mut whole = vec![0.0f32; 3 * h * w * s * s];
                m.forward(&input, h, w, &mut whole);
                assert!(whole.iter().all(|v| v.is_finite()));
                for (threads, tile) in [
                    (1usize, 32usize),
                    (3, 32),
                    (2, 40),
                    (1, 64),
                    (4, 0),
                    (16, 0),
                ] {
                    let tiled = m.upscale_tiled(&input, h, w, threads, tile);
                    if let Some(i) = tiled
                        .iter()
                        .zip(&whole)
                        .position(|(a, b)| a.to_bits() != b.to_bits())
                    {
                        panic!(
                            "nf={nf} nc={nc} s={s} {h}x{w} threads={threads} tile={tile}: element {i} {:e} != whole {:e}",
                            tiled[i], whole[i]
                        );
                    }
                }
            }
        }
    }
}
