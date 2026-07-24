//! Adopted-model support: SRVGGNetCompact ("compact") and official SPAN-48
//! ("span48") graphs running on the shared kernel set, with a generic
//! exact-halo tiled runner.
//!
//! Weight files come from tools/dump_adopted.py (fixed order, f32 LE);
//! Conv3XC branches are pre-merged and verified at dump time.
#![allow(clippy::too_many_arguments)]

use crate::simd::{
    conv1x1_gen_dispatch, conv3x3_packed_dispatch, gate_all_dispatch, prelu_dispatch,
    silu_all_dispatch,
};
use crate::{nearest_add, pack_conv1x1, pack_conv3x3, pixel_shuffle_s_strided};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

pub enum Arch {
    /// [conv3x3+PReLU] x (nc+1) -> conv3x3(3*s^2) -> shuffle -> +nearest(x)
    Compact { nf: usize, nc: usize },
    /// conv_1 + 6 gated SPAB blocks + conv_2 + 4-way cat + conv1x1 + upsampler
    Span48,
}

pub struct AdoptedModel {
    pub arch: Arch,
    pub scale: usize,
    /// packed conv weights in graph order
    packed: Vec<Vec<f32>>,
    biases: Vec<Vec<f32>>,
    /// per-channel PReLU slopes (compact only), in order
    slopes: Vec<Vec<f32>>,
}

const SPAN_FC: usize = 48;

fn take<'a>(buf: &mut &'a [f32], n: usize) -> &'a [f32] {
    let (h, t) = buf.split_at(n);
    *buf = t;
    h
}

impl AdoptedModel {
    /// Receptive-field radius in input pixels (number of chained 3x3 convs).
    pub fn halo(&self) -> usize {
        match self.arch {
            Arch::Compact { nc, .. } => nc + 2,
            Arch::Span48 => 21,
        }
    }

    pub fn load_compact(raw: &[f32], nf: usize, nc: usize, scale: usize) -> Result<Self, String> {
        let s2 = scale * scale;
        let expect = (3 * nf * 9 + nf + nf)
            + nc * (nf * nf * 9 + nf + nf)
            + (nf * 3 * s2 * 9 + 3 * s2);
        if raw.len() != expect {
            return Err(format!("compact: expected {expect} floats, got {}", raw.len()));
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
        for _ in 0..nc {
            packed.push(pack_conv3x3(take(&mut buf, nf * nf * 9), nf, nf));
            biases.push(take(&mut buf, nf).to_vec());
            slopes.push(take(&mut buf, nf).to_vec());
        }
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
        Ok(AdoptedModel { arch: Arch::Compact { nf, nc }, scale, packed, biases, slopes })
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
            return Err(format!("span48: expected {expect} floats, got {}", raw.len()));
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
        Ok(AdoptedModel { arch: Arch::Span48, scale, packed, biases, slopes: Vec::new() })
    }

    /// Whole-tile forward: input [3,h,w] tight -> out [3,s*h,s*w] tight.
    pub fn forward(&self, input: &[f32], h: usize, w: usize, out: &mut [f32]) {
        let plane = h * w;
        let cs = plane; // identity stride (padding measured 2x slower — see PLAN appendix)
        let s = self.scale;
        assert_eq!(input.len(), 3 * plane);
        assert_eq!(out.len(), 3 * plane * s * s);
        match self.arch {
            Arch::Compact { nf, nc } => {
                let s2 = s * s;
                let mut inp3 = vec![0.0f32; 3 * cs];
                for c in 0..3 {
                    inp3[c * cs..c * cs + plane].copy_from_slice(&input[c * plane..(c + 1) * plane]);
                }
                let mut cur = vec![0.0f32; nf * cs];
                let mut nxt = vec![0.0f32; nf * cs];
                let span3 = 2 * cs + plane;
                let span_nf = (nf - 1) * cs + plane;
                conv3x3_packed_dispatch(&inp3[..span3], 3, &self.packed[0], &self.biases[0], &mut cur, nf, h, w, cs);
                prelu_dispatch(&mut cur[..span_nf], &self.slopes[0], nf, plane, cs);
                for i in 0..nc {
                    conv3x3_packed_dispatch(&cur[..span_nf], nf, &self.packed[1 + i], &self.biases[1 + i], &mut nxt, nf, h, w, cs);
                    prelu_dispatch(&mut nxt[..span_nf], &self.slopes[1 + i], nf, plane, cs);
                    core::mem::swap(&mut cur, &mut nxt);
                }
                let cpad = (3 * s2).next_multiple_of(4);
                let mut pre = vec![0.0f32; cpad * cs];
                conv3x3_packed_dispatch(&cur[..span_nf], nf, &self.packed[1 + nc], &self.biases[1 + nc], &mut pre, cpad, h, w, cs);
                pixel_shuffle_s_strided(&pre, cs, out, h, w, s);
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
                let mut inp3 = vec![0.0f32; 3 * cs];
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
                let mut feat = vec![0.0f32; fc * cs];
                conv3x3_packed_dispatch(&inp3[..span3], 3, &self.packed[0], &self.biases[0], &mut feat, fc, h, w, cs);

                let mut b_prev = vec![0.0f32; fc * cs];
                b_prev[..span_fc].copy_from_slice(&feat[..span_fc]);
                let mut b1_keep = vec![0.0f32; fc * cs];
                let mut b6_o1 = vec![0.0f32; fc * cs];
                let mut t1 = vec![0.0f32; fc * cs];
                let mut t2 = vec![0.0f32; fc * cs];
                for blk in 0..6 {
                    let base = 1 + blk * 3;
                    conv3x3_packed_dispatch(&b_prev[..span_fc], fc, &self.packed[base], &self.biases[base], &mut t1, fc, h, w, cs);
                    silu_all_dispatch(&mut t1[..span_fc]);
                    // official SPAB uses SiLU(inplace=True): the out1 returned to the
                    // final concat is the POST-activation tensor (keep for block 6)
                    if blk == 5 {
                        b6_o1[..span_fc].copy_from_slice(&t1[..span_fc]);
                    }
                    conv3x3_packed_dispatch(&t1[..span_fc], fc, &self.packed[base + 1], &self.biases[base + 1], &mut t2, fc, h, w, cs);
                    silu_all_dispatch(&mut t2[..span_fc]);
                    conv3x3_packed_dispatch(&t2[..span_fc], fc, &self.packed[base + 2], &self.biases[base + 2], &mut t1, fc, h, w, cs);
                    // gate: out = (o3 + x) * (sigmoid(o3) - 0.5)   (always, official SPAB)
                    gate_all_dispatch(&t1[..span_fc], &b_prev[..span_fc], &mut t2[..span_fc]);
                    core::mem::swap(&mut b_prev, &mut t2);
                    if blk == 0 {
                        b1_keep[..span_fc].copy_from_slice(&b_prev[..span_fc]);
                    }
                }
                // conv_2 on b6
                conv3x3_packed_dispatch(&b_prev[..span_fc], fc, &self.packed[19], &self.biases[19], &mut t1, fc, h, w, cs);
                // cat [feat, conv_2(b6), b1, b6_o1] -> conv1x1 -> upsampler
                let mut catd = vec![0.0f32; fc * cs];
                conv1x1_gen_dispatch(
                    &[
                        (&feat[..span_fc], fc),
                        (&t1[..span_fc], fc),
                        (&b1_keep[..span_fc], fc),
                        (&b6_o1[..span_fc], fc),
                    ],
                    &self.packed[20],
                    &self.biases[20],
                    &mut catd,
                    fc,
                    4 * fc,
                    plane,
                    cs,
                );
                let mut pre = vec![0.0f32; 3 * s2 * cs];
                conv3x3_packed_dispatch(&catd[..span_fc], fc, &self.packed[21], &self.biases[21], &mut pre, 3 * s2, h, w, cs);
                pixel_shuffle_s_strided(&pre, cs, out, h, w, s);
            }
        }
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
        let halo = self.halo();
        let tile = if tile == 0 { 128 } else { tile };
        assert!(tile >= 32);
        assert_eq!(input.len(), 3 * h * w);
        let (oh, ow) = (s * h, s * w);
        let tiles_x = w.div_ceil(tile);
        let tiles_y = h.div_ceil(tile);
        let n_tiles = tiles_x * tiles_y;
        let threads = threads.max(1).min(n_tiles);
        let out = Mutex::new(vec![0.0f32; 3 * oh * ow]);
        let next = AtomicUsize::new(0);

        let worker = || {
            let mut ext_in: Vec<f32> = Vec::new();
            let mut tile_out: Vec<f32> = Vec::new();
            loop {
                let idx = next.fetch_add(1, Ordering::Relaxed);
                if idx >= n_tiles {
                    break;
                }
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
                ext_in.clear();
                ext_in.resize(3 * eh * ew, 0.0);
                for c in 0..3 {
                    for y in 0..eh {
                        let src = &input[c * h * w + (ey0 + y) * w + ex0..][..ew];
                        ext_in[c * eh * ew + y * ew..][..ew].copy_from_slice(src);
                    }
                }
                tile_out.clear();
                tile_out.resize(3 * eh * ew * s * s, 0.0);
                self.forward(&ext_in, eh, ew, &mut tile_out);
                let (cx, cy) = ((x0 - ex0) * s, (y0 - ey0) * s);
                let (cw, ch) = ((x1 - x0) * s, (y1 - y0) * s);
                let mut guard = out.lock().unwrap();
                for c in 0..3 {
                    for y in 0..ch {
                        let src = &tile_out
                            [c * (s * eh) * (s * ew) + (cy + y) * (s * ew) + cx..][..cw];
                        guard[c * oh * ow + (y0 * s + y) * ow + x0 * s..][..cw]
                            .copy_from_slice(src);
                    }
                }
            }
        };
        if threads == 1 {
            worker();
        } else {
            std::thread::scope(|sc| {
                for _ in 0..threads {
                    sc.spawn(worker);
                }
            });
        }
        out.into_inner().unwrap()
    }
}
