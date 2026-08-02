//! Multithreaded tiled SPANF x4: exact halo tiling + std::thread::scope pool.
//!
//! SPANF's deepest path is 16 stacked 3x3 convs (b1..b5 x3 + conv_2), so a
//! 16-px halo per tile side makes cropped tile outputs match whole-image
//! inference exactly (zero padding only ever influences cropped-away pixels;
//! residual differences are FMA-association noise at the 1e-6 level because a
//! pixel's tile-relative column decides which vector path computes it).
//!
//! Threading: worker threads pull tile indices from an atomic counter; each
//! owns a `Scratch`; finished tiles paste into the shared output under a
//! mutex (paste is a ~ms memcpy vs ~200ms tile compute).
#![allow(clippy::too_many_arguments)]

use crate::simd::forward_dispatch;
use crate::{Scratch, SpanfModel};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Receptive-field radius of the whole net (16 chained 3x3 convs).
pub const HALO: usize = 16;

/// Multithreaded tiled x4 upscale. `input` is [3,h,w] NCHW f32.
/// `tile` is the CORE tile size (extended by up to HALO per side internally);
/// pass 0 for the measured default (128 — best across 1..24 threads on the
/// 2026-07-22 sweep; small tiles win on both cache footprint and granularity).
/// `threads >= 1`.
pub fn spanf_x4_tiled(
    model: &SpanfModel,
    input: &[f32],
    h: usize,
    w: usize,
    threads: usize,
    tile: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), 3 * h * w);
    let tile = if tile == 0 { 128 } else { tile };
    assert!(tile >= 32, "tile must be >= 32");
    let (oh, ow) = (4 * h, 4 * w);
    let tiles_x = w.div_ceil(tile);
    let tiles_y = h.div_ceil(tile);
    let n_tiles = tiles_x * tiles_y;
    let threads = threads.max(1).min(n_tiles);

    let out = Mutex::new(vec![0.0f32; 3 * oh * ow]);
    let next = AtomicUsize::new(0);

    let run_worker = || {
        let mut scratch: Option<(usize, usize, Scratch)> = None;
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
            // extended (halo) rect, clamped to the image
            let ex0 = x0.saturating_sub(HALO);
            let ey0 = y0.saturating_sub(HALO);
            let ex1 = (x1 + HALO).min(w);
            let ey1 = (y1 + HALO).min(h);
            let (ew, eh) = (ex1 - ex0, ey1 - ey0);

            // gather extended input planes
            ext_in.clear();
            ext_in.resize(3 * eh * ew, 0.0);
            for c in 0..3 {
                for y in 0..eh {
                    let src = &input[c * h * w + (ey0 + y) * w + ex0..][..ew];
                    ext_in[c * eh * ew + y * ew..][..ew].copy_from_slice(src);
                }
            }

            // per-thread scratch, rebuilt only when the extended shape changes
            let sc = match &mut scratch {
                Some((sh, sw, sc)) if *sh == eh && *sw == ew => sc,
                slot => {
                    *slot = Some((eh, ew, Scratch::new(eh, ew)));
                    &mut slot.as_mut().unwrap().2
                }
            };
            tile_out.clear();
            tile_out.resize(3 * eh * ew * 16, 0.0);
            forward_dispatch(&ext_in, eh, ew, model, sc, &mut tile_out);

            // paste the core crop (scaled x4) into the shared output
            let (cx, cy) = ((x0 - ex0) * 4, (y0 - ey0) * 4);
            let (cw, ch) = ((x1 - x0) * 4, (y1 - y0) * 4);
            let mut guard = out.lock().unwrap();
            for c in 0..3 {
                for y in 0..ch {
                    let src = &tile_out[c * (4 * eh) * (4 * ew) + (cy + y) * (4 * ew) + cx..][..cw];
                    guard[c * oh * ow + (y0 * 4 + y) * ow + x0 * 4..][..cw].copy_from_slice(src);
                }
            }
        }
    };

    if threads == 1 {
        run_worker();
    } else {
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(run_worker);
            }
        });
    }
    out.into_inner().unwrap()
}
