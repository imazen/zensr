//! zenpixels entry points: borrowed [`PixelSlice`] in (stride-aware), owned
//! [`PixelBuffer`] out. Interleaved RGB f32 (`RGBF32` / `RGBF32_LINEAR`);
//! the input's descriptor (and thus colorimetry tag) is carried to the output.
//!
//! De/re-interleave happens at the staging boundary that already exists for
//! tiling, so strided input costs nothing extra versus tightly-packed input.

use crate::tiled::spanf_x4_tiled;
use crate::SpanfModel;
use archmage::prelude::*;
use magetypes::simd::backends::F32x8Backend;
use magetypes::simd::generic::f32x8 as V8;
use zenpixels::{At, BufferError, PixelBuffer, PixelDescriptor, PixelSlice};

#[derive(Debug)]
pub enum PxError {
    /// Only RGBF32 / RGBF32_LINEAR interleaved input is supported (v1).
    UnsupportedFormat(PixelDescriptor),
    /// Zero-sized input.
    EmptyInput,
    /// Output allocation failed.
    Alloc(At<BufferError>),
}

impl core::fmt::Display for PxError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PxError::UnsupportedFormat(d) => {
                write!(
                    f,
                    "unsupported pixel format {d:?}; need RGBF32 or RGBF32_LINEAR"
                )
            }
            PxError::EmptyInput => write!(f, "empty input"),
            PxError::Alloc(e) => write!(f, "output allocation failed: {e}"),
        }
    }
}

impl std::error::Error for PxError {}

fn supported(d: PixelDescriptor) -> bool {
    d == PixelDescriptor::RGBF32 || d == PixelDescriptor::RGBF32_LINEAR
}

/// SPANF x4 upscale of a borrowed, possibly strided RGB-f32 [`PixelSlice`]
/// into a new owned [`PixelBuffer`] (4x width, 4x height, same descriptor).
///
/// `threads >= 1`; `tile == 0` selects the measured default (128).
pub fn spanf_x4_px(
    model: &SpanfModel,
    src: &PixelSlice<'_>,
    threads: usize,
    tile: usize,
) -> Result<PixelBuffer, PxError> {
    let d = src.descriptor();
    if !supported(d) {
        return Err(PxError::UnsupportedFormat(d));
    }
    let (w, h) = (src.width() as usize, src.rows() as usize);
    if w == 0 || h == 0 {
        return Err(PxError::EmptyInput);
    }

    // De-interleave (strided rows -> planar NCHW). PixelSlice guarantees each
    // row start is aligned for the channel type, so the f32 cast is safe.
    let plane = h * w;
    let mut planar = vec![0.0f32; 3 * plane];
    attuned!(
        deinterleave_rgb_rows(src, &mut planar, w, h),
        [
            v4x_t(cfg(avx512)),
            v4_t(cfg(tier_v4)),
            v3_t,
            neon_t,
            wasm128_t,
            scalar_t
        ]
    );

    let out_planar = spanf_x4_tiled(model, &planar, h, w, threads, tile);

    // Re-interleave into an owned buffer with the same descriptor.
    let (oh, ow) = (4 * h, 4 * w);
    let mut buf = PixelBuffer::try_new(ow as u32, oh as u32, d).map_err(PxError::Alloc)?;
    {
        let mut view = buf.rows_mut(0, oh as u32);
        attuned!(
            interleave_rgb_rows(&out_planar, &mut view, ow, oh),
            [
                v4x_t(cfg(avx512)),
                v4_t(cfg(tier_v4)),
                v3_t,
                neon_t,
                wasm128_t,
                scalar_t
            ]
        );
    }
    Ok(buf)
}

#[attune(_*_t, +v4x(cfg(avx512)), +v4(cfg(tier_v4)))]
fn deinterleave_rgb_rows(
    token: Token,
    src: &PixelSlice<'_>,
    planar: &mut [f32],
    w: usize,
    h: usize,
) {
    deinterleave_rgb_rows_generic(token, src, planar, w, h);
}

#[inline(always)]
fn deinterleave_rgb_rows_generic<T: F32x8Backend>(
    token: T,
    src: &PixelSlice<'_>,
    planar: &mut [f32],
    w: usize,
    h: usize,
) {
    let plane = h * w;
    let (pr_all, rest) = planar.split_at_mut(plane);
    let (pg_all, pb_all) = rest.split_at_mut(plane);
    let v_end = w & !7;
    for y in 0..h {
        let row: &[f32] = bytemuck::cast_slice(src.row(y as u32));
        let pr = &mut pr_all[y * w..y * w + w];
        let pg = &mut pg_all[y * w..y * w + w];
        let pb = &mut pb_all[y * w..y * w + w];
        let mut x = 0usize;
        while x < v_end {
            let c: &[f32; 24] = (&row[3 * x..3 * x + 24]).try_into().unwrap();
            let v0 = V8::<T>::load_t(token, (&c[0..8]).try_into().unwrap());
            let v1 = V8::<T>::load_t(token, (&c[8..16]).try_into().unwrap());
            let v2 = V8::<T>::load_t(token, (&c[16..24]).try_into().unwrap());
            let [r, g, b] = V8::<T>::deinterleave_3ch([v0, v1, v2]);
            r.store((&mut pr[x..x + 8]).try_into().unwrap());
            g.store((&mut pg[x..x + 8]).try_into().unwrap());
            b.store((&mut pb[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        while x < w {
            pr[x] = row[3 * x];
            pg[x] = row[3 * x + 1];
            pb[x] = row[3 * x + 2];
            x += 1;
        }
    }
}

#[attune(_*_t, +v4x(cfg(avx512)), +v4(cfg(tier_v4)))]
fn interleave_rgb_rows(
    token: Token,
    planar: &[f32],
    view: &mut zenpixels::PixelSliceMut<'_>,
    ow: usize,
    oh: usize,
) {
    interleave_rgb_rows_generic(token, planar, view, ow, oh);
}

#[inline(always)]
fn interleave_rgb_rows_generic<T: F32x8Backend>(
    token: T,
    planar: &[f32],
    view: &mut zenpixels::PixelSliceMut<'_>,
    ow: usize,
    oh: usize,
) {
    let oplane = oh * ow;
    let (pr_all, rest) = planar.split_at(oplane);
    let (pg_all, pb_all) = rest.split_at(oplane);
    let v_end = ow & !7;
    for y in 0..oh {
        let row: &mut [f32] = bytemuck::cast_slice_mut(view.row_mut(y as u32));
        let pr = &pr_all[y * ow..y * ow + ow];
        let pg = &pg_all[y * ow..y * ow + ow];
        let pb = &pb_all[y * ow..y * ow + ow];
        let mut x = 0usize;
        while x < v_end {
            let r = V8::<T>::load_t(token, (&pr[x..x + 8]).try_into().unwrap());
            let g = V8::<T>::load_t(token, (&pg[x..x + 8]).try_into().unwrap());
            let b = V8::<T>::load_t(token, (&pb[x..x + 8]).try_into().unwrap());
            let [v0, v1, v2] = V8::<T>::interleave_3ch([r, g, b]);
            let dst: &mut [f32; 24] = (&mut row[3 * x..3 * x + 24]).try_into().unwrap();
            let (d0, rest_d) = dst.split_at_mut(8);
            let (d1, d2) = rest_d.split_at_mut(8);
            v0.store(d0.try_into().unwrap());
            v1.store(d1.try_into().unwrap());
            v2.store(d2.try_into().unwrap());
            x += 8;
        }
        while x < ow {
            row[3 * x] = pr[x];
            row[3 * x + 1] = pg[x];
            row[3 * x + 2] = pb[x];
            x += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strided_rgb_f32_deinterleave_reinterleave_bit_exact() {
        let w = 19usize;
        let h = 11usize;
        let row_floats = w * 3;
        let stride_floats = row_floats + 9; // non-packed stride in floats (multiple of 1 f32, aligned bytes)
        let stride_bytes = stride_floats * 4;
        let mut src_floats = vec![f32::NAN; h * stride_floats];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    src_floats[y * stride_floats + 3 * x + c] =
                        ((y * 41 + x * 13 + c * 7) % 256) as f32 / 255.0;
                }
            }
        }
        let src_bytes: &[u8] = bytemuck::cast_slice(&src_floats);
        let slice = PixelSlice::new(
            src_bytes,
            w as u32,
            h as u32,
            stride_bytes,
            PixelDescriptor::RGBF32_LINEAR,
        )
        .unwrap();

        let model = SpanfModel::new(vec![0.01f32; crate::TOTAL_FLOATS]).unwrap();

        let out = spanf_x4_px(&model, &slice, 1, 0).unwrap();

        // Scalar reference path
        let plane = w * h;
        let mut ref_planar = vec![0.0f32; 3 * plane];
        for y in 0..h {
            let row: &[f32] = bytemuck::cast_slice(slice.row(y as u32));
            for x in 0..w {
                ref_planar[y * w + x] = row[3 * x];
                ref_planar[plane + y * w + x] = row[3 * x + 1];
                ref_planar[2 * plane + y * w + x] = row[3 * x + 2];
            }
        }
        let ref_out_planar = spanf_x4_tiled(&model, &ref_planar, h, w, 1, 0);
        let (ow, oh) = (4 * w, 4 * h);
        let oplane = ow * oh;
        let view = out.rows(0, oh as u32);
        for y in 0..oh {
            let got: &[f32] = bytemuck::cast_slice(view.row(y as u32));
            for x in 0..ow {
                assert_eq!(
                    got[3 * x].to_bits(),
                    ref_out_planar[y * ow + x].to_bits(),
                    "R mismatch at ({x},{y})"
                );
                assert_eq!(
                    got[3 * x + 1].to_bits(),
                    ref_out_planar[oplane + y * ow + x].to_bits(),
                    "G mismatch at ({x},{y})"
                );
                assert_eq!(
                    got[3 * x + 2].to_bits(),
                    ref_out_planar[2 * oplane + y * ow + x].to_bits(),
                    "B mismatch at ({x},{y})"
                );
            }
        }
    }
}
