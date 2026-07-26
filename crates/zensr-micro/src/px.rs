//! zenpixels entry points: borrowed [`PixelSlice`] in (stride-aware), owned
//! [`PixelBuffer`] out. Interleaved RGB f32 (`RGBF32` / `RGBF32_LINEAR`);
//! the input's descriptor (and thus colorimetry tag) is carried to the output.
//!
//! De/re-interleave happens at the staging boundary that already exists for
//! tiling, so strided input costs nothing extra versus tightly-packed input.

use crate::tiled::spanf_x4_tiled;
use crate::SpanfModel;
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
                write!(f, "unsupported pixel format {d:?}; need RGBF32 or RGBF32_LINEAR")
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
    for y in 0..h {
        let row: &[f32] = bytemuck::cast_slice(src.row(y as u32));
        for x in 0..w {
            planar[y * w + x] = row[3 * x];
            planar[plane + y * w + x] = row[3 * x + 1];
            planar[2 * plane + y * w + x] = row[3 * x + 2];
        }
    }

    let out_planar = spanf_x4_tiled(model, &planar, h, w, threads, tile);

    // Re-interleave into an owned buffer with the same descriptor.
    let (oh, ow) = (4 * h, 4 * w);
    let oplane = oh * ow;
    let mut buf = PixelBuffer::try_new(ow as u32, oh as u32, d).map_err(PxError::Alloc)?;
    {
        let mut view = buf.rows_mut(0, oh as u32);
        for y in 0..oh {
            let row: &mut [f32] = bytemuck::cast_slice_mut(view.row_mut(y as u32));
            for x in 0..ow {
                row[3 * x] = out_planar[y * ow + x];
                row[3 * x + 1] = out_planar[oplane + y * ow + x];
                row[3 * x + 2] = out_planar[2 * oplane + y * ow + x];
            }
        }
    }
    Ok(buf)
}
