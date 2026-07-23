//! Shared eval helpers (decode, crop, resize, metrics) for the bench bins.
#![allow(clippy::too_many_arguments)]

use std::path::{Path, PathBuf};

pub struct Rgb8Img {
    pub px: Vec<u8>, // interleaved RGB8
    pub w: usize,
    pub h: usize,
}

/// Convert a zenpixels PixelBuffer (any 8/16-bit gray/rgb/rgba layout) to RGB8.
pub fn pixelbuffer_to_rgb8(buf: &zenpixels::PixelBuffer) -> Option<Rgb8Img> {
    let v = buf.as_slice();
    let (w, h) = (v.width() as usize, v.rows() as usize);
    let d = v.descriptor();
    let bpp = d.bytes_per_pixel();
    let bpc = d.bytes_per_channel().max(1);
    let ch = bpp / bpc;
    let mut px = vec![0u8; w * h * 3];
    for y in 0..h {
        let row = v.row(y as u32);
        for x in 0..w {
            let s = x * bpp;
            let dsti = (y * w + x) * 3;
            let sample = |c: usize| -> u8 {
                let o = s + c * bpc;
                if bpc == 2 {
                    u16::from_le_bytes([row[o], row[o + 1]]).to_be_bytes()[0]
                } else {
                    row[o]
                }
            };
            match ch {
                3 | 4 => {
                    px[dsti] = sample(0);
                    px[dsti + 1] = sample(1);
                    px[dsti + 2] = sample(2);
                }
                1 | 2 => {
                    let g = sample(0);
                    px[dsti] = g;
                    px[dsti + 1] = g;
                    px[dsti + 2] = g;
                }
                _ => return None,
            }
        }
    }
    Some(Rgb8Img { px, w, h })
}

pub fn decode_any(path: &Path) -> Option<Rgb8Img> {
    let data = std::fs::read(path).ok()?;
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "png" => {
            let out = zenpng::decode(
                &data,
                &zenpng::PngDecodeConfig::default(),
                &enough::Unstoppable,
            )
            .ok()?;
            pixelbuffer_to_rgb8(&out.pixels)
        }
        "jpg" | "jpeg" => {
            let r = zenjpeg::decoder::Decoder::new()
                .decode(&data, zenjpeg::encoder::Unstoppable)
                .ok()?;
            let (w, h) = r.dimensions();
            let px = r.pixels_u8()?;
            if px.len() != w as usize * h as usize * 3 {
                return None; // grayscale/CMYK: skip for eval v1
            }
            Some(Rgb8Img { px: px.to_vec(), w: w as usize, h: h as usize })
        }
        _ => None,
    }
}

pub fn center_crop(img: &Rgb8Img, cap: usize) -> Option<Rgb8Img> {
    let cw = (img.w.min(cap) / 4) * 4;
    let ch = (img.h.min(cap) / 4) * 4;
    if cw < 64 || ch < 64 {
        return None;
    }
    let x0 = (img.w - cw) / 2;
    let y0 = (img.h - ch) / 2;
    let mut px = Vec::with_capacity(cw * ch * 3);
    for y in 0..ch {
        let s = ((y0 + y) * img.w + x0) * 3;
        px.extend_from_slice(&img.px[s..s + cw * 3]);
    }
    Some(Rgb8Img { px, w: cw, h: ch })
}

pub fn resize_rgb8(img: &Rgb8Img, dw: usize, dh: usize, filter: zenresize::Filter) -> Rgb8Img {
    let config = zenresize::ResizeConfig::builder(
        img.w as u32,
        img.h as u32,
        dw as u32,
        dh as u32,
    )
    .filter(filter)
    .format(zenresize::PixelDescriptor::RGB8_SRGB)
    .build();
    let out = zenresize::Resizer::new(&config).resize(&img.px);
    Rgb8Img { px: out, w: dw, h: dh }
}

pub fn to_planar_f32(img: &Rgb8Img) -> Vec<f32> {
    let plane = img.w * img.h;
    let mut p = vec![0.0f32; 3 * plane];
    for i in 0..plane {
        p[i] = img.px[3 * i] as f32 / 255.0;
        p[plane + i] = img.px[3 * i + 1] as f32 / 255.0;
        p[2 * plane + i] = img.px[3 * i + 2] as f32 / 255.0;
    }
    p
}

pub fn planar_to_rgb8(p: &[f32], w: usize, h: usize) -> Rgb8Img {
    let plane = w * h;
    let mut px = vec![0u8; 3 * plane];
    for i in 0..plane {
        px[3 * i] = (p[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        px[3 * i + 1] = (p[plane + i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        px[3 * i + 2] = (p[2 * plane + i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
    Rgb8Img { px, w, h }
}

pub fn psnr_rgb8(a: &Rgb8Img, b: &Rgb8Img) -> f64 {
    let mut se = 0.0f64;
    for (x, y) in a.px.iter().zip(b.px.iter()) {
        let d = *x as f64 - *y as f64;
        se += d * d;
    }
    let mse = se / a.px.len() as f64 / (255.0 * 255.0);
    if mse == 0.0 {
        99.0
    } else {
        -10.0 * mse.log10()
    }
}

pub fn ssim2(a: &Rgb8Img, b: &Rgb8Img) -> f64 {
    use imgref::Img;
    let conv = |i: &Rgb8Img| -> Img<Vec<[u8; 3]>> {
        let v: Vec<[u8; 3]> = i.px.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        Img::new(v, i.w, i.h)
    };
    let (ra, rb) = (conv(a), conv(b));
    fast_ssim2::compute_ssimulacra2(ra.as_ref(), rb.as_ref()).unwrap_or(f64::NAN)
}

pub fn butter_n3(a: &Rgb8Img, b: &Rgb8Img) -> f64 {
    use butteraugli::{butteraugli, ButteraugliParams, Img, RGB8};
    use rgb::FromSlice;
    let ia: Img<Vec<RGB8>> = Img::new(a.px.as_rgb().to_vec(), a.w, a.h);
    let ib: Img<Vec<RGB8>> = Img::new(b.px.as_rgb().to_vec(), b.w, b.h);
    match butteraugli(ia.as_ref(), ib.as_ref(), &ButteraugliParams::default()) {
        Ok(r) => r.pnorm_3,
        Err(_) => f64::NAN,
    }
}

pub fn list_images(dir: &Path) -> Vec<PathBuf> {
    // Recursive: several subcorpora nest by resolution/source folders.
    fn walk(dir: &Path, out: &mut Vec<PathBuf>, depth: usize) {
        if depth > 4 {
            return;
        }
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out, depth + 1);
            } else if p
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg"))
                .unwrap_or(false)
            {
                out.push(p);
            }
        }
    }
    let mut v = Vec::new();
    walk(dir, &mut v, 0);
    v.sort();
    v
}
