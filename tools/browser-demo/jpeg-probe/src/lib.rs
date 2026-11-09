//! Bounded browser JPEG admission and sRGB pixel decode through zenjpeg.
#![forbid(unsafe_code)]
use wasm_bindgen::prelude::*;
#[wasm_bindgen]
pub fn dimensions(bytes: &[u8]) -> Result<Vec<u32>, JsValue> {
    let info = zenjpeg::decoder::Decoder::new()
        .read_info(bytes)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    if info.dimensions.width == 0
        || info.dimensions.height == 0
        || info.dimensions.width > 8192
        || info.dimensions.height > 8192
        || u64::from(info.dimensions.width) * u64::from(info.dimensions.height) > 12_600_000
    {
        return Err(JsValue::from_str(
            "Choose an image up to 12.6 MP and 8192 pixels per side",
        ));
    }
    Ok(vec![info.dimensions.width, info.dimensions.height])
}

#[wasm_bindgen]
pub struct DecodedJpeg {
    width: u32,
    height: u32,
    rgb: Vec<u8>,
}
#[wasm_bindgen]
impl DecodedJpeg {
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn take_pixels(self) -> Vec<u8> {
        self.rgb
    }
}
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn memory_bytes() -> usize {
    core::arch::wasm32::memory_size::<0>() * 65536
}
#[wasm_bindgen]
pub fn decode(bytes: &[u8]) -> Result<DecodedJpeg, JsValue> {
    dimensions(bytes)?;
    let error = |e: String| JsValue::from_str(&e);
    let result = zenjpeg::decoder::Decoder::new()
        .max_pixels(12_600_000)
        .max_memory(450 * 1024 * 1024)
        .output_format(zenjpeg::decoder::PixelFormat::Rgb)
        .auto_orient(false)
        .deblock(zenjpeg::decoder::DeblockMode::Off)
        .correct_color(Some(zenjpeg::decoder::TargetColorSpace::Srgb))
        .decode(bytes, enough::Unstoppable)
        .map_err(|e| error(e.to_string()))?;
    let (width, height) = result.dimensions();
    let rgb = result
        .into_pixels_u8()
        .ok_or_else(|| error("JPEG decoder did not return RGB8".into()))?;
    if rgb.len() != width as usize * height as usize * 3 {
        return Err(error("JPEG RGB layout differs".into()));
    }
    let mut orientation = 1;
    for span in zenjpeg::container::MarkerIter::new(bytes) {
        if span.kind == zenjpeg::container::MarkerKind::Eoi {
            break;
        }
        if span.kind == zenjpeg::container::MarkerKind::App(1)
            && span.payload.starts_with(b"Exif\0\0")
        {
            orientation = zenjpeg::lossless::parse_exif_orientation(span.payload).unwrap_or(1);
            break;
        }
    }
    if orientation == 1 {
        return Ok(DecodedJpeg { width, height, rgb });
    }
    let source = zenpixels::PixelSlice::new(
        &rgb,
        width,
        height,
        width as usize * 3,
        zenpixels::PixelDescriptor::RGB8_SRGB,
    )
    .map_err(|e| error(e.to_string()))?;
    let orient = zenpixels::Orientation::from_exif(orientation)
        .ok_or_else(|| error("Invalid JPEG orientation".into()))?;
    let result = zenpixels_convert::orient::apply_orientation(source, orient);
    let (width, height) = orient.output_dimensions(width, height);
    let rgb = result
        .as_contiguous_bytes()
        .ok_or_else(|| error("JPEG result is not contiguous".into()))?
        .to_vec();
    Ok(DecodedJpeg { width, height, rgb })
}
