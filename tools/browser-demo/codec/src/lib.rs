//! Browser demo glue around the shared streaming PNG encoder.
#![forbid(unsafe_code)]
use sha2::{Digest, Sha256};
use wasm_bindgen::prelude::*;
use zencodec::ThreadingPolicy;
use zencodec::encode::{EncodeJob, Encoder, EncoderConfig};
use zenpixels::{PixelDescriptor, PixelSlice};
use zenpng::{Compression, DowncastFlags, PngEncoder, PngEncoderConfig};

fn error(e: impl core::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

#[wasm_bindgen]
pub fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Convert only a finished core. The x2 path resizes float samples before
/// quantizing, using the shared resampler in the model's encoded-RGB domain.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn pack_tile(
    data: &[f32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    core_width: usize,
    core_height: usize,
    scale: usize,
) -> Result<Vec<u8>, JsValue> {
    if width > 768
        || height > 768
        || data.len() != 3 * width * height
        || !matches!(scale, 2 | 4)
        || x + core_width > width
        || y + core_height > height
        || core_width == 0
        || core_height == 0
        || !data.iter().all(|v| v.is_finite())
    {
        return Err(error("Invalid model tile"));
    }
    let mut core = vec![0.0; core_width * core_height * 3];
    for cy in 0..core_height {
        for cx in 0..core_width {
            for c in 0..3 {
                core[(cy * core_width + cx) * 3 + c] =
                    data[c * width * height + (y + cy) * width + x + cx];
            }
        }
    }
    let final_pixels = if scale == 2 {
        if !core_width.is_multiple_of(2) || !core_height.is_multiple_of(2) {
            return Err(error("Unaligned x2 tile"));
        }
        let config = zenresize::ResizeConfig::builder(
            core_width as u32,
            core_height as u32,
            (core_width / 2) as u32,
            (core_height / 2) as u32,
        )
        .filter(zenresize::Filter::Box)
        .format(
            zenresize::PixelDescriptor::RGBF32_LINEAR
                .with_transfer(zenresize::TransferFunction::Srgb),
        )
        .srgb()
        .build();
        zenresize::Resizer::new(&config).resize_f32(&core)
    } else {
        core
    };
    Ok(final_pixels
        .into_iter()
        .map(|v| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8)
        .collect())
}

#[wasm_bindgen]
pub fn dimensions(data: &[u8]) -> Result<Vec<u32>, JsValue> {
    let info = zenpng::probe(data).map_err(error)?;
    let (w, h) = (info.width, info.height);
    if w == 0 || h == 0 || u64::from(w) * u64::from(h) > 12_600_000 || w > 8192 || h > 8192 {
        return Err(error(
            "Choose a JPEG or PNG up to 12.6 MP and 8192 pixels per side",
        ));
    }
    Ok(vec![w, h])
}

#[wasm_bindgen]
pub struct PngRows {
    encoder: Option<PngEncoder>,
    width: u32,
    height: u32,
    rows: u32,
}

#[wasm_bindgen]
impl PngRows {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32) -> Result<PngRows, JsValue> {
        if width == 0
            || height == 0
            || width > 32768
            || height > 32768
            || u64::from(width) * u64::from(height) > 201_600_000
        {
            return Err(error("Output dimensions exceed demo bounds"));
        }
        let mut encoder = PngEncoderConfig::new()
            .with_compression(Compression::Effort(1))
            .with_downcast(DowncastFlags::none())
            .with_threading(ThreadingPolicy::Sequential)
            .job()
            .encoder()
            .map_err(error)?;
        encoder
            .begin(width, height, PixelDescriptor::RGB8_SRGB)
            .map_err(error)?;
        Ok(Self {
            encoder: Some(encoder),
            width,
            height,
            rows: 0,
        })
    }
    pub fn push(&mut self, bytes: &[u8], rows: u32) -> Result<(), JsValue> {
        if rows == 0
            || self.rows + rows > self.height
            || rows > 512
            || bytes.len() != self.width as usize * rows as usize * 3
        {
            return Err(error("PNG strip shape differs"));
        }
        let pixels = PixelSlice::new(
            bytes,
            self.width,
            rows,
            self.width as usize * 3,
            PixelDescriptor::RGB8_SRGB,
        )
        .map_err(error)?;
        self.encoder
            .as_mut()
            .ok_or_else(|| error("Encoder already finished"))?
            .push_rows(pixels)
            .map_err(error)?;
        self.rows += rows;
        Ok(())
    }
    pub fn drain(&mut self) -> Result<Vec<u8>, JsValue> {
        let mut output = Vec::new();
        self.encoder
            .as_mut()
            .ok_or_else(|| error("Encoder already finished"))?
            .take_output(&mut output);
        Ok(output)
    }
    pub fn finish(mut self) -> Result<Vec<u8>, JsValue> {
        if self.rows != self.height {
            return Err(error("PNG has missing rows"));
        }
        Ok(self
            .encoder
            .take()
            .ok_or_else(|| error("Encoder already finished"))?
            .finish()
            .map_err(error)?
            .into_vec())
    }
}
