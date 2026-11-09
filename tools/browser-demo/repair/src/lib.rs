//! Browser glue for the unchanged native canon4 restoration request.
#![forbid(unsafe_code)]
use wasm_bindgen::prelude::*;
use zenpixels_convert::cms::PluggableCms;
use zensr_micro::{adopted::AdoptedModel, decode_all_f16};
use zensr_zenjpeg::api::{Budget, Outcome, Restorer};

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn memory_bytes() -> usize {
    core::arch::wasm32::memory_size::<0>() * 65536
}

#[wasm_bindgen]
pub struct RepairResult {
    width: u32,
    height: u32,
    applied: bool,
    reason: String,
    rgb: Vec<u8>,
}

#[wasm_bindgen]
impl RepairResult {
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }
    #[wasm_bindgen(getter)]
    pub fn applied(&self) -> bool {
        self.applied
    }
    #[wasm_bindgen(getter)]
    pub fn reason(&self) -> String {
        self.reason.clone()
    }
    pub fn take_pixels(self) -> Vec<u8> {
        self.rgb
    }
}

#[wasm_bindgen]
pub fn repair(jpeg: &[u8], weights: &[u8]) -> Result<RepairResult, JsValue> {
    let error = |e: String| JsValue::from_str(&e);
    let info = zenjpeg::decoder::Decoder::new()
        .read_info(jpeg)
        .map_err(|e| error(e.to_string()))?;
    let icc = zenjpeg::decoder::extract_icc_profile(jpeg);
    if info.has_icc_profile && icc.is_none() {
        return Err(error("JPEG ICC profile is incomplete".into()));
    }
    let (width, height) = (info.dimensions.width, info.dimensions.height);
    if width == 0
        || height == 0
        || width > 8192
        || height > 8192
        || u64::from(width) * u64::from(height) > 12_600_000
    {
        return Err(error("JPEG repair input exceeds demo bounds".into()));
    }
    if weights.len() != 86_406 {
        return Err(error("Unexpected canon4 weights length".into()));
    }
    let expanded = decode_all_f16(weights);
    let model = AdoptedModel::load_compact(&expanded, 24, 8, 1).map_err(error)?;
    // Keep coefficient projection and decoded pixels in the same physical orientation.
    // Use the shared marker walker and EXIF setter; bake orientation after repair.
    let mut normalized = jpeg.to_vec();
    let mut orientation = 1;
    for span in zenjpeg::container::MarkerIter::new(jpeg) {
        if span.kind == zenjpeg::container::MarkerKind::Eoi {
            break;
        }
        if span.kind == zenjpeg::container::MarkerKind::App(1)
            && span.payload.starts_with(b"Exif\0\0")
        {
            orientation = zenjpeg::lossless::parse_exif_orientation(span.payload).unwrap_or(1);
            if orientation != 1
                && !zenjpeg::lossless::set_exif_orientation(
                    &mut normalized[span.offset + 4..span.offset + 4 + span.payload.len()],
                    1,
                )
            {
                return Err(error("Could not normalize JPEG orientation".into()));
            }
            break;
        }
    }
    let restorer = Restorer::new().with_tier(Budget::Realtime, model);
    let outcome = restorer
        .restore(&normalized)
        .threads(1)
        .run()
        .map_err(|e| error(e.to_string()))?;
    let (applied, reason) = match &outcome {
        Outcome::Restored { .. } => (
            true,
            "Canon4 repair and coefficient projection applied".into(),
        ),
        Outcome::Unchanged { why, .. } => (false, format!("Router left JPEG unchanged: {why:?}")),
    };
    let pixels = outcome.pixels();
    let (width, height) = (pixels.width() as u32, pixels.height() as u32);
    let mut rgb = pixels.to_rgb8();
    // Projection stays in JPEG sample space. Convert repaired samples to display sRGB afterward.
    if let Some(icc) = icc {
        let mut transform = zenpixels_convert::MoxCms
            .build_source_transform(
                zenpixels::ColorProfileSource::Icc(&icc),
                zenpixels::ColorProfileSource::Cicp(zenpixels::Cicp::SRGB),
                zenpixels::PixelFormat::Rgb8,
                zenpixels::PixelFormat::Rgb8,
                &zenpixels_convert::ConvertOptions::retain_channels(),
            )
            .ok_or_else(|| error("JPEG ICC color space is unsupported for RGB repair".into()))?
            .map_err(|e| error(e.to_string()))?;
        transform.prepare(width).map_err(|e| error(e.to_string()))?;
        let row_bytes = width as usize * 3;
        let mut row = vec![0; row_bytes];
        for source in rgb.chunks_exact_mut(row_bytes) {
            transform
                .try_transform_row(source, &mut row, width)
                .map_err(|e| error(e.to_string()))?;
            source.copy_from_slice(&row);
        }
    }
    let (width, height, rgb) = if orientation != 1 {
        let source = zenpixels::PixelSlice::new(
            &rgb,
            width,
            height,
            width as usize * 3,
            zenpixels::PixelDescriptor::RGB8_SRGB,
        )
        .map_err(|e| error(e.to_string()))?;
        let orient = zenpixels::Orientation::from_exif(orientation)
            .ok_or_else(|| error("Invalid EXIF orientation".into()))?;
        let out = zenpixels_convert::orient::apply_orientation(source, orient);
        let (w, h) = orient.output_dimensions(width, height);
        (
            w,
            h,
            out.as_contiguous_bytes()
                .ok_or_else(|| error("Oriented output is not contiguous".into()))?
                .to_vec(),
        )
    } else {
        (width, height, rgb)
    };
    Ok(RepairResult {
        width,
        height,
        applied,
        reason,
        rgb,
    })
}
