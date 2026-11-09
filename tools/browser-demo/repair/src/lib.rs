//! Thin WASM bindings over the same zenjpeg + zensr integration used natively.
#![forbid(unsafe_code)]
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use zensr::api::{Budget, Restorer};
use zensr::{AdoptedModel, decode_all_f16};

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
    if weights.len() != 86_406 {
        return Err(error("Unexpected canon4 weights length".into()));
    }
    let model = AdoptedModel::load_compact(&decode_all_f16(weights), 24, 8, 1).map_err(error)?;
    let restorer = Arc::new(Restorer::new().with_tier(Budget::Realtime, model));
    let decoder = zenjpeg::decoder::Decoder::new()
        .max_pixels(12_600_000)
        .max_memory(950 * 1024 * 1024)
        .num_threads(1)
        .correct_color(Some(zenjpeg::decoder::TargetColorSpace::Srgb))
        .artifact_removal(restorer);
    let result = decoder
        .decode(jpeg, enough::Unstoppable)
        .map_err(|e| error(e.to_string()))?;
    let (width, height) = result.dimensions();
    let report = result
        .artifact_removal_report()
        .ok_or_else(|| error("Missing codec restoration report".into()))?;
    let applied = report.applied;
    let reason = report.reason.clone();
    let rgb = result
        .into_pixels_u8()
        .ok_or_else(|| error("Codec did not return RGB8".into()))?;
    Ok(RepairResult {
        width,
        height,
        applied,
        reason,
        rgb,
    })
}
