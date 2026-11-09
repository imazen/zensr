//! Header-only browser input admission; shared JPEG parser, no pixel decode.
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
