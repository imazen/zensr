//! Thin C ABI over zensr-micro (zentract-abi pattern: FFI unsafe isolated here;
//! the algorithm crate stays forbid(unsafe_code)). Size-probe target.

use zensr_micro::{SpanfWeights, spanf_x4_simd, TOTAL_FLOATS};

/// SPANF x4 on [3,h,w] f32 NCHW. `weights` = TOTAL_FLOATS f32, dump order.
/// `out` must hold 3*h*w*16 f32. Returns 0 on success, negative on error.
///
/// # Safety
/// Pointers must be valid for the documented lengths.
#[no_mangle]
pub unsafe extern "C" fn zensr_spanf_x4(
    input: *const f32,
    h: usize,
    w: usize,
    weights: *const f32,
    out: *mut f32,
) -> i32 {
    if input.is_null() || weights.is_null() || out.is_null() || h == 0 || w == 0 {
        return -1;
    }
    let input = core::slice::from_raw_parts(input, 3 * h * w);
    let wbuf = core::slice::from_raw_parts(weights, TOTAL_FLOATS);
    let out = core::slice::from_raw_parts_mut(out, 3 * h * w * 16);
    let Ok(parsed) = SpanfWeights::parse(wbuf) else {
        return -2;
    };
    let result = spanf_x4_simd(input, h, w, &parsed);
    out.copy_from_slice(&result);
    0
}
