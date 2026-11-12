//! Score-only RGB8 comparison against an independent SR runtime.
#![forbid(unsafe_code)]
use std::{error::Error, fs};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("sr_diff reference.rgb8 candidate.rgb8 width height".into());
    }
    let w: usize = args[3].parse()?;
    let h: usize = args[4].parse()?;
    let a = fs::read(&args[1])?;
    let b = fs::read(&args[2])?;
    if w == 0
        || h == 0
        || w.checked_mul(h).and_then(|n| n.checked_mul(3)) != Some(a.len())
        || a.len() != b.len()
    {
        return Err("RGB8 shape mismatch".into());
    }
    let av: Vec<[u8; 3]> = a.as_chunks::<3>().0.to_vec();
    let bv: Vec<[u8; 3]> = b.as_chunks::<3>().0.to_vec();
    let historical_score = fast_ssim2::compute_ssimulacra2(
        imgref::Img::new(av, w, h).as_ref(),
        imgref::Img::new(bv, w, h).as_ref(),
    )?;
    let reference = ssim2_reference::PixelSlice::new(
        &a,
        w as u32,
        h as u32,
        w * 3,
        ssim2_reference::PixelDescriptor::RGB8_SRGB,
    )?;
    let candidate = ssim2_reference::PixelSlice::new(
        &b,
        w as u32,
        h as u32,
        w * 3,
        ssim2_reference::PixelDescriptor::RGB8_SRGB,
    )?;
    let score = ssim2_reference::compute_ssimulacra2(&reference, &candidate)?;
    let mut max = 0u8;
    let mut different = 0usize;
    let mut sse = 0u64;
    for (&x, &y) in a.iter().zip(&b) {
        let d = x.abs_diff(y);
        max = max.max(d);
        different += usize::from(d != 0);
        sse += u64::from(d) * u64::from(d);
    }
    println!(
        "{}",
        serde_json::json!({"ssim2":score,"ssim2_0_8_2":historical_score,"max_rgb8_difference":max,"different_values":different,"values":a.len(),"sse":sse})
    );
    Ok(())
}
