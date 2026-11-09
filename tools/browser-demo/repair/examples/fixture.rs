//! Bounded synthetic fixture using the shared JPEG encoder; no quality-study claim.
use std::{error::Error, fs::OpenOptions, io::Write};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("fixture width height quality output.jpg".into());
    }
    let width: u32 = args[1].parse()?;
    let height: u32 = args[2].parse()?;
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > 12_600_000 {
        return Err("fixture bounds".into());
    }
    let mut rgb = vec![0u8; width as usize * height as usize * 3];
    for y in 0..height {
        for x in 0..width {
            let i = (y as usize * width as usize + x as usize) * 3;
            rgb[i] = ((x + y * 3) % 256) as u8;
            rgb[i + 1] = ((x * 3 + y) % 256) as u8;
            rgb[i + 2] = ((x ^ y) % 256) as u8;
        }
    }
    let config = zenjpeg::encoder::EncoderConfig::ycbcr(
        args[3].parse::<f32>()?,
        zenjpeg::encoder::ChromaSubsampling::Quarter,
    );
    let jpeg = config.encode_bytes(&rgb, width, height, zenjpeg::encoder::PixelLayout::Rgb8Srgb)?;
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[4])?
        .write_all(&jpeg)?;
    Ok(())
}
