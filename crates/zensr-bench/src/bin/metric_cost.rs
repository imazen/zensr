//! What actually costs time per eval cell.
//!
//! The encoder is 30x faster in-process, but the sweep needs only ~4 encodes/sec
//! against a 200/sec subprocess floor, so encoding was never the bottleneck.
//! This measures the pieces that are.
use std::path::PathBuf;
use std::time::Instant;
use zensr_bench::*;
fn main() {
    let root = PathBuf::from(std::env::args().nth(1).expect("root"));
    let n: usize = std::env::args()
        .nth(2)
        .map(|s| s.parse().unwrap())
        .unwrap_or(20);
    let mut imgs = Vec::new();
    for (_, dir) in &subcorpora_for(&root) {
        for f in list_images(&root.join(dir)) {
            if imgs.len() >= n {
                break;
            }
            if let Some(i) = decode_any(&f).and_then(|i| center_crop(&i, 512)) {
                imgs.push(i);
            }
        }
        if imgs.len() >= n {
            break;
        }
    }
    let pairs: Vec<(Rgb8Img, Rgb8Img)> = imgs
        .iter()
        .map(|i| {
            let px: &[rgb::Rgb<u8>] = bytemuck::cast_slice(&i.px);
            let jpg = zenjpeg::encoder::EncoderConfig::ycbcr(
                75.0,
                zenjpeg::encoder::ChromaSubsampling::Quarter,
            )
            .encode(px, i.w as u32, i.h as u32)
            .expect("enc");
            let d = zenjpeg::decoder::Decoder::new()
                .decode(&jpg, enough::Unstoppable)
                .expect("dec");
            let (w, h) = d.dimensions();
            (
                i.clone(),
                Rgb8Img {
                    px: d.pixels_u8().unwrap().to_vec(),
                    w: w as usize,
                    h: h as usize,
                },
            )
        })
        .collect();
    let n = pairs.len();
    println!("{n} pairs at 512x512, single-threaded\n");
    for (label, f) in [
        (
            "psnr",
            (|a: &Rgb8Img, b: &Rgb8Img| {
                psnr_rgb8(a, b);
            }) as fn(&Rgb8Img, &Rgb8Img),
        ),
        ("ssim2", |a, b| {
            ssim2(a, b);
        }),
        ("butteraugli n3", |a, b| {
            butter_n3(a, b);
        }),
    ] {
        let t = Instant::now();
        for (a, b) in &pairs {
            f(a, b);
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / n as f64;
        println!("{label:<18} {ms:7.2} ms/call");
    }
    let t = Instant::now();
    for (_, b) in &pairs {
        let px: &[rgb::Rgb<u8>] = bytemuck::cast_slice(&b.px);
        let _ = zenjpeg::encoder::EncoderConfig::ycbcr(
            75.0,
            zenjpeg::encoder::ChromaSubsampling::Quarter,
        )
        .scan_mode(zenjpeg::encoder::ProgressiveScanMode::Baseline)
        .encode(px, b.w as u32, b.h as u32);
    }
    println!(
        "{:<18} {:7.2} ms/call",
        "zenjpeg encode",
        t.elapsed().as_secs_f64() * 1000.0 / n as f64
    );
}
