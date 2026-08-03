//! Dump SR comparison crops as PNG for visual inspection.
//!
//! Exists because `renders` is the one subcorpus where the metrics disagree:
//! paired per file, PSNR says SPANF wins on 8 of 8 and butteraugli on 7 of 8,
//! while SSIM2 says the median image gets worse
//! (`benchmarks/sr_pinned_2026-08-03.md`). A number cannot settle that; the
//! pixels can.
//!
//! Same protocol as the `eval` bin — centre-crop HR, CatmullRom /4 down, then
//! up by each method — so what is written here is exactly what was scored.
//!
//! Usage: sr_dump <corpus-root> <outdir> [sub=renders] [crop=512]

use std::path::PathBuf;
use zensr_bench::*;
use zensr_micro::{spanf_x4_tiled, SpanfModel};

fn main() {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(
        args.next()
            .expect("usage: sr_dump <root> <outdir> [sub] [crop]"),
    );
    let outdir = PathBuf::from(args.next().expect("outdir"));
    let sub = args.next().unwrap_or_else(|| "unsplash-renders".into());
    let crop: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(512);
    std::fs::create_dir_all(&outdir).expect("mkdir");

    let wdir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let wbytes = std::fs::read(wdir.join("spanf_weights.raw")).expect("spanf_weights.raw");
    let wbuf: Vec<f32> = wbytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let model = SpanfModel::new(wbuf).expect("model");

    let pinned = load_pinned(&pin_path());
    let want = pinned.as_ref().and_then(|m| m.get(&sub));

    for f in list_images(&root.join(&sub)) {
        let fname = f.file_name().unwrap().to_string_lossy().to_string();
        if let Some(w) = want {
            if !w.contains(&pinned_stem(&fname)) {
                continue;
            }
        }
        let Some(img) = decode_any(&f) else { continue };
        let Some(hr) = center_crop(&img, crop) else {
            continue;
        };
        let (lw, lh) = (hr.w / 4, hr.h / 4);
        let lr = resize_rgb8(&hr, lw, lh, zenresize::Filter::CatmullRom);

        let lr_planar = to_planar_f32(&lr);
        let sr_planar = spanf_x4_tiled(&model, &lr_planar, lh, lw, 8, 0);
        let sr = planar_to_rgb8(&sr_planar, hr.w, hr.h);
        let lanczos = resize_rgb8(&lr, hr.w, hr.h, zenresize::Filter::Lanczos);

        let stem = pinned_stem(&fname);
        for (tag, im) in [("0ref", &hr), ("1lanczos", &lanczos), ("2spanf", &sr)] {
            write_png(im, &outdir.join(format!("{stem}__{tag}.png")));
        }
        // Amplified absolute difference against the reference, so where each
        // method departs is visible rather than inferred. Same gain for both,
        // or the panels would not be comparable.
        for (tag, im) in [("3diff_lanczos", &lanczos), ("4diff_spanf", &sr)] {
            write_png(
                &diff_x8(&hr, im),
                &outdir.join(format!("{stem}__{tag}.png")),
            );
        }
        eprintln!(
            "{stem}: {}x{}  ssim2 lanczos {:.2} spanf {:.2}  psnr {:.2}/{:.2}",
            hr.w,
            hr.h,
            ssim2(&hr, &lanczos),
            ssim2(&hr, &sr),
            psnr_rgb8(&hr, &lanczos),
            psnr_rgb8(&hr, &sr),
        );
    }
    eprintln!("wrote to {}", outdir.display());
}

/// |a-b| * 8, clamped — visible without being saturated into uselessness.
fn diff_x8(a: &Rgb8Img, b: &Rgb8Img) -> Rgb8Img {
    let px =
        a.px.iter()
            .zip(&b.px)
            .map(|(&x, &y)| ((x as i16 - y as i16).unsigned_abs() * 8).min(255) as u8)
            .collect();
    Rgb8Img { px, w: a.w, h: a.h }
}

fn write_png(img: &Rgb8Img, path: &PathBuf) {
    let px: &[rgb::Rgb<u8>] = bytemuck::cast_slice(&img.px);
    let bytes = zenpng::encode_rgb8(
        imgref::ImgRef::new(px, img.w, img.h),
        None,
        &zenpng::EncodeConfig::default(),
        &enough::Unstoppable,
        &enough::Unstoppable,
    )
    .expect("png encode");
    std::fs::write(path, bytes).expect("write png");
}
