//! Visual gallery: dump original / decoded / restored / restored-strict crops
//! as PPM for the measurement console. Includes deliberately-chosen REGRESSION
//! cases (files where the model loses) alongside wins — the console should
//! show what a -9 ssim2 outlier actually looks like, not just count it.
//!
//! Usage: gallery_dump <imazen26-root> <outdir> <model> [crop=192]
//!   env ZENSR_GAL_CASES="sub:file:q,sub:file:q,..."

use std::path::PathBuf;
use std::process::Command;
use zensr_bench::*;
use zensr_zenjpeg::{restore_jpeg, Projection, ProjectionConfig, RestoreConfig};

fn write_ppm(img: &Rgb8Img, p: &PathBuf) {
    let mut buf = format!("P6\n{} {}\n255\n", img.w, img.h).into_bytes();
    buf.extend_from_slice(&img.px);
    std::fs::write(p, buf).unwrap();
}

fn crop_at(img: &Rgb8Img, x0: usize, y0: usize, s: usize) -> Rgb8Img {
    let (w, h) = (s.min(img.w - x0), s.min(img.h - y0));
    let mut px = Vec::with_capacity(3 * w * h);
    for y in 0..h {
        px.extend_from_slice(&img.px[((y0 + y) * img.w + x0) * 3..][..w * 3]);
    }
    Rgb8Img { px, w, h }
}

/// Most-detailed sub-crop (max mean |laplacian| on luma) — where artifacts show.
fn busiest(img: &Rgb8Img, s: usize) -> (usize, usize) {
    let l: Vec<f32> = (0..img.w * img.h)
        .map(|i| 0.299 * img.px[i * 3] as f32 + 0.587 * img.px[i * 3 + 1] as f32 + 0.114 * img.px[i * 3 + 2] as f32)
        .collect();
    let (mut best, mut bxy) = (-1.0f32, (0, 0));
    let step = (s / 2).max(16);
    for y0 in (0..img.h.saturating_sub(s)).step_by(step) {
        for x0 in (0..img.w.saturating_sub(s)).step_by(step) {
            let mut e = 0.0f32;
            for y in y0 + 1..(y0 + s - 1).min(img.h - 1) {
                for x in x0 + 1..(x0 + s - 1).min(img.w - 1) {
                    let c = l[y * img.w + x];
                    e += (4.0 * c - l[(y - 1) * img.w + x] - l[(y + 1) * img.w + x]
                        - l[y * img.w + x - 1] - l[y * img.w + x + 1]).abs();
                }
            }
            if e > best { best = e; bxy = (x0, y0); }
        }
    }
    bxy
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().expect("root"));
    let outdir = PathBuf::from(args.next().expect("outdir"));
    let model_name = args.next().unwrap_or_else(|| "dejpeg_rt24g".into());
    let crop: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(192);
    let model = load_adopted(&model_name).expect("model");
    std::fs::create_dir_all(&outdir).unwrap();
    let td = outdir.join("_tmp");
    std::fs::create_dir_all(&td).unwrap();

    let cases: Vec<(String, String, u32)> = std::env::var("ZENSR_GAL_CASES")
        .expect("ZENSR_GAL_CASES")
        .split(',')
        .map(|c| {
            let p: Vec<&str> = c.split(':').collect();
            (p[0].to_string(), p[1].to_string(), p[2].parse().unwrap())
        })
        .collect();

    for (sub, fname, q) in cases {
        let dir = SUBCORPORA.iter().find(|(s, _)| *s == sub).map(|(_, d)| *d).expect("sub");
        let Some(path) = list_images(&root.join(dir)).into_iter().find(|p| {
            p.file_name().unwrap().to_string_lossy().contains(&fname)
        }) else { eprintln!("MISS {sub}/{fname}"); continue };
        let Some(img) = decode_any(&path) else { continue };
        let Some(gt512) = center_crop(&img, 512) else { continue };
        let ppm = td.join("g.ppm");
        let jpg = td.join("g.jpg");
        write_ppm(&gt512, &ppm);
        assert!(Command::new("cjpeg")
            .args(["-quality", &q.to_string(), "-sample", "2x2", "-optimize", "-outfile"])
            .arg(&jpg).arg(&ppm).status().unwrap().success());
        let data = std::fs::read(&jpg).unwrap();
        let dec = zenjpeg::decoder::Decoder::new().decode(&data, enough::Unstoppable).unwrap();
        let ident = Rgb8Img { px: dec.pixels_u8().unwrap().to_vec(), w: gt512.w, h: gt512.h };
        let r = restore_jpeg(&data, &model, &RestoreConfig::default().with_threads(12)).unwrap();
        let rest = Rgb8Img { px: r.to_rgb8(), w: r.width, h: r.height };
        let rs = restore_jpeg(&data, &model, &RestoreConfig::default().with_threads(12)
            .with_projection(Projection::Fixed(
                ProjectionConfig::with_slack_q(0.0).with_slack_abs(0.0)))).unwrap();
        let strict = Rgb8Img { px: rs.to_rgb8(), w: rs.width, h: rs.height };
        let (cx, cy) = busiest(&gt512, crop);
        let tag = format!("{sub}_{}_{q}", fname.split('.').next().unwrap_or("f").replace('-', "_"));
        for (name, im) in [("gt", &gt512), ("jpeg", &ident), ("restored", &rest), ("strict", &strict)] {
            write_ppm(&crop_at(im, cx, cy, crop), &outdir.join(format!("{tag}__{name}.ppm")));
        }
        let s_i = score(&gt512, &ident);
        let s_r = score(&gt512, &rest);
        let s_s = score(&gt512, &strict);
        println!("{tag}\tident {:.2}\trestored {:.2} ({:+.2})\tstrict {:.2} ({:+.2})",
                 s_i.ssim2, s_r.ssim2, s_r.ssim2 - s_i.ssim2, s_s.ssim2, s_s.ssim2 - s_i.ssim2);
    }
}
