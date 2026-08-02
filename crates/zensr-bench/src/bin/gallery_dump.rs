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

/// (label, slack_q, slack_abs) — None = projection off entirely.
const SLACKS: &[(&str, Option<(f32, f32)>)] = &[
    ("noproj", None),
    ("strict", Some((0.0, 0.0))),
    ("p99", Some((0.05, 1.0))),
    ("calibrated", Some((0.15, 1.5))),
];

/// Amplified |a-b| error map (x8, clamped) — shows WHERE strategies differ.
fn diffmap(a: &Rgb8Img, b: &Rgb8Img) -> Rgb8Img {
    let px =
        a.px.iter()
            .zip(b.px.iter())
            .map(|(x, y)| (((*x as i32 - *y as i32).abs() * 8).min(255)) as u8)
            .collect();
    Rgb8Img { px, w: a.w, h: a.h }
}

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
        .map(|i| {
            0.299 * img.px[i * 3] as f32
                + 0.587 * img.px[i * 3 + 1] as f32
                + 0.114 * img.px[i * 3 + 2] as f32
        })
        .collect();
    let (mut best, mut bxy) = (-1.0f32, (0, 0));
    let step = (s / 2).max(16);
    for y0 in (0..img.h.saturating_sub(s)).step_by(step) {
        for x0 in (0..img.w.saturating_sub(s)).step_by(step) {
            let mut e = 0.0f32;
            for y in y0 + 1..(y0 + s - 1).min(img.h - 1) {
                for x in x0 + 1..(x0 + s - 1).min(img.w - 1) {
                    let c = l[y * img.w + x];
                    e += (4.0 * c
                        - l[(y - 1) * img.w + x]
                        - l[(y + 1) * img.w + x]
                        - l[y * img.w + x - 1]
                        - l[y * img.w + x + 1])
                        .abs();
                }
            }
            if e > best {
                best = e;
                bxy = (x0, y0);
            }
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
        let dir = SUBCORPORA
            .iter()
            .find(|(s, _)| *s == sub)
            .map(|(_, d)| *d)
            .expect("sub");
        let Some(path) = list_images(&root.join(dir))
            .into_iter()
            .find(|p| p.file_name().unwrap().to_string_lossy().contains(&fname))
        else {
            eprintln!("MISS {sub}/{fname}");
            continue;
        };
        let Some(img) = decode_any(&path) else {
            continue;
        };
        let Some(gt512) = center_crop(&img, 512) else {
            continue;
        };
        let ppm = td.join("g.ppm");
        let jpg = td.join("g.jpg");
        write_ppm(&gt512, &ppm);
        assert!(Command::new("cjpeg")
            .args([
                "-quality",
                &q.to_string(),
                "-sample",
                "2x2",
                "-optimize",
                "-outfile"
            ])
            .arg(&jpg)
            .arg(&ppm)
            .status()
            .unwrap()
            .success());
        // ZENSR_GAL_GENS=2 -> decode and re-encode once more (aligned recompress)
        if std::env::var("ZENSR_GAL_GENS").as_deref() == Ok("2") {
            let d1 = zenjpeg::decoder::Decoder::new()
                .decode(&std::fs::read(&jpg).unwrap(), enough::Unstoppable)
                .unwrap();
            let (w1, h1) = d1.dimensions();
            let mid = Rgb8Img {
                px: d1.pixels_u8().unwrap().to_vec(),
                w: w1 as usize,
                h: h1 as usize,
            };
            write_ppm(&mid, &ppm);
            assert!(Command::new("cjpeg")
                .args([
                    "-quality",
                    &q.to_string(),
                    "-sample",
                    "2x2",
                    "-optimize",
                    "-outfile"
                ])
                .arg(&jpg)
                .arg(&ppm)
                .status()
                .unwrap()
                .success());
        }
        let data = std::fs::read(&jpg).unwrap();
        let dec = zenjpeg::decoder::Decoder::new()
            .decode(&data, enough::Unstoppable)
            .unwrap();
        let ident = Rgb8Img {
            px: dec.pixels_u8().unwrap().to_vec(),
            w: gt512.w,
            h: gt512.h,
        };
        let (cx, cy) = busiest(&gt512, crop);
        let gens = std::env::var("ZENSR_GAL_GENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1usize);
        let tag = format!(
            "{sub}_{}_{q}_g{gens}",
            fname.split('.').next().unwrap_or("f").replace('-', "_")
        );
        write_ppm(
            &crop_at(&gt512, cx, cy, crop),
            &outdir.join(format!("{tag}__gt.ppm")),
        );
        write_ppm(
            &crop_at(&ident, cx, cy, crop),
            &outdir.join(format!("{tag}__jpeg.ppm")),
        );
        write_ppm(
            &diffmap(
                &crop_at(&gt512, cx, cy, crop),
                &crop_at(&ident, cx, cy, crop),
            ),
            &outdir.join(format!("{tag}__jpegdiff.ppm")),
        );
        let s_i = score(&gt512, &ident);
        let mut line = format!("{tag}\tident {:.2}", s_i.ssim2);
        for (label, sl) in SLACKS {
            let mut rc = RestoreConfig::default().with_threads(12);
            rc = match sl {
                None => rc.with_projection(Projection::Off),
                Some((sq, sa)) => rc.with_projection(Projection::Fixed(
                    ProjectionConfig::with_slack_q(*sq).with_slack_abs(*sa),
                )),
            };
            let r = restore_jpeg(&data, &model, &rc).unwrap();
            let im = Rgb8Img {
                px: r.to_rgb8(),
                w: r.width,
                h: r.height,
            };
            let c = crop_at(&im, cx, cy, crop);
            write_ppm(&c, &outdir.join(format!("{tag}__{label}.ppm")));
            write_ppm(
                &diffmap(&crop_at(&gt512, cx, cy, crop), &c),
                &outdir.join(format!("{tag}__{label}diff.ppm")),
            );
            let sc = score(&gt512, &im);
            line += &format!("\t{label} {:.2} ({:+.2})", sc.ssim2, sc.ssim2 - s_i.ssim2);
        }
        println!("{line}");
    }
}
