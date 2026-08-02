//! Oracle-chroma ceiling for the S5 two-trunk question (SYSTEMS.md scope
//! correction, 2026-07-27): if chroma restoration were PERFECT, how much
//! ssim2 is left on the table by the current pipeline at 4:2:0?
//!
//! Arms per (file, q), turbo 420:
//!   decode          — pixel-exact zenjpeg decode
//!   decode_oc       — decode Y + GROUND-TRUTH Cb/Cr (oracle chroma)
//!   restored        — restore_jpeg(model) full pipeline
//!   restored_oc     — restored Y + ground-truth Cb/Cr
//!
//! restored_oc − restored = hard upper bound on ANY chroma-side
//! architecture gain (incl. S5 half-res trunk). decode_oc − decode = the
//! same bound before the model. If these gaps are small, S5 is capped.
//!
//! TSV: sub file q arm psnr ssim2 butter_n3
//! Usage: chroma_ceiling <imazen26-root> <out-tsv> [per-sub=3] [threads=12]
//!        [model=dejpeg7_graphics]

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;
use zensr_bench::*;
use zensr_micro::consist::{rgb_to_ycbcr_planes, ycbcr_to_rgb_planes};
use zensr_zenjpeg::{restore_jpeg, RestoreConfig};

fn encode_turbo(img: &Rgb8Img, q: u32, td: &PathBuf) -> Vec<u8> {
    let ppm = td.join("c.ppm");
    let jpg = td.join("c.jpg");
    let mut buf = format!("P6\n{} {}\n255\n", img.w, img.h).into_bytes();
    buf.extend_from_slice(&img.px);
    std::fs::write(&ppm, buf).unwrap();
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
    std::fs::read(&jpg).unwrap()
}

/// planes = planar RGB f32 [3,h,w] in [0,1] -> (y, cb, cr)
fn split_ycc(planes: &[f32], plane: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (mut y, mut cb, mut cr) = (vec![0.0; plane], vec![0.0; plane], vec![0.0; plane]);
    rgb_to_ycbcr_planes(planes, plane, &mut y, &mut cb, &mut cr);
    (y, cb, cr)
}

fn rgb8_to_planar(img: &Rgb8Img) -> Vec<f32> {
    to_planar_f32(img)
}

fn merge_y_with_oracle_chroma(
    y: &[f32],
    gt_cb: &[f32],
    gt_cr: &[f32],
    w: usize,
    h: usize,
) -> Rgb8Img {
    let plane = w * h;
    let mut rgb = vec![0.0f32; 3 * plane];
    ycbcr_to_rgb_planes(y, gt_cb, gt_cr, &mut rgb, plane);
    planar_to_rgb8(&rgb, w, h)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().expect("corpus root"));
    let out_path = PathBuf::from(args.next().expect("out tsv"));
    let per_sub: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(3);
    let threads: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(12);
    let model_name = args.next().unwrap_or_else(|| "dejpeg7_graphics".into());
    let model = load_adopted(&model_name).expect("model");
    let td = PathBuf::from(std::env::var("HOME").unwrap())
        .join("tmp")
        .join(format!("zensr-cc-{}", std::process::id()));
    std::fs::create_dir_all(&td).unwrap();

    let mut tsv = String::from("sub\tfile\tq\tarm\tpsnr\tssim2\tbutter_n3\n");
    for (sub, dir) in SUBCORPORA {
        let mut used = 0usize;
        for f in list_images(&root.join(dir)) {
            if used >= per_sub {
                break;
            }
            let Some(img) = decode_any(&f) else { continue };
            let Some(hr) = center_crop(&img, 512) else {
                continue;
            };
            used += 1;
            let fname = f.file_name().unwrap().to_string_lossy().to_string();
            let plane = hr.w * hr.h;
            let gt_planes = rgb8_to_planar(&hr);
            let (_gy, gcb, gcr) = split_ycc(&gt_planes, plane);
            for q in [15u32, 35, 55, 75, 90] {
                let jpg = encode_turbo(&hr, q, &td);
                // decode arm
                let dec = zenjpeg::decoder::Decoder::new()
                    .decode(&jpg, enough::Unstoppable)
                    .unwrap();
                let px = dec.pixels_u8().unwrap();
                let dimg = Rgb8Img {
                    px: px.to_vec(),
                    w: hr.w,
                    h: hr.h,
                };
                let dplanes = rgb8_to_planar(&dimg);
                let (dy, _, _) = split_ycc(&dplanes, plane);
                // restored arm
                let r = restore_jpeg(
                    &jpg,
                    &model,
                    &RestoreConfig::default().with_threads(threads),
                )
                .expect("restore");
                let rimg = Rgb8Img {
                    px: r.to_rgb8(),
                    w: r.width,
                    h: r.height,
                };
                let (ry, _, _) = split_ycc(&r.planes, plane);
                // lattice floor: GT chroma box-downsampled 2x2 + bilinear-up,
                // GT luma untouched, NO quantization — the pure subsampling loss.
                let (hw, hh) = (hr.w / 2, hr.h / 2);
                let mut lcb = vec![0.0f32; hw * hh];
                let mut lcr = vec![0.0f32; hw * hh];
                for yy in 0..hh {
                    for xx in 0..hw {
                        let i00 = (2 * yy) * hr.w + 2 * xx;
                        let i10 = (2 * yy + 1) * hr.w + 2 * xx;
                        lcb[yy * hw + xx] =
                            0.25 * (gcb[i00] + gcb[i00 + 1] + gcb[i10] + gcb[i10 + 1]);
                        lcr[yy * hw + xx] =
                            0.25 * (gcr[i00] + gcr[i00 + 1] + gcr[i10] + gcr[i10 + 1]);
                    }
                }
                let up = |half: &Vec<f32>| -> Vec<f32> {
                    let mut full = vec![0.0f32; plane];
                    for y in 0..hr.h {
                        let fy = ((y as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, (hh - 1) as f32);
                        let (y0, ty) = (fy.floor() as usize, fy.fract());
                        let y1 = (y0 + 1).min(hh - 1);
                        for x in 0..hr.w {
                            let fx = ((x as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, (hw - 1) as f32);
                            let (x0, tx) = (fx.floor() as usize, fx.fract());
                            let x1 = (x0 + 1).min(hw - 1);
                            let top = half[y0 * hw + x0] * (1.0 - tx) + half[y0 * hw + x1] * tx;
                            let bot = half[y1 * hw + x0] * (1.0 - tx) + half[y1 * hw + x1] * tx;
                            full[y * hr.w + x] = top * (1.0 - ty) + bot * ty;
                        }
                    }
                    full
                };
                let (gy_full, _, _) = split_ycc(&gt_planes, plane);
                let lattice =
                    merge_y_with_oracle_chroma(&gy_full, &up(&lcb), &up(&lcr), hr.w, hr.h);
                // joint-bilateral 2x upsample of the SAME clean half-res chroma,
                // guided by full-res GT luma: the classic guided-upsampling bound.
                let jbu = |half: &Vec<f32>| -> Vec<f32> {
                    let mut full = vec![0.0f32; plane];
                    let sig_r = 0.06f32; // luma range sigma ([0,1] units)
                    for y in 0..hr.h {
                        for x in 0..hr.w {
                            let g = gy_full[y * hr.w + x];
                            let cy = ((y as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, (hh - 1) as f32);
                            let cx = ((x as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, (hw - 1) as f32);
                            let (y0, x0) = (cy.floor() as isize, cx.floor() as isize);
                            let mut num = 0.0f32;
                            let mut den = 0.0f32;
                            for dy in -1..=2isize {
                                let sy = (y0 + dy).clamp(0, hh as isize - 1) as usize;
                                for dx in -1..=2isize {
                                    let sx = (x0 + dx).clamp(0, hw as isize - 1) as usize;
                                    // guide luma at the chroma sample's center (2x2 mean)
                                    let gy2 = (2 * sy).min(hr.h - 1);
                                    let gx2 = (2 * sx).min(hr.w - 1);
                                    let gs = 0.25
                                        * (gy_full[gy2 * hr.w + gx2]
                                            + gy_full[gy2 * hr.w + (gx2 + 1).min(hr.w - 1)]
                                            + gy_full[(gy2 + 1).min(hr.h - 1) * hr.w + gx2]
                                            + gy_full[(gy2 + 1).min(hr.h - 1) * hr.w
                                                + (gx2 + 1).min(hr.w - 1)]);
                                    let ds = ((sy as f32 - cy).powi(2) + (sx as f32 - cx).powi(2))
                                        / (2.0 * 0.8f32.powi(2));
                                    let dr = (g - gs).powi(2) / (2.0 * sig_r * sig_r);
                                    let w = (-ds - dr).exp();
                                    num += w * half[sy * hw + sx];
                                    den += w;
                                }
                            }
                            full[y * hr.w + x] = num / den.max(1e-9);
                        }
                    }
                    full
                };
                let lattice_jbu =
                    merge_y_with_oracle_chroma(&gy_full, &jbu(&lcb), &jbu(&lcr), hr.w, hr.h);
                let arms: Vec<(&str, Rgb8Img)> = vec![
                    ("lattice_floor", lattice),
                    ("lattice_jbu", lattice_jbu),
                    ("decode", dimg),
                    (
                        "decode_oc",
                        merge_y_with_oracle_chroma(&dy, &gcb, &gcr, hr.w, hr.h),
                    ),
                    ("restored", rimg),
                    (
                        "restored_oc",
                        merge_y_with_oracle_chroma(&ry, &gcb, &gcr, hr.w, hr.h),
                    ),
                ];
                for (arm, o) in &arms {
                    let s = score(&hr, o);
                    let _ = writeln!(
                        tsv,
                        "{sub}\t{fname}\t{q}\t{arm}\t{:.3}\t{:.3}\t{:.4}",
                        s.psnr, s.ssim2, s.butter
                    );
                }
            }
            eprintln!("done {sub}/{fname}");
        }
    }
    std::fs::write(&out_path, &tsv).expect("write tsv");
}
