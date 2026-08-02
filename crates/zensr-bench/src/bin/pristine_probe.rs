//! How much downscaling makes a JPEG source effectively pristine?
//! (user proposal 2026-08-01: recover JPEG-only corpus dirs as clean sources)
//!
//! JPEG artifacts are 8x8-block structured. Downscaling by N shrinks the
//! block footprint to 8/N pixels and averages ringing across neighbours, so
//! past some N the residue drops below the level that would bias training or
//! eval. This measures that N directly, on sources we KNOW are clean:
//!
//!   A = downscale(pristine PNG, N)             <- true reference
//!   B = downscale(JPEG(pristine PNG, q), N)    <- what a JPEG source yields
//!
//! PSNR(A,B) is the residual contamination. It is an upper bound on the error
//! a downscaled-JPEG source would introduce, and it is directly comparable to
//! the gains we are trying to measure (a restoration gain of +3 ssim2 is
//! meaningless if the reference itself carries more error than that).
//!
//! Also reports the SAME comparison after the pipeline's own degradation, i.e.
//! whether a downscaled-JPEG reference changes a measured restoration gain —
//! which is the number that actually decides whether the trick is usable.
//!
//! TSV: sub file q scale kernel psnr_ref ssim2_ref gain_true gain_jpegsrc gain_err
//! Usage: pristine_probe <imazen26-root> <out-tsv> [per-sub=3] [threads=12]

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;
use zensr_bench::*;
use zensr_zenjpeg::{restore_jpeg, RestoreConfig};

/// Source-JPEG qualities to simulate (what an unsplash-class file looks like).
const SRC_QS: &[u32] = &[75, 85, 92];
/// Downscale factors to test.
const SCALES: &[usize] = &[1, 2, 3, 4];
/// Quality the pipeline is then evaluated at.
const EVAL_Q: u32 = 35;

fn write_ppm(img: &Rgb8Img, p: &PathBuf) {
    let mut buf = format!("P6\n{} {}\n255\n", img.w, img.h).into_bytes();
    buf.extend_from_slice(&img.px);
    std::fs::write(p, buf).unwrap();
}

fn enc_turbo(img: &Rgb8Img, q: u32, td: &PathBuf, ss: &str) -> Vec<u8> {
    let ppm = td.join("p.ppm");
    let jpg = td.join("p.jpg");
    write_ppm(img, &ppm);
    assert!(Command::new("cjpeg")
        .args([
            "-quality",
            &q.to_string(),
            "-sample",
            ss,
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

fn dec(data: &[u8]) -> Rgb8Img {
    let d = zenjpeg::decoder::Decoder::new()
        .decode(data, enough::Unstoppable)
        .expect("decode");
    let (w, h) = d.dimensions();
    Rgb8Img {
        px: d.pixels_u8().expect("u8").to_vec(),
        w: w as usize,
        h: h as usize,
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().expect("corpus root"));
    let out_path = PathBuf::from(args.next().expect("out tsv"));
    let per_sub: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(3);
    let threads: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(12);
    let model = load_adopted("dejpeg_rt24g").expect("model");
    let td = PathBuf::from(std::env::var("HOME").unwrap())
        .join("tmp")
        .join(format!("zensr-pp-{}", std::process::id()));
    std::fs::create_dir_all(&td).unwrap();

    let mut tsv = String::from(
        "sub\tfile\tsrc_q\tscale\tpsnr_ref\tssim2_ref\tgain_true\tgain_jpegsrc\tgain_err\n",
    );
    for (sub, dir) in SUBCORPORA {
        let mut used = 0usize;
        for f in list_images(&root.join(dir)) {
            // only PNG sources — we need a KNOWN-clean reference to measure against
            if !f.to_string_lossy().to_lowercase().ends_with(".png") {
                continue;
            }
            if used >= per_sub {
                break;
            }
            let Some(img) = decode_any(&f) else { continue };
            let Some(base) = center_crop(&img, 1024) else {
                continue;
            };
            used += 1;
            let fname = f.file_name().unwrap().to_string_lossy().to_string();
            for &sq in SRC_QS {
                // simulate an unsplash-class JPEG source
                let src_jpeg = dec(&enc_turbo(&base, sq, &td, "1x1"));
                for &n in SCALES {
                    if base.w / n < 256 {
                        continue;
                    }
                    let (dw, dh) = (base.w / n, base.h / n);
                    let a = if n == 1 {
                        base.clone()
                    } else {
                        resize_rgb8(&base, dw, dh, zenresize::Filter::Lanczos)
                    };
                    let b = if n == 1 {
                        src_jpeg.clone()
                    } else {
                        resize_rgb8(&src_jpeg, dw, dh, zenresize::Filter::Lanczos)
                    };
                    // residual contamination of the would-be reference
                    let s_ref = score(&a, &b);
                    // does using B as the reference change a MEASURED gain?
                    let jb = enc_turbo(&a, EVAL_Q, &td, "2x2");
                    let id_a = dec(&jb);
                    let r =
                        restore_jpeg(&jb, &model, &RestoreConfig::default().with_threads(threads))
                            .expect("restore");
                    let rest = Rgb8Img {
                        px: r.to_rgb8(),
                        w: r.width,
                        h: r.height,
                    };
                    let gain_true = score(&a, &rest).ssim2 - score(&a, &id_a).ssim2;
                    let gain_jpegsrc = score(&b, &rest).ssim2 - score(&b, &id_a).ssim2;
                    let _ = writeln!(
                        tsv,
                        "{sub}\t{fname}\t{sq}\t{n}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}",
                        s_ref.psnr,
                        s_ref.ssim2,
                        gain_true,
                        gain_jpegsrc,
                        gain_jpegsrc - gain_true
                    );
                }
            }
            eprintln!("done {sub}/{fname}");
        }
    }
    std::fs::write(&out_path, &tsv).expect("write tsv");
}
