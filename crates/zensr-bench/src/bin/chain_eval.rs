//! Chained vs direct restoration+SR (user question 2026-07-26):
//! should artifact removal (x1 dejpeg, S10-projected) and SR be separate
//! steps — so SR needn't JPEG-specialize — or does chaining lose quality
//! vs running a (jpeg-capable) SR directly on the degraded input?
//!
//! Protocol matches systems_eval: HR = center-crop 1024; LR = catmullrom
//! half-size -> SYSTEM cjpeg (turbo, 420, -optimize) at q; scored vs HR.
//! Arms per SR family (span48 = quality tier, compact = adopted tier):
//!   <fam>_direct : SR straight on the degraded LR pixels
//!   <fam>_chain  : restore_jpeg(dejpeg4_policy, full prod pipeline incl.
//!                  S10 projection) on the LR JPEG BYTES -> SR on restored
//! plus lanczos baseline.
//!
//! TSV: sub file q arm psnr ssim2 butter_n3
//! Usage: chain_eval <imazen26-root> <out-tsv> [per-sub=3] [threads=12]

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;
use zensr_bench::*;
use zensr_micro::guards::{guarded_merge, GuardConfig};
use zensr_zenjpeg::{restore_jpeg, RestoreConfig};

fn encode_turbo(img: &Rgb8Img, q: u32, td: &PathBuf) -> Vec<u8> {
    let ppm = td.join("c.ppm");
    let jpg = td.join("c.jpg");
    let mut buf = format!("P6\n{} {}\n255\n", img.w, img.h).into_bytes();
    buf.extend_from_slice(&img.px);
    std::fs::write(&ppm, buf).unwrap();
    // ZENSR_CHAIN_SS=444 switches the whole experiment to 4:4:4 (default 420)
    let samp = if std::env::var("ZENSR_CHAIN_SS").as_deref() == Ok("444") {
        "1x1"
    } else {
        "2x2"
    };
    assert!(Command::new("cjpeg")
        .args([
            "-quality",
            &q.to_string(),
            "-sample",
            samp,
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

fn sr_x2(m: &zensr_micro::adopted::AdoptedModel, lr: &Rgb8Img, threads: usize) -> Rgb8Img {
    let lp = to_planar_f32(lr);
    let mut up = m.upscale_tiled(&lp, lr.h, lr.w, threads, 0);
    guarded_merge(&mut up, &lp, lr.h, lr.w, m.scale, &GuardConfig::default());
    planar_to_rgb8(&up, lr.w * m.scale, lr.h * m.scale)
}

fn sr_x2_planes(
    m: &zensr_micro::adopted::AdoptedModel,
    planes: &[f32],
    w: usize,
    h: usize,
    threads: usize,
) -> Rgb8Img {
    let mut up = m.upscale_tiled(planes, h, w, threads, 0);
    guarded_merge(&mut up, planes, h, w, m.scale, &GuardConfig::default());
    planar_to_rgb8(&up, w * m.scale, h * m.scale)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().expect("corpus root"));
    let out_path = PathBuf::from(args.next().expect("out tsv"));
    let per_sub: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(3);
    let threads: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(12);
    let dejpeg = load_adopted("dejpeg4_policy").expect("dejpeg4_policy");
    let span = load_adopted("nomosuni_span_2x").expect("span2x");
    let compact = load_adopted("nomosuni_compact_2x").expect("compact2x");
    let td = PathBuf::from(std::env::var("HOME").unwrap())
        .join("tmp")
        .join(format!("zensr-chain-{}", std::process::id()));
    std::fs::create_dir_all(&td).unwrap();

    let mut tsv = String::from("sub\tfile\tq\tarm\tpsnr\tssim2\tbutter_n3\tgt_src\n");
    for (sub, dir) in SUBCORPORA {
        let mut used = 0usize;
        for f in list_images(&root.join(dir)) {
            if used >= per_sub {
                break;
            }
            let Some(img) = decode_any(&f) else { continue };
            let Some(hr) = center_crop(&img, 1024) else {
                continue;
            };
            if hr.w % 2 != 0 || hr.h % 2 != 0 {
                continue;
            }
            used += 1;
            let fname = f.file_name().unwrap().to_string_lossy().to_string();
            let gt_src = if fname.to_ascii_lowercase().ends_with(".png") {
                "png"
            } else {
                "jpg"
            };
            if gt_src != "png" && std::env::var("ZENSR_EVAL_CLEAN_GT").as_deref() == Ok("1") {
                continue;
            }
            let lr_clean = resize_rgb8(&hr, hr.w / 2, hr.h / 2, zenresize::Filter::CatmullRom);
            for q in [35u32, 50, 75] {
                let jpg = encode_turbo(&lr_clean, q, &td);
                let lr = zj_decode_off(&jpg);
                let mut outs: Vec<(String, Rgb8Img)> = vec![
                    (
                        "lanczos".into(),
                        resize_rgb8(&lr, hr.w, hr.h, zenresize::Filter::Lanczos),
                    ),
                    ("span_direct".into(), sr_x2(&span, &lr, threads)),
                    ("compact_direct".into(), sr_x2(&compact, &lr, threads)),
                ];
                // chained: full prod x1 restore (probe/policy/projection) then SR
                let r = restore_jpeg(
                    &jpg,
                    &dejpeg,
                    &RestoreConfig::default().with_threads(threads),
                )
                .expect("restore");
                outs.push((
                    "span_chain".into(),
                    sr_x2_planes(&span, &r.planes, r.width, r.height, threads),
                ));
                outs.push((
                    "compact_chain".into(),
                    sr_x2_planes(&compact, &r.planes, r.width, r.height, threads),
                ));
                for (arm, o) in &outs {
                    let s = score(&hr, o);
                    let _ = writeln!(
                        tsv,
                        "{sub}\t{fname}\t{q}\t{arm}\t{:.3}\t{:.3}\t{:.4}\t{gt_src}",
                        s.psnr, s.ssim2, s.butter
                    );
                }
            }
            eprintln!("done {sub}/{fname}");
        }
    }
    std::fs::write(&out_path, &tsv).expect("write tsv");
}

fn zj_decode_off(data: &[u8]) -> Rgb8Img {
    let r = zenjpeg::decoder::Decoder::new()
        .decode(data, enough::Unstoppable)
        .expect("decode");
    let (w, h) = r.dimensions();
    Rgb8Img {
        px: r.pixels_u8().expect("u8").to_vec(),
        w: w as usize,
        h: h as usize,
    }
}
