//! Contrast-stratified error analysis (user question 2026-07-30: are
//! high-contrast regions — text, lines, edges — where restoration matters
//! most, and where do we actually stand there?).
//!
//! Whole-image SSIM2 averages over flat and edge regions alike. This bin
//! buckets every pixel by the GROUND TRUTH's local contrast (max-min over a
//! 8x8 block, i.e. exactly the JPEG block the artifacts live in) and reports
//! per-bucket PSNR for identity vs the model. That says whether the model's
//! gain is concentrated in flat areas (where artifacts are cheap to remove
//! but invisible) or in the edge blocks that dominate perception.
//!
//! Also reports the DEGRADATION profile (identity vs GT per bucket) so the
//! two questions stay separate: where does JPEG hurt, and where do we help.
//!
//! TSV: sub file q bucket n_px gt_contrast identity_psnr model_psnr gain
//! Usage: contrast_eval <imazen26-root> <out-tsv> [per-sub=3] [threads=12]
//!        [model=dejpeg_rt24f]

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;
use zensr_bench::*;
use zensr_zenjpeg::{restore_jpeg, RestoreConfig};

const QS: &[u32] = &[15, 35, 55, 75];
/// GT block-contrast bucket edges (max-min of luma over 8x8, 0..255).
const EDGES: &[f32] = &[8.0, 24.0, 64.0];
const NAMES: &[&str] = &["flat(<8)", "low(8-24)", "mid(24-64)", "high(>64)"];

fn encode_turbo(img: &Rgb8Img, q: u32, td: &PathBuf) -> Vec<u8> {
    let ppm = td.join("c.ppm");
    let jpg = td.join("c.jpg");
    let mut buf = format!("P6\n{} {}\n255\n", img.w, img.h).into_bytes();
    buf.extend_from_slice(&img.px);
    std::fs::write(&ppm, buf).unwrap();
    assert!(Command::new("cjpeg")
        .args(["-quality", &q.to_string(), "-sample", "2x2", "-optimize", "-outfile"])
        .arg(&jpg)
        .arg(&ppm)
        .status()
        .unwrap()
        .success());
    std::fs::read(&jpg).unwrap()
}

fn luma(img: &Rgb8Img) -> Vec<f32> {
    (0..img.w * img.h)
        .map(|i| {
            0.299 * img.px[i * 3] as f32
                + 0.587 * img.px[i * 3 + 1] as f32
                + 0.114 * img.px[i * 3 + 2] as f32
        })
        .collect()
}

/// Per-pixel bucket id from the GT's own 8x8 block contrast (max-min luma).
fn bucket_map(gt: &Rgb8Img) -> Vec<u8> {
    let y = luma(gt);
    let (w, h) = (gt.w, gt.h);
    let mut b = vec![0u8; w * h];
    for by in (0..h).step_by(8) {
        for bx in (0..w).step_by(8) {
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for yy in by..(by + 8).min(h) {
                for xx in bx..(bx + 8).min(w) {
                    let v = y[yy * w + xx];
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            let c = hi - lo;
            let id = EDGES.iter().position(|e| c < *e).unwrap_or(EDGES.len()) as u8;
            for yy in by..(by + 8).min(h) {
                for xx in bx..(bx + 8).min(w) {
                    b[yy * w + xx] = id;
                }
            }
        }
    }
    b
}

/// Per-bucket MSE over RGB.
fn bucket_mse(gt: &Rgb8Img, other: &Rgb8Img, buckets: &[u8], nb: usize) -> Vec<(f64, usize)> {
    let mut acc = vec![(0.0f64, 0usize); nb];
    for i in 0..gt.w * gt.h {
        let b = buckets[i] as usize;
        let mut e = 0.0f64;
        for c in 0..3 {
            let d = gt.px[i * 3 + c] as f64 - other.px[i * 3 + c] as f64;
            e += d * d;
        }
        acc[b].0 += e / 3.0;
        acc[b].1 += 1;
    }
    acc
}

fn psnr(mse: f64) -> f64 {
    if mse <= 1e-9 {
        99.0
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().expect("corpus root"));
    let out_path = PathBuf::from(args.next().expect("out tsv"));
    let per_sub: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(3);
    let threads: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(12);
    let model_name = args.next().unwrap_or_else(|| "dejpeg_rt24f".into());
    let model = load_adopted(&model_name).expect("model");
    let td = PathBuf::from(std::env::var("HOME").unwrap())
        .join("tmp")
        .join(format!("zensr-ce-{}", std::process::id()));
    std::fs::create_dir_all(&td).unwrap();

    let nb = EDGES.len() + 1;
    let mut tsv = String::from(
        "sub\tfile\tq\tbucket\tn_px\tidentity_psnr\tmodel_psnr\tgain_db\tpx_share\tgt_src\n",
    );
    for (sub, dir) in SUBCORPORA {
        let mut used = 0usize;
        for f in list_images(&root.join(dir)) {
            if used >= per_sub {
                break;
            }
            let Some(img) = decode_any(&f) else { continue };
            let Some(gt) = center_crop(&img, 512) else { continue };
            used += 1;
            let fname = f.file_name().unwrap().to_string_lossy().to_string();
            let gt_src = if fname.to_ascii_lowercase().ends_with(".png") { "png" } else { "jpg" };
            if gt_src != "png" && std::env::var("ZENSR_EVAL_CLEAN_GT").as_deref() == Ok("1") { continue; }
            let buckets = bucket_map(&gt);
            let total = (gt.w * gt.h) as f64;
            for &q in QS {
                let jpg = encode_turbo(&gt, q, &td);
                let dec = zenjpeg::decoder::Decoder::new()
                    .decode(&jpg, enough::Unstoppable)
                    .unwrap();
                let ident =
                    Rgb8Img { px: dec.pixels_u8().unwrap().to_vec(), w: gt.w, h: gt.h };
                let r = restore_jpeg(&jpg, &model, &RestoreConfig::default().with_threads(threads))
                    .expect("restore");
                let m = Rgb8Img { px: r.to_rgb8(), w: r.width, h: r.height };
                let bi = bucket_mse(&gt, &ident, &buckets, nb);
                let bm = bucket_mse(&gt, &m, &buckets, nb);
                for b in 0..nb {
                    if bi[b].1 == 0 {
                        continue;
                    }
                    let pi = psnr(bi[b].0 / bi[b].1 as f64);
                    let pm = psnr(bm[b].0 / bm[b].1 as f64);
                    let _ = writeln!(
                        tsv,
                        "{sub}\t{fname}\t{q}\t{}\t{}\t{pi:.3}\t{pm:.3}\t{:.3}\t{:.4}\t{gt_src}",
                        NAMES[b],
                        bi[b].1,
                        pm - pi,
                        bi[b].1 as f64 / total
                    );
                }
            }
            eprintln!("done {sub}/{fname}");
        }
    }
    std::fs::write(&out_path, &tsv).expect("write tsv");
}
