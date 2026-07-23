//! Five-system evaluation on imazen-26 with a REAL libjpeg-turbo degradation
//! axis (system cjpeg, 4:2:0) and bounded-downside stats.
//!
//! Tracks: x2 (A2 fast-photo, E realtime-distill), x4 (A4 fast, B quality
//! wdn-blend, D anime), x1 repair (C = 2x-restore then box-down).
//! All systems run GUARDED (zensr_micro::guards). Baselines: bilinear,
//! catmullrom, lanczos. Stats per (system, subcorpus, degradation):
//! median + p10 ssim2/psnr/butteraugli + worse-than-lanczos rate.
//!
//! Usage: systems_eval <corpus-root> <out-tsv> [per-sub=8] [threads=12]
#![allow(clippy::too_many_arguments)]

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;
use zensr_bench::*;
use zensr_micro::adopted::AdoptedModel;
use zensr_micro::guards::{guarded_merge, GuardConfig};

const SUBCORPORA: &[(&str, &str)] = &[
    ("photos", "lilith"),
    ("people", "unsplash-people"),
    ("screen", "screen"),
    ("documents", "office-documents"),
    ("art-scans", "internet-archive-scans"),
    ("maps", "national-park-service"),
    ("renders", "unsplash-renders"),
    ("textures", "unsplash-textures"),
];

fn read_f32_file(p: &std::path::Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn load_adopted(dir: &str) -> Option<AdoptedModel> {
    let d = PathBuf::from("models/adopted").join(dir);
    let meta = std::fs::read_to_string(d.join("meta.json")).ok()?;
    let f = |k: &str| -> String {
        let pat = format!("\"{k}\":");
        match meta.find(&pat) {
            Some(i) => meta[i + pat.len()..]
                .trim_start()
                .trim_start_matches('"')
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect(),
            None => String::new(),
        }
    };
    let raw = read_f32_file(&d.join("weights.raw"));
    let scale: usize = f("scale").parse().ok()?;
    match f("arch").as_str() {
        "compact" => {
            AdoptedModel::load_compact(&raw, f("nf").parse().ok()?, f("nc").parse().ok()?, scale)
                .ok()
        }
        "span48" => AdoptedModel::load_span48(&raw, scale).ok(),
        _ => None,
    }
}

/// Lerp two compact weight sets (wdn severity blend).
fn lerp_compact(a: &[f32], b: &[f32], t: f32, nf: usize, nc: usize, s: usize) -> AdoptedModel {
    let mixed: Vec<f32> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x * (1.0 - t) + y * t)
        .collect();
    AdoptedModel::load_compact(&mixed, nf, nc, s).unwrap()
}

/// Real libjpeg-turbo round trip at quality q, 4:2:0, via system cjpeg.
fn turbo_jpeg(img: &Rgb8Img, q: u32) -> Rgb8Img {
    let dir = std::env::temp_dir().join(format!("zensr-se-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let ppm = dir.join("t.ppm");
    let jpg = dir.join("t.jpg");
    let mut buf = format!("P6\n{} {}\n255\n", img.w, img.h).into_bytes();
    buf.extend_from_slice(&img.px);
    std::fs::write(&ppm, &buf).unwrap();
    let st = Command::new("cjpeg")
        .args([
            "-quality",
            &q.to_string(),
            "-sample",
            "2x2",
            "-optimize",
            "-outfile",
        ])
        .arg(&jpg)
        .arg(&ppm)
        .status()
        .expect("cjpeg");
    assert!(st.success());
    decode_any(&jpg).expect("decode turbo jpeg")
}

struct Scored {
    psnr: f64,
    ssim2: f64,
    butter: f64,
}

fn score(hr: &Rgb8Img, out: &Rgb8Img) -> Scored {
    Scored {
        psnr: psnr_rgb8(hr, out),
        ssim2: ssim2(hr, out),
        butter: butter_n3(hr, out),
    }
}

fn run_guarded(m: &AdoptedModel, lr: &Rgb8Img, threads: usize, guard: bool) -> Rgb8Img {
    let lp = to_planar_f32(lr);
    let mut sr = m.upscale_tiled(&lp, lr.h, lr.w, threads, 0);
    if guard {
        guarded_merge(&mut sr, &lp, lr.h, lr.w, m.scale, &GuardConfig::default());
    }
    planar_to_rgb8(&sr, lr.w * m.scale, lr.h * m.scale)
}

fn run_guarded_spanf(m: &zensr_micro::SpanfModel, lr: &Rgb8Img, threads: usize) -> Rgb8Img {
    let lp = to_planar_f32(lr);
    let mut sr = zensr_micro::tiled::spanf_x4_tiled(m, &lp, lr.h, lr.w, threads, 0);
    guarded_merge(&mut sr, &lp, lr.h, lr.w, 4, &GuardConfig::default());
    planar_to_rgb8(&sr, lr.w * 4, lr.h * 4)
}

/// Box-downscale by 2 (for the x1 repair system).
fn box_down2(img: &Rgb8Img) -> Rgb8Img {
    let (w, h) = (img.w / 2, img.h / 2);
    let mut px = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let s = |yy: usize, xx: usize| img.px[(yy * img.w + xx) * 3 + c] as u32;
                px[(y * w + x) * 3 + c] =
                    ((s(2 * y, 2 * x) + s(2 * y, 2 * x + 1) + s(2 * y + 1, 2 * x) + s(2 * y + 1, 2 * x + 1) + 2)
                        / 4) as u8;
            }
        }
    }
    Rgb8Img { px, w, h }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().expect("corpus root"));
    let out_path = PathBuf::from(args.next().expect("out tsv"));
    let per_sub: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(8);
    let threads: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(12);

    let span2 = load_adopted("nomosuni_span_2x").expect("span2x");
    let span4 = load_adopted("nomosuni_span_4x").expect("span4x");
    let compact2 = load_adopted("nomosuni_compact_2x").expect("compact2x");
    let anime4 = load_adopted("animevideo_x4v3").expect("anime");
    let gen_raw = read_f32_file(&PathBuf::from("models/adopted/general_x4v3/weights.raw"));
    let wdn_raw = read_f32_file(&PathBuf::from("models/adopted/general_wdn_x4v3/weights.raw"));
    let rt = load_adopted("rt_distill_2x"); // may not exist yet (training)
    let spanf = zensr_micro::SpanfModel::new(read_f32_file(&PathBuf::from("models/spanf_weights.raw")))
        .expect("spanf");
    // wdn severity blends keyed by degradation level
    let b_for = |deg: &str| -> AdoptedModel {
        let t = match deg {
            "clean" => 0.0,
            "q75" => 0.15,
            "q50" => 0.40,
            _ => 0.65,
        };
        lerp_compact(&gen_raw, &wdn_raw, t, 64, 32, 4)
    };

    let mut tsv = String::new();
    let _ = writeln!(tsv, "# systems eval; LR degraded with SYSTEM cjpeg (libjpeg-turbo, 4:2:0, -optimize); guards=default; per_sub={per_sub}");
    let _ = writeln!(tsv, "subcorpus\tfile\ttrack\tdeg\tsystem\tpsnr\tssim2\tbutter_n3");

    for (sub, dir) in SUBCORPORA {
        let files = list_images(&root.join(dir));
        let mut used = 0usize;
        for f in files {
            if used >= per_sub {
                break;
            }
            let Some(img) = decode_any(&f) else { continue };
            let Some(hr) = center_crop(&img, 512) else { continue };
            let fname = f.file_name().unwrap().to_string_lossy().to_string();
            used += 1;

            for deg in ["clean", "q75", "q50", "q35"] {
                let degrade = |im: &Rgb8Img| -> Rgb8Img {
                    match deg {
                        "clean" => Rgb8Img { px: im.px.clone(), w: im.w, h: im.h },
                        "q75" => turbo_jpeg(im, 75),
                        "q50" => turbo_jpeg(im, 50),
                        _ => turbo_jpeg(im, 35),
                    }
                };
                // ---- x2 track
                {
                    let lr = degrade(&resize_rgb8(&hr, hr.w / 2, hr.h / 2, zenresize::Filter::CatmullRom));
                    let mut outs: Vec<(String, Rgb8Img)> = vec![
                        ("lanczos".into(), resize_rgb8(&lr, hr.w, hr.h, zenresize::Filter::Lanczos)),
                        ("catmullrom".into(), resize_rgb8(&lr, hr.w, hr.h, zenresize::Filter::CatmullRom)),
                        ("A2_span".into(), run_guarded(&span2, &lr, threads, true)),
                        ("A2_span_raw".into(), run_guarded(&span2, &lr, threads, false)),
                        ("A2c_compact".into(), run_guarded(&compact2, &lr, threads, true)),
                    ];
                    if let Some(rtm) = &rt {
                        outs.push(("E_rt".into(), run_guarded(rtm, &lr, threads, true)));
                    }
                    for (name, o) in &outs {
                        let s = score(&hr, o);
                        let _ = writeln!(tsv, "{sub}\t{fname}\tx2\t{deg}\t{name}\t{:.3}\t{:.3}\t{:.4}", s.psnr, s.ssim2, s.butter);
                    }
                }
                // ---- x4 track
                {
                    let lr = degrade(&resize_rgb8(&hr, hr.w / 4, hr.h / 4, zenresize::Filter::CatmullRom));
                    let outs: Vec<(String, Rgb8Img)> = vec![
                        ("lanczos".into(), resize_rgb8(&lr, hr.w, hr.h, zenresize::Filter::Lanczos)),
                        ("A4_span".into(), run_guarded(&span4, &lr, threads, true)),
                        ("F_spanf".into(), run_guarded_spanf(&spanf, &lr, threads)),
                        ("B_quality".into(), run_guarded(&b_for(deg), &lr, threads, true)),
                        ("D_anime".into(), run_guarded(&anime4, &lr, threads, true)),
                    ];
                    for (name, o) in &outs {
                        let s = score(&hr, o);
                        let _ = writeln!(tsv, "{sub}\t{fname}\tx4\t{deg}\t{name}\t{:.3}\t{:.3}\t{:.4}", s.psnr, s.ssim2, s.butter);
                    }
                }
                // ---- x1 repair track (jpeg the full-res image, restore at 1x)
                if deg != "clean" {
                    let lr = degrade(&hr);
                    let sr2 = run_guarded(&compact2, &lr, threads, true);
                    let repaired = box_down2(&sr2);
                    // box-down softens; CatmullRom preserves more of the restored detail
                    let repaired_cr = resize_rgb8(&sr2, lr.w, lr.h, zenresize::Filter::CatmullRom);
                    for (name, o) in [
                        ("identity", &lr),
                        ("C_repair", &repaired),
                        ("C_repair_cr", &repaired_cr),
                    ] {
                        let s = score(&hr, o);
                        let _ = writeln!(tsv, "{sub}\t{fname}\tx1\t{deg}\t{name}\t{:.3}\t{:.3}\t{:.4}", s.psnr, s.ssim2, s.butter);
                    }
                }
            }
            eprintln!("{sub}/{fname} done");
        }
        eprintln!("== {sub}: {used}");
    }
    std::fs::write(&out_path, &tsv).unwrap();
    eprintln!("wrote {}", out_path.display());
    summarize(&tsv);
}

fn summarize(tsv: &str) {
    use std::collections::BTreeMap;
    // rows: (track, deg, system) -> per-image ssim2 + psnr; lanczos captured for rate
    let mut per: BTreeMap<(String, String, String), Vec<(f64, f64, f64)>> = BTreeMap::new();
    let mut lanc: BTreeMap<(String, String, String), f64> = BTreeMap::new(); // (track,deg,file)
    for l in tsv.lines().skip(2) {
        let c: Vec<&str> = l.split('\t').collect();
        if c.len() < 8 {
            continue;
        }
        let (file, track, deg, sys) = (c[1], c[2], c[3], c[4]);
        let p: f64 = c[5].parse().unwrap_or(f64::NAN);
        let s2: f64 = c[6].parse().unwrap_or(f64::NAN);
        let b: f64 = c[7].parse().unwrap_or(f64::NAN);
        if sys == "lanczos" || sys == "identity" {
            lanc.insert((track.into(), deg.into(), file.into()), s2);
        }
        per.entry((track.into(), deg.into(), sys.into()))
            .or_default()
            .push((p, s2, b));
    }
    // second pass for worse-than-baseline rate
    let mut rates: BTreeMap<(String, String, String), (usize, usize)> = BTreeMap::new();
    for l in tsv.lines().skip(2) {
        let c: Vec<&str> = l.split('\t').collect();
        if c.len() < 8 || c[4] == "lanczos" || c[4] == "identity" {
            continue;
        }
        let key = (c[2].to_string(), c[3].to_string(), c[1].to_string());
        if let Some(base) = lanc.get(&key) {
            let s2: f64 = c[6].parse().unwrap_or(f64::NAN);
            let e = rates.entry((c[2].into(), c[3].into(), c[4].into())).or_default();
            e.1 += 1;
            if s2 < *base {
                e.0 += 1;
            }
        }
    }
    println!("track\tdeg\tsystem\tn\tpsnr_med\tssim2_med\tssim2_p10\tbutter_med\tworse_than_base%");
    for ((track, deg, sys), v) in &per {
        let col = |i: usize| -> Vec<f64> {
            let mut x: Vec<f64> = v
                .iter()
                .map(|t| [t.0, t.1, t.2][i])
                .filter(|x| x.is_finite())
                .collect();
            x.sort_by(|a, b| a.partial_cmp(b).unwrap());
            x
        };
        let med = |x: &Vec<f64>| if x.is_empty() { f64::NAN } else { x[x.len() / 2] };
        let p10 = |x: &Vec<f64>| if x.is_empty() { f64::NAN } else { x[x.len() / 10] };
        let (worse, tot) = rates.get(&(track.clone(), deg.clone(), sys.clone())).copied().unwrap_or((0, 0));
        let rate = if tot > 0 { 100.0 * worse as f64 / tot as f64 } else { f64::NAN };
        let (p, s2, b) = (col(0), col(1), col(2));
        println!(
            "{track}\t{deg}\t{sys}\t{}\t{:.2}\t{:.2}\t{:.2}\t{:.3}\t{rate:.0}",
            v.len(),
            med(&p),
            med(&s2),
            p10(&s2),
            med(&b)
        );
    }
}
