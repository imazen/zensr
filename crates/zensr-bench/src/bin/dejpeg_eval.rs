//! Dejpeg-v2 survey: {turbo, mozjpeg, jpegli, zenjpeg} x {420,444} x q-grid,
//! decoded by zenjpeg, arms {identity_off, identity_auto, model_off, model_auto}.
//! Records the zenjpeg fingerprint per encode (family + estimated quality).
//!
//! TSV: sub file encoder ss q arm psnr ssim2 butter probe_family probe_q
//!
//! Usage: dejpeg_eval <imazen26-root> <out-tsv> [per-sub=3] [threads=12]
//!        [model_off=dejpeg2_off] [model_auto=dejpeg2_auto]

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;
use zensr_bench::*;
use zensr_micro::guards::{guarded_merge, GuardConfig};

const QS: &[u32] = &[15, 35, 55, 75, 90];
const QS_HIGH: &[u32] = &[85, 90, 93, 96];
const ENCODERS: &[&str] = &["turbo", "mozjpeg", "jpegli", "zenjpeg"];

fn tmpdir() -> PathBuf {
    let d = PathBuf::from(std::env::var("HOME").unwrap())
        .join("tmp")
        .join(format!("zensr-dejpeg-eval-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write_ppm(img: &Rgb8Img, p: &PathBuf) {
    let mut buf = format!("P6\n{} {}\n255\n", img.w, img.h).into_bytes();
    buf.extend_from_slice(&img.px);
    std::fs::write(p, buf).unwrap();
}

fn encode(ppm: &PathBuf, jpg: &PathBuf, enc: &str, q: u32, ss: &str) -> bool {
    let home = std::env::var("HOME").unwrap();
    let st = match enc {
        "turbo" => Command::new("cjpeg")
            .args(["-quality", &q.to_string(), "-sample", if ss == "420" { "2x2" } else { "1x1" }, "-optimize", "-outfile"])
            .arg(jpg)
            .arg(ppm)
            .status(),
        "mozjpeg" => Command::new(format!("{home}/tmp/ati-bin/mozjpeg-cjpeg"))
            .env("LD_LIBRARY_PATH", format!("{home}/tmp/ati-bin/mozjpeg-lib64"))
            .args(["-quality", &q.to_string(), "-sample", if ss == "420" { "2x2" } else { "1x1" }, "-outfile"])
            .arg(jpg)
            .arg(ppm)
            .status(),
        "jpegli" => Command::new("cjpegli")
            .arg(ppm)
            .arg(jpg)
            .args(["-q", &q.to_string(), &format!("--chroma_subsampling={ss}")])
            .status(),
        _ => {
            let me = std::env::current_exe().unwrap();
            Command::new(me.parent().unwrap().join("zjtool"))
                .arg("enc")
                .arg(ppm)
                .arg(jpg)
                .args([&q.to_string(), ss])
                .status()
        }
    };
    st.map(|s| s.success()).unwrap_or(false)
}

fn zj_decode(data: &[u8], deblock_auto: bool) -> Rgb8Img {
    let mode = if deblock_auto {
        zenjpeg::decoder::DeblockMode::Auto
    } else {
        zenjpeg::decoder::DeblockMode::Off
    };
    let r = zenjpeg::decoder::Decoder::new()
        .deblock(mode)
        .decode(data, enough::Unstoppable)
        .expect("zenjpeg decode");
    let (w, h) = r.dimensions();
    Rgb8Img { px: r.pixels_u8().expect("u8").to_vec(), w: w as usize, h: h as usize }
}

fn probe_cols(data: &[u8]) -> (String, String) {
    match zenjpeg::detect::probe(data) {
        Ok(p) => (format!("{:?}", p.encoder), format!("{:.1}", p.quality.value)),
        Err(_) => ("ERR".into(), "-".into()),
    }
}

fn run_x1(m: &zensr_micro::adopted::AdoptedModel, lr: &Rgb8Img, threads: usize) -> Rgb8Img {
    let lp = to_planar_f32(lr);
    let mut sr = m.upscale_tiled(&lp, lr.h, lr.w, threads, 0);
    guarded_merge(&mut sr, &lp, lr.h, lr.w, 1, &GuardConfig::default());
    planar_to_rgb8(&sr, lr.w, lr.h)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().expect("corpus root"));
    let out_path = PathBuf::from(args.next().expect("out tsv"));
    let per_sub: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(3);
    let threads: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(12);
    let m_off_dir = args.next().unwrap_or_else(|| "dejpeg2_off".into());
    let m_auto_dir = args.next().unwrap_or_else(|| "dejpeg2_auto".into());
    let qs: &[u32] = if std::env::var("ZENSR_EVAL_HIGHQ").is_ok() { QS_HIGH } else { QS };
    let m_off = load_adopted(&m_off_dir).expect("model off");
    let m_auto = load_adopted(&m_auto_dir).expect("model auto");
    assert!(m_off.scale == 1 && m_auto.scale == 1);
    let td = tmpdir();

    let mut tsv = String::from(
        "sub\tfile\tencoder\tss\tq\tarm\tpsnr\tssim2\tbutter_n3\tprobe_family\tprobe_q\n",
    );
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
            let ppm = td.join("hr.ppm");
            write_ppm(&hr, &ppm);
            for enc in ENCODERS {
                for ss in ["420", "444"] {
                    for &q in qs {
                        let jpg = td.join("e.jpg");
                        let _ = std::fs::remove_file(&jpg);
                        if !encode(&ppm, &jpg, enc, q, ss) {
                            eprintln!("ENCODE FAIL {enc} {ss} q{q} {fname}");
                            continue;
                        }
                        let data = std::fs::read(&jpg).unwrap();
                        let (pf, pq) = probe_cols(&data);
                        let d_off = zj_decode(&data, false);
                        let d_auto = zj_decode(&data, true);
                        let outs = [
                            ("identity_off", &d_off, None),
                            ("identity_auto", &d_auto, None),
                            ("model_off", &d_off, Some(&m_off)),
                            ("model_auto", &d_auto, Some(&m_auto)),
                        ];
                        for (arm, src, model) in outs {
                            let o = match model {
                                Some(m) => run_x1(m, src, threads),
                                None => (*src).clone(),
                            };
                            let s = score(&hr, &o);
                            let _ = writeln!(
                                tsv,
                                "{sub}\t{fname}\t{enc}\t{ss}\t{q}\t{arm}\t{:.3}\t{:.3}\t{:.4}\t{pf}\t{pq}",
                                s.psnr, s.ssim2, s.butter
                            );
                        }
                    }
                }
            }
            eprintln!("done {sub}/{fname}");
        }
    }
    std::fs::write(&out_path, &tsv).expect("write tsv");
    eprintln!("wrote {}", out_path.display());
}
