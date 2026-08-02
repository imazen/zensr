//! Generation-loss survey: multi-generation JPEG re-encode chains (gen2/gen3),
//! scored against the pristine original at EVERY generation, with the full
//! production pipeline (restore_jpeg: policy + guarded model + S10 projection)
//! and a no-projection arm.
//!
//! Key questions this answers (SYSTEMS.md "generation loss"):
//! 1. How much does the model under-correct on gen2/3 (artifacts baked into
//!    the signal the FINAL encode quantized — the file's DQT understates
//!    true damage)?
//! 2. Does the S10 projection (valid only w.r.t. the FINAL generation's
//!    coefficients) help or hinder on multi-gen input?
//! 3. Grid misalignment: a crop/shift between generations doubles block
//!    boundaries — the classic recompressed-meme signature.
//!
//! TSV: sub file chain gen enc q ss arm psnr ssim2 butter probe_family probe_q
//!
//! Usage: gen_eval <imazen26-root> <out-tsv> [per-sub=3] [threads=8]
//!        [model=dejpeg4_policy]

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;
use zensr_bench::*;
use zensr_zenjpeg::{restore_jpeg, Projection, RestoreConfig};

/// One generation step: encoder, quality, subsampling, and a pixel shift
/// applied BEFORE this encode (crops shift x shift from the top-left; 0 = none)
/// to model crop-recompress grid misalignment.
#[derive(Clone, Copy)]
struct Gen(&'static str, u32, &'static str, usize);

/// (name, generations). Singles double as matched-final-q baselines.
const CHAINS: &[(&str, &[Gen])] = &[
    // matched single-generation baselines
    ("single-t75", &[Gen("turbo", 75, "420", 0)]),
    ("single-t90", &[Gen("turbo", 90, "420", 0)]),
    ("single-t50", &[Gen("turbo", 50, "420", 0)]),
    ("single-t35", &[Gen("turbo", 35, "420", 0)]),
    ("single-m70", &[Gen("mozjpeg", 70, "420", 0)]),
    // gen2: common web flows
    (
        "g2-social",
        &[Gen("turbo", 85, "420", 0), Gen("turbo", 75, "420", 0)],
    ),
    (
        "g2-cdn",
        &[Gen("turbo", 92, "420", 0), Gen("mozjpeg", 70, "420", 0)],
    ),
    (
        "g2-upq",
        &[Gen("turbo", 60, "420", 0), Gen("turbo", 90, "420", 0)],
    ),
    (
        "g2-low",
        &[Gen("turbo", 35, "420", 0), Gen("turbo", 35, "420", 0)],
    ),
    (
        "g2-444to420",
        &[Gen("jpegli", 85, "444", 0), Gen("turbo", 75, "420", 0)],
    ),
    // gen2 with 2px crop between generations (block grids misaligned)
    (
        "g2-shift2",
        &[Gen("turbo", 75, "420", 0), Gen("turbo", 75, "420", 2)],
    ),
    // gen3
    (
        "g3-meme",
        &[
            Gen("turbo", 75, "420", 0),
            Gen("mozjpeg", 60, "420", 0),
            Gen("turbo", 50, "420", 0),
        ],
    ),
    (
        "g3-deep",
        &[
            Gen("turbo", 35, "420", 0),
            Gen("turbo", 35, "420", 0),
            Gen("turbo", 35, "420", 0),
        ],
    ),
    (
        "g3-shift",
        &[
            Gen("turbo", 85, "420", 0),
            Gen("turbo", 75, "420", 2),
            Gen("turbo", 65, "420", 2),
        ],
    ),
];

fn tmpdir() -> PathBuf {
    let d = PathBuf::from(std::env::var("HOME").unwrap())
        .join("tmp")
        .join(format!("zensr-gen-eval-{}", std::process::id()));
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
    let samp = if ss == "420" { "2x2" } else { "1x1" };
    let st = match enc {
        "turbo" => Command::new("cjpeg")
            .args([
                "-quality",
                &q.to_string(),
                "-sample",
                samp,
                "-optimize",
                "-outfile",
            ])
            .arg(jpg)
            .arg(ppm)
            .status(),
        "mozjpeg" => Command::new(format!("{home}/tmp/ati-bin/mozjpeg-cjpeg"))
            .env(
                "LD_LIBRARY_PATH",
                format!("{home}/tmp/ati-bin/mozjpeg-lib64"),
            )
            .args(["-quality", &q.to_string(), "-sample", samp, "-outfile"])
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

fn zj_decode(data: &[u8]) -> Rgb8Img {
    let r = zenjpeg::decoder::Decoder::new()
        .decode(data, enough::Unstoppable)
        .expect("zenjpeg decode");
    let (w, h) = r.dimensions();
    Rgb8Img {
        px: r.pixels_u8().expect("u8").to_vec(),
        w: w as usize,
        h: h as usize,
    }
}

fn probe_cols(data: &[u8]) -> (String, String) {
    match zenjpeg::detect::probe(data) {
        Ok(p) => (
            format!("{:?}", p.encoder),
            format!("{:.1}", p.quality.value),
        ),
        Err(_) => ("ERR".into(), "-".into()),
    }
}

/// Crop `off` pixels from top and left (both GT and chain image move together).
fn shift_crop(img: &Rgb8Img, off: usize) -> Rgb8Img {
    let (w, h) = (img.w - off, img.h - off);
    let mut px = Vec::with_capacity(3 * w * h);
    for y in 0..h {
        let row = &img.px[((y + off) * img.w + off) * 3..][..w * 3];
        px.extend_from_slice(row);
    }
    Rgb8Img { px, w, h }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().expect("corpus root"));
    let out_path = PathBuf::from(args.next().expect("out tsv"));
    let per_sub: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(3);
    let threads: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(8);
    let model_dir = args.next().unwrap_or_else(|| "dejpeg4_policy".into());
    let model = load_adopted(&model_dir).expect("model");
    let td = tmpdir();

    let mut tsv = String::from(
        "sub\tfile\tchain\tgen\tenc\tq\tss\tarm\tpsnr\tssim2\tbutter_n3\tprobe_family\tprobe_q\n",
    );
    for (sub, dir) in SUBCORPORA {
        let mut used = 0usize;
        for f in list_images(&root.join(dir)) {
            if used >= per_sub {
                break;
            }
            let Some(img) = decode_any(&f) else { continue };
            let Some(hr0) = center_crop(&img, 512) else {
                continue;
            };
            used += 1;
            let fname = f.file_name().unwrap().to_string_lossy().to_string();
            for (chain, gens) in CHAINS {
                // gt tracks cumulative shifts so every generation scores
                // against the exactly-aligned pristine pixels
                let mut gt = hr0.clone();
                let mut cur = hr0.clone();
                for (gi, g) in gens.iter().enumerate() {
                    let Gen(enc, q, ss, shift) = *g;
                    if shift > 0 {
                        gt = shift_crop(&gt, shift);
                        cur = shift_crop(&cur, shift);
                    }
                    let ppm = td.join("g.ppm");
                    let jpg = td.join("g.jpg");
                    write_ppm(&cur, &ppm);
                    if !encode(&ppm, &jpg, enc, q, ss) {
                        eprintln!("ENCODE-FAIL {chain} gen{gi} {enc} q{q}");
                        break;
                    }
                    let data = std::fs::read(&jpg).unwrap();
                    let (pf, pq) = probe_cols(&data);
                    let decoded = zj_decode(&data);
                    let gen_no = gi + 1;
                    let last = gi + 1 == gens.len();
                    // identity at every generation; model arms at the final
                    let mut outs: Vec<(&str, Rgb8Img)> = vec![("identity", decoded.clone())];
                    if last {
                        let r = restore_jpeg(
                            &data,
                            &model,
                            &RestoreConfig::default().with_threads(threads),
                        )
                        .expect("restore");
                        let (w, h) = (r.width, r.height);
                        outs.push((
                            "model",
                            Rgb8Img {
                                px: r.to_rgb8(),
                                w,
                                h,
                            },
                        ));
                        let rn = restore_jpeg(
                            &data,
                            &model,
                            &RestoreConfig::default()
                                .with_threads(threads)
                                .with_projection(Projection::Off),
                        )
                        .expect("restore-noproj");
                        outs.push((
                            "model_noproj",
                            Rgb8Img {
                                px: rn.to_rgb8(),
                                w,
                                h,
                            },
                        ));
                    }
                    for (arm, o) in &outs {
                        let s = score(&gt, o);
                        let _ = writeln!(
                            tsv,
                            "{sub}\t{fname}\t{chain}\t{gen_no}\t{enc}\t{q}\t{ss}\t{arm}\t{:.3}\t{:.3}\t{:.4}\t{pf}\t{pq}",
                            s.psnr, s.ssim2, s.butter
                        );
                    }
                    // next generation starts from this generation's decode
                    cur = decoded;
                }
            }
            eprintln!("done {sub}/{fname}");
        }
    }
    std::fs::write(&out_path, &tsv).expect("write tsv");
}
