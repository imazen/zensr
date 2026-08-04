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
const QS_LOW: &[u32] = &[5, 8, 12];
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
            .args([
                "-quality",
                &q.to_string(),
                "-sample",
                if ss == "420" { "2x2" } else { "1x1" },
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
            .args([
                "-quality",
                &q.to_string(),
                "-sample",
                if ss == "420" { "2x2" } else { "1x1" },
                "-outfile",
            ])
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
    Rgb8Img {
        px: r.pixels_u8().expect("u8").to_vec(),
        w: w as usize,
        h: h as usize,
    }
}

/// Deployment policy: Knusperli (Auto) only for Annex-K-family files at
/// probe-estimated q <= 9 (exact at those q); AQ-family (Cjpegli*) never.
fn policy_wants_auto(data: &[u8]) -> bool {
    // kept for arm construction; canonical copy lives in zensr-zenjpeg
    match zenjpeg::detect::probe(data) {
        Ok(p) => {
            let fam = format!("{:?}", p.encoder);
            let scale = format!("{:?}", p.quality.scale);
            !fam.starts_with("Cjpegli")
                && (scale == "IjgQuality" || scale == "MozjpegQuality")
                && p.quality.value <= 9.5
        }
        Err(_) => false,
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

fn run_x1(m: &zensr_micro::adopted::AdoptedModel, lr: &Rgb8Img, threads: usize) -> Rgb8Img {
    let lp = to_planar_f32(lr);
    let mut sr = m.upscale_tiled(&lp, lr.h, lr.w, threads, 0);
    guarded_merge(&mut sr, &lp, lr.h, lr.w, 1, &GuardConfig::default());
    planar_to_rgb8(&sr, lr.w, lr.h)
}

/// Longest edge the eval crops to. 512 by default, which matches every ladder
/// measured so far. A corpus that is deliberately size-diverse — the picker
/// renditions run 64..1024 — must raise it, or the crop flattens the size axis
/// the corpus exists to provide.
fn crop_cap() -> usize {
    std::env::var("ZENSR_EVAL_CROP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().expect("corpus root"));
    let out_path = PathBuf::from(args.next().expect("out tsv"));
    let per_sub: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(3);
    let threads: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(12);
    let m_off_dir = args.next().unwrap_or_else(|| "dejpeg2_off".into());
    let m_auto_dir = args.next().unwrap_or_else(|| "dejpeg2_auto".into());
    // ZENSR_EVAL_QS=40,50,60,... overrides the named grids entirely
    // ZENSR_EVAL_ENCODERS / ZENSR_EVAL_SS trim the grid the same way
    // ZENSR_EVAL_QS does. A full 4-encoder x 2-subsampling sweep is 8x the
    // cheapest useful cell, which is the difference between a run that fits
    // beside another session's build and one that takes the whole box.
    let encoders: Vec<String> = std::env::var("ZENSR_EVAL_ENCODERS")
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_else(|_| ENCODERS.iter().map(|s| s.to_string()).collect());
    let subsamplings: Vec<String> = std::env::var("ZENSR_EVAL_SS")
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_else(|_| vec!["420".into(), "444".into()]);
    let custom_qs: Option<Vec<u32>> = std::env::var("ZENSR_EVAL_QS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect());
    let qs: &[u32] = match (
        custom_qs.as_deref(),
        std::env::var("ZENSR_EVAL_GRID").as_deref(),
    ) {
        (Some(c), _) => c,
        (None, Ok("high")) => QS_HIGH,
        (None, Ok("low")) => QS_LOW,
        _ => QS,
    };
    let m_policy_dir = args.next();
    let m_off = load_adopted(&m_off_dir).expect("model off");
    let m_auto = load_adopted(&m_auto_dir).expect("model auto");
    let m_policy = m_policy_dir.map(|d| load_adopted(&d).expect("model policy"));
    assert!(m_off.scale == 1 && m_auto.scale == 1);
    let td = tmpdir();

    let mut tsv = String::from(
        "sub\tfile\tencoder\tss\tq\tarm\tpsnr\tssim2\tbutter_n3\tprobe_family\tprobe_q\tgt_src\n",
    );
    // ZENSR_EVAL_PIN=<tsv> (default eval_split/imazen26_eval_files.tsv) restricts
    // every subcorpus to the frozen eval files. Without it, "first N sorted"
    // silently admits training images whenever the directory listing differs
    // from the one the split was frozen against.
    let pinned = resolve_pinned(&root);
    for (sub, dir) in &subcorpora_for(&root) {
        let (sub, dir) = (sub.as_str(), dir.as_str());
        let files = list_images(&root.join(dir));
        let want = pinned.as_ref().and_then(|m| m.get(dir));
        let mut seen_pinned = std::collections::HashSet::new();
        let mut used = 0usize;
        for f in files {
            if used >= per_sub {
                break;
            }
            if let Some(w) = want {
                let stem = pinned_stem(&f.file_name().unwrap().to_string_lossy());
                if !w.contains(&stem) {
                    continue;
                }
                seen_pinned.insert(stem);
            }
            let Some(img) = decode_any(&f) else { continue };
            let Some(hr) = center_crop(&img, crop_cap()) else {
                continue;
            };
            let fname = f.file_name().unwrap().to_string_lossy().to_string();
            // Recorded per row; ZENSR_EVAL_CLEAN_GT=1 skips JPEG references.
            let gt_src = gt_src_of(&fname);
            if gt_src != "png" && std::env::var("ZENSR_EVAL_CLEAN_GT").as_deref() == Ok("1") {
                continue;
            }
            used += 1;
            let ppm = td.join("hr.ppm");
            for enc in encoders.iter().map(|s| s.as_str()) {
                for ss in subsamplings.iter().map(|s| s.as_str()) {
                    for &q in qs {
                        let jpg = td.join("e.jpg");
                        let _ = std::fs::remove_file(&jpg);
                        write_ppm(&hr, &ppm); // fresh source each cell (gen chains rewrite it)
                        if !encode(&ppm, &jpg, enc, q, ss) {
                            eprintln!("ENCODE FAIL {enc} {ss} q{q} {fname}");
                            continue;
                        }
                        // ZENSR_EVAL_GENS=N: re-encode N-1 more times (aligned
                        // recompression). Multi-gen inputs violate the S10 box
                        // by 2-4 Q, so gate + slack must be set against these,
                        // not just pristine single-generation encodes.
                        let gens: usize = std::env::var("ZENSR_EVAL_GENS")
                            .ok()
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(1);
                        for _ in 1..gens {
                            let d = zenjpeg::decoder::Decoder::new()
                                .decode(&std::fs::read(&jpg).unwrap(), enough::Unstoppable)
                                .expect("decode");
                            let (dw, dh) = d.dimensions();
                            let mid = Rgb8Img {
                                px: d.pixels_u8().expect("u8").to_vec(),
                                w: dw as usize,
                                h: dh as usize,
                            };
                            write_ppm(&mid, &ppm);
                            if !encode(&ppm, &jpg, enc, q, ss) {
                                break;
                            }
                        }
                        let data = std::fs::read(&jpg).unwrap();
                        let (pf, pq) = probe_cols(&data);
                        let d_off = zj_decode(&data, false);
                        let d_auto = zj_decode(&data, true);
                        // ZENSR_EVAL_ARMS=policy skips arms that are invariant
                        // across policy-model A/Bs (identity_auto, model_off,
                        // model_auto — reuse them from the committed baseline
                        // TSV of the same grid) — those are ~2/3 of the compute.
                        let policy_only =
                            std::env::var("ZENSR_EVAL_ARMS").as_deref() == Ok("policy");
                        let mut outs: Vec<(
                            &str,
                            &Rgb8Img,
                            Option<&zensr_micro::adopted::AdoptedModel>,
                        )> = vec![("identity_off", &d_off, None)];
                        if !policy_only {
                            outs.push(("identity_auto", &d_auto, None));
                            outs.push(("model_off", &d_off, Some(&m_off)));
                            outs.push(("model_auto", &d_auto, Some(&m_auto)));
                        }
                        if let Some(mp) = &m_policy {
                            let src = if policy_wants_auto(&data) {
                                &d_auto
                            } else {
                                &d_off
                            };
                            outs.push(("model_policy", src, Some(mp)));
                        }
                        let proj_out = m_policy.as_ref().map(|mp| {
                            // ZENSR_SLACK_Q / ZENSR_SLACK_ABS override the
                            // family-calibrated projection slack (tail study)
                            // ZENSR_EVAL_NOGATE=1 disables the shipped high-q
                            // identity gate. Without it the gate short-circuits
                            // restoration above its threshold and every cell up
                            // there reads exactly 0.000 — measuring the gate,
                            // not the crossover it is supposed to sit at.
                            let mut rc = zensr_zenjpeg::RestoreConfig::default()
                                .with_threads(threads)
                                .with_high_q_identity(
                                    std::env::var("ZENSR_EVAL_NOGATE").as_deref() != Ok("1"),
                                );
                            if let (Ok(sq), Ok(sa)) = (
                                std::env::var("ZENSR_SLACK_Q"),
                                std::env::var("ZENSR_SLACK_ABS"),
                            ) {
                                rc = rc.with_projection(zensr_zenjpeg::Projection::Fixed(
                                    zensr_zenjpeg::ProjectionConfig::with_slack_q(
                                        sq.parse().unwrap(),
                                    )
                                    .with_slack_abs(sa.parse().unwrap()),
                                ));
                            }
                            let r =
                                zensr_zenjpeg::restore_jpeg(&data, mp, &rc).expect("restore_jpeg");
                            Rgb8Img {
                                px: r.to_rgb8(),
                                w: r.width,
                                h: r.height,
                            }
                        });
                        if let Some(po) = &proj_out {
                            outs.push(("model_proj", po, None));
                        }
                        for (arm, src, model) in outs {
                            let o = match model {
                                Some(m) => run_x1(m, src, threads),
                                None => (*src).clone(),
                            };
                            let s = score(&hr, &o);
                            let _ = writeln!(
                                tsv,
                                "{sub}\t{fname}\t{enc}\t{ss}\t{q}\t{arm}\t{:.3}\t{:.3}\t{:.4}\t{pf}\t{pq}\t{gt_src}",
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
