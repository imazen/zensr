//! Dejpeg-v2 survey: {turbo, mozjpeg, jpegli, zenjpeg} x {420,444} x q-grid,
//! decoded by zenjpeg, arms {identity_off, identity_auto, model_off, model_auto}.
//! Records the zenjpeg fingerprint per encode (family + estimated quality).
//!
//! TSV: sub file encoder ss q arm psnr ssim2 butter probe_family probe_q
//!
//! Usage: dejpeg_eval <imazen26-root> <out-tsv> [per-sub=3] [threads=12]
//!        [model_off=dejpeg2_off] [model_auto=dejpeg2_auto] [model_policy]
//!
//! ZENSR_EVAL_JOBS=P runs P files concurrently (default 1, sequential), each
//! running model inference with `threads` threads — P*T should stay at or
//! below the core count. TSV row order is unchanged (sub, file, encoder, ss,
//! q, arm) regardless of P.

use rayon::prelude::*;
use std::collections::HashMap;
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
            // "zenjpeg" or "zenjpeg:<variant>" (knob settings: zjtool.rs)
            let variant = enc.split_once(':').map(|(_, v)| v).unwrap_or("default");
            let me = std::env::current_exe().unwrap();
            Command::new(me.parent().unwrap().join("zjtool"))
                .arg("enc")
                .arg(ppm)
                .arg(jpg)
                .args([&q.to_string(), ss, variant])
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
    let jobs: usize = std::env::var("ZENSR_EVAL_JOBS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
        .max(1);
    let m_off = load_adopted(&m_off_dir).expect("model off");
    let m_auto = load_adopted(&m_auto_dir).expect("model auto");
    let m_policy = m_policy_dir
        .as_ref()
        .map(|d| load_adopted(d).expect("model policy"));
    assert!(m_off.scale == 1 && m_auto.scale == 1);
    let metadata_restorer =
        (std::env::var("ZENSR_EVAL_METADATA").as_deref() == Ok("1")).then(|| {
            zensr_zenjpeg::api::Restorer::new().with_tier(
                zensr_zenjpeg::api::Budget::Quality,
                load_adopted(&m_off_dir).expect("metadata model"),
            )
        });
    let td = tmpdir();

    // Restricts every subcorpus to the held-out files — validate + test of
    // eval_split/imazen26_effective_split.tsv (`just split`). ZENSR_EVAL_PIN
    // overrides with a two-column dir<TAB>filename list for a one-off. Without
    // either, "first N sorted" silently admits training images whenever the
    // directory listing differs from the one the split was frozen against.
    let pinned = resolve_pinned(&root);
    // Held-out files this run could not score, by reason. Reported at the end:
    // `decode_any` handles PNG and JPEG only, so every held-out HEIC (90 in the
    // canonical corpus) and both DNGs fall out here, and a silent `continue`
    // would quietly shrink the eval set without changing any printed number.
    let mut skipped: std::collections::BTreeMap<String, usize> = Default::default();
    // File selection, decode and crop stay sequential: the task list order is
    // the TSV's file order, identical to the sequential walk.
    let mut tasks: Vec<Task> = Vec::new();
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
            let ext = f
                .extension()
                .map(|e| e.to_string_lossy().to_ascii_lowercase())
                .unwrap_or_default();
            let Some(img) = decode_any(&f) else {
                *skipped.entry(format!("undecodable .{ext}")).or_default() += 1;
                continue;
            };
            let Some(hr) = center_crop(&img, crop_cap()) else {
                *skipped.entry("smaller than the crop".into()).or_default() += 1;
                continue;
            };
            let fname = f.file_name().unwrap().to_string_lossy().to_string();
            // Recorded per row; ZENSR_EVAL_CLEAN_GT=1 skips JPEG references.
            let gt_src = gt_src_of(&fname);
            if gt_src != "png" && std::env::var("ZENSR_EVAL_CLEAN_GT").as_deref() == Ok("1") {
                *skipped
                    .entry("non-PNG reference (ZENSR_EVAL_CLEAN_GT=1)".into())
                    .or_default() += 1;
                continue;
            }
            used += 1;
            tasks.push(Task {
                sub: sub.to_string(),
                fname,
                gt_src: gt_src.to_string(),
                hr,
            });
        }
    }

    let sw = Sweep {
        metadata_restorer: metadata_restorer.as_ref(),
        encoders: &encoders,
        subsamplings: &subsamplings,
        qs,
        m_off: &m_off,
        m_auto: &m_auto,
        m_policy: m_policy.as_ref(),
        model_canon: model_canon(&m_off_dir, &m_auto_dir, m_policy_dir.as_deref()),
        threads,
        gens: std::env::var("ZENSR_EVAL_GENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1),
        // ZENSR_EVAL_ARMS=policy skips arms that are invariant across
        // policy-model A/Bs (identity_auto, model_off, model_auto — reuse
        // them from the committed baseline TSV of the same grid) — those are
        // ~2/3 of the compute.
        policy_only: std::env::var("ZENSR_EVAL_ARMS").as_deref() == Ok("policy"),
        // ZENSR_EVAL_NOGATE=1 disables the shipped high-q identity gate.
        // Without it the gate short-circuits restoration above its threshold
        // and every cell up there reads exactly 0.000 — measuring the gate,
        // not the crossover it is supposed to sit at.
        nogate: std::env::var("ZENSR_EVAL_NOGATE").as_deref() == Ok("1"),
        slack: match (
            std::env::var("ZENSR_SLACK_Q"),
            std::env::var("ZENSR_SLACK_ABS"),
        ) {
            (Ok(sq), Ok(sa)) => Some((sq.parse().unwrap(), sa.parse().unwrap())),
            _ => None,
        },
        td,
    };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .expect("rayon pool");
    // Indexed par_iter collects in task order, so the TSV is byte-identical
    // to the sequential walk for any ZENSR_EVAL_JOBS.
    let chunks: Vec<String> = pool.install(|| {
        tasks
            .par_iter()
            .enumerate()
            .map(|(i, t)| sweep_file(&sw, t, i))
            .collect()
    });

    let mut tsv = String::from(
        "sub\tfile\tencoder\tss\tq\tarm\tpsnr\tssim2\tbutter_n3\tprobe_family\tprobe_q\tgt_src\n",
    );
    if metadata_restorer.is_some() {
        tsv = "sub\tfile\tencoder\tss\tq\tprobe_family\tprobe_q\trouting_family\trouting_scale\trouting_q\tzero_ac_frac\tidentity_ssim2\testimated_gain\tgate\tconservative_gate\tmargin_gates\n".into();
    }
    for c in &chunks {
        tsv.push_str(c);
    }
    std::fs::write(&out_path, &tsv).expect("write tsv");
    if !skipped.is_empty() {
        let total: usize = skipped.values().sum();
        eprintln!("SKIPPED {total} held-out files that could not be scored:");
        for (why, n) in &skipped {
            eprintln!("  {n:>4}  {why}");
        }
    }
    eprintln!("wrote {}", out_path.display());
}

struct Task {
    sub: String,
    fname: String,
    gt_src: String,
    hr: Rgb8Img,
}

struct Sweep<'a> {
    metadata_restorer: Option<&'a zensr_zenjpeg::api::Restorer>,
    encoders: &'a [String],
    subsamplings: &'a [String],
    qs: &'a [u32],
    m_off: &'a zensr_micro::adopted::AdoptedModel,
    m_auto: &'a zensr_micro::adopted::AdoptedModel,
    m_policy: Option<&'a zensr_micro::adopted::AdoptedModel>,
    /// Canonical inference slot per model dir: equal dirs share a slot so a
    /// (model, source) pair is computed once per cell. [off, auto, policy]
    model_canon: [usize; 3],
    threads: usize,
    gens: usize,
    policy_only: bool,
    nogate: bool,
    /// (slack_q, slack_abs) when both ZENSR_SLACK_* are set.
    slack: Option<(f32, f32)>,
    td: PathBuf,
}

fn model_canon(off: &str, auto: &str, policy: Option<&str>) -> [usize; 3] {
    let dirs = [off, auto, policy.unwrap_or("\0none")];
    let mut canon = [0usize; 3];
    for (i, d) in dirs.iter().enumerate() {
        canon[i] = dirs.iter().position(|x| x == d).unwrap();
    }
    canon
}

/// Runs the full encoder x ss x q grid for one file. Returns its TSV rows in
/// cell/arm order. Uses a per-task tmpdir (parallel workers cannot share the
/// e.jpg/hr.ppm scratch) and removes it before returning.
fn sweep_file(sw: &Sweep, t: &Task, idx: usize) -> String {
    let td = sw.td.join(format!("task-{idx:04}"));
    std::fs::create_dir_all(&td).unwrap();
    let ppm = td.join("hr.ppm");
    let mut rows = String::new();
    for enc in sw.encoders.iter().map(|s| s.as_str()) {
        for ss in sw.subsamplings.iter().map(|s| s.as_str()) {
            for &q in sw.qs {
                let jpg = td.join("e.jpg");
                let _ = std::fs::remove_file(&jpg);
                write_ppm(&t.hr, &ppm); // fresh source each cell (gen chains rewrite it)
                if !encode(&ppm, &jpg, enc, q, ss) {
                    assert!(
                        sw.metadata_restorer.is_none(),
                        "metadata encode failed: {enc} {ss} {q}"
                    );
                    eprintln!("ENCODE FAIL {enc} {ss} q{q} {}", t.fname);
                    continue;
                }
                // ZENSR_EVAL_GENS=N: re-encode N-1 more times (aligned
                // recompression). Multi-gen inputs violate the S10 box
                // by 2-4 Q, so gate + slack must be set against these,
                // not just pristine single-generation encodes.
                for _ in 1..sw.gens {
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
                if let Some(restorer) = sw.metadata_restorer {
                    let outcome = restorer
                        .restore(&data)
                        .budget(zensr_zenjpeg::api::Budget::Quality)
                        .routing(zensr_zenjpeg::api::Routing::Auto {
                            min_gain: f32::INFINITY,
                        })
                        .run()
                        .expect("routing metadata");
                    let estimated = match &outcome {
                        zensr_zenjpeg::api::Outcome::Unchanged {
                            why: zensr_zenjpeg::api::SkipReason::NotWorthIt { estimated_gain },
                            ..
                        } => *estimated_gain,
                        _ => panic!("metadata request must skip"),
                    };
                    let probe = zenjpeg::detect::probe(&data).unwrap();
                    let gate = zensr_zenjpeg::policy_high_q_identity(&probe);
                    let conservative_gate =
                        zensr_zenjpeg::policy_high_q_identity_with_margin(&probe, 6.0);
                    // Audit every candidate margin using the real record-corrected gate.
                    let margin_gates = [0.0_f32, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
                        .into_iter()
                        .map(|margin| {
                            format!(
                                "{margin}:{}",
                                zensr_zenjpeg::policy_high_q_identity_with_margin(&probe, margin)
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(",");
                    let report = outcome.report();
                    let coeffs = zenjpeg::decoder::Decoder::new()
                        .decode_coefficients(&data, enough::Unstoppable)
                        .expect("coefficients");
                    let blocks = coeffs.components[0].coeffs.as_chunks::<64>().0;
                    assert!(!blocks.is_empty());
                    let frac = blocks
                        .iter()
                        .filter(|b| b[1..].iter().all(|&v| v == 0))
                        .count() as f64
                        / blocks.len() as f64;
                    let identity = score(&t.hr, &d_off);
                    writeln!(
                        rows,
                        "{}\t{}\t{enc}\t{ss}\t{q}\t{pf}\t{pq}\t{}\t{}\t{:.8}\t{frac:.8}\t{:.3}\t{estimated:.8}\t{gate}\t{conservative_gate}\t{margin_gates}",
                        t.sub,
                        t.fname,
                        report.encoder_family,
                        report.quality_scale,
                        report.quality,
                        identity.ssim2
                    )
                    .unwrap();
                    continue;
                }
                let d_auto = zj_decode(&data, true);
                let srcs = [&d_off, &d_auto];
                // (arm, src idx, model canon slot). Model arms pull from a
                // per-cell cache so a repeated (model, source) inference —
                // model_policy pointing at the off/auto model dir, or equal
                // model dirs — runs only once.
                let mut outs: Vec<(&str, usize, Option<usize>)> = vec![("identity_off", 0, None)];
                if !sw.policy_only {
                    outs.push(("identity_auto", 1, None));
                    outs.push(("model_off", 0, Some(sw.model_canon[0])));
                    outs.push(("model_auto", 1, Some(sw.model_canon[1])));
                }
                if sw.m_policy.is_some() {
                    let s = if policy_wants_auto(&data) { 1 } else { 0 };
                    outs.push(("model_policy", s, Some(sw.model_canon[2])));
                }
                let proj_out = sw.m_policy.map(|mp| {
                    // ZENSR_SLACK_Q / ZENSR_SLACK_ABS override the
                    // family-calibrated projection slack (tail study).
                    let mut rc = zensr_zenjpeg::RestoreConfig::default()
                        .with_threads(sw.threads)
                        .with_high_q_identity(!sw.nogate);
                    if let Some((sq, sa)) = sw.slack {
                        rc = rc.with_projection(zensr_zenjpeg::Projection::Fixed(
                            zensr_zenjpeg::ProjectionConfig::with_slack_q(sq).with_slack_abs(sa),
                        ));
                    }
                    let r = zensr_zenjpeg::restore_jpeg(&data, mp, &rc).expect("restore_jpeg");
                    Rgb8Img {
                        px: r.to_rgb8(),
                        w: r.width,
                        h: r.height,
                    }
                });
                let mut cache: HashMap<(usize, usize), Rgb8Img> = HashMap::new();
                let model_of = |slot| match slot {
                    0 => sw.m_off,
                    1 => sw.m_auto,
                    _ => sw.m_policy.expect("policy model"),
                };
                for (arm, sidx, mslot) in outs {
                    let oref: &Rgb8Img = match mslot {
                        Some(slot) => cache
                            .entry((slot, sidx))
                            .or_insert_with(|| run_x1(model_of(slot), srcs[sidx], sw.threads)),
                        None => srcs[sidx],
                    };
                    let s = score(&t.hr, oref);
                    let _ = writeln!(
                        rows,
                        "{}\t{}\t{enc}\t{ss}\t{q}\t{arm}\t{:.3}\t{:.3}\t{:.4}\t{pf}\t{pq}\t{}",
                        t.sub, t.fname, s.psnr, s.ssim2, s.butter, t.gt_src
                    );
                }
                if let Some(po) = &proj_out {
                    let s = score(&t.hr, po);
                    let _ = writeln!(
                        rows,
                        "{}\t{}\t{enc}\t{ss}\t{q}\tmodel_proj\t{:.3}\t{:.3}\t{:.4}\t{pf}\t{pq}\t{}",
                        t.sub, t.fname, s.psnr, s.ssim2, s.butter, t.gt_src
                    );
                }
            }
        }
    }
    eprintln!("done {}/{}", t.sub, t.fname);
    if sw.metadata_restorer.is_none() {
        let _ = std::fs::remove_dir_all(&td);
    }
    rows
}
