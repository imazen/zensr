//! zenanalyze feature extraction for the model-class chooser
//! (photo vs graphics/text/lineart specialist routing).
//!
//! The chooser runs at RESTORE time, on the decoded JPEG — so features are
//! extracted from compressed-then-decoded crops (turbo 420) across the q
//! range, not from pristine originals. Labels come from the subcorpus:
//! graphics = screen/documents/art-scans/maps, photo = the rest.
//!
//! split column: files in eval_split/imazen26_eval_files.tsv (or the
//! first-8-sorted convention) = "eval" (chooser VALIDATION set — never fit
//! thresholds on it); the rest = "train".
//!
//! TSV: sub label split file q feat_<name>...
//!
//! Usage: chooser_probe <imazen26-root> <out-tsv> [per-sub=100] [qs=35,75,92]

use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;
use zenanalyze::analyze_features_rgb8;
use zenanalyze::feature::{AnalysisQuery, FeatureSet};
use zensr_bench::*;

const GRAPHICS: &[&str] = &["screen", "documents", "art-scans", "maps"];

fn eval_pinned(root: &PathBuf) -> HashSet<(String, String)> {
    // repo-relative pin file, same convention as tools/make_distill_data.py
    let mut out = HashSet::new();
    let pin = PathBuf::from("eval_split/imazen26_eval_files.tsv");
    if let Ok(s) = std::fs::read_to_string(&pin) {
        for line in s.lines() {
            let c: Vec<&str> = line.split('\t').collect();
            if c.len() == 2 && !c[0].starts_with('#') {
                out.insert((c[0].to_string(), c[1].to_string()));
            }
        }
    } else {
        eprintln!(
            "WARN: no {} (root {}), using first-8 only",
            pin.display(),
            root.display()
        );
    }
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().expect("corpus root"));
    let out_path = PathBuf::from(args.next().expect("out tsv"));
    let per_sub: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(100);
    let qs: Vec<u32> = args
        .next()
        .unwrap_or_else(|| "35,75,92".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let pinned = eval_pinned(&root);
    let td = PathBuf::from(std::env::var("HOME").unwrap())
        .join("tmp")
        .join(format!("zensr-chooser-{}", std::process::id()));
    std::fs::create_dir_all(&td).unwrap();

    let query = AnalysisQuery::new(FeatureSet::SUPPORTED);
    // header from the feature set itself so it tracks the zenanalyze build
    let mut header = String::from("sub\tlabel\tsplit\tfile\tq");
    for f in FeatureSet::SUPPORTED.iter() {
        let _ = write!(header, "\tfeat_{}", f.name());
    }
    let mut tsv = header + "\n";

    for (sub, dir) in SUBCORPORA {
        let label = if GRAPHICS.contains(sub) {
            "graphics"
        } else {
            "photo"
        };
        let files = list_images(&root.join(dir));
        let mut used = 0usize;
        for (fi, f) in files.iter().enumerate() {
            if used >= per_sub {
                break;
            }
            let Some(img) = decode_any(f) else { continue };
            let Some(hr) = center_crop(&img, 512) else {
                continue;
            };
            used += 1;
            let fname = f.file_name().unwrap().to_string_lossy().to_string();
            // eval = pinned union first-8-sorted (the frozen model-eval files)
            let split = if fi < 8 || pinned.contains(&(sub.to_string(), fname.clone())) {
                "eval"
            } else {
                "train"
            };
            for &q in &qs {
                let ppm = td.join("c.ppm");
                let jpg = td.join("c.jpg");
                let mut buf = format!("P6\n{} {}\n255\n", hr.w, hr.h).into_bytes();
                buf.extend_from_slice(&hr.px);
                std::fs::write(&ppm, &buf).unwrap();
                let ok = Command::new("cjpeg")
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
                    .map(|s| s.success())
                    .unwrap_or(false);
                if !ok {
                    eprintln!("ENCODE-FAIL {sub}/{fname} q{q}");
                    continue;
                }
                let data = std::fs::read(&jpg).unwrap();
                let dec = zenjpeg::decoder::Decoder::new()
                    .decode(&data, enough::Unstoppable)
                    .expect("decode");
                let (w, h) = dec.dimensions();
                let px = dec.pixels_u8().expect("u8");
                let res = analyze_features_rgb8(px, w, h, &query);
                let _ = write!(tsv, "{sub}\t{label}\t{split}\t{fname}\t{q}");
                for feat in FeatureSet::SUPPORTED.iter() {
                    match res.get(feat) {
                        Some(v) => {
                            let _ = write!(tsv, "\t{:.6}", v.to_f32());
                        }
                        None => {
                            let _ = write!(tsv, "\t");
                        }
                    }
                }
                tsv.push('\n');
            }
        }
        eprintln!("done {sub} ({used} files)");
    }
    std::fs::write(&out_path, &tsv).expect("write tsv");
}
