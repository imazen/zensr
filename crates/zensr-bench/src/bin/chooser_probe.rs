//! zenanalyze feature extraction for the model-class chooser
//! (photo vs graphics/text/lineart specialist routing).
//!
//! The chooser runs at RESTORE time, on the decoded JPEG — so features are
//! extracted from compressed-then-decoded crops (turbo 420) across the q
//! range, not from pristine originals. Labels come from the subcorpus:
//! graphics = screen/documents/art-scans/maps, photo = the rest.
//!
//! split column: the canonical corpus split (validate ∪ test) = "eval" (chooser
//! VALIDATION set — never fit thresholds on it); train bucket = "train". Read
//! from the corpus repo via `zensr_bench::resolve_pinned`, not from a list kept
//! here — this binary previously carried its own loader for a repo-relative pin
//! file, and when that file went away it fell back to "first 8 sorted" with a
//! warning, which is the exact mechanism that leaked training images into an
//! eval twice.
//!
//! TSV: sub label split file q feat_<name>...
//!
//! Usage: chooser_probe <imazen26-root> <out-tsv> [per-sub=100] [qs=35,75,92]

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;
use zenanalyze::analyze_features_rgb8;
use zenanalyze::feature::{AnalysisQuery, FeatureSet};
use zensr_bench::*;

// Oracle content labels. `documents` and `maps` keep their meaning under the
// canonical layout; `patents`, `plots` and `clipart` are new classes that are
// unambiguously graphic. `illustrations` and `ai-products` are deliberately NOT
// here — see AMBIGUOUS_SUBS in tools/routing_headroom.py; they are a measurement
// to make, not a name to read.
const GRAPHICS: &[&str] = &[
    "screen",
    "documents",
    "art-scans",
    "maps",
    "patents",
    "plots",
    "clipart",
];

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
    // The canonical held-out set, keyed content_class -> {stem}. Panics rather
    // than degrading to sorted order: a chooser threshold fitted on its own
    // validation set is worse than no chooser.
    let pinned = resolve_pinned(&root).expect(
        "no eval split: clone github.com/imazen/imazen-26 (or set IMAZEN26_REPO), \
         or point ZENSR_EVAL_PIN at a two-column dir<TAB>filename list",
    );
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
        for f in files.iter() {
            if used >= per_sub {
                break;
            }
            let Some(img) = decode_any(f) else { continue };
            let Some(hr) = center_crop(&img, 512) else {
                continue;
            };
            used += 1;
            let fname = f.file_name().unwrap().to_string_lossy().to_string();
            // eval = the canonical validate+test buckets. No first-N rule: it
            // slid past decode-skipped files and admitted training images.
            // Keyed on `dir` (the folder), NOT `sub` (the content label):
            // canonical_holdout keys by content_class, which is the folder name.
            // "photos" spans six folders, so keying on the label finds nothing.
            let split = if pinned
                .get(*dir)
                .is_some_and(|s| s.contains(&pinned_stem(&fname)))
            {
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
