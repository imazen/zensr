//! zenanalyze source-image features, one row per reference image.
//!
//! The routing work has been testing nine hand-rolled coefficient features
//! against per-image restore gain. zenanalyze already has a designed feature
//! set, and the canonical picker datasets show what it is worth: they carry
//! ~469 `feat_*` columns per row, and those are **source-image** features —
//! constant across all 330 encodes of a given reference. So they join to any
//! encode, at any quality, by filename alone.
//!
//! Those datasets themselves cannot be reused here: they are built from
//! imazen-26 origins (id range 1000-9999) and the dejpeg model trains on
//! imazen-26, with no origin→path map available to filter by. Their documented
//! `leakage_check` covers picker-internal train/val/test overlap, which is a
//! different question. The *feature set* transfers even though the data does
//! not.
//!
//! Cost is per IMAGE, not per cell — that is the whole point. A source feature
//! is computed once and reused across every quality and encoder in the sweep,
//! where the coefficient features have to be recomputed per variant.
//!
//! Usage: source_features <corpus-root> <out-tsv> [per-sub=100000] [crop=512]

use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::Instant;
use zenalyze_shim::*;
use zensr_bench::*;

mod zenalyze_shim {
    pub use zenanalyze::analyze_features_rgb8;
    pub use zenanalyze::feature::{AnalysisQuery, FeatureSet};
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
    let root = PathBuf::from(
        args.next()
            .expect("usage: source_features <root> <out.tsv> [per-sub] [crop]"),
    );
    let out_path = PathBuf::from(args.next().expect("out tsv"));
    let per_sub: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(100_000);
    let crop: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(512);

    // Same geometry the ladder scores, so a feature describes the pixels that
    // were actually encoded rather than the whole source image.
    // ZENSR_SF_ONLY=a,b,c extracts just those features. Routing needs one
    // (edge_slope_stdev), and asking for SUPPORTED computes ~101 including the
    // expensive DCT and alpha passes — gating is per-pass at runtime, so a
    // narrow request skips whole passes rather than just discarding columns.
    let only: Option<Vec<String>> = std::env::var("ZENSR_SF_ONLY")
        .ok()
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect());
    let set = match &only {
        Some(names) => FeatureSet::SUPPORTED
            .iter()
            .filter(|f| names.iter().any(|n| n == f.name()))
            .fold(FeatureSet::new(), |acc, f| acc.with(f)),
        None => FeatureSet::SUPPORTED,
    };
    let feats: Vec<_> = set.iter().collect();
    eprintln!("{} features per image, centre-{crop} crop", feats.len());

    let pinned = resolve_pinned(&root);
    let mut tsv = String::new();
    let _ = writeln!(
        tsv,
        "# zenanalyze source features, centre-{crop} crop, one row per reference image"
    );
    let _ = write!(tsv, "sub\tfile\twidth\theight");
    for f in &feats {
        let _ = write!(tsv, "\tsf_{}", f.name());
    }
    let _ = writeln!(tsv);

    let t0 = Instant::now();
    let mut n = 0usize;
    for (name, dir) in &subcorpora_for(&root) {
        let want = pinned.as_ref().and_then(|m| m.get(dir));
        let mut used = 0usize;
        for f in list_images(&root.join(dir)) {
            if used >= per_sub {
                break;
            }
            let fname = f.file_name().unwrap().to_string_lossy().to_string();
            if let Some(w) = want {
                if !w.contains(&pinned_stem(&fname)) {
                    continue;
                }
            }
            let Some(img) = decode_any(&f) else { continue };
            let Some(c) = center_crop(&img, crop) else {
                continue;
            };
            let res =
                analyze_features_rgb8(&c.px, c.w as u32, c.h as u32, &AnalysisQuery::new(set));
            let _ = write!(tsv, "{name}\t{fname}\t{}\t{}", c.w, c.h);
            for ft in &feats {
                // A feature the extractor declines to produce is written empty,
                // never zero — zero is a legitimate value for most of these and
                // would silently become data.
                match res.get(*ft) {
                    Some(v) => {
                        let _ = write!(tsv, "\t{:.6}", v.to_f32());
                    }
                    None => {
                        let _ = write!(tsv, "\t");
                    }
                }
            }
            let _ = writeln!(tsv);
            used += 1;
            n += 1;
        }
        eprintln!("== {name}: {used}");
    }
    std::fs::write(&out_path, &tsv).expect("write tsv");
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "wrote {} — {n} images, {:.1} ms/image, {:.1}s total",
        out_path.display(),
        ms / n.max(1) as f64,
        ms / 1000.0
    );
}
