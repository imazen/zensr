//! imazen-26 SR quality eval: per-subcorpus, multi-metric.
//!
//! Protocol per image: decode -> center-crop HR (multiple of 4, cap 512) ->
//! LR = CatmullRom 4x downscale -> methods {spanf (zensr-micro), lanczos up,
//! catmullrom up} -> metrics vs HR {psnr, ssimulacra2, butteraugli n_3}.
//!
//! NOTE on degradation space: zenresize resamples u8 sRGB in linear light.
//! SPANF's NTIRE training LR uses bicubic in ENCODED space, so this protocol
//! is slightly off-distribution for the network (scores are conservative).
//! Recorded in the TSV header; revisit with encoded-space f32 resize.
//!
//! Usage: eval <corpus-root> <out-tsv> [per-sub=8] [threads=16]

use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::Instant;
use zensr_bench::*;

use zensr_micro::{spanf_x4_tiled, SpanfModel};

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

fn main() {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(
        args.next()
            .expect("usage: eval <corpus-root> <out-tsv> [per-sub] [threads]"),
    );
    let out_path = PathBuf::from(args.next().expect("out tsv"));
    let per_sub: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(8);
    let threads: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(16);

    let wdir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let wbytes =
        std::fs::read(wdir.join("spanf_weights.raw")).expect("spanf_weights.raw (just dump)");
    let wbuf: Vec<f32> = wbytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let model = SpanfModel::new(wbuf).expect("model");

    let mut tsv = String::new();
    let _ = writeln!(
        tsv,
        "# zensr imazen-26 eval; LR=CatmullRom/4 (linear-light; NOTE: SPANF trained on encoded-space bicubic — conservative); HR cap 512; per-sub {per_sub}; threads {threads}"
    );
    let _ = writeln!(
        tsv,
        "subcorpus\tfile\tmethod\tpsnr\tssim2\tbutter_n3\tsr_ms\tgt_src"
    );

    // Pinned selection. This eval drops images that fail to decode AND images
    // smaller than the crop, and every drop used to slide selection one file
    // deeper into the training set — the same defect as the 2026-07-23
    // postmortem, but more likely to fire here because the size filter rejects
    // far more files than a decode failure does.
    let pin_path = pin_path();
    let pinned = load_pinned(&pin_path);
    match &pinned {
        Some(m) => eprintln!("pinned eval split: {} ({} dirs)", pin_path, m.len()),
        None => eprintln!(
            "WARNING: no pinned eval split at {pin_path} — falling back to sorted \
             order, which can admit training images. Results are not comparable \
             to pinned runs."
        ),
    }

    for (name, dir) in SUBCORPORA {
        let files = list_images(&root.join(dir));
        let want = pinned.as_ref().and_then(|m| m.get(*dir));
        let mut seen_pinned = std::collections::HashSet::new();
        let mut used = 0usize;
        for f in files {
            if used >= per_sub {
                break;
            }
            let fname_raw = f.file_name().unwrap().to_string_lossy().to_string();
            if let Some(w) = want {
                let stem = pinned_stem(&fname_raw);
                if !w.contains(&stem) {
                    continue;
                }
                seen_pinned.insert(stem);
            }
            // Reference provenance. Super-resolution is penalised by a JPEG
            // reference in the opposite direction to restoration: the detail it
            // correctly reconstructs was quantised away in the reference, so
            // sharper output scores worse. ZENSR_EVAL_CLEAN_GT=1 drops them.
            let gt_src = gt_src_of(&fname_raw);
            if gt_src != "png" && std::env::var("ZENSR_EVAL_CLEAN_GT").as_deref() == Ok("1") {
                continue;
            }
            let Some(img) = decode_any(&f) else {
                eprintln!("skip (decode): {}", f.display());
                continue;
            };
            let Some(hr) = center_crop(&img, 512) else {
                eprintln!("skip (too small {}x{}): {}", img.w, img.h, f.display());
                continue;
            };
            let (lw, lh) = (hr.w / 4, hr.h / 4);
            let lr = resize_rgb8(&hr, lw, lh, zenresize::Filter::CatmullRom);

            let lr_planar = to_planar_f32(&lr);
            let t = Instant::now();
            let sr_planar = spanf_x4_tiled(&model, &lr_planar, lh, lw, threads, 0);
            let sr_ms = t.elapsed().as_secs_f64() * 1e3;
            let sr = planar_to_rgb8(&sr_planar, hr.w, hr.h);

            let lanczos = resize_rgb8(&lr, hr.w, hr.h, zenresize::Filter::Lanczos);
            let catrom = resize_rgb8(&lr, hr.w, hr.h, zenresize::Filter::CatmullRom);

            let fname = f.file_name().unwrap().to_string_lossy();
            for (m, out, ms) in [
                ("spanf", &sr, sr_ms),
                ("lanczos", &lanczos, 0.0),
                ("catmullrom", &catrom, 0.0),
            ] {
                let _ = writeln!(
                    tsv,
                    "{name}\t{fname}\t{m}\t{:.3}\t{:.3}\t{:.4}\t{ms:.1}\t{gt_src}",
                    psnr_rgb8(&hr, out),
                    ssim2(&hr, out),
                    butter_n3(&hr, out),
                );
            }
            used += 1;
            eprintln!("{name}/{fname}: {}x{} done ({sr_ms:.0}ms SR)", hr.w, hr.h);
        }
        // A pinned file that never got scored means the split and the corpus
        // disagree. Silence here would look identical to full coverage, which
        // is how a short eval passes for a complete one.
        //
        // Only meaningful once the quota is unfilled: with per_sub below the
        // pin count the loop stops early by design, and warning about the files
        // it never reached would cry wolf on every short run.
        if let (Some(w), true) = (want, used < per_sub) {
            let missing: Vec<&String> = w.iter().filter(|s| !seen_pinned.contains(*s)).collect();
            if !missing.is_empty() {
                eprintln!(
                    "== {name}: {used}/{per_sub} images; WARNING {} pinned file(s) missing from \
                     corpus (split and corpus disagree): {:?}",
                    missing.len(),
                    missing
                );
                continue;
            }
        }
        eprintln!("== {name}: {used} images");
    }
    std::fs::write(&out_path, &tsv).expect("write tsv");
    eprintln!("wrote {}", out_path.display());

    // Per-(subcorpus, method) medians.
    print_summary(&tsv);
}

fn print_summary(tsv: &str) {
    use std::collections::BTreeMap;
    let mut acc: BTreeMap<(String, String), Vec<[f64; 3]>> = BTreeMap::new();
    for line in tsv.lines().skip(2) {
        let c: Vec<&str> = line.split('\t').collect();
        if c.len() < 7 {
            continue;
        }
        let vals = [
            c[3].parse().unwrap_or(f64::NAN),
            c[4].parse().unwrap_or(f64::NAN),
            c[5].parse().unwrap_or(f64::NAN),
        ];
        acc.entry((c[0].into(), c[2].into()))
            .or_default()
            .push(vals);
    }
    println!("subcorpus\tmethod\tn\tpsnr_med\tssim2_med\tbutter_n3_med");
    for ((sub, m), rows) in acc {
        let med = |i: usize| {
            let mut v: Vec<f64> = rows
                .iter()
                .map(|r| r[i])
                .filter(|x| x.is_finite())
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if v.is_empty() {
                f64::NAN
            } else {
                v[v.len() / 2]
            }
        };
        println!(
            "{sub}\t{m}\t{}\t{:.2}\t{:.2}\t{:.3}",
            rows.len(),
            med(0),
            med(1),
            med(2)
        );
    }
}
