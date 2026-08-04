//! Head-to-head on the two content classifiers zensr could route on.
//!
//! ROADMAP §1.12 measured that content type is the largest remaining routing
//! lever — a binary graphic/photographic split captures 44% of the headroom a
//! quality-only curve leaves. That figure used ground-truth labels, so it is a
//! ceiling. This binary produces the labels a real router would see.
//!
//! Two candidates, cheapest first:
//!
//! - `zenjpeg::detect::content::classify_from_luma_coefficients` — counts
//!   all-DC (zero-AC) blocks in the file's own luma coefficients. zensr already
//!   decodes those for the S10 projection, so the marginal cost is a pass over
//!   data it holds: no pixel work, no feature extraction.
//! - `zensr_zenjpeg::chooser::classify_rgb8` — a 21-feature logistic rule over
//!   zenanalyze features on the centre-512 crop. Trained and validated
//!   (`benchmarks/chooser_fit_2026-07-26.txt`), but needs a decode plus feature
//!   extraction.
//!
//! If the coefficient signal separates the corpus adequately, the pixel pass is
//! unnecessary. That is the question this answers.
//!
//! Ground truth is the subcorpus: documents/maps/screen are graphic, the rest
//! photographic — the same grouping §1.12's measurement used, so the two are
//! directly comparable.
//!
//! Usage: content_probe <corpus-root> <out-tsv> [per-sub=8] [threads=8]

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;
use zensr_bench::*;

/// Synthetic content: flat regions, sharp edges, limited palette. Same split as
/// `tools/routing_headroom.py::GRAPHIC_SUBS`; keep them in step.
const GRAPHIC: &[&str] = &["documents", "maps", "screen"];

const QS: &[u32] = &[15, 35, 55, 75, 90];

fn tmpdir() -> PathBuf {
    let d = std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap()
        .join("tmp");
    let _ = std::fs::create_dir_all(&d);
    d
}

fn encode(ppm: &PathBuf, jpg: &PathBuf, q: u32) -> bool {
    Command::new("cjpeg")
        .args(["-quality", &q.to_string(), "-sample", "2x2", "-outfile"])
        .arg(jpg)
        .arg(ppm)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().expect("usage: content_probe <root> <out.tsv>"));
    let out_path = PathBuf::from(args.next().expect("out tsv"));
    let per_sub: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(8);
    let _threads: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(8);

    let pinned = resolve_pinned(&root);

    let td = tmpdir();
    let ppm = td.join("cprobe.ppm");
    let jpg = td.join("cprobe.jpg");

    let mut tsv = String::new();
    let _ = writeln!(
        tsv,
        "# content classifier head-to-head; truth = subcorpus (graphic = {GRAPHIC:?})"
    );
    let _ = writeln!(
        tsv,
        "sub\tfile\tq\ttruth\tp_graphics\tchooser_class\tzero_ac_frac\tcoef_class\tchooser_ms\tcoef_ms\tgt_src"
    );

    for (name, dir) in &subcorpora_for(&root) {
        let (name, dir) = (name.as_str(), dir.as_str());
        let files = list_images(&root.join(dir));
        let want = pinned.as_ref().and_then(|m| m.get(dir));
        let truth = if GRAPHIC.contains(&name) {
            "graphic"
        } else {
            "photo"
        };
        let mut used = 0usize;
        for f in files {
            if used >= per_sub {
                break;
            }
            let fname = f.file_name().unwrap().to_string_lossy().to_string();
            if let Some(w) = want {
                if !w.contains(&pinned_stem(&fname)) {
                    continue;
                }
            }
            let gt_src = gt_src_of(&fname);
            let Some(img) = decode_any(&f) else { continue };
            let Some(hr) = center_crop(&img, 512) else {
                continue;
            };
            write_ppm_rgb(&hr, &ppm);

            for &q in QS {
                if !encode(&ppm, &jpg, q) {
                    continue;
                }
                let data = std::fs::read(&jpg).unwrap();

                // Pixel path: the 21-feature logistic rule. Timed WITHOUT the
                // decode — the restore path decodes pixels regardless, so the
                // decode is not marginal cost for this decision. Timing it in
                // would flatter the coefficient path, which does pay for its
                // own separate parse.
                let dec = zenjpeg::decoder::Decoder::new()
                    .decode(&data, enough::Unstoppable)
                    .expect("decode");
                let (dw, dh) = dec.dimensions();
                let px = dec.pixels_u8().expect("u8").to_vec();
                let t = Instant::now();
                let rep = zensr_zenjpeg::chooser::classify_rgb8(&px, dw as usize, dh as usize);
                let chooser_ms = t.elapsed().as_secs_f64() * 1e3;

                // Coefficient path: zero-AC block fraction over luma, on
                // coefficients the restore pipeline already reads.
                let t = Instant::now();
                let (coef_class, frac) = match zenjpeg::decoder::Decoder::new()
                    .decode_coefficients(&data, enough::Unstoppable)
                {
                    Ok(c) if !c.components.is_empty() => {
                        let luma = &c.components[0];
                        let nblocks = luma.coeffs.len() / 64;
                        let (ct, fr) = zenjpeg::detect::content::classify_from_luma_coefficients(
                            &luma.coeffs,
                            nblocks,
                        );
                        (format!("{ct:?}"), fr)
                    }
                    _ => ("Error".to_string(), f32::NAN),
                };
                let coef_ms = t.elapsed().as_secs_f64() * 1e3;

                let _ = writeln!(
                    tsv,
                    "{name}\t{fname}\t{q}\t{truth}\t{:.4}\t{:?}\t{:.4}\t{coef_class}\t{chooser_ms:.2}\t{coef_ms:.2}\t{gt_src}",
                    rep.p_graphics, rep.class, frac
                );
            }
            used += 1;
            eprintln!("{name}/{fname}: done");
        }
        eprintln!("== {name}: {used} images");
    }
    std::fs::write(&out_path, &tsv).expect("write tsv");
    eprintln!("wrote {}", out_path.display());
}

fn write_ppm_rgb(img: &Rgb8Img, path: &PathBuf) {
    let mut v = format!("P6\n{} {}\n255\n", img.w, img.h).into_bytes();
    v.extend_from_slice(&img.px);
    std::fs::write(path, v).expect("write ppm");
}
