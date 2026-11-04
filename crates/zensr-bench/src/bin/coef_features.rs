//! Cheap coefficient-domain features, measured against per-image restore gain.
//!
//! Content class (zero-AC-block fraction) is shipped and buys +0.15 ssim2 of
//! routing quality, but the per-image oracle is another +0.26 above that, and
//! resolving the same signal into more buckets does not close it — 5 quantile
//! buckets score within noise of a binary threshold. The signal carries about
//! one bit. Getting further needs a *different* signal.
//!
//! Everything here comes out of one pass over coefficients the restore path
//! already parses, so the marginal cost is a few arithmetic ops per block:
//!
//! - `zero_ac_blocks` — blocks with no AC at all. The shipped content signal.
//! - `zero_ac_coefs` — individual AC coefficients quantised to zero. Measures
//!   how much was *discarded* rather than how flat the image is, which is
//!   closer to what the model has to reconstruct.
//! - `hf_survival` — share of surviving AC energy in zigzag 32..64. Fine detail
//!   is what quantisation kills first and what the model most has to invent.
//! - `mean_abs_ac` — average magnitude of surviving AC.
//! - `dc_spread` — stdev of block DC. Large-scale structure, which distinguishes
//!   a flat UI from a flat-but-shaded render.
//! - `bpp` — bits per pixel. Free, and jointly encodes content complexity and
//!   quality in a way neither alone does.
//!
//! Emitted on the same grid as the restore ladder so rows join directly on
//! (file, encoder, ss, q) with no nearest-quality approximation.
//!
//! Usage: coef_features <corpus-root> <out-tsv> [per-sub=8]
//!   env ZENSR_QS, ZENSR_ENCODERS, ZENSR_SS

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;
use zensr_bench::*;

fn tmpdir() -> PathBuf {
    let d = std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap()
        .join("tmp");
    let _ = std::fs::create_dir_all(&d);
    d
}

fn encode(ppm: &PathBuf, jpg: &PathBuf, enc: &str, q: u32, ss: &str) -> bool {
    let sample = if ss == "444" { "1x1" } else { "2x2" };
    let st = match enc {
        "turbo" => Command::new("cjpeg")
            .args(["-quality", &q.to_string(), "-sample", sample, "-outfile"])
            .arg(jpg)
            .arg(ppm)
            .status(),
        // The prebuilt mozjpeg needs its own libs on the path, exactly as in
        // `dejpeg_eval` — without this it silently fails and the whole encoder
        // vanishes from the output rather than erroring.
        "mozjpeg" => {
            let home = std::env::var("HOME").unwrap();
            Command::new(format!("{home}/tmp/ati-bin/mozjpeg-cjpeg"))
                .env(
                    "LD_LIBRARY_PATH",
                    format!("{home}/tmp/ati-bin/mozjpeg-lib64"),
                )
                .args(["-quality", &q.to_string(), "-sample", sample, "-outfile"])
                .arg(jpg)
                .arg(ppm)
                .status()
        }
        _ => Command::new("cjpegli")
            .arg(ppm)
            .arg(jpg)
            .args(["-q", &q.to_string(), &format!("--chroma_subsampling={ss}")])
            .status(),
    };
    st.map(|s| s.success()).unwrap_or(false)
}

#[path = "coef_features/metrics.rs"]
mod metrics;
use metrics::features;

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
    let root = PathBuf::from(args.next().expect("usage: coef_features <root> <out.tsv>"));
    let out_path = PathBuf::from(args.next().expect("out tsv"));
    let per_sub: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(8);

    let qs: Vec<u32> = std::env::var("ZENSR_QS")
        .unwrap_or_else(|_| "15,35,55,75,85,90,94,96,98,100".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let encs: Vec<String> = std::env::var("ZENSR_ENCODERS")
        .unwrap_or_else(|_| "turbo,mozjpeg".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let subs: Vec<String> = std::env::var("ZENSR_SS")
        .unwrap_or_else(|_| "420,444".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    let pinned = resolve_pinned(&root);
    let td = tmpdir();
    let ppm = td.join("cfeat.ppm");
    let jpg = td.join("cfeat.jpg");

    let mut tsv = String::new();
    let _ = writeln!(tsv, "# cheap coefficient-domain features vs restore gain");
    let _ = writeln!(
        tsv,
        "sub\tfile\tencoder\tss\tq\tbytes\tbpp\tzero_ac_blocks\tzero_ac_coefs\thf_survival\tmean_abs_ac\tdc_spread\tac_count_spread\tgutted_block_frac\tgutted_clustering"
    );

    for (name, dir) in &subcorpora_for(&root) {
        let (name, dir) = (name.as_str(), dir.as_str());
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
            let Some(hr) = center_crop(&img, crop_cap()) else {
                continue;
            };
            write_ppm_rgb(&hr, &ppm);
            let pixels = (hr.w * hr.h) as f64;

            for e in &encs {
                for ss in &subs {
                    for &q in &qs {
                        if !encode(&ppm, &jpg, e, q, ss) {
                            continue;
                        }
                        let data = std::fs::read(&jpg).unwrap();
                        let Ok(c) = zenjpeg::decoder::Decoder::new()
                            .decode_coefficients(&data, enough::Unstoppable)
                        else {
                            continue;
                        };
                        let Some(luma) = c.components.first() else {
                            continue;
                        };
                        let nb = luma.coeffs.len() / 64;
                        let Some(ft) = features(&luma.coeffs, nb, luma.blocks_wide) else {
                            continue;
                        };
                        let _ = writeln!(
                            tsv,
                            "{name}\t{fname}\t{e}\t{ss}\t{q}\t{}\t{:.4}\t{:.5}\t{:.5}\t{:.5}\t{:.4}\t{:.2}\t{:.4}\t{:.5}\t{:.5}",
                            data.len(),
                            data.len() as f64 * 8.0 / pixels,
                            ft.zero_ac_blocks,
                            ft.zero_ac_coefs,
                            ft.hf_survival,
                            ft.mean_abs_ac,
                            ft.dc_spread,
                            ft.ac_count_spread,
                            ft.gutted_block_frac,
                            ft.gutted_clustering,
                        );
                    }
                }
            }
            used += 1;
            eprintln!("{name}/{fname}");
        }
        eprintln!("== {name}: {used}");
    }
    std::fs::write(&out_path, &tsv).expect("write");
    eprintln!("wrote {}", out_path.display());
}

fn write_ppm_rgb(img: &Rgb8Img, path: &PathBuf) {
    let mut v = format!("P6\n{} {}\n255\n", img.w, img.h).into_bytes();
    v.extend_from_slice(&img.px);
    std::fs::write(path, v).expect("write ppm");
}
