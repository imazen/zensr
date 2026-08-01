//! Rebuild JPEG-only corpus sources as pristine PNG references by downscaling
//! (user proposal 2026-08-01, measured by `pristine_probe`).
//!
//! JPEG artifacts are 8x8-block structured; downscaling shrinks their footprint
//! and averages ringing away. Measured residual bias in a restoration gain when
//! the reference is a downscaled JPEG instead of a true pristine source
//! (benchmarks/pristine_probe_*.tsv):
//!
//!   source q75:  1x -1.18   2x -0.37   3x -0.22
//!   source q85:  1x -0.72   2x -0.23   3x -0.15
//!   source q92:  1x -0.36   2x -0.12   3x -0.11
//!
//! Policy implemented here: **3x for sources below q90, 2x for q90+**, which
//! holds residual bias at ~0.1-0.2 ssim2 — below our metric floor and a
//! twentieth of the effects we measure. Source quality comes from zenjpeg's
//! own probe, so the decision is per-file rather than assumed.
//!
//! Writes PNG + a provenance manifest. Never overwrites the input corpus.
//!
//! Usage: make_pristine <src-root> <dst-root> [min_output_dim=512]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use zenpng::{encode_rgb8, Compression, EncodeConfig};

fn probe_quality(bytes: &[u8]) -> Option<(String, f32, String)> {
    let p = zenjpeg::detect::probe(bytes).ok()?;
    Some((
        format!("{:?}", p.encoder),
        p.quality.value,
        format!("{:?}", p.quality.scale),
    ))
}

/// 3x below q90, 2x at q90+; Butteraugli-distance scales (cjpegli family) use
/// d<=1.0 as the "high quality" equivalent.
fn scale_for(q: f32, scale_kind: &str) -> usize {
    let high = match scale_kind {
        "ButteraugliDistance" => q <= 1.0,
        _ => q >= 90.0,
    };
    if high {
        2
    } else {
        3
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let src = PathBuf::from(args.next().expect("src root"));
    let dst = PathBuf::from(args.next().expect("dst root"));
    let min_dim: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(512);

    let mut manifest = String::from(
        "# make_pristine: JPEG sources -> downscaled pristine PNG references\n\
         # policy: 3x below q90, 2x at q90+ (pristine_probe 2026-08-01)\n\
         subdir\tsrc_file\tdst_file\tsrc_encoder\tsrc_q\tsrc_scale_kind\tdownscale\tsrc_wh\tdst_wh\n",
    );
    let mut n_done = 0usize;
    let mut n_skip = 0usize;
    for (_, dir) in zensr_bench::SUBCORPORA {
        let sdir = src.join(dir);
        if !sdir.is_dir() {
            continue;
        }
        let ddir = dst.join(dir);
        std::fs::create_dir_all(&ddir).unwrap();
        for f in zensr_bench::list_images(&sdir) {
            let name = f.file_name().unwrap().to_string_lossy().to_string();
            let lower = name.to_ascii_lowercase();
            let is_jpeg = lower.ends_with(".jpg") || lower.ends_with(".jpeg");
            if !is_jpeg {
                continue; // PNG sources are already pristine — left in place
            }
            let Ok(bytes) = std::fs::read(&f) else { continue };
            let Some((enc, q, kind)) = probe_quality(&bytes) else {
                eprintln!("SKIP (probe failed) {name}");
                n_skip += 1;
                continue;
            };
            let Some(img) = zensr_bench::decode_any(&f) else {
                n_skip += 1;
                continue;
            };
            let n = scale_for(q, &kind);
            let (dw, dh) = (img.w / n, img.h / n);
            if dw < min_dim || dh < min_dim {
                eprintln!("SKIP (too small after {n}x: {dw}x{dh}) {name}");
                n_skip += 1;
                continue;
            }
            let small = zensr_bench::resize_rgb8(&img, dw, dh, zenresize::Filter::Lanczos);
            let pix: Vec<rgb::Rgb<u8>> = small
                .px
                .chunks_exact(3)
                .map(|c| rgb::Rgb { r: c[0], g: c[1], b: c[2] })
                .collect();
            let iref = imgref::ImgRef::new(&pix, dw, dh);
            let cfg = EncodeConfig::default().with_compression(Compression::High);
            let png = match encode_rgb8(iref, None, &cfg, &enough::Unstoppable, &enough::Unstoppable)
            {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("SKIP (encode {e:?}) {name}");
                    n_skip += 1;
                    continue;
                }
            };
            let stem = Path::new(&name).file_stem().unwrap().to_string_lossy();
            let out_name = format!("{stem}__pristine{n}x.png");
            std::fs::write(ddir.join(&out_name), &png).unwrap();
            let _ = writeln!(
                manifest,
                "{dir}\t{name}\t{out_name}\t{enc}\t{q:.1}\t{kind}\t{n}\t{}x{}\t{dw}x{dh}",
                img.w, img.h
            );
            n_done += 1;
            if n_done % 10 == 0 {
                eprintln!("{n_done} converted");
            }
        }
    }
    std::fs::write(dst.join("PRISTINE_MANIFEST.tsv"), &manifest).unwrap();
    println!("converted {n_done}, skipped {n_skip} -> {}", dst.display());
}
