//! Build the CLEAN-REFERENCE corpus: every canonical source turned into a PNG
//! reference that carries no JPEG artifacts, with per-file provenance.
//!
//! Three source kinds, three treatments, and the manifest records which one each
//! file got — because the ladder has to be readable split by provenance. The
//! 2026-07 record understated every gain precisely because that column did not
//! exist and JPEG-sourced references were mixed in unmarked.
//!
//!   native-png  copied through byte-for-byte. Already a clean reference.
//!   heic        decoded to PNG. HEVC artifacts are not JPEG artifacts, so a
//!               restoration model has no tendency to remove them — this is the
//!               cheap win, no downscale needed. The decoder bakes the
//!               container's irot/imir itself; NOTHING may apply EXIF on top.
//!   jpeg        downscaled until the source's own artifacts fall below the
//!               quantiser floor, then treated as clean (below).
//!
//! The JPEG rule is not optional cleanup: a JPEG ground truth penalises the
//! model for removing artifacts that are present in the reference (ROADMAP
//! §0.1). But dropping JPEGs is not an option either — the entire photographic
//! half of this corpus is JPEG-sourced, so dropping them would leave screenshots,
//! plots, AI renders and scans, and silently turn the "photo" leg of the
//! content-split curves into a fiction.
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
//!
//! `<src-root>` is the canonical corpus repo (github.com/imazen/imazen-26); the
//! subdirectory list is `zensr_bench::SUBCORPORA`, which names the canonical
//! numbered folders.

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
        "# make_pristine: canonical sources -> clean PNG references\n\
         # ref_kind: native-png (copied) | heic (decoded) | jpeg (downscaled)\n\
         # jpeg policy: 3x below q90, 2x at q90+ (pristine_probe 2026-08-01)\n\
         subdir\tsrc_file\tdst_file\tref_kind\tsrc_encoder\tsrc_q\tsrc_scale_kind\tdownscale\tsrc_wh\tdst_wh\n",
    );
    let mut n_done = 0usize;
    let mut n_skip = 0usize;
    let mut n_png = 0usize;
    let mut n_heic = 0usize;
    let mut n_jpeg = 0usize;
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
            let is_heic = lower.ends_with(".heic") || lower.ends_with(".heif");
            let stem = Path::new(&name).file_stem().unwrap().to_string_lossy();
            let Ok(bytes) = std::fs::read(&f) else {
                continue;
            };

            // native PNG: already a clean reference. Copy the bytes rather than
            // re-encoding — a re-encode would be lossless but not byte-identical,
            // which costs the ability to prove the reference is untouched.
            if !is_jpeg && !is_heic {
                let out_name = name.clone();
                if std::fs::write(ddir.join(&out_name), &bytes).is_err() {
                    n_skip += 1;
                    continue;
                }
                let (w, h) = zensr_bench::decode_any(&f).map_or((0, 0), |i| (i.w, i.h));
                let _ = writeln!(
                    manifest,
                    "{dir}\t{name}\t{out_name}\tnative-png\t-\t-\t-\t1\t{w}x{h}\t{w}x{h}"
                );
                n_done += 1;
                n_png += 1;
                continue;
            }

            // HEIC: decode straight to PNG. No downscale — HEVC artifacts are
            // not JPEG artifacts, so the restoration model has no tendency to
            // remove them and they cost the reference nothing.
            if is_heic {
                let Some(img) = zensr_bench::decode_any(&f) else {
                    eprintln!("SKIP (heic decode) {name}");
                    n_skip += 1;
                    continue;
                };
                let Some(png) = encode_png(&img) else {
                    eprintln!("SKIP (encode) {name}");
                    n_skip += 1;
                    continue;
                };
                let out_name = format!("{stem}__heic.png");
                std::fs::write(ddir.join(&out_name), &png).unwrap();
                let _ = writeln!(
                    manifest,
                    "{dir}\t{name}\t{out_name}\theic\theic\t-\t-\t1\t{}x{}\t{}x{}",
                    img.w, img.h, img.w, img.h
                );
                n_done += 1;
                n_heic += 1;
                continue;
            }
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
            let Some(png) = encode_png(&small) else {
                eprintln!("SKIP (encode) {name}");
                n_skip += 1;
                continue;
            };
            let out_name = format!("{stem}__pristine{n}x.png");
            std::fs::write(ddir.join(&out_name), &png).unwrap();
            let _ = writeln!(
                manifest,
                "{dir}\t{name}\t{out_name}\tjpeg\t{enc}\t{q:.1}\t{kind}\t{n}\t{}x{}\t{dw}x{dh}",
                img.w, img.h
            );
            n_done += 1;
            n_jpeg += 1;
            if n_done % 10 == 0 {
                eprintln!("{n_done} converted");
            }
        }
    }
    std::fs::write(dst.join("PRISTINE_MANIFEST.tsv"), &manifest).unwrap();
    println!(
        "clean corpus: {n_done} references ({n_png} native-png, {n_heic} heic, \
         {n_jpeg} jpeg-downscaled), {n_skip} skipped -> {}",
        dst.display()
    );
}

/// One PNG encode, shared by all three source kinds so they cannot drift apart.
fn encode_png(img: &zensr_bench::Rgb8Img) -> Option<Vec<u8>> {
    let pix: Vec<rgb::Rgb<u8>> = img
        .px
        .chunks_exact(3)
        .map(|c| rgb::Rgb {
            r: c[0],
            g: c[1],
            b: c[2],
        })
        .collect();
    let iref = imgref::ImgRef::new(&pix, img.w, img.h);
    let cfg = EncodeConfig::default().with_compression(Compression::High);
    encode_rgb8(iref, None, &cfg, &enough::Unstoppable, &enough::Unstoppable).ok()
}
