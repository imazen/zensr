//! Build the CLEAN-REFERENCE corpus: every canonical source turned into a PNG
//! reference that carries no JPEG artifacts, with per-file provenance.
//!
//! Three source kinds, three treatments, and the manifest records which one each
//! file got — because the ladder has to be readable split by provenance. The
//! 2026-07 record understated every gain precisely because that column did not
//! exist and JPEG-sourced references were mixed in unmarked.
//!
//!   native-png  copied through byte-for-byte. Already a clean reference, and
//!               copying rather than re-encoding keeps the ability to PROVE it
//!               is untouched.
//!   heic        decoded to PNG, no downscale. HEVC artifacts are not JPEG
//!               artifacts, so a restoration model has no tendency to remove
//!               them — this is the cheap win.
//!   jpeg        downscaled until the source's own artifacts fall below the
//!               quantiser floor, then treated as clean (policy below).
//!
//! The JPEG rule is not optional cleanup: a JPEG ground truth penalises the
//! model for removing artifacts that are present in the reference (ROADMAP
//! §0.1). But dropping JPEGs is not the alternative — the entire photographic
//! half of this corpus is JPEG-sourced, so dropping it would leave screenshots,
//! plots, AI renders and scans, and silently turn the "photo" leg of the
//! content-split curves into a fiction.
//!
//! JPEG artifacts are 8x8-block structured; downscaling shrinks their footprint
//! and averages ringing away. Measured residual bias in a restoration gain when
//! the reference is a downscaled JPEG instead of a true pristine source
//! (`benchmarks/pristine_probe_*.tsv`):
//!
//!   source q75:  1x -1.18   2x -0.37   3x -0.22
//!   source q85:  1x -0.72   2x -0.23   3x -0.15
//!   source q92:  1x -0.36   2x -0.12   3x -0.11
//!
//! Policy: **3x for sources below q90, 2x for q90+**, holding residual bias at
//! ~0.1-0.2 ssim2 — below our metric floor and a twentieth of the effects we
//! measure. Source quality comes from zenjpeg's own probe, so the decision is
//! per-file rather than assumed.
//!
//! Never overwrites the input corpus.
//!
//! Usage: `make_pristine <src-root> <dst-root> [min_output_dim=512]`
//!
//! `<src-root>` is the canonical corpus repo (github.com/imazen/imazen-26); the
//! subdirectory list is [`zensr_bench::SUBCORPORA`], which names the canonical
//! numbered folders. Conversion is parallel across files — single-threaded this
//! measured 5 files/min on a 16-core box, about 6.7 hours for the corpus.

use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use zenpng::{encode_rgb8, Compression, EncodeConfig};

/// What treatment a reference received. Written to the manifest's `ref_kind`
/// column, which is the whole point of the manifest.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RefKind {
    NativePng,
    Heic,
    Jpeg,
}

impl RefKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::NativePng => "native-png",
            Self::Heic => "heic",
            Self::Jpeg => "jpeg",
        }
    }
}

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

/// One PNG encode, shared by every source kind so they cannot drift apart.
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

/// Convert one source. Returns its `(kind, manifest line)`, or `None` if it was
/// skipped. Pure per-file work — no shared state, which is what lets the caller
/// run these in parallel.
fn convert_one(dir: &str, f: &Path, ddir: &Path, min_dim: usize) -> Option<(RefKind, String)> {
    let name = f.file_name()?.to_string_lossy().to_string();
    let lower = name.to_ascii_lowercase();
    let is_jpeg = lower.ends_with(".jpg") || lower.ends_with(".jpeg");
    let is_heic = lower.ends_with(".heic") || lower.ends_with(".heif");
    let stem = Path::new(&name).file_stem()?.to_string_lossy().to_string();
    let bytes = std::fs::read(f).ok()?;

    // Native PNG: already a clean reference. Copy the bytes.
    if !is_jpeg && !is_heic {
        std::fs::write(ddir.join(&name), &bytes).ok()?;
        let (w, h) = zensr_bench::decode_any(f).map_or((0, 0), |i| (i.w, i.h));
        return Some((
            RefKind::NativePng,
            format!("{dir}\t{name}\t{name}\tnative-png\t-\t-\t-\t1\t{w}x{h}\t{w}x{h}\n"),
        ));
    }

    // HEIC: decode straight to PNG, no downscale.
    if is_heic {
        let img = zensr_bench::decode_any(f).or_else(|| {
            eprintln!("SKIP (heic decode) {name}");
            None
        })?;
        let png = encode_png(&img)?;
        let out_name = format!("{stem}__heic.png");
        std::fs::write(ddir.join(&out_name), &png).ok()?;
        return Some((
            RefKind::Heic,
            format!(
                "{dir}\t{name}\t{out_name}\theic\theic\t-\t-\t1\t{}x{}\t{}x{}\n",
                img.w, img.h, img.w, img.h
            ),
        ));
    }

    // JPEG: downscale to pristine.
    let Some((enc, q, kind)) = probe_quality(&bytes) else {
        eprintln!("SKIP (probe failed) {name}");
        return None;
    };
    let img = zensr_bench::decode_any(f)?;
    let n = scale_for(q, &kind);
    let (dw, dh) = (img.w / n, img.h / n);
    if dw < min_dim || dh < min_dim {
        eprintln!("SKIP (too small after {n}x: {dw}x{dh}) {name}");
        return None;
    }
    let small = zensr_bench::resize_rgb8(&img, dw, dh, zenresize::Filter::Lanczos);
    let png = encode_png(&small).or_else(|| {
        eprintln!("SKIP (encode) {name}");
        None
    })?;
    let out_name = format!("{stem}__pristine{n}x.png");
    std::fs::write(ddir.join(&out_name), &png).ok()?;
    Some((
        RefKind::Jpeg,
        format!(
            "{dir}\t{name}\t{out_name}\tjpeg\t{enc}\t{q:.1}\t{kind}\t{n}\t{}x{}\t{dw}x{dh}\n",
            img.w, img.h
        ),
    ))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let src = PathBuf::from(args.next().expect("src root"));
    let dst = PathBuf::from(args.next().expect("dst root"));
    let min_dim: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(512);

    // Gather the work first, so the conversion itself is a flat parallel map.
    let mut work: Vec<(&str, PathBuf, PathBuf)> = Vec::new();
    for (_, dir) in zensr_bench::SUBCORPORA {
        let sdir = src.join(dir);
        if !sdir.is_dir() {
            continue;
        }
        let ddir = dst.join(dir);
        std::fs::create_dir_all(&ddir).unwrap();
        for f in zensr_bench::list_images(&sdir) {
            work.push((dir, f, ddir.clone()));
        }
    }
    let total = work.len();
    eprintln!("{total} sources to convert");

    let done = AtomicUsize::new(0);
    let rows: Vec<Option<(RefKind, String)>> = work
        .par_iter()
        .map(|(dir, f, ddir)| {
            let r = convert_one(dir, f, ddir, min_dim);
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            if n % 50 == 0 {
                eprintln!("{n}/{total}");
            }
            r
        })
        .collect();

    let mut manifest = String::from(
        "# make_pristine: canonical sources -> clean PNG references\n\
         # ref_kind: native-png (copied) | heic (decoded) | jpeg (downscaled)\n\
         # jpeg policy: 3x below q90, 2x at q90+ (pristine_probe 2026-08-01)\n\
         subdir\tsrc_file\tdst_file\tref_kind\tsrc_encoder\tsrc_q\tsrc_scale_kind\tdownscale\tsrc_wh\tdst_wh\n",
    );
    let (mut n_png, mut n_heic, mut n_jpeg, mut n_skip) = (0usize, 0usize, 0usize, 0usize);
    let mut lines: Vec<String> = Vec::new();
    for r in rows {
        match r {
            Some((kind, line)) => {
                match kind {
                    RefKind::NativePng => n_png += 1,
                    RefKind::Heic => n_heic += 1,
                    RefKind::Jpeg => n_jpeg += 1,
                }
                lines.push(line);
            }
            None => n_skip += 1,
        }
    }
    // Sorted, so the manifest is byte-identical run to run regardless of how the
    // threads interleaved. A manifest that reorders itself cannot be diffed.
    lines.sort();
    for l in &lines {
        manifest.push_str(l);
    }
    std::fs::write(dst.join("PRISTINE_MANIFEST.tsv"), &manifest).unwrap();
    println!(
        "clean corpus: {} references ({n_png} native-png, {n_heic} heic, \
         {n_jpeg} jpeg-downscaled), {n_skip} skipped -> {}",
        n_png + n_heic + n_jpeg,
        dst.display()
    );
}
