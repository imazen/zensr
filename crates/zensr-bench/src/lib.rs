//! Shared eval helpers (decode, crop, resize, metrics) for the bench bins.
#![allow(clippy::too_many_arguments)]

use std::path::{Path, PathBuf};

#[derive(Clone)]
pub struct Rgb8Img {
    pub px: Vec<u8>, // interleaved RGB8
    pub w: usize,
    pub h: usize,
}

/// Convert a zenpixels PixelBuffer (any 8/16-bit gray/rgb/rgba layout) to RGB8.
pub fn pixelbuffer_to_rgb8(buf: &zenpixels::PixelBuffer) -> Option<Rgb8Img> {
    let v = buf.as_slice();
    let (w, h) = (v.width() as usize, v.rows() as usize);
    let d = v.descriptor();
    let bpp = d.bytes_per_pixel();
    let bpc = d.bytes_per_channel().max(1);
    let ch = bpp / bpc;
    let mut px = vec![0u8; w * h * 3];
    for y in 0..h {
        let row = v.row(y as u32);
        for x in 0..w {
            let s = x * bpp;
            let dsti = (y * w + x) * 3;
            let sample = |c: usize| -> u8 {
                let o = s + c * bpc;
                if bpc == 2 {
                    u16::from_le_bytes([row[o], row[o + 1]]).to_be_bytes()[0]
                } else {
                    row[o]
                }
            };
            match ch {
                3 | 4 => {
                    px[dsti] = sample(0);
                    px[dsti + 1] = sample(1);
                    px[dsti + 2] = sample(2);
                }
                1 | 2 => {
                    let g = sample(0);
                    px[dsti] = g;
                    px[dsti + 1] = g;
                    px[dsti + 2] = g;
                }
                _ => return None,
            }
        }
    }
    Some(Rgb8Img { px, w, h })
}

pub fn decode_any(path: &Path) -> Option<Rgb8Img> {
    let data = std::fs::read(path).ok()?;
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "png" => {
            let out = zenpng::decode(
                &data,
                &zenpng::PngDecodeConfig::default(),
                &enough::Unstoppable,
            )
            .ok()?;
            pixelbuffer_to_rgb8(&out.pixels)
        }
        "jpg" | "jpeg" => {
            let r = zenjpeg::decoder::Decoder::new()
                .decode(&data, zenjpeg::encoder::Unstoppable)
                .ok()?;
            let (w, h) = r.dimensions();
            let px = r.pixels_u8()?;
            if px.len() != w as usize * h as usize * 3 {
                return None; // grayscale/CMYK: skip for eval v1
            }
            Some(Rgb8Img {
                px: px.to_vec(),
                w: w as usize,
                h: h as usize,
            })
        }
        _ => None,
    }
}

pub fn center_crop(img: &Rgb8Img, cap: usize) -> Option<Rgb8Img> {
    let cw = (img.w.min(cap) / 4) * 4;
    let ch = (img.h.min(cap) / 4) * 4;
    if cw < 64 || ch < 64 {
        return None;
    }
    let x0 = (img.w - cw) / 2;
    let y0 = (img.h - ch) / 2;
    let mut px = Vec::with_capacity(cw * ch * 3);
    for y in 0..ch {
        let s = ((y0 + y) * img.w + x0) * 3;
        px.extend_from_slice(&img.px[s..s + cw * 3]);
    }
    Some(Rgb8Img { px, w: cw, h: ch })
}

pub fn resize_rgb8(img: &Rgb8Img, dw: usize, dh: usize, filter: zenresize::Filter) -> Rgb8Img {
    let config = zenresize::ResizeConfig::builder(img.w as u32, img.h as u32, dw as u32, dh as u32)
        .filter(filter)
        .format(zenresize::PixelDescriptor::RGB8_SRGB)
        .build();
    let out = zenresize::Resizer::new(&config).resize(&img.px);
    Rgb8Img {
        px: out,
        w: dw,
        h: dh,
    }
}

pub fn to_planar_f32(img: &Rgb8Img) -> Vec<f32> {
    let plane = img.w * img.h;
    let mut p = vec![0.0f32; 3 * plane];
    for i in 0..plane {
        p[i] = img.px[3 * i] as f32 / 255.0;
        p[plane + i] = img.px[3 * i + 1] as f32 / 255.0;
        p[2 * plane + i] = img.px[3 * i + 2] as f32 / 255.0;
    }
    p
}

pub fn planar_to_rgb8(p: &[f32], w: usize, h: usize) -> Rgb8Img {
    let plane = w * h;
    let mut px = vec![0u8; 3 * plane];
    for i in 0..plane {
        px[3 * i] = (p[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        px[3 * i + 1] = (p[plane + i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        px[3 * i + 2] = (p[2 * plane + i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
    Rgb8Img { px, w, h }
}

pub fn psnr_rgb8(a: &Rgb8Img, b: &Rgb8Img) -> f64 {
    let mut se = 0.0f64;
    for (x, y) in a.px.iter().zip(b.px.iter()) {
        let d = *x as f64 - *y as f64;
        se += d * d;
    }
    let mse = se / a.px.len() as f64 / (255.0 * 255.0);
    if mse == 0.0 {
        99.0
    } else {
        -10.0 * mse.log10()
    }
}

pub fn ssim2(a: &Rgb8Img, b: &Rgb8Img) -> f64 {
    use imgref::Img;
    let conv = |i: &Rgb8Img| -> Img<Vec<[u8; 3]>> {
        let v: Vec<[u8; 3]> = i.px.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        Img::new(v, i.w, i.h)
    };
    let (ra, rb) = (conv(a), conv(b));
    fast_ssim2::compute_ssimulacra2(ra.as_ref(), rb.as_ref()).unwrap_or(f64::NAN)
}

pub fn butter_n3(a: &Rgb8Img, b: &Rgb8Img) -> f64 {
    use butteraugli::{butteraugli, ButteraugliParams, Img, RGB8};
    use rgb::FromSlice;
    let ia: Img<Vec<RGB8>> = Img::new(a.px.as_rgb().to_vec(), a.w, a.h);
    let ib: Img<Vec<RGB8>> = Img::new(b.px.as_rgb().to_vec(), b.w, b.h);
    match butteraugli(ia.as_ref(), ib.as_ref(), &ButteraugliParams::default()) {
        Ok(r) => r.pnorm_3,
        Err(_) => f64::NAN,
    }
}

pub fn list_images(dir: &Path) -> Vec<PathBuf> {
    // Recursive: several subcorpora nest by resolution/source folders.
    fn walk(dir: &Path, out: &mut Vec<PathBuf>, depth: usize) {
        if depth > 4 {
            return;
        }
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out, depth + 1);
            } else if p
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg"))
                .unwrap_or(false)
            {
                out.push(p);
            }
        }
    }
    let mut v = Vec::new();
    walk(dir, &mut v, 0);
    v.sort();
    v
}

/// Stem used to match a file against the pinned eval list: basename without
/// extension, with any `__pristineNx` suffix removed so a downscaled-to-pristine
/// replacement still matches the original it was derived from.
pub fn pinned_stem(fname: &str) -> String {
    let base = fname.rsplit_once('.').map(|(a, _)| a).unwrap_or(fname);
    match base.find("__pristine") {
        Some(i) => base[..i].to_string(),
        None => base.to_string(),
    }
}

/// The pinned eval split, keyed by directory.
///
/// Selecting files by "first N sorted" is not safe: the sort slides past
/// decode-skipped and format-filtered files, which is how training images
/// reached an eval run (2026-07-23 postmortem; again 2026-08-02 when a pristine
/// directory held only the JPEG-sourced subset). Any eval that filters files
/// for ANY reason — decode failure, minimum size, extension — needs this, since
/// each skip slides selection one file deeper into the training set.
///
/// Returns None when no list is available, so the caller can fall back to
/// sorted order and say so loudly.
pub fn load_pinned(
    path: &str,
) -> Option<std::collections::HashMap<String, std::collections::HashSet<String>>> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut m: std::collections::HashMap<String, std::collections::HashSet<String>> =
        std::collections::HashMap::new();
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let mut it = line.split('\t');
        if let (Some(d), Some(f)) = (it.next(), it.next()) {
            m.entry(d.to_string()).or_default().insert(pinned_stem(f));
        }
    }
    if m.is_empty() {
        None
    } else {
        Some(m)
    }
}

/// Default location of the pinned split, overridable with `ZENSR_EVAL_PIN`.
pub fn pin_path() -> String {
    std::env::var("ZENSR_EVAL_PIN")
        .unwrap_or_else(|_| "eval_split/imazen26_eval_files.tsv".to_string())
}

/// The canonical imazen-26 corpus checkout (github.com/imazen/imazen-26).
///
/// The repository IS the corpus: it carries the manifests, the canonical
/// train/validate/test split and the variant registry, and versions them as one
/// unit. Image bytes are synced into its (gitignored) class folders from R2.
/// Override with `IMAZEN26_REPO`.
pub fn imazen26_repo() -> PathBuf {
    if let Ok(p) = std::env::var("IMAZEN26_REPO") {
        return PathBuf::from(p);
    }
    match std::env::var("HOME") {
        Ok(h) => PathBuf::from(h).join("work").join("imazen-26"),
        Err(_) => PathBuf::from("imazen-26"),
    }
}

/// Path of the effective split — the contract between the trainer and the eval.
pub const EFFECTIVE_SPLIT: &str = "eval_split/imazen26_effective_split.tsv";

/// The held-out half of the effective split — `validate` ∪ `test` — as
/// `content_class -> {file stem}`.
///
/// Reads `EFFECTIVE_SPLIT`, which `tools/corpus_split.py --write` generates from
/// the corpus repo (`just split`). It is deliberately NOT the repo's raw
/// `manifests/{validate,test}.tsv`, and that distinction is load-bearing: the
/// near-duplicate same-bucketing the trainer applies moves 309 files, **180 of
/// which cross between held-out and train**. Reading the raw buckets here would
/// score 180 files the model was trained on — measured, not hypothesised.
///
/// Returns None when the file is absent; callers must fail rather than fall back
/// to sorted order or to the raw manifests. Both fallbacks have leaked training
/// images into an eval in this repo already.
pub fn canonical_holdout() -> Option<std::collections::HashMap<String, std::collections::HashSet<String>>>
{
    let text = std::fs::read_to_string(EFFECTIVE_SPLIT).ok()?;
    // Warn if either input moved after the split was generated — the corpus
    // itself, or the rule that derives the buckets. A stale split is the same
    // trainer/eval disagreement in slow motion.
    if let Ok(a) = std::fs::metadata(EFFECTIVE_SPLIT).and_then(|m| m.modified()) {
        for newer in [
            imazen26_repo().join("CORPUS-MANIFEST.tsv"),
            PathBuf::from("tools/corpus_split.py"),
        ] {
            if let Ok(b) = std::fs::metadata(&newer).and_then(|m| m.modified()) {
                if b > a {
                    eprintln!(
                        "WARNING: {EFFECTIVE_SPLIT} is older than {} — run `just split`",
                        newer.display()
                    );
                }
            }
        }
    }
    let mut m: std::collections::HashMap<String, std::collections::HashSet<String>> =
        Default::default();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split('\t');
        let (Some(path), Some(bucket)) = (it.next(), it.next()) else {
            continue;
        };
        if bucket == "train" {
            continue;
        }
        let Some((class, file)) = path.split_once('/') else {
            continue;
        };
        let stem = file.rsplit('/').next().unwrap_or(file);
        m.entry(class.to_string())
            .or_default()
            .insert(pinned_stem(stem));
    }
    if m.is_empty() {
        None
    } else {
        Some(m)
    }
}

/// Resolve the pinned split for a corpus, and say what was resolved.
///
/// A corpus none of which is in any training set needs no pin — the pin exists
/// to keep training images out of an eval, and there are none to keep out. Such
/// a corpus declares itself with a `NO_PIN_REQUIRED` file in its root whose
/// contents explain why; the explanation is echoed so the claim is auditable
/// rather than taken on trust.
///
/// Without that marker and without a pin list, this warns loudly: "first N
/// sorted" silently admitting training images is the failure this whole
/// mechanism exists to prevent, and a warning nobody reads is how it recurred
/// twice.
pub fn resolve_pinned(
    root: &Path,
) -> Option<std::collections::HashMap<String, std::collections::HashSet<String>>> {
    let marker = root.join("NO_PIN_REQUIRED");
    if let Ok(why) = std::fs::read_to_string(&marker) {
        let why = why
            .lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("")
            .trim();
        eprintln!("no pin needed for this corpus — {why}");
        return None;
    }
    // The corpus repo's own split first — it is the source of truth, and a
    // vendored copy is exactly what drifted last time.
    if std::env::var("ZENSR_EVAL_PIN").is_err() {
        if let Some(m) = canonical_holdout() {
            eprintln!(
                "eval split: effective validate+test from {EFFECTIVE_SPLIT} ({} classes)",
                m.len()
            );
            return Some(m);
        }
    }
    let path = pin_path();
    let pinned = load_pinned(&path);
    match &pinned {
        Some(m) => eprintln!("pinned eval split: {} ({} dirs)", path, m.len()),
        None => eprintln!(
            "WARNING: no eval split — {EFFECTIVE_SPLIT} is missing (run `just split`; \
             it needs the canonical corpus at {}), and there is no pin at {path} or \
             NO_PIN_REQUIRED marker in {}. Falling back to sorted order, which can \
             admit training images.",
            imazen26_repo().display(),
            root.display()
        ),
    }
    pinned
}

/// Provenance of a reference image: `"png"` if the ground truth is lossless,
/// `"jpg"` if the reference is ITSELF a JPEG.
///
/// A JPEG reference is compressed, so it contains the very artifacts a restore
/// model is asked to remove and the very detail a super-resolution model is
/// asked to invent — the metric penalises correct output in both directions.
/// On 2026-07-31 this was 39% of the pinned eval split and it understated every
/// absolute gain. Record it per row so any run can be audited after the fact.
pub fn gt_src_of(fname: &str) -> &'static str {
    if fname.to_ascii_lowercase().ends_with(".png") {
        "png"
    } else {
        "jpg"
    }
}

/// zensim score (PreviewV0_2 default profile), 0-100-ish scale.
pub fn zensim_score(a: &Rgb8Img, b: &Rgb8Img) -> f64 {
    use zensim::source::RgbSlice;
    use zensim::{Zensim, ZensimProfile};
    let (ca, _) = a.px.as_chunks::<3>();
    let (cb, _) = b.px.as_chunks::<3>();
    let sa = RgbSlice::new(ca, a.w, a.h);
    let sb = RgbSlice::new(cb, b.w, b.h);
    match Zensim::new(ZensimProfile::PreviewV0_2).compute(&sa, &sb) {
        Ok(r) => r.score(),
        Err(_) => f64::NAN,
    }
}

// ---- shared eval-system helpers (used by systems_eval, ert_eval) ----

pub struct Scored {
    pub psnr: f64,
    pub ssim2: f64,
    pub butter: f64,
}

/// Score a pair on all three metrics.
///
/// `ZENSR_EVAL_NO_BUTTER=1` skips butteraugli and reports NaN for it. Measured
/// per call at 512²: ssim2 19.75 ms, butteraugli 16.58 ms, psnr 0.00 ms — so
/// butteraugli is **46% of the metric cost**, and every routing curve zensr
/// ships is fitted on ssim2. A sweep whose only consumer is the routing work
/// pays a large bill for a column it will not read.
///
/// Deliberately opt-*out*: butteraugli disagreeing with ssim2 is a real finding
/// (§0.3, the `renders` case), so it stays on by default and is dropped only
/// when a run's purpose is known. NaN rather than a plausible number, so a
/// skipped column can never be mistaken for a measured one.
pub fn score(hr: &Rgb8Img, out: &Rgb8Img) -> Scored {
    let skip_butter = std::env::var("ZENSR_EVAL_NO_BUTTER").as_deref() == Ok("1");
    Scored {
        psnr: psnr_rgb8(hr, out),
        ssim2: ssim2(hr, out),
        butter: if skip_butter {
            f64::NAN
        } else {
            butter_n3(hr, out)
        },
    }
}

pub fn read_f32_file(p: &std::path::Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Load an adopted model dir (models/adopted/<dir>) via its meta.json.
pub fn load_adopted(dir: &str) -> Option<zensr_micro::adopted::AdoptedModel> {
    use zensr_micro::adopted::AdoptedModel;
    let d = std::path::PathBuf::from("models/adopted").join(dir);
    let meta = std::fs::read_to_string(d.join("meta.json")).ok()?;
    let f = |k: &str| -> String {
        let pat = format!("\"{k}\":");
        match meta.find(&pat) {
            Some(i) => meta[i + pat.len()..]
                .trim_start()
                .trim_start_matches('"')
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect(),
            None => String::new(),
        }
    };
    let f16p = d.join("weights_f16.raw");
    let raw = if f16p.exists()
        && meta.contains("\"f16_goldens\"")
        && std::env::var("ZENSR_LOAD_F32").is_err()
    {
        zensr_micro::decode_all_f16(&std::fs::read(&f16p).ok()?)
    } else {
        read_f32_file(&d.join("weights.raw"))
    };
    let scale: usize = f("scale").parse().ok()?;
    let mut m = match f("arch").as_str() {
        "compact" => {
            AdoptedModel::load_compact(&raw, f("nf").parse().ok()?, f("nc").parse().ok()?, scale)
                .ok()?
        }
        "span48" => AdoptedModel::load_span48(&raw, scale).ok()?,
        _ => return None,
    };
    if f("space") == "ycbcr" {
        m.set_space(zensr_micro::adopted::ModelSpace::Ycbcr);
    }
    Some(m)
}

/// Real libjpeg-turbo round trip at quality q, 4:2:0 -optimize, via system cjpeg.
/// Scratch under ~/tmp (never /tmp — see global CLAUDE.md ban).
pub fn turbo_jpeg(img: &Rgb8Img, q: u32) -> Rgb8Img {
    let home = std::env::var("HOME").expect("HOME");
    let dir = std::path::PathBuf::from(home)
        .join("tmp")
        .join(format!("zensr-se-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let ppm = dir.join("t.ppm");
    let jpg = dir.join("t.jpg");
    let mut buf = format!("P6\n{} {}\n255\n", img.w, img.h).into_bytes();
    buf.extend_from_slice(&img.px);
    std::fs::write(&ppm, &buf).unwrap();
    let st = std::process::Command::new("cjpeg")
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
        .expect("cjpeg");
    assert!(st.success());
    decode_any(&jpg).expect("decode turbo jpeg")
}

pub fn run_guarded(
    m: &zensr_micro::adopted::AdoptedModel,
    lr: &Rgb8Img,
    threads: usize,
    guard: bool,
) -> Rgb8Img {
    use zensr_micro::guards::{guarded_merge, GuardConfig};
    let lp = to_planar_f32(lr);
    let mut sr = m.upscale_tiled(&lp, lr.h, lr.w, threads, 0);
    if guard {
        guarded_merge(&mut sr, &lp, lr.h, lr.w, m.scale, &GuardConfig::default());
    }
    planar_to_rgb8(&sr, lr.w * m.scale, lr.h * m.scale)
}

/// imazen-26 eval subcorpora: (label, directory).
///
/// The CANONICAL corpus layout (github.com/imazen/imazen-26, see
/// `imazen26_repo()`), repointed 2026-09-07/08. The previous list named the flat
/// directories of `/mnt/v/imazen-26`, which has been deleted — pointed at the
/// canonical corpus it resolved every entry to a missing directory and evaluated
/// zero images without saying so. See `docs/CORPUS-REPOINT-IMPACT.md`.
///
/// Several labels intentionally span more than one directory, because the
/// content-split curves are fit per LABEL and the canonical corpus splits some
/// classes finer than the curves need (`photos` is six directories).
/// `office-documents` has no entry: it did not survive curation.
pub const SUBCORPORA: &[(&str, &str)] = &[
    ("photos", "1000-lilith-photos-general"),
    ("photos", "1200-lilith-interiors"),
    ("photos", "1400-lilith-nature"),
    ("photos", "1600-lilith-food"),
    ("photos", "3000-art-institute-of-chicago-photos"),
    ("photos", "3300-met-museum-photos"),
    ("people", "2000-unsplash-people"),
    ("renders", "2200-unsplash-renders"),
    ("textures", "2400-unsplash-textures"),
    ("maps", "5000-national-park-service-brochures"),
    ("documents", "5200-epa-climate-impact-2021-report"),
    ("documents", "5300-noaa-hurricane-documents"),
    ("documents", "6800-ia-scans-manuscript-text"),
    ("art-scans", "6600-ia-scans-manuscript-illustrations"),
    ("patents", "6000-lilith-scans-public-patents"),
    ("plots", "7000-lilith-plots"),
    ("screen", "8000-lilith-mobile-screenshots"),
    ("screen", "8100-lilith-web-screenshots"),
    ("clipart", "9000-lilith-ai-clipart"),
    ("illustrations", "9094-lilith-ai-illustrations"),
    ("ai-products", "9226-lilith-ai-products"),
];

/// The subcorpora to evaluate, for whichever corpus is being pointed at.
///
/// A corpus may ship its own list as `SUBCORPORA.tsv` in its root
/// (`label<TAB>directory` per line, `#` comments ignored). Without one the
/// imazen-26 layout above is assumed, which is what every eval before
/// 2026-08-03 hardcoded.
///
/// This exists so a second corpus can be evaluated without editing and
/// rebuilding five binaries — the alternative was to keep the layout of one
/// corpus compiled into the tools, which is how the tools ended up only ever
/// being pointed at that corpus.
pub fn subcorpora_for(root: &Path) -> Vec<(String, String)> {
    let manifest = root.join("SUBCORPORA.tsv");
    if let Ok(text) = std::fs::read_to_string(&manifest) {
        let mut v = Vec::new();
        for line in text.lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let mut it = line.split('\t');
            if let (Some(label), Some(dir)) = (it.next(), it.next()) {
                v.push((label.trim().to_string(), dir.trim().to_string()));
            }
        }
        if !v.is_empty() {
            eprintln!("subcorpora: {} from {}", v.len(), manifest.display());
            return v;
        }
    }
    // Keep only entries that exist under this root, and refuse to return an
    // empty list. Falling back to a hardcoded layout that does not match the
    // corpus is precisely how an eval scores zero images and reports success.
    let present: Vec<(String, String)> = SUBCORPORA
        .iter()
        .filter(|(_, d)| root.join(d).is_dir())
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
    assert!(
        !present.is_empty(),
        "no subcorpora found under {}: it has no SUBCORPORA.tsv and none of the \
         canonical imazen-26 directories. Point it at the canonical corpus \
         (github.com/imazen/imazen-26, by default ~/work/imazen-26; override with \
         IMAZEN26_REPO) or give the corpus a SUBCORPORA.tsv.",
        root.display()
    );
    if present.len() < SUBCORPORA.len() {
        eprintln!(
            "subcorpora: {} of {} canonical directories present under {}",
            present.len(),
            SUBCORPORA.len(),
            root.display()
        );
    }
    present
}
