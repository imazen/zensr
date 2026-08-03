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

/// Zigzag index at which "high frequency" starts. 32 splits the 64 coefficients
/// in half by zigzag order, which is coarse but needs no tuning — the point is
/// to separate detail from structure, not to find an optimal cut.
const HF_START: usize = 32;

struct Feats {
    zero_ac_blocks: f64,
    zero_ac_coefs: f64,
    hf_survival: f64,
    mean_abs_ac: f64,
    dc_spread: f64,
    // Spatial statistics of damage. The six scalar features above all say how
    // MUCH was quantised away; none says WHERE, and a uniformly-degraded image
    // and one with a few destroyed regions score identically under all of them.
    // These are the cheapest way to ask that question.
    /// Stdev across blocks of the per-block surviving-AC count. High when
    /// damage is concentrated, low when it is spread evenly.
    ac_count_spread: f64,
    /// Share of blocks whose surviving-AC count is below a quarter of the
    /// image's own mean — locally destroyed regions, measured relative to the
    /// image so it does not just re-measure global quality.
    gutted_block_frac: f64,
    /// Share of gutted blocks that neighbour another gutted block
    /// (4-connected). Separates scattered loss from contiguous flattened areas,
    /// which is the distinction a spatial signal has to make to be new.
    gutted_clustering: f64,
}

fn features(coeffs: &[i16], nblocks: usize, blocks_wide: usize) -> Option<Feats> {
    if nblocks == 0 {
        return None;
    }
    // Per-block surviving-AC count, kept in raster order so neighbours can be
    // found for the clustering measure.
    let accounts: Vec<f64> = coeffs[..nblocks * 64]
        .chunks_exact(64)
        .map(|b| b[1..].iter().filter(|&&c| c != 0).count() as f64)
        .collect();
    let (mut zblk, mut zcoef, mut hf, mut lf, mut absac, mut nac) =
        (0u64, 0u64, 0f64, 0f64, 0f64, 0u64);
    let mut dcs: Vec<f64> = Vec::with_capacity(nblocks);
    for b in coeffs[..nblocks * 64].chunks_exact(64) {
        dcs.push(b[0] as f64);
        if b[1..].iter().all(|&c| c == 0) {
            zblk += 1;
        }
        for (i, &c) in b.iter().enumerate().skip(1) {
            if c == 0 {
                zcoef += 1;
            } else {
                let e = (c as f64) * (c as f64);
                if i >= HF_START {
                    hf += e;
                } else {
                    lf += e;
                }
                absac += (c as f64).abs();
                nac += 1;
            }
        }
    }
    let n = nblocks as f64;
    let mean_dc = dcs.iter().sum::<f64>() / n;
    let var_dc = dcs.iter().map(|d| (d - mean_dc).powi(2)).sum::<f64>() / n;

    let mean_ac = accounts.iter().sum::<f64>() / n;
    let var_ac = accounts.iter().map(|a| (a - mean_ac).powi(2)).sum::<f64>() / n;
    let cut = mean_ac * 0.25;
    let gutted: Vec<bool> = accounts.iter().map(|&a| a <= cut).collect();
    let ngut = gutted.iter().filter(|&&g| g).count();
    let bw = blocks_wide.max(1);
    let mut clustered = 0usize;
    for (i, &g) in gutted.iter().enumerate() {
        if !g {
            continue;
        }
        let (x, y) = (i % bw, i / bw);
        let neigh = [
            (x > 0).then(|| i - 1),
            (x + 1 < bw).then(|| i + 1),
            (y > 0).then(|| i.wrapping_sub(bw)),
            (i + bw < gutted.len()).then(|| i + bw),
        ];
        if neigh.into_iter().flatten().any(|j| gutted[j]) {
            clustered += 1;
        }
    }

    Some(Feats {
        zero_ac_blocks: zblk as f64 / n,
        zero_ac_coefs: zcoef as f64 / (n * 63.0),
        hf_survival: if hf + lf > 0.0 { hf / (hf + lf) } else { 0.0 },
        mean_abs_ac: if nac > 0 { absac / nac as f64 } else { 0.0 },
        dc_spread: var_dc.sqrt(),
        ac_count_spread: var_ac.sqrt(),
        gutted_block_frac: ngut as f64 / n,
        gutted_clustering: if ngut > 0 {
            clustered as f64 / ngut as f64
        } else {
            0.0
        },
    })
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

    let pinned = load_pinned(&pin_path());
    let td = tmpdir();
    let ppm = td.join("cfeat.ppm");
    let jpg = td.join("cfeat.jpg");

    let mut tsv = String::new();
    let _ = writeln!(tsv, "# cheap coefficient-domain features vs restore gain");
    let _ = writeln!(
        tsv,
        "sub\tfile\tencoder\tss\tq\tbytes\tbpp\tzero_ac_blocks\tzero_ac_coefs\thf_survival\tmean_abs_ac\tdc_spread\tac_count_spread\tgutted_block_frac\tgutted_clustering"
    );

    for (name, dir) in SUBCORPORA {
        let want = pinned.as_ref().and_then(|m| m.get(*dir));
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
            let Some(hr) = center_crop(&img, 512) else {
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
                        let Some(ft) = features(&luma.coeffs, nb, luma.blocks_wide as usize) else {
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
