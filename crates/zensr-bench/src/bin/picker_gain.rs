//! Restore gain over pre-encoded JPEGs, reusing the canonical picker metrics.
//!
//! The canonical picker datasets already hold, for every cell, the encoded JPEG
//! and `score_ssim2` — which is exactly the `identity_off` arm every ladder here
//! recomputes. Re-encoding and re-scoring it is redundant work: measured, an
//! identity arm costs ~36 ms of metrics against ~2 ms for the encode it
//! replaces.
//!
//! So this does only the part nobody precomputed: run the model on the stored
//! variant and score the result. Gain is then
//! `ssim2(ref, restored) - score_ssim2`.
//!
//! **It verifies the borrowed number instead of trusting it.** The first thing
//! each cell does is recompute `ssim2(ref, encoded)` and compare against the
//! stored value. If the stored score were against a different reference — a
//! different crop, a different colour handling — every gain derived from it
//! would be silently wrong, and the whole point of reusing it would be lost.
//! Cells whose identity disagrees by more than `--tol` are reported and
//! excluded rather than quietly averaged in.
//!
//! Input is JSONL, one cell per line, with `ref_filename`, `q`, `score_ssim2`,
//! `variant_r2_url`. Variants are read from `--variants` by basename.
//!
//! Usage: picker_gain <cells.jsonl> <refs-dir> <variants-dir> <out.tsv> [model] [threads]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use zensr_bench::*;

fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("\"{key}\":");
    let i = line.find(&pat)? + pat.len();
    let rest = line[i..].trim_start();
    if let Some(r) = rest.strip_prefix('"') {
        r.find('"').map(|j| &r[..j])
    } else {
        let j = rest.find([',', '}']).unwrap_or(rest.len());
        Some(rest[..j].trim())
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let cells = PathBuf::from(args.next().expect("cells.jsonl"));
    let refs = PathBuf::from(args.next().expect("refs dir"));
    let vars = PathBuf::from(args.next().expect("variants dir"));
    let out = PathBuf::from(args.next().expect("out tsv"));
    let model_dir = args.next().unwrap_or_else(|| "dejpeg_rt24g".into());
    let threads: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(8);
    let tol: f64 = std::env::var("ZENSR_IDENTITY_TOL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.5);

    let model = load_adopted(&model_dir).expect("model");
    let text = std::fs::read_to_string(&cells).expect("cells");

    let mut tsv = String::from(
        "ref\tq\twidth\theight\tstored_identity\trecomputed_identity\tidentity_delta\trestored\tgain\n",
    );
    let (mut n, mut skipped, mut mismatch) = (0usize, 0usize, 0usize);
    let mut worst = 0.0f64;

    for line in text.lines() {
        let (Some(rf), Some(q), Some(sc), Some(url)) = (
            field(line, "ref_filename"),
            field(line, "q"),
            field(line, "score_ssim2"),
            field(line, "variant_r2_url"),
        ) else {
            continue;
        };
        let vpath = vars.join(Path::new(url).file_name().unwrap());
        if !vpath.exists() {
            skipped += 1;
            continue;
        }
        let Some(reference) = decode_any(&refs.join(rf)) else {
            skipped += 1;
            continue;
        };
        let Ok(jpg) = std::fs::read(&vpath) else {
            skipped += 1;
            continue;
        };
        let Ok(dec) = zenjpeg::decoder::Decoder::new().decode(&jpg, enough::Unstoppable) else {
            skipped += 1;
            continue;
        };
        let (w, h) = dec.dimensions();
        let Some(px) = dec.pixels_u8() else {
            skipped += 1;
            continue;
        };
        let decoded = Rgb8Img {
            px: px.to_vec(),
            w: w as usize,
            h: h as usize,
        };
        if decoded.w != reference.w || decoded.h != reference.h {
            skipped += 1;
            continue;
        }

        // Verify the borrowed identity score before using it.
        let stored: f64 = sc.parse().unwrap_or(f64::NAN);
        let recomputed = ssim2(&reference, &decoded);
        let delta = recomputed - stored;
        if delta.abs() > tol {
            mismatch += 1;
            worst = worst.max(delta.abs());
            continue;
        }

        let lp = to_planar_f32(&decoded);
        let mut sr = model.upscale_tiled(&lp, decoded.h, decoded.w, threads, 0);
        zensr_micro::guards::guarded_merge(
            &mut sr,
            &lp,
            decoded.h,
            decoded.w,
            1,
            &zensr_micro::guards::GuardConfig::default(),
        );
        let restored = planar_to_rgb8(&sr, decoded.w, decoded.h);
        let rs = ssim2(&reference, &restored);

        let _ = writeln!(
            tsv,
            "{rf}\t{q}\t{}\t{}\t{stored:.4}\t{recomputed:.4}\t{delta:+.4}\t{rs:.4}\t{:+.4}",
            decoded.w,
            decoded.h,
            rs - stored
        );
        n += 1;
        if n % 500 == 0 {
            eprintln!("{n} cells");
        }
    }
    std::fs::write(&out, &tsv).expect("write");
    eprintln!(
        "wrote {} — {n} cells, {skipped} skipped, {mismatch} identity mismatches (worst {worst:.3}, tol {tol})",
        out.display()
    );
    if mismatch > n / 20 {
        eprintln!(
            "WARNING: {mismatch} mismatches is over 5% — the stored identity arm may be \
             against a different reference than {}, which would invalidate every gain here",
            refs.display()
        );
    }
}
