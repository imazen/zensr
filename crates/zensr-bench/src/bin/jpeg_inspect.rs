//! What is this JPEG, and what would the restoration pipeline do with it?
//!
//! Prints the probe's view of each file — encoder family, estimated quality on
//! that family's own scale, chroma subsampling, coding mode — and then the
//! policy decisions that follow from it: whether the high-q identity gate
//! fires (which since 2026-08-02 depends on subsampling, not quality alone),
//! whether the deblock policy wants Knusperli, and the projection slack the
//! family calibration would use.
//!
//! Usage: jpeg_inspect <dir-or-file>...

use std::path::{Path, PathBuf};
use zensr_bench::load_adopted;

fn collect(p: &Path, out: &mut Vec<PathBuf>) {
    if p.is_dir() {
        let mut e: Vec<_> = std::fs::read_dir(p)
            .into_iter()
            .flatten()
            .flatten()
            .map(|d| d.path())
            .collect();
        e.sort();
        for c in e {
            collect(&c, out);
        }
    } else if p.is_file() {
        out.push(p.to_path_buf());
    }
}

/// JPEG files start FF D8 FF; anything else here is reported, not guessed at.
fn is_jpeg(b: &[u8]) -> bool {
    b.len() > 3 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF
}

fn sha16(b: &[u8]) -> String {
    // Tiny FNV-1a; enough to spot byte-identical duplicates in a listing.
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: jpeg_inspect <dir-or-file>...");
        std::process::exit(2);
    }
    let mut files = Vec::new();
    for a in &args {
        collect(Path::new(a), &mut files);
    }

    println!(
        "file\tbytes\tw\th\tMP\tbpp\tencoder\tquality\tscale\tsubsampling\tmode\t\
         gate_identity\tdeblock\tslack_q\tslack_abs\tfnv16"
    );
    let (mut n_jpeg, mut n_other, mut n_gated) = (0usize, 0usize, 0usize);
    for f in &files {
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        let Ok(bytes) = std::fs::read(f) else {
            println!("{name}\tUNREADABLE");
            continue;
        };
        if !is_jpeg(&bytes) {
            n_other += 1;
            let kind = if bytes.len() > 12 && &bytes[4..8] == b"ftyp" {
                format!(
                    "not-jpeg (ISOBMFF/{})",
                    String::from_utf8_lossy(&bytes[8..12])
                )
            } else {
                "not-jpeg".to_string()
            };
            println!("{name}\t{}\t-\t-\t-\t-\t{kind}", bytes.len());
            continue;
        }
        n_jpeg += 1;
        let p = match zenjpeg::detect::probe(&bytes) {
            Ok(p) => p,
            Err(e) => {
                println!("{name}\t{}\tPROBE FAILED: {e:?}", bytes.len());
                continue;
            }
        };
        let (w, h) = (p.dimensions.width as usize, p.dimensions.height as usize);
        let mp = (w * h) as f64 / 1e6;
        let bpp = if w * h > 0 {
            bytes.len() as f64 * 8.0 / (w * h) as f64
        } else {
            0.0
        };
        let gate = zensr_zenjpeg::policy_high_q_identity(&p);
        if gate {
            n_gated += 1;
        }
        println!(
            "{name}\t{}\t{w}\t{h}\t{mp:.2}\t{bpp:.3}\t{:?}\t{:.1}\t{:?}\t{:?}\t{:?}\t{}\t{}\t{:.2}\t{:.1}\t{}",
            bytes.len(),
            p.encoder,
            p.quality.value,
            p.quality.scale,
            p.subsampling,
            p.mode,
            if gate { "SKIP (near-pristine)" } else { "restore" },
            if zensr_zenjpeg::policy_wants_auto(&p) { "Auto(Knusperli)" } else { "Off" },
            zensr_zenjpeg::slack_for(&p),
            zensr_zenjpeg::slack_abs_for(&p),
            sha16(&bytes),
        );
    }
    eprintln!(
        "\n{} JPEG(s), {} non-JPEG, {} would skip the model at the identity gate",
        n_jpeg, n_other, n_gated
    );

    // --restore: run the real pipeline and report how far it actually moves
    // the pixels. Without a pristine original no absolute quality is knowable,
    // but "how much does restoration change this file" is measurable and is
    // what decides whether these inputs are worth processing at all.
    if std::env::args().any(|a| a == "--restore") {
        let Some(model) = load_adopted("dejpeg_rt24g") else {
            eprintln!("--restore needs models/adopted/dejpeg_rt24g");
            return;
        };
        println!("\n# --restore: change vs the plain decode, dejpeg_rt24g + projection");
        println!("file\tmean_abs_delta\tmax_abs_delta\tpx_changed_pct\tskipped_high_q");
        for f in &files {
            let Ok(bytes) = std::fs::read(f) else {
                continue;
            };
            if !is_jpeg(&bytes) {
                continue;
            }
            let name = f.file_name().unwrap().to_string_lossy().to_string();
            let cfg = zensr_zenjpeg::RestoreConfig::default().with_threads(8);
            let Ok(r) = zensr_zenjpeg::restore_jpeg(&bytes, &model, &cfg) else {
                println!("{name}\tRESTORE FAILED");
                continue;
            };
            let out = r.to_rgb8();
            let Ok(dec) = zenjpeg::decoder::Decoder::new().decode(&bytes, enough::Unstoppable)
            else {
                continue;
            };
            let Some(base) = dec.pixels_u8() else {
                continue;
            };
            let (mut sum, mut mx, mut ch) = (0u64, 0u8, 0usize);
            for (a, b) in out.iter().zip(base.iter()) {
                let d = a.abs_diff(*b);
                sum += d as u64;
                mx = mx.max(d);
                if d > 0 {
                    ch += 1;
                }
            }
            let n = out.len().max(1);
            println!(
                "{name}\t{:.3}\t{mx}\t{:.1}\t{}",
                sum as f64 / n as f64,
                100.0 * ch as f64 / n as f64,
                r.report.skipped_model_high_q
            );
        }
    }
}
