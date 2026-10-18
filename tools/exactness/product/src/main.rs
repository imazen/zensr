//! Restore JPEGs through `zensr_zenjpeg::restore_jpeg` — the shipped x1
//! pipeline: guarded model, quantisation-consistency projection, RGB8 — at
//! several thread counts with the default automatic tile, and write each
//! final RGB8 buffer so a driver can compare them byte for byte.
//!
//! Usage: zensr-exactness-product <weights_f16.raw> <nf> <nc> <threads,..> <out-dir> <jpeg>...
//! Writes <out-dir>/<jpeg-stem>-t<threads>.rgb8 (refuses to overwrite).
//! ZENSR_EXACT_TIER=auto (default) | v3 | scalar forces the dispatch tier for
//! the whole process before any model work; a tier that cannot be forced is an
//! error, never a silent fallback.
#![forbid(unsafe_code)]

use std::io::Write;
use zensr_zenjpeg::RestoreConfig;

fn force_tier() {
    let tier = std::env::var("ZENSR_EXACT_TIER").unwrap_or_else(|_| "auto".into());
    #[cfg(target_arch = "x86_64")]
    {
        let off = |r: Result<(), archmage::CompileTimeGuaranteedError>, what: &str| {
            r.unwrap_or_else(|e| {
                panic!("cannot force ZENSR_EXACT_TIER={tier}: disabling {what}: {e:?}")
            })
        };
        match tier.as_str() {
            "auto" => {}
            "v3" => {
                off(
                    archmage::X64V4xToken::dangerously_disable_token_process_wide(true),
                    "v4x",
                );
                off(
                    archmage::X64V4Token::dangerously_disable_token_process_wide(true),
                    "v4",
                );
            }
            "scalar" => {
                off(
                    archmage::X64V4xToken::dangerously_disable_token_process_wide(true),
                    "v4x",
                );
                off(
                    archmage::X64V4Token::dangerously_disable_token_process_wide(true),
                    "v4",
                );
                off(
                    archmage::X64V3Token::dangerously_disable_token_process_wide(true),
                    "v3",
                );
            }
            other => panic!("unknown ZENSR_EXACT_TIER={other}"),
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    assert_eq!(tier, "auto", "tier forcing is implemented for x86_64 only");
    eprintln!("tier={tier}");
}

fn main() {
    force_tier();
    let a: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        a.len() >= 6,
        "usage: <weights_f16.raw> <nf> <nc> <threads,..> <out-dir> <jpeg>..."
    );
    let raw = zensr_micro::decode_all_f16(&std::fs::read(&a[0]).expect("read weights"));
    let nf: usize = a[1].parse().expect("nf");
    let nc: usize = a[2].parse().expect("nc");
    let model =
        zensr_micro::adopted::AdoptedModel::load_compact(&raw, nf, nc, 1).expect("load model");
    let threads: Vec<usize> = a[3]
        .split(',')
        .map(|t| t.parse().expect("threads"))
        .collect();
    let out_dir = std::path::Path::new(&a[4]);
    for jpeg_path in &a[5..] {
        let jpeg = std::fs::read(jpeg_path).expect("read jpeg");
        let stem = std::path::Path::new(jpeg_path)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        for &t in &threads {
            let cfg = RestoreConfig::default().with_threads(t);
            let r = zensr_zenjpeg::restore_jpeg(&jpeg, &model, &cfg).expect("restore");
            assert!(!r.report.skipped_model_high_q, "{stem}: the model must run");
            let path = out_dir.join(format!("{stem}-t{t}.rgb8"));
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .expect("create output");
            f.write_all(&r.to_rgb8()).expect("write output");
            println!(
                "{stem} t={t} {}x{} est_q={} -> {}",
                r.width,
                r.height,
                r.report.est_quality,
                path.display()
            );
        }
    }
}
