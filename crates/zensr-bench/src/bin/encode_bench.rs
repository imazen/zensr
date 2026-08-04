//! Encode throughput: in-process zenjpeg vs the `cjpeg` subprocess.
//!
//! The XL classification sweep shells out to `cjpeg` for every cell, which pays
//! process startup, a PPM write, and two file reads per encode — on a grid of
//! 913 images x 29 qualities x 2 subsamplings that is 53k process spawns per
//! encoder. zenjpeg is a library, so the same work can run in-process on
//! in-memory buffers with none of that.
//!
//! For the *classification* question — does a cheap feature separate images
//! within a quality cell — the encoder only has to produce representative JPEG
//! damage. It does not have to be libjpeg-turbo. So if in-process is markedly
//! faster, the sweep should use it and the saved hours buy more images, which
//! is the axis the whole exercise is short on.
//!
//! Configs measured, fastest-first: Baseline (sequential, single scan) without
//! Huffman optimization is the cheapest thing the encoder can do; progressive
//! plus optimized Huffman is the default and costs a second entropy pass.
//!
//! Usage: encode_bench <corpus-root> [images=40] [threads=1]

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, ProgressiveScanMode};
use zensr_bench::*;

const QS: &[f32] = &[15.0, 35.0, 55.0, 75.0, 85.0, 90.0, 94.0, 98.0];

fn main() {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(
        args.next()
            .expect("usage: encode_bench <root> [n] [threads]"),
    );
    let n: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(40);
    let threads: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(1);

    // Same geometry the sweep uses, so the numbers transfer directly.
    let mut imgs = Vec::new();
    for (_, dir) in &subcorpora_for(&root) {
        for f in list_images(&root.join(dir)) {
            if imgs.len() >= n {
                break;
            }
            if let Some(i) = decode_any(&f).and_then(|i| center_crop(&i, 512)) {
                imgs.push(i);
            }
        }
        if imgs.len() >= n {
            break;
        }
    }
    let cells = imgs.len() * QS.len();
    eprintln!(
        "{} images x {} qualities = {cells} encodes per config, {threads} thread(s)",
        imgs.len(),
        QS.len()
    );

    let td = PathBuf::from(std::env::var("HOME").unwrap()).join("tmp");
    let _ = std::fs::create_dir_all(&td);
    let ppm = td.join("encbench.ppm");
    let jpg = td.join("encbench.jpg");

    // Subprocess baseline: what the sweep does today. The PPM write is part of
    // the cost — a subprocess cannot take an in-memory buffer.
    let t = Instant::now();
    let mut sub_bytes = 0usize;
    for im in &imgs {
        let mut v = format!("P6\n{} {}\n255\n", im.w, im.h).into_bytes();
        v.extend_from_slice(&im.px);
        std::fs::write(&ppm, &v).unwrap();
        for q in QS {
            let ok = Command::new("cjpeg")
                .args([
                    "-quality",
                    &(*q as u32).to_string(),
                    "-sample",
                    "2x2",
                    "-outfile",
                ])
                .arg(&jpg)
                .arg(&ppm)
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if ok {
                sub_bytes += std::fs::metadata(&jpg)
                    .map(|m| m.len() as usize)
                    .unwrap_or(0);
            }
        }
    }
    let sub_s = t.elapsed().as_secs_f64();
    report(
        "cjpeg subprocess (what the sweep does)",
        cells,
        sub_s,
        sub_bytes,
    );

    for (label, mk) in [
        (
            "zenjpeg sequential, no huffman opt",
            (|q: f32| {
                EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
                    .scan_mode(ProgressiveScanMode::Baseline)
                    .optimize_huffman(false)
            }) as fn(f32) -> EncoderConfig,
        ),
        ("zenjpeg sequential, huffman opt", |q: f32| {
            EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
                .scan_mode(ProgressiveScanMode::Baseline)
                .optimize_huffman(true)
        }),
        ("zenjpeg progressive (config default)", |q: f32| {
            EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
        }),
    ] {
        let t = Instant::now();
        let bytes: usize = run_inproc(&imgs, threads, &mk);
        report(label, cells, t.elapsed().as_secs_f64(), bytes);
    }
}

fn run_inproc(
    imgs: &[Rgb8Img],
    threads: usize,
    mk: &(dyn Fn(f32) -> EncoderConfig + Sync),
) -> usize {
    let work = |chunk: &[Rgb8Img]| -> usize {
        let mut b = 0usize;
        for im in chunk {
            let px: &[rgb::Rgb<u8>] = bytemuck::cast_slice(&im.px);
            for q in QS {
                // An encode that errors must not look like a fast encode. The
                // first version of this bench swallowed the error and reported
                // 26 million enc/s on a config that produced nothing.
                match mk(*q).encode(px, im.w as u32, im.h as u32) {
                    Ok(v) => b += v.len(),
                    Err(e) => panic!("encode failed at q{q}: {e:?}"),
                }
            }
        }
        b
    };
    if threads <= 1 {
        return work(imgs);
    }
    // Multi-stream: one image per worker. Each encode is independent, so this
    // scales with cores without touching the encoder's own threading.
    std::thread::scope(|s| {
        let chunk = imgs.len().div_ceil(threads);
        let hs: Vec<_> = imgs
            .chunks(chunk)
            .map(|c| s.spawn(move || work(c)))
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).sum()
    })
}

fn report(label: &str, cells: usize, secs: f64, bytes: usize) {
    println!(
        "{label:<40} {secs:7.2}s  {:8.1} enc/s  {:6.2} ms/enc  {:.0} KB total",
        cells as f64 / secs,
        secs * 1000.0 / cells as f64,
        bytes as f64 / 1024.0
    );
}
