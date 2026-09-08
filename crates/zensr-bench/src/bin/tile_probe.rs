//! Tile-size sweep for the tiled model runner.
//!
//! `upscale_tiled` defaults to 128px tiles, and each tile recomputes a `halo`
//! border it then discards: for the realtime tier (halo = nc + 2 = 10) that is
//! (128+20)^2 / 128^2 = 1.34, i.e. 34% wasted work. Bigger tiles amortise the
//! halo — 256px is 1.16, 512px is 1.08 — but were previously exposed to the
//! conv3x3 TLB cliff (see benchmarks/realtime_kernels_x86_2026-09-08.md), which
//! channel blocking removed. This measures whether that trade now pays.
//!
//! Usage: tile_probe [model=dejpeg_rt24g] [side=1024] [threads=12] [reps=3]
use std::time::Instant;
use zensr_bench::*;

fn main() {
    let mut a = std::env::args().skip(1);
    let model = a.next().unwrap_or_else(|| "dejpeg_rt24g".into());
    let side: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let threads: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(12);
    let reps: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(3);
    let m = load_adopted(&model).expect("model");
    let halo = m.halo();

    // Deterministic planar RGB input in [0,1].
    let mut s = 12345u32;
    let planes: Vec<f32> = (0..3 * side * side)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / 16_777_216.0
        })
        .collect();

    println!("model {model} halo {halo}, {side}px, {threads} threads, {reps} reps");
    println!(
        "{:>6} {:>10} {:>9} {:>6} {:>8}",
        "tile", "median ms", "halo cost", "tail", "checksum"
    );
    // Aligned candidates alongside the round numbers: when tile + 2*halo is a
    // multiple of the vector width the kernel needs no overlapping tail tile,
    // which otherwise recomputes (W - width%W) columns per row.
    for &tile in &[76usize, 108, 140, 172, 268, 396, 524] {
        if tile > side {
            continue;
        }
        let mut ts = Vec::new();
        let mut sum = 0.0f64;
        for _ in 0..reps {
            let t = Instant::now();
            let out = m.upscale_tiled(&planes, side, side, threads, tile);
            ts.push(t.elapsed().as_secs_f64() * 1e3);
            sum = out.iter().take(4096).map(|&v| v as f64).sum();
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let overhead = ((tile + 2 * halo) as f64 / tile as f64).powi(2);
        let tail = (tile + 2 * halo) % 16;
        println!(
            "{tile:>6} {:>10.1} {:>8.3}x {:>6} {:>8.3}",
            ts[ts.len() / 2],
            overhead,
            if tail == 0 { "-".to_string() } else { format!("{tail}") },
            sum
        );
    }
}
