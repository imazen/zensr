//! Tile-size sweep for the tiled model runner.
//!
//! `upscale_tiled` defaults to 128px tiles, and each tile recomputes a `halo`
//! border it then discards: for the realtime tier (halo = nc + 2 = 10) that is
//! (128+20)^2 / 128^2 = 1.34, i.e. 34% wasted work. Bigger tiles amortise the
//! halo — 256px is 1.16, 512px is 1.08 — but were previously exposed to the
//! conv3x3 TLB cliff (see benchmarks/realtime_kernels_x86_2026-09-08.md), which
//! channel blocking removed. This measures whether that trade now pays.
//!
//! Usage: tile_probe [model=dejpeg_rt24g] [size=1024] [threads=12] [reps=3]
//!
//! `size` is `N` for a square image or `WxH` for a rectangle. Non-square is not
//! a curiosity: the tile is SQUARE, so it cannot divide both axes evenly, and a
//! rule tuned only on squares can be choosing a tile that leaves a runt on the
//! short axis without ever showing it.
use std::time::Instant;
use zensr_bench::*;

fn main() {
    let mut a = std::env::args().skip(1);
    let model = a.next().unwrap_or_else(|| "dejpeg_rt24g".into());
    let size = a.next().unwrap_or_else(|| "1024".into());
    let (w, h) = match size.split_once('x') {
        Some((a, b)) => (
            a.parse().expect("width"),
            b.parse::<usize>().expect("height"),
        ),
        None => {
            let n: usize = size.parse().expect("size");
            (n, n)
        }
    };
    let side = w.max(h);
    let threads: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(12);
    let reps: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(3);
    let m = load_adopted(&model).expect("model");
    let halo = m.halo();

    // Deterministic planar RGB input in [0,1].
    let mut s = 12345u32;
    let planes: Vec<f32> = (0..3 * w * h)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / 16_777_216.0
        })
        .collect();

    println!("model {model} halo {halo}, {w}x{h}px, {threads} threads, {reps} reps");
    println!(
        "{:>6} {:>10} {:>9} {:>6} {:>8}",
        "tile", "median ms", "halo cost", "tail", "checksum"
    );
    // Aligned candidates alongside the round numbers: when tile + 2*halo is a
    // multiple of the vector width the kernel needs no overlapping tail tile,
    // which otherwise recomputes (W - width%W) columns per row.
    // `ZENSR_TP_TILES=a,b,c` replaces the sweep with an explicit list.
    //
    // This exists because comparing two BUILDS of the tile rule does not work:
    // editing the rule changes code layout and inlining, and two builds that
    // choose the identical tile were measured 6% apart on it (1024px, 12
    // threads, tile 140: 109.2 vs 115.9 ms). Every delta from such an A/B
    // carries that bias. The sound comparison is both rules' CHOSEN TILES
    // measured inside ONE binary, which is what this flag is for.
    let tiles: Vec<usize> = match std::env::var("ZENSR_TP_TILES") {
        Ok(v) => v
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.trim()
                    .parse()
                    .expect("ZENSR_TP_TILES: comma-separated integers")
            })
            .collect(),
        Err(_) => vec![0, 76, 108, 140, 172, 268, 396, 524],
    };
    let explicit = std::env::var("ZENSR_TP_TILES").is_ok();
    for &tile in &tiles {
        // A tile larger than the image is not nonsense — it is one tile, which
        // is what the shipped default does on a small image, so an explicitly
        // requested one must be measurable. Only the default sweep skips them,
        // to keep its table from repeating the same row.
        if tile > side && !explicit {
            continue;
        }
        let mut ts = Vec::new();
        let mut sum = 0.0f64;
        for _ in 0..reps {
            let t = Instant::now();
            let out = m.upscale_tiled(&planes, h, w, threads, tile);
            ts.push(t.elapsed().as_secs_f64() * 1e3);
            sum = out.iter().take(4096).map(|&v| v as f64).sum();
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // The default row has no tile size of its own to report a halo cost
        // for; print it as `dflt` and leave the derived columns blank rather
        // than computing them from 0 and printing nonsense.
        let label = if tile == 0 {
            "  dflt".to_string()
        } else {
            format!("{tile:>6}")
        };
        let overhead = if tile == 0 {
            f64::NAN
        } else {
            ((tile + 2 * halo) as f64 / tile as f64).powi(2)
        };
        let tail = if tile == 0 { 0 } else { (tile + 2 * halo) % 16 };
        println!(
            "{label} {:>10.1} {:>8.3}x {:>6} {:>8.3}",
            ts[ts.len() / 2],
            overhead,
            if tail == 0 {
                "-".to_string()
            } else {
                format!("{tail}")
            },
            sum
        );
    }
}
