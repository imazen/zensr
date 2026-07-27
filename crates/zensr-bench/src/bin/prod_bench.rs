//! Production-runtime bench: the FULL restore_jpeg pipeline (probe -> policy
//! decode -> guarded x1 model -> S10 projection -> RGB) on real JPEG bytes,
//! size-swept per the sweep discipline (fit total = alpha + beta*MP; the
//! intercept matters at thumbnail sizes), 1 and 12 threads, plus the chained
//! x2 SR step for the dejpeg->SR pipeline cost.
//!
//! RAM: run the single largest case under /usr/bin/time -v (ZENSR_PB_ONE=4096
//! env) — peak RSS is the only reported memory number (no estimates).
//!
//! Usage: prod_bench [reps=3]   (or ZENSR_PB_ONE=<side> for the RSS run)

use std::process::Command;
use std::time::Instant;
use zensr_bench::*;
use zensr_zenjpeg::{restore_jpeg, RestoreConfig};

fn synth(n: usize) -> Rgb8Img {
    let mut px = vec![0u8; 3 * n * n];
    for y in 0..n {
        for x in 0..n {
            let i = (y * n + x) * 3;
            px[i] = ((x * 7 + y * 3) % 256) as u8;
            px[i + 1] = ((x * 2 + y * 11) % 256) as u8;
            px[i + 2] = ((x * 13 + (y * y) % 97) % 256) as u8;
        }
    }
    Rgb8Img { px, w: n, h: n }
}

fn encode_turbo(img: &Rgb8Img, q: u32) -> Vec<u8> {
    let d = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join("tmp");
    std::fs::create_dir_all(&d).unwrap();
    let ppm = d.join(format!("pb-{}.ppm", std::process::id()));
    let jpg = d.join(format!("pb-{}.jpg", std::process::id()));
    let mut buf = format!("P6\n{} {}\n255\n", img.w, img.h).into_bytes();
    buf.extend_from_slice(&img.px);
    std::fs::write(&ppm, buf).unwrap();
    assert!(Command::new("cjpeg")
        .args(["-quality", &q.to_string(), "-sample", "2x2", "-optimize", "-outfile"])
        .arg(&jpg)
        .arg(&ppm)
        .status()
        .unwrap()
        .success());
    std::fs::read(&jpg).unwrap()
}

fn main() {
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let model_name = std::env::var("ZENSR_PB_MODEL").unwrap_or_else(|_| "dejpeg4_policy".into());
    let model = load_adopted(&model_name).expect("model");
    let sr = load_adopted("nomosuni_span_2x");
    let one: Option<usize> = std::env::var("ZENSR_PB_ONE").ok().and_then(|s| s.parse().ok());
    let sizes: Vec<usize> = one.map(|n| vec![n]).unwrap_or_else(|| vec![64, 256, 1024, 2048, 4096]);

    println!("# prod_bench: restore_jpeg(dejpeg4_policy) on turbo q75 420 synth; sr=nomosuni_span_2x");
    println!("stage\tside\tmp\tthreads\tmin_ms\tmp_per_s");
    let mut fitpts: Vec<(f64, f64, usize)> = Vec::new(); // (mp, ms, threads)
    for &n in &sizes {
        let img = synth(n);
        let data = encode_turbo(&img, 75);
        let mp = (n * n) as f64 / 1e6;
        // big sizes: 1 rep; 4096 single-thread skipped (would add ~12 min for
        // a point the fit doesn't need — truncation noted per sweep discipline)
        let reps = if n <= 1024 { reps } else { 1 };
        for threads in [1usize, 12] {
            if one.is_some() && threads == 1 {
                continue; // RSS run: one config only
            }
            if n >= 4096 && threads == 1 {
                println!("# skipped: {n} at 1 thread (cost/benefit; fit uses 64..2048)");
                continue;
            }
            // decode-only reference (zenjpeg, pixel-exact)
            let mut dec_best = f64::INFINITY;
            for _ in 0..reps {
                let t = Instant::now();
                let r = zenjpeg::decoder::Decoder::new()
                    .decode(&data, enough::Unstoppable)
                    .unwrap();
                std::hint::black_box(r.pixels_u8());
                dec_best = dec_best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            println!("decode\t{n}\t{mp:.2}\t{threads}\t{dec_best:.1}\t{:.2}", mp / (dec_best / 1e3));
            // full restore
            let cfg = RestoreConfig::default().with_threads(threads);
            let mut best = f64::INFINITY;
            let mut restored = None;
            for _ in 0..reps {
                let t = Instant::now();
                let r = restore_jpeg(&data, &model, &cfg).unwrap();
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
                restored = Some(r);
            }
            println!("restore\t{n}\t{mp:.2}\t{threads}\t{best:.1}\t{:.2}", mp / (best / 1e3));
            fitpts.push((mp, best, threads));
            // chained x2 SR on the restored planes
            let no_sr = std::env::var("ZENSR_PB_NOSR").is_ok();
            if no_sr {
                continue;
            }
            if let (Some(srm), Some(r)) = (&sr, &restored) {
                let mut sbest = f64::INFINITY;
                for _ in 0..reps {
                    let t = Instant::now();
                    let up = srm.upscale_tiled(&r.planes, r.height, r.width, threads, 0);
                    std::hint::black_box(&up);
                    sbest = sbest.min(t.elapsed().as_secs_f64() * 1e3);
                }
                println!("sr_x2\t{n}\t{mp:.2}\t{threads}\t{sbest:.1}\t{:.2}", mp / (sbest / 1e3));
                println!(
                    "chain\t{n}\t{mp:.2}\t{threads}\t{:.1}\t{:.2}",
                    best + sbest,
                    mp / ((best + sbest) / 1e3)
                );
            }
        }
    }
    // least-squares fit total = alpha + beta*mp per thread count (restore stage)
    if one.is_none() {
        for threads in [1usize, 12] {
            let pts: Vec<_> = fitpts.iter().filter(|p| p.2 == threads).collect();
            let n = pts.len() as f64;
            let (sx, sy): (f64, f64) = (pts.iter().map(|p| p.0).sum(), pts.iter().map(|p| p.1).sum());
            let sxx: f64 = pts.iter().map(|p| p.0 * p.0).sum();
            let sxy: f64 = pts.iter().map(|p| p.0 * p.1).sum();
            let beta = (n * sxy - sx * sy) / (n * sxx - sx * sx);
            let alpha = (sy - beta * sx) / n;
            println!("# restore fit t={threads}: total_ms = {alpha:.1} + {beta:.1} * MP");
        }
    }
}
