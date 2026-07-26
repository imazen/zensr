//! Calibrate the S10 projection slack per encoder family.
//!
//! For each encoder x q: encode eval crops, decode coefficients via zenjpeg,
//! compute the TRUE luma DCT coefficients from the pre-encode original, and
//! measure excess = (|c_true - c_hat| - Q/2) / Q. Round-to-nearest encoders
//! should show excess <= ~0 (+ tiny fdct implementation skew); trellis
//! quantizers (mozjpeg) deliberately exceed it. p99/max set ProjectionConfig
//! slack per family.
//!
//! Luma-only (chroma of subsampled files has no per-block ground truth
//! without replicating each encoder's downsample). 444 encodes included so
//! chroma-capable slack can reuse the luma numbers.
//!
//! Usage: slack_probe <imazen26-root> [per-sub=2]

use std::path::PathBuf;
use std::process::Command;
use zensr_bench::*;
use zensr_micro::consist::{rgb_to_ycbcr_planes, ZIGZAG_TO_NATURAL};

const QS: &[u32] = &[8, 35, 75, 92];
const ENCODERS: &[&str] = &["turbo", "mozjpeg", "jpegli", "zenjpeg"];

fn encode(ppm: &PathBuf, jpg: &PathBuf, enc: &str, q: u32) -> bool {
    let home = std::env::var("HOME").unwrap();
    let st = match enc {
        "turbo" => Command::new("cjpeg")
            .args(["-quality", &q.to_string(), "-sample", "1x1", "-optimize", "-outfile"])
            .arg(jpg).arg(ppm).status(),
        "mozjpeg" => Command::new(format!("{home}/tmp/ati-bin/mozjpeg-cjpeg"))
            .env("LD_LIBRARY_PATH", format!("{home}/tmp/ati-bin/mozjpeg-lib64"))
            .args(["-quality", &q.to_string(), "-sample", "1x1", "-outfile"])
            .arg(jpg).arg(ppm).status(),
        "jpegli" => Command::new("cjpegli")
            .arg(ppm).arg(jpg)
            .args(["-q", &q.to_string(), "--chroma_subsampling=444"]).status(),
        _ => {
            let me = std::env::current_exe().unwrap();
            Command::new(me.parent().unwrap().join("zjtool"))
                .arg("enc").arg(ppm).arg(jpg).args([&q.to_string(), "444"]).status()
        }
    };
    st.map(|s| s.success()).unwrap_or(false)
}

fn fdct_luma(y: &[f32], w: usize, h: usize, bx: usize, by: usize) -> [f32; 64] {
    // same basis as consist::basis (kept private there; tiny duplicate is fine
    // for a calibration tool — asserts against the same JPEG scaling)
    let mut m = [[0.0f32; 8]; 8];
    for (u, row) in m.iter_mut().enumerate() {
        let cu = if u == 0 { (0.5f32).sqrt() } else { 1.0 };
        for (x, v) in row.iter_mut().enumerate() {
            *v = 0.5 * cu * (((2 * x + 1) as f32) * (u as f32) * core::f32::consts::PI / 16.0).cos();
        }
    }
    let mut px = [[0.0f32; 8]; 8];
    for yy in 0..8 {
        let sy = (by * 8 + yy).min(h - 1);
        for xx in 0..8 {
            let sx = (bx * 8 + xx).min(w - 1);
            px[yy][xx] = y[sy * w + sx] * 255.0 - 128.0;
        }
    }
    let mut tmp = [[0.0f32; 8]; 8];
    let mut f = [0.0f32; 64];
    for u in 0..8 {
        for x in 0..8 {
            tmp[u][x] = (0..8).map(|yy| m[u][yy] * px[yy][x]).sum();
        }
    }
    for u in 0..8 {
        for v in 0..8 {
            f[u * 8 + v] = (0..8).map(|x| tmp[u][x] * m[v][x]).sum();
        }
    }
    f
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().expect("corpus root"));
    let per_sub: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(2);
    let home = std::env::var("HOME").unwrap();
    let td = PathBuf::from(&home).join("tmp").join(format!("zensr-slack-{}", std::process::id()));
    std::fs::create_dir_all(&td).unwrap();

    println!("encoder\tq\tn_coeffs\tp50_excess\tp99_excess\tmax_excess\tviolation%");
    for enc in ENCODERS {
        for &q in QS {
            let mut excess: Vec<f32> = Vec::new();
            for (_, dir) in SUBCORPORA {
                let mut used = 0usize;
                for f in list_images(&root.join(dir)) {
                    if used >= per_sub { break; }
                    let Some(img) = decode_any(&f) else { continue };
                    let Some(hr) = center_crop(&img, 256) else { continue };
                    used += 1;
                    let ppm = td.join("s.ppm");
                    let jpg = td.join("s.jpg");
                    let mut buf = format!("P6\n{} {}\n255\n", hr.w, hr.h).into_bytes();
                    buf.extend_from_slice(&hr.px);
                    std::fs::write(&ppm, &buf).unwrap();
                    if !encode(&ppm, &jpg, enc, q) { continue; }
                    let data = std::fs::read(&jpg).unwrap();
                    let Ok(dc) = zenjpeg::decoder::Decoder::new()
                        .decode_coefficients(&data, enough::Unstoppable) else { continue };
                    let comp = &dc.components[0];
                    let Some(qt) = dc.quant_tables[comp.quant_table_idx as usize] else { continue };
                    // true luma from the ORIGINAL (pre-encode) pixels
                    let plane = hr.w * hr.h;
                    let mut rgbp = vec![0.0f32; 3 * plane];
                    for i in 0..plane {
                        for c in 0..3 {
                            rgbp[c * plane + i] = hr.px[i * 3 + c] as f32 / 255.0;
                        }
                    }
                    let (mut y, mut cb, mut cr) = (vec![0.0; plane], vec![0.0; plane], vec![0.0; plane]);
                    rgb_to_ycbcr_planes(&rgbp, plane, &mut y, &mut cb, &mut cr);
                    for by in 0..comp.blocks_high {
                        for bx in 0..comp.blocks_wide {
                            let f_true = fdct_luma(&y, hr.w, hr.h, bx, by);
                            let blk = &comp.coeffs[(by * comp.blocks_wide + bx) * 64..][..64];
                            for k in 0..64 {
                                let nat = ZIGZAG_TO_NATURAL[k];
                                let qv = qt[nat] as f32;
                                let c_hat = blk[k] as f32 * qv;
                                excess.push(((f_true[nat] - c_hat).abs() - qv * 0.5) / qv);
                            }
                        }
                    }
                }
            }
            excess.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = excess.len();
            if n == 0 { continue; }
            let pct = |p: f64| excess[((n as f64 - 1.0) * p) as usize];
            let viol = excess.iter().filter(|e| **e > 0.0).count() as f64 / n as f64 * 100.0;
            println!(
                "{enc}\t{q}\t{n}\t{:.3}\t{:.3}\t{:.3}\t{:.2}",
                pct(0.5), pct(0.99), excess[n - 1], viol
            );
        }
    }
}
