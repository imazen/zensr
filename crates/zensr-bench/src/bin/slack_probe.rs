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

fn env_list(name: &str, default: Vec<String>) -> Vec<String> {
    std::env::var(name)
        .map(|v| {
            v.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or(default)
}

fn encode(ppm: &PathBuf, jpg: &PathBuf, enc: &str, q: u32) -> bool {
    let home = std::env::var("HOME").unwrap();
    let st = match enc {
        "turbo" => Command::new("cjpeg")
            .args([
                "-quality",
                &q.to_string(),
                "-sample",
                "1x1",
                "-optimize",
                "-outfile",
            ])
            .arg(jpg)
            .arg(ppm)
            .status(),
        "mozjpeg" => Command::new(format!("{home}/tmp/ati-bin/mozjpeg-cjpeg"))
            .env(
                "LD_LIBRARY_PATH",
                format!("{home}/tmp/ati-bin/mozjpeg-lib64"),
            )
            .args(["-quality", &q.to_string(), "-sample", "1x1", "-outfile"])
            .arg(jpg)
            .arg(ppm)
            .status(),
        "jpegli" => Command::new("cjpegli")
            .arg(ppm)
            .arg(jpg)
            .args(["-q", &q.to_string(), "--chroma_subsampling=444"])
            .status(),
        _ => {
            let me = std::env::current_exe().unwrap();
            Command::new(me.parent().unwrap().join("zjtool"))
                .arg("enc")
                .arg(ppm)
                .arg(jpg)
                .args([&q.to_string(), "444"])
                .status()
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
            *v =
                0.5 * cu * (((2 * x + 1) as f32) * (u as f32) * core::f32::consts::PI / 16.0).cos();
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
    let td = PathBuf::from(&home)
        .join("tmp")
        .join(format!("zensr-slack-{}", std::process::id()));
    std::fs::create_dir_all(&td).unwrap();

    // provenance (sweep discipline): commit, host, config
    let commit = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let host = std::fs::read_to_string("/etc/hostname").unwrap_or_default();
    let encoders = env_list(
        "ZENSR_SLACK_ENCODERS",
        ENCODERS.iter().map(|s| s.to_string()).collect(),
    );
    let qs: Vec<u32> = env_list("ZENSR_SLACK_QS", QS.iter().map(|q| q.to_string()).collect())
        .iter()
        .map(|s| s.parse().expect("q"))
        .collect();
    println!(
        "# slack_probe commit={commit} host={} per_sub={per_sub} encoders={encoders:?} qs={qs:?}",
        host.trim()
    );
    println!("encoder\tq\tn_coeffs\tp50_excess\tp99_excess\tmax_excess\tviolation%");
    let mut perq_rows: Vec<String> = Vec::new();
    for enc in &encoders {
        let enc = enc.as_str();
        for &q in &qs {
            let mut excess: Vec<f32> = Vec::new();
            // (quantizer value, normalized excess) for the per-Q breakdown
            let mut by_qv: std::collections::BTreeMap<u16, Vec<f32>> = Default::default();
            // nonzero-coded coefficients only (tests the skip_zeroed rule:
            // trellis/AQ violations should concentrate on zeroed bands)
            let mut excess_nz: Vec<f32> = Vec::new();
            for (_, dir) in SUBCORPORA {
                let mut used = 0usize;
                for f in list_images(&root.join(dir)) {
                    if used >= per_sub {
                        break;
                    }
                    let Some(img) = decode_any(&f) else { continue };
                    let Some(hr) = center_crop(&img, 256) else {
                        continue;
                    };
                    used += 1;
                    let ppm = td.join("s.ppm");
                    let jpg = td.join("s.jpg");
                    let mut buf = format!("P6\n{} {}\n255\n", hr.w, hr.h).into_bytes();
                    buf.extend_from_slice(&hr.px);
                    std::fs::write(&ppm, &buf).unwrap();
                    // ZENSR_SLACK_GENS=N: re-encode the crop N times (each
                    // generation adds its own pre-FDCT u8 rounding). The final
                    // generation's coefficients are compared against the TRUE
                    // DCT of the PRISTINE original — i.e. how far outside the
                    // box the truth sits for a multi-generation file, which is
                    // what the projection must tolerate on real web images.
                    let gens: usize = std::env::var("ZENSR_SLACK_GENS")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(1);
                    let mut cur_ppm = ppm.clone();
                    let mut ok = true;
                    for gi in 0..gens {
                        if !encode(&cur_ppm, &jpg, enc, q) {
                            ok = false;
                            break;
                        }
                        if gi + 1 < gens {
                            let mid = td.join(format!("m{gi}.ppm"));
                            let me = std::env::current_exe().unwrap();
                            let st = Command::new(me.parent().unwrap().join("zjtool"))
                                .arg("dec")
                                .arg(&jpg)
                                .arg(&mid)
                                .arg("off")
                                .status();
                            if !st.map(|s| s.success()).unwrap_or(false) {
                                ok = false;
                                break;
                            }
                            // ZENSR_SLACK_RESIZE=1: resample between generations
                            // (the CDN thumbnail flow). Destroys the previous
                            // block grid, so the next encode sees a resampled
                            // signal rather than an aligned re-quantization —
                            // the violation physics should differ.
                            if std::env::var("ZENSR_SLACK_RESIZE").as_deref() == Ok("1") {
                                let Some(dimg) = decode_any(&mid) else {
                                    ok = false;
                                    break;
                                };
                                let half = resize_rgb8(
                                    &dimg,
                                    dimg.w * 3 / 4,
                                    dimg.h * 3 / 4,
                                    zenresize::Filter::CatmullRom,
                                );
                                let back =
                                    resize_rgb8(&half, hr.w, hr.h, zenresize::Filter::CatmullRom);
                                let mut b2 =
                                    format!("P6\n{} {}\n255\n", back.w, back.h).into_bytes();
                                b2.extend_from_slice(&back.px);
                                std::fs::write(&mid, &b2).unwrap();
                            }
                            cur_ppm = mid;
                        }
                    }
                    if !ok {
                        continue;
                    }
                    let data = std::fs::read(&jpg).unwrap();
                    let Ok(dc) = zenjpeg::decoder::Decoder::new()
                        .decode_coefficients(&data, enough::Unstoppable)
                    else {
                        continue;
                    };
                    let comp = &dc.components[0];
                    let Some(qt) = dc.quant_tables[comp.quant_table_idx as usize] else {
                        continue;
                    };
                    // true luma from the ORIGINAL (pre-encode) pixels
                    let plane = hr.w * hr.h;
                    let mut rgbp = vec![0.0f32; 3 * plane];
                    for i in 0..plane {
                        for c in 0..3 {
                            rgbp[c * plane + i] = hr.px[i * 3 + c] as f32 / 255.0;
                        }
                    }
                    let (mut y, mut cb, mut cr) =
                        (vec![0.0; plane], vec![0.0; plane], vec![0.0; plane]);
                    rgb_to_ycbcr_planes(&rgbp, plane, &mut y, &mut cb, &mut cr);
                    for by in 0..comp.blocks_high {
                        for bx in 0..comp.blocks_wide {
                            let f_true = fdct_luma(&y, hr.w, hr.h, bx, by);
                            let blk = &comp.coeffs[(by * comp.blocks_wide + bx) * 64..][..64];
                            for k in 0..64 {
                                let nat = ZIGZAG_TO_NATURAL[k];
                                let qv = qt[nat] as f32;
                                let c_hat = blk[k] as f32 * qv;
                                let e = ((f_true[nat] - c_hat).abs() - qv * 0.5) / qv;
                                excess.push(e);
                                by_qv.entry(qt[nat]).or_default().push(e);
                                if blk[k] != 0 {
                                    excess_nz.push(e);
                                }
                            }
                        }
                    }
                }
            }
            excess.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = excess.len();
            if n == 0 {
                continue;
            }
            let pct = |p: f64| excess[((n as f64 - 1.0) * p) as usize];
            let viol = excess.iter().filter(|e| **e > 0.0).count() as f64 / n as f64 * 100.0;
            println!(
                "{enc}\t{q}\t{n}\t{:.3}\t{:.3}\t{:.3}\t{:.2}",
                pct(0.5),
                pct(0.99),
                excess[n - 1],
                viol
            );
            excess_nz.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let nn = excess_nz.len();
            if nn > 0 {
                let pcn = |p: f64| excess_nz[((nn as f64 - 1.0) * p) as usize];
                let violn =
                    excess_nz.iter().filter(|e| **e > 0.0).count() as f64 / nn as f64 * 100.0;
                println!(
                    "{enc}-nz\t{q}\t{nn}\t{:.3}\t{:.3}\t{:.3}\t{:.2}",
                    pcn(0.5),
                    pcn(0.99),
                    excess_nz[nn - 1],
                    violn
                );
            }
            // per-quantizer-value tail stats: if the violation tail is an
            // ABSOLUTE noise floor (fdct/idct implementation skew), then
            // p99_abs (= p99_excess * Q) should be ~constant across Q while
            // p99_excess blows up as Q -> 1.
            for (qv, mut v) in std::mem::take(&mut by_qv) {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let m = v.len();
                let pc = |p: f64| v[((m as f64 - 1.0) * p) as usize];
                let vi = v.iter().filter(|e| **e > 0.0).count() as f64 / m as f64 * 100.0;
                perq_rows.push(format!(
                    "{enc}\t{q}\t{qv}\t{m}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.2}",
                    pc(0.99),
                    pc(0.999),
                    v[m - 1],
                    pc(0.99) * qv as f32,
                    v[m - 1] * qv as f32,
                    vi
                ));
            }
        }
    }
    println!("\n# per-quantizer-value breakdown");
    println!("encoder\tq\tQval\tn\tp99_exc\tp999_exc\tmax_exc\tp99_abs\tmax_abs\tviolation%");
    for r in perq_rows {
        println!("{r}");
    }
}
