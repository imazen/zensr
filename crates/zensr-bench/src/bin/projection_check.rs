//! Matched projection ablation on retained JPEG/reference pairs.
//! Usage: projection_check manifest.tsv output-dir
//! Columns: id, jpeg, reference_rgb8, width, height (tab separated).
//! All model arms keep guards and decode policy identical and disable routing
//! and the high-quality gate. This is a diagnostic, not a routing frontier.
use std::{fs, io::Write, path::Path};
use zensr_bench::{ssim2, Rgb8Img};
use zensr_micro::consist::{rgb_to_ycbcr_planes, ZIGZAG_TO_NATURAL};
use zensr_zenjpeg::{Projection, RestoreConfig};
// Share the exact record correction without expanding the product's public API.
#[path = "../../../zensr-zenjpeg/src/record/probe.rs"]
mod record;

#[derive(Default, Debug)]
struct Residual {
    coefficients: usize,
    outside: usize,
    max_excess: f64,
    sum_excess: f64,
}

// Independent f64 DCT measurement: do not infer consistency from the projector's
// clamped fraction. Measure only visible blocks, replicating partial edges.
#[allow(clippy::too_many_arguments)]
fn residual(
    pixels: &[f32],
    width: usize,
    height: usize,
    stride: usize,
    coefficients: &[i16],
    blocks_wide: usize,
    quant: &[u16; 64],
    slack: (f64, f64),
) -> Residual {
    assert!(stride >= width && pixels.len() >= (height - 1) * stride + width);
    let basis: [[f64; 8]; 8] = core::array::from_fn(|u| {
        core::array::from_fn(|x| {
            let scale = if u == 0 { 0.5f64.sqrt() } else { 1.0 };
            0.5 * scale * (((2 * x + 1) * u) as f64 * std::f64::consts::PI / 16.).cos()
        })
    });
    let mut out = Residual::default();
    for by in 0..height.div_ceil(8) {
        for bx in 0..width.div_ceil(8) {
            let mut tmp = [[0.; 8]; 8];
            for (u, row) in tmp.iter_mut().enumerate() {
                for (x, value) in row.iter_mut().enumerate() {
                    for y in 0..8 {
                        let p = pixels
                            [(by * 8 + y).min(height - 1) * stride + (bx * 8 + x).min(width - 1)];
                        *value += basis[u][y] * (f64::from(p) * 255. - 128.);
                    }
                }
            }
            for (k, &natural) in ZIGZAG_TO_NATURAL.iter().enumerate() {
                let (u, v) = (natural / 8, natural % 8);
                let value: f64 = (0..8).map(|x| tmp[u][x] * basis[v][x]).sum();
                let q = f64::from(quant[natural]);
                let stored = f64::from(coefficients[(by * blocks_wide + bx) * 64 + k]) * q;
                let excess = ((value - stored).abs() - q * (0.5 + slack.0) - slack.1).max(0.);
                out.coefficients += 1;
                // Explicit numeric tolerance in JPEG coefficient units.
                out.outside += usize::from(excess > 1e-3);
                out.max_excess = out.max_excess.max(excess);
                out.sum_excess += excess;
            }
        }
    }
    out
}

fn box_down(plane: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.; w.div_ceil(2) * h.div_ceil(2)];
    for y in 0..h.div_ceil(2) {
        for x in 0..w.div_ceil(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    out[y * w.div_ceil(2) + x] +=
                        0.25 * plane[(2 * y + dy).min(h - 1) * w + (2 * x + dx).min(w - 1)];
                }
            }
        }
    }
    out
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "projection_check manifest.tsv output-dir");
    let root = Path::new(&args[2]);
    fs::create_dir(root).expect("output directory must be new");
    fs::copy(&args[1], root.join("inputs.tsv")).unwrap();
    let model = zensr_bench::load_adopted("dejpeg_rt24i_canon4").expect("canon4 weights required");
    let mut report = fs::File::create(root.join("results.tsv")).unwrap();
    writeln!(
        report,
        "id\tarm\tssim2\tcomponent\tcoefficients\toutside\tmax_excess\tmean_excess\twidth\theight"
    )
    .unwrap();
    let manifest = fs::read_to_string(&args[1]).unwrap();
    assert_eq!(
        manifest.lines().next(),
        Some("id\tjpeg\treference_rgb8\twidth\theight")
    );
    let mut ids = std::collections::HashSet::new();
    for line in manifest.lines().skip(1) {
        let f: Vec<_> = line.split('\t').collect();
        assert_eq!(f.len(), 5);
        assert!(f[0]
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
        assert!(!f[0].is_empty() && ids.insert(f[0]));
        let (w, h): (usize, usize) = (f[3].parse().unwrap(), f[4].parse().unwrap());
        assert!(w > 0 && h > 0);
        let reference = Rgb8Img {
            px: fs::read(f[2]).unwrap(),
            w,
            h,
        };
        assert_eq!(reference.px.len(), w * h * 3);
        let data = fs::read(f[1]).unwrap();
        let probe = zenjpeg::detect::probe(&data).unwrap();
        let coeffs = zenjpeg::decoder::Decoder::new()
            .decode_coefficients(&data, enough::Unstoppable)
            .unwrap();
        assert_eq!((coeffs.width as usize, coeffs.height as usize), (w, h));
        assert_eq!(coeffs.components.len(), 3, "RGB YCbCr JPEG required");
        // RestoreConfig::Auto applies encode-record overrides internally; use
        // the same record-corrected family for diagnostic slack below.
        let mut slack_probe = probe;
        record::apply_record(&mut slack_probe);
        let slack = (
            f64::from(zensr_zenjpeg::slack_for(&slack_probe)),
            f64::from(zensr_zenjpeg::slack_abs_for(&slack_probe)),
        );
        for (arm, projection, skip) in [
            ("identity", Projection::Off, true),
            ("projection_off", Projection::Off, false),
            ("projection_on", Projection::Auto, false),
        ] {
            let cfg = RestoreConfig::default()
                .with_threads(1)
                .with_high_q_identity(false)
                .with_projection(projection);
            let pixels = if skip {
                let dec = zenjpeg::decoder::Decoder::new()
                    .deblock(zenjpeg::decoder::DeblockMode::Off)
                    .decode(&data, enough::Unstoppable)
                    .unwrap();
                dec.pixels_u8().unwrap().to_vec()
            } else {
                zensr_zenjpeg::restore_jpeg(&data, &model, &cfg)
                    .unwrap()
                    .to_rgb8()
            };
            assert_eq!(pixels.len(), w * h * 3);
            fs::write(root.join(format!("{}-{arm}.rgb8", f[0])), &pixels).unwrap();
            let image = Rgb8Img { px: pixels, w, h };
            let quality = ssim2(&reference, &image);
            assert!(quality.is_finite(), "nonfinite SSIM2");
            let rgb = zensr_bench::to_planar_f32(&image);
            let (mut y, mut cb, mut cr) = (vec![0.; w * h], vec![0.; w * h], vec![0.; w * h]);
            rgb_to_ycbcr_planes(&rgb, w * h, &mut y, &mut cb, &mut cr);
            for (ci, mut plane) in [y, cb, cr].into_iter().enumerate() {
                let component = &coeffs.components[ci];
                let max_h = coeffs.components.iter().map(|c| c.h_samp).max().unwrap();
                let max_v = coeffs.components.iter().map(|c| c.v_samp).max().unwrap();
                let (pw, ph) = if component.h_samp == max_h && component.v_samp == max_v {
                    (w, h)
                } else {
                    assert!(
                        2 * component.h_samp == max_h && 2 * component.v_samp == max_v,
                        "only 420/444 supported"
                    );
                    plane = box_down(&plane, w, h);
                    (w.div_ceil(2), h.div_ceil(2))
                };
                let r = residual(
                    &plane,
                    pw,
                    ph,
                    pw,
                    &component.coeffs,
                    component.blocks_wide,
                    &coeffs.quant_tables[component.quant_table_idx as usize].unwrap(),
                    slack,
                );
                writeln!(
                    report,
                    "{}\t{arm}\t{quality:.12}\t{ci}\t{}\t{}\t{:.9}\t{:.9}\t{w}\t{h}",
                    f[0],
                    r.coefficients,
                    r.outside,
                    r.max_excess,
                    r.sum_excess / r.coefficients as f64
                )
                .unwrap();
            }
            report.flush().unwrap();
        }
    }
    assert!(!ids.is_empty(), "empty input manifest");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn box_sampling_replicates_odd_edges() {
        assert_eq!(
            box_down(&[0., 1., 2., 3., 4., 5., 6., 7., 8.], 3, 3),
            vec![2., 3.5, 6.5, 8.]
        );
    }

    #[test]
    fn independent_dct_measures_dc_and_strided_rows() {
        let pixels = vec![1.; 8 * 11];
        let mut coefficients = [0; 64];
        coefficients[0] = 1016; // (255-128)*8
        let r = residual(&pixels, 8, 8, 11, &coefficients, 1, &[1; 64], (0., 0.));
        assert_eq!(r.outside, 0);
        coefficients[0] = 1000;
        let r = residual(&pixels, 8, 8, 11, &coefficients, 1, &[1; 64], (0., 0.));
        assert_eq!(r.outside, 1);
        assert!((r.max_excess - 15.5).abs() < 1e-8);
    }
}
