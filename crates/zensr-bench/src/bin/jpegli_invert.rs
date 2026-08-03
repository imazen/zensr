//! Recover the butteraugli distance from a jpegli-lineage quant table, in
//! closed form — no stored tables.
//!
//! zenjpeg builds each luma quantizer as
//!     q[i] = round(base[i] * global_scale * freq_scale(d, i))
//!     freq_scale(d, i) = d                                   if d < T
//!                      = max(0.5d, T^(1-e[i]) * d^e[i])      otherwise
//! (T = DIST_THRESHOLD = 1.5). Each coefficient therefore inverts on its own:
//!     d = q[i] / (base[i] * G)                               low branch
//!     d = (q[i] / (base[i] * G * T^(1-e[i])))^(1/e[i])        high branch
//! Rounding makes each estimate noisy, so the median over 64 coefficients is
//! the estimator. This replaces ~3,690 sampled tables (ceiling ~11,400, and
//! the parameter is continuous so no sampling ever closes it) with 129 floats.
use zensr_zenjpeg::jpegli_params::{
    BASE_QUANT_LUMA, DIST_THRESHOLD, FREQUENCY_EXPONENT, GLOBAL_SCALE_YCBCR,
};

/// Median of per-coefficient distance estimates.
pub fn distance_from_luma(q: &[u16; 64]) -> f32 {
    let base = &BASE_QUANT_LUMA;
    let mut est: Vec<f32> = Vec::with_capacity(64);
    for i in 0..64 {
        let denom = base[i] * GLOBAL_SCALE_YCBCR;
        if denom <= 0.0 || q[i] == 0 {
            continue;
        }
        let v = q[i] as f32;
        // low branch is linear in d
        let d_low = v / denom;
        let d = if d_low < DIST_THRESHOLD {
            d_low
        } else {
            let e = FREQUENCY_EXPONENT[i];
            let mul = DIST_THRESHOLD.powf(1.0 - e);
            (v / (denom * mul)).powf(1.0 / e)
        };
        if d.is_finite() && d > 0.0 {
            est.push(d);
        }
    }
    if est.is_empty() {
        return f32::NAN;
    }
    est.sort_by(|a, b| a.partial_cmp(b).unwrap());
    est[est.len() / 2]
}

/// Refine the closed-form seed to the exact distance interval.
///
/// The forward map is monotone in `d` at every coefficient, so the set of
/// distances producing a given table is a contiguous interval. The median
/// estimate lands within a rounding step of it; a short bisection on the
/// number of coefficients that are too high converges onto the interval.
pub fn refine(seed: f32, target: &[u16; 64]) -> f32 {
    let too_high = |d: f32| {
        let t = forward_luma(d);
        (0..64).filter(|&i| t[i] > target[i]).count() as i32
            - (0..64).filter(|&i| t[i] < target[i]).count() as i32
    };
    let (mut lo, mut hi) = (seed * 0.9, seed * 1.1);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if forward_luma(mid) == *target {
            return mid;
        }
        if too_high(mid) > 0 {
            hi = mid
        } else {
            lo = mid
        }
    }
    0.5 * (lo + hi)
}

fn main() {
    use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig};
    let (w, h) = (64usize, 64usize);
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            rgb[i] = ((x * 7 + y * 3) % 256) as u8;
            rgb[i + 1] = ((x * 2 + y * 11) % 256) as u8;
            rgb[i + 2] = ((x * 13) % 256) as u8;
        }
    }
    let px: &[rgb::Rgb<u8>] = bytemuck::cast_slice(&rgb);
    // Validation that does not depend on knowing the "true" distance: run the
    // recovered distance back through the FORWARD formula and compare tables.
    // If forward(invert(T)) == T exactly, the inversion has recovered the
    // parameter to the full resolution the table can carry.
    println!("quality\tprobe_d\tinverted_d\tforward_roundtrip_max_delta");
    let (mut worst, mut n, mut exact) = (0u16, 0u32, 0u32);
    for qi in 1..=100u32 {
        let jpg = EncoderConfig::ycbcr(qi as f32, ChromaSubsampling::None)
            .encode(px, w as u32, h as u32)
            .expect("encode");
        let p = zenjpeg::detect::probe(&jpg).expect("probe");
        let t = super_dqt(&jpg).expect("dqt");
        let d = refine(distance_from_luma(&t), &t);
        let rebuilt = forward_luma(d);
        let delta = (0..64)
            .map(|i| rebuilt[i].abs_diff(t[i]))
            .max()
            .unwrap_or(0);
        worst = worst.max(delta);
        if delta == 0 {
            exact += 1;
        }
        n += 1;
        if qi % 10 == 0 {
            println!("{qi}\t{:.4}\t{:.4}\t{delta}", p.quality.value, d);
        }
    }
    println!("\n{exact}/{n} qualities round-trip EXACTLY; worst coefficient delta {worst}");
}

/// The forward law, for round-trip validation.
fn forward_luma(d: f32) -> [u16; 64] {
    let mut out = [0u16; 64];
    for i in 0..64 {
        let fs = if d < DIST_THRESHOLD {
            d
        } else {
            let e = FREQUENCY_EXPONENT[i];
            (0.5 * d).max(DIST_THRESHOLD.powf(1.0 - e) * d.powf(e))
        };
        out[i] = (BASE_QUANT_LUMA[i] * GLOBAL_SCALE_YCBCR * fs).round() as u16;
    }
    out
}

fn super_dqt(data: &[u8]) -> Option<[u16; 64]> {
    const ZZ: [usize; 64] = [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
        20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    ];
    let mut i = 2usize;
    while i + 3 < data.len() {
        if data[i] != 0xFF {
            i += 1;
            continue;
        }
        let m = data[i + 1];
        if m == 0xD8 || m == 0x01 || (0xD0..=0xD7).contains(&m) {
            i += 2;
            continue;
        }
        if m == 0xD9 {
            break;
        }
        let len = ((data[i + 2] as usize) << 8) | data[i + 3] as usize;
        if i + 2 + len > data.len() {
            break;
        }
        if m == 0xDB {
            let seg = &data[i + 4..i + 2 + len];
            if seg[0] == 0 && seg.len() >= 65 {
                let mut out = [0u16; 64];
                for k in 0..64 {
                    out[ZZ[k]] = seg[1 + k] as u16;
                }
                return Some(out);
            }
        }
        if m == 0xDA {
            break;
        }
        i += 2 + len;
    }
    None
}
