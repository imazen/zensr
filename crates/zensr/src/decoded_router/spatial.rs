//! Development-only selector. No runtime policy or trained thresholds change.
//! One representative center plus maxima of DC structure, erased structure,
//! AC population and AC-population variance. Maxima may overlap substantially.
use super::{Window, windows};

pub(super) fn select(
    width: usize,
    height: usize,
    coefficients: &[i16],
    blocks_wide: usize,
) -> Vec<Window> {
    let baseline = windows(width, height);
    if width <= 512 && height <= 512 {
        return baseline.to_vec();
    }
    let (bw, bh) = (width.div_ceil(8), height.div_ceil(8));
    assert!(blocks_wide >= bw && coefficients.len() >= blocks_wide * bh * 64);
    let stride = bw + 1;
    let mut sums: [Vec<f64>; 4] = core::array::from_fn(|_| vec![0.; stride * (bh + 1)]);
    for y in 0..bh {
        for x in 0..bw {
            let offset = (y * blocks_wide + x) * 64;
            let block = &coefficients[offset..offset + 64];
            let ac = block[1..].iter().filter(|&&c| c != 0).count() as f64;
            let dc = f64::from(block[0]);
            let structure = if x > 0 {
                (dc - f64::from(coefficients[offset - 64])).abs()
            } else {
                0.
            } + if y > 0 {
                (dc - f64::from(coefficients[offset - blocks_wide * 64])).abs()
            } else {
                0.
            };
            let values = [structure, structure * (63. - ac), ac, ac * ac];
            let i = (y + 1) * stride + x + 1;
            for k in 0..4 {
                sums[k][i] =
                    values[k] + sums[k][i - 1] + sums[k][i - stride] - sums[k][i - stride - 1];
            }
        }
    }
    let (w, h) = (width.min(512), height.min(512));
    let query = |map: &[f64], x: usize, y: usize, cw: usize, ch: usize| {
        map[(y + ch) * stride + x + cw] - map[y * stride + x + cw] - map[(y + ch) * stride + x]
            + map[y * stride + x]
    };
    let mut candidates = Vec::new();
    for y in (0..=height - h).step_by(16) {
        for x in (0..=width - w).step_by(16) {
            let (bx, by, cw, ch) = (x / 8, y / 8, w.div_ceil(8), h.div_ceil(8));
            let totals: [f64; 4] = core::array::from_fn(|k| query(&sums[k], bx, by, cw, ch));
            let variance = (totals[3] - totals[2] * totals[2] / (cw * ch) as f64).max(0.);
            candidates.push((
                Window {
                    x,
                    y,
                    width: w,
                    height: h,
                },
                [totals[0], totals[1], totals[2], variance],
            ));
        }
    }
    let mut selected = vec![baseline[0]];
    for metric in 0..4 {
        let mut best = None;
        for (window, scores) in &candidates {
            if selected.contains(window) || scores[metric] <= 0. {
                continue;
            }
            if best
                .as_ref()
                .is_none_or(|(_, score)| scores[metric] > *score)
            {
                best = Some((*window, scores[metric]));
            }
        }
        if let Some((window, _)) = best {
            selected.push(window);
        }
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_structure_outside_fixed_positions_gets_sampled() {
        let (w, h) = (2048, 1536);
        let (bw, bh) = (w / 8, h / 8);
        let mut coefficients = vec![0; bw * bh * 64];
        let (x, y) = (600, 600);
        coefficients[(y / 8 * bw + x / 8) * 64] = 200;
        let covers = |window: &Window| {
            window.x <= x
                && x < window.x + window.width
                && window.y <= y
                && y < window.y + window.height
        };
        assert!(!windows(w, h).iter().any(covers));
        let selected = select(w, h, &coefficients, bw);
        assert!(selected.iter().any(covers));
        assert!(selected.len() <= 5);
        for (i, window) in selected.iter().enumerate() {
            assert_eq!((window.x % 16, window.y % 16), (0, 0));
            assert!(window.x + window.width <= w && window.y + window.height <= h);
            assert!(!selected[..i].contains(window));
        }
    }
    #[test]
    fn full_image_and_flat_inputs_do_not_invent_extra_evidence() {
        for (w, h) in [(8usize, 8usize), (127, 63), (512, 341), (1024, 768)] {
            let bw = w.div_ceil(16) * 2;
            let bh = h.div_ceil(16) * 2;
            let selected = select(w, h, &vec![0; bw * bh * 64], bw);
            assert_eq!(selected, vec![windows(w, h)[0]]);
        }
    }
}
