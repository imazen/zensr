//! Winograd F(2x2, 3x3) for the compact-model middle layers (cin==cout==nf).
//!
//! 2.25x multiply reduction vs direct 3x3. Interior output pixels are
//! computed in 2x2 tiles (16-tile GEMM batches over the transform domain,
//! written with fixed [f32; T] blocks so LLVM vectorizes the inner loops);
//! the 1-pixel border frame (and any h/w<4 buffer) uses a scalar direct
//! conv with identical zero-pad semantics. Overlapped edge tiles recompute
//! the same math (idempotent stores, same trick as the direct kernel).
//!
//! Numerics: transforms are exact-in-f32 (B/A entries 0,±1; G entries
//! 0,±0.5); accumulated error vs direct conv measured < 1e-4 on the golden
//! tolerance scale (see `wino_matches_direct` test).

const T: usize = 16; // tiles per GEMM batch (2 AVX2 regs per [f32; T] row)

/// U = G g G^T for every (oc, ic) 3x3 kernel; QUAD-major layout
/// [16][cout/4][cin][4] (matches pack_conv3x3's broadcast-4-outputs GEMM
/// pattern; cout must be a multiple of 4).
pub(crate) fn wino_weights(raw: &[f32], cin: usize, cout: usize) -> Vec<f32> {
    assert_eq!(raw.len(), cout * cin * 9);
    assert_eq!(cout % 4, 0, "wino: cout must be quad-aligned");
    let mut u = vec![0.0f32; 16 * cout * cin];
    for oc in 0..cout {
        for ic in 0..cin {
            let g = &raw[(oc * cin + ic) * 9..][..9];
            // Gg: 4x3
            let mut gg = [[0.0f32; 3]; 4];
            for c in 0..3 {
                let (g0, g1, g2) = (g[c], g[3 + c], g[6 + c]);
                gg[0][c] = g0;
                gg[1][c] = 0.5 * (g0 + g1 + g2);
                gg[2][c] = 0.5 * (g0 - g1 + g2);
                gg[3][c] = g2;
            }
            // (Gg)G^T: 4x4
            for r in 0..4 {
                let (a, b, c2) = (gg[r][0], gg[r][1], gg[r][2]);
                let row = [a, 0.5 * (a + b + c2), 0.5 * (a - b + c2), c2];
                for (p, v) in row.iter().enumerate() {
                    let pos = r * 4 + p;
                    u[((pos * (cout / 4) + oc / 4) * cin + ic) * 4 + (oc % 4)] = *v;
                }
            }
        }
    }
    u
}

/// B^T d B for a 4x4 input tile (rows are 4 consecutive f32 quads).
#[inline(always)]
fn input_transform(d: &[[f32; 4]; 4]) -> [f32; 16] {
    // rows: B^T d  (B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]])
    let mut t = [[0.0f32; 4]; 4];
    for c in 0..4 {
        t[0][c] = d[0][c] - d[2][c];
        t[1][c] = d[1][c] + d[2][c];
        t[2][c] = d[2][c] - d[1][c];
        t[3][c] = d[1][c] - d[3][c];
    }
    let mut v = [0.0f32; 16];
    for r in 0..4 {
        v[r * 4] = t[r][0] - t[r][2];
        v[r * 4 + 1] = t[r][1] + t[r][2];
        v[r * 4 + 2] = t[r][2] - t[r][1];
        v[r * 4 + 3] = t[r][1] - t[r][3];
    }
    v
}

/// A^T m A -> 2x2 (A^T = [[1,1,1,0],[0,1,-1,-1]]).
#[inline(always)]
fn output_transform(m: &[f32; 16]) -> [f32; 4] {
    let mut t = [[0.0f32; 4]; 2];
    for c in 0..4 {
        t[0][c] = m[c] + m[4 + c] + m[8 + c];
        t[1][c] = m[4 + c] - m[8 + c] - m[12 + c];
    }
    [
        t[0][0] + t[0][1] + t[0][2],
        t[0][1] - t[0][2] - t[0][3],
        t[1][0] + t[1][1] + t[1][2],
        t[1][1] - t[1][2] - t[1][3],
    ]
}

/// Scalar direct 3x3 (zero pad) for one output pixel — border path.
#[inline(always)]
pub(crate) fn direct_px(
    inp: &[f32],
    cin: usize,
    raw: &[f32],
    bias_oc: f32,
    oc: usize,
    y: usize,
    x: usize,
    h: usize,
    wd: usize,
    cs: usize,
) -> f32 {
    let mut s = bias_oc;
    for ic in 0..cin {
        let k = &raw[(oc * cin + ic) * 9..][..9];
        for ky in 0..3usize {
            let yy = y as isize + ky as isize - 1;
            if yy < 0 || yy >= h as isize {
                continue;
            }
            let row = &inp[ic * cs + yy as usize * wd..][..wd];
            for kx in 0..3usize {
                let xx = x as isize + kx as isize - 1;
                if xx < 0 || xx >= wd as isize {
                    continue;
                }
                s += k[ky * 3 + kx] * row[xx as usize];
            }
        }
    }
    s
}

/// Full same-size conv3x3 (zero pad, planar, bias) — Winograd interior +
/// scalar borders. Drop-in equivalent of `conv3x3_packed_dispatch`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv3x3_wino(
    inp: &[f32],
    cin: usize,
    u: &[f32],
    raw: &[f32],
    bias: &[f32],
    out: &mut [f32],
    cout: usize,
    h: usize,
    wd: usize,
    cs: usize,
) {
    if h < 4 || wd < 4 {
        for oc in 0..cout {
            for y in 0..h {
                for x in 0..wd {
                    out[oc * cs + y * wd + x] =
                        direct_px(inp, cin, raw, bias[oc], oc, y, x, h, wd, cs);
                }
            }
        }
        return;
    }
    // 1px border frame via scalar direct
    for oc in 0..cout {
        for x in 0..wd {
            out[oc * cs + x] = direct_px(inp, cin, raw, bias[oc], oc, 0, x, h, wd, cs);
            out[oc * cs + (h - 1) * wd + x] =
                direct_px(inp, cin, raw, bias[oc], oc, h - 1, x, h, wd, cs);
        }
        for y in 1..h - 1 {
            out[oc * cs + y * wd] = direct_px(inp, cin, raw, bias[oc], oc, y, 0, h, wd, cs);
            out[oc * cs + y * wd + wd - 1] =
                direct_px(inp, cin, raw, bias[oc], oc, y, wd - 1, h, wd, cs);
        }
    }
    // interior tile grid: output rows 1..h-1, cols 1..w-1, 2x2 tiles with
    // overlapped last row/col (idempotent recompute)
    let mut y_starts: Vec<usize> = (1..h.saturating_sub(2)).step_by(2).collect();
    if *y_starts.last().unwrap_or(&0) + 2 < h - 1 || y_starts.is_empty() {
        y_starts.push(h - 3);
    }
    if y_starts.last() == Some(&0) {
        y_starts.pop();
    }
    let mut x_starts: Vec<usize> = (1..wd.saturating_sub(2)).step_by(2).collect();
    if *x_starts.last().unwrap_or(&0) + 2 < wd - 1 || x_starts.is_empty() {
        x_starts.push(wd - 3);
    }
    // scratch: v/m layouts [16][ch][T]
    let mut v = vec![0.0f32; 16 * cin * T];
    let mut mbuf = vec![0.0f32; 16 * cout * T];
    for &y0 in &y_starts {
        let mut bx = 0usize;
        while bx < x_starts.len() {
            let nt = (x_starts.len() - bx).min(T);
            // gather + input transform
            for ic in 0..cin {
                let base = ic * cs + (y0 - 1) * wd;
                for (t, &x0) in x_starts[bx..bx + nt].iter().enumerate() {
                    let mut d = [[0.0f32; 4]; 4];
                    for r in 0..4 {
                        let row = &inp[base + r * wd + (x0 - 1)..][..4];
                        d[r] = [row[0], row[1], row[2], row[3]];
                    }
                    let tv = input_transform(&d);
                    for p in 0..16 {
                        v[(p * cin + ic) * T + t] = tv[p];
                    }
                }
            }
            // GEMM per transform position
            for p in 0..16 {
                let up = &u[p * cout * cin..][..cout * cin];
                let vp = &v[p * cin * T..][..cin * T];
                let mp = &mut mbuf[p * cout * T..][..cout * T];
                for q in 0..cout / 4 {
                    let mut acc = [[0.0f32; T]; 4];
                    for ic in 0..cin {
                        let w4 = &up[(q * cin + ic) * 4..][..4];
                        let vv: &[f32; T] = (&vp[ic * T..ic * T + T]).try_into().unwrap();
                        for ob in 0..4 {
                            for tt in 0..T {
                                acc[ob][tt] += w4[ob] * vv[tt];
                            }
                        }
                    }
                    for ob in 0..4 {
                        mp[(q * 4 + ob) * T..(q * 4 + ob) * T + T].copy_from_slice(&acc[ob]);
                    }
                }
            }
            // inverse transform + bias
            for oc in 0..cout {
                let b = bias[oc];
                for (t, &x0) in x_starts[bx..bx + nt].iter().enumerate() {
                    let mut m = [0.0f32; 16];
                    for p in 0..16 {
                        m[p] = mbuf[(p * cout + oc) * T + t];
                    }
                    let yq = output_transform(&m);
                    let o = oc * cs + y0 * wd + x0;
                    out[o] = yq[0] + b;
                    out[o + 1] = yq[1] + b;
                    out[o + wd] = yq[2] + b;
                    out[o + wd + 1] = yq[3] + b;
                }
            }
            bx += nt;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    #[test]
    fn wino_dispatch_matches_direct() {
        // sizes chosen so nt >= 16 (exercises the VECTOR kernel incl. the
        // f32x16 tier), plus odd/even w and h and non-tight stride
        for &(cin, cout, h, wd) in
            &[(8usize, 8usize, 40usize, 70usize), (16, 16, 23, 37), (8, 8, 24, 36)]
        {
            let cs = h * wd + 5;
            let raw = lcg(cout * cin * 9, 3);
            let bias = lcg(cout, 5);
            let inp = lcg(cin * cs, 9);
            let u = wino_weights(&raw, cin, cout);
            let mut got = vec![0.0f32; cout * cs];
            crate::simd::conv3x3_wino_dispatch(
                &inp, cin, &u, &raw, &bias, &mut got, cout, h, wd, cs,
            );
            let mut mx = 0.0f32;
            for oc in 0..cout {
                for y in 0..h {
                    for x in 0..wd {
                        let want = direct_px(&inp, cin, &raw, bias[oc], oc, y, x, h, wd, cs);
                        mx = mx.max((got[oc * cs + y * wd + x] - want).abs());
                    }
                }
            }
            assert!(mx < 1e-4, "dispatch cin{cin} {h}x{wd}: max diff {mx}");
        }
    }

    #[test]
    fn wino_matches_direct() {
        for &(cin, cout, h, wd) in
            &[(8usize, 8usize, 13usize, 11usize), (16, 16, 12, 20), (4, 8, 5, 4), (8, 4, 4, 9)]
        {
            let cs = h * wd + 3; // non-tight channel stride
            let raw = lcg(cout * cin * 9, 7);
            let bias = lcg(cout, 11);
            let inp = lcg(cin * cs, 13);
            let u = wino_weights(&raw, cin, cout);
            let mut got = vec![0.0f32; cout * cs];
            conv3x3_wino(&inp, cin, &u, &raw, &bias, &mut got, cout, h, wd, cs);
            // reference: scalar direct everywhere
            let mut want = vec![0.0f32; cout * cs];
            for oc in 0..cout {
                for y in 0..h {
                    for x in 0..wd {
                        want[oc * cs + y * wd + x] =
                            direct_px(&inp, cin, &raw, bias[oc], oc, y, x, h, wd, cs);
                    }
                }
            }
            let mut mx = 0.0f32;
            for oc in 0..cout {
                for i in 0..h * wd {
                    mx = mx.max((got[oc * cs + i] - want[oc * cs + i]).abs());
                }
            }
            assert!(mx < 1e-4, "cin{cin} cout{cout} {h}x{wd}: max diff {mx}");
        }
    }
}
