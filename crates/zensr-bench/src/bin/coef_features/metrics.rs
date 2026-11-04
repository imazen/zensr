/// Zigzag index at which "high frequency" starts. 32 splits the 64 coefficients
/// in half by zigzag order, which is coarse but needs no tuning — the point is
/// to separate detail from structure, not to find an optimal cut.
const HF_START: usize = 32;

pub(super) struct Feats {
    pub(super) zero_ac_blocks: f64,
    pub(super) zero_ac_coefs: f64,
    pub(super) hf_survival: f64,
    pub(super) mean_abs_ac: f64,
    pub(super) dc_spread: f64,
    // Spatial statistics of damage. The six scalar features above all say how
    // MUCH was quantised away; none says WHERE, and a uniformly-degraded image
    // and one with a few destroyed regions score identically under all of them.
    // These are the cheapest way to ask that question.
    /// Stdev across blocks of the per-block surviving-AC count. High when
    /// damage is concentrated, low when it is spread evenly.
    pub(super) ac_count_spread: f64,
    /// Share of blocks whose surviving-AC count is below a quarter of the
    /// image's own mean — locally destroyed regions, measured relative to the
    /// image so it does not just re-measure global quality.
    pub(super) gutted_block_frac: f64,
    /// Share of gutted blocks that neighbour another gutted block
    /// (4-connected). Separates scattered loss from contiguous flattened areas,
    /// which is the distinction a spatial signal has to make to be new.
    pub(super) gutted_clustering: f64,
}

pub(super) fn features(coeffs: &[i16], nblocks: usize, blocks_wide: usize) -> Option<Feats> {
    if nblocks == 0 {
        return None;
    }
    // Per-block surviving-AC count, kept in raster order so neighbours can be
    // found for the clustering measure.
    let accounts: Vec<f64> = coeffs[..nblocks * 64]
        .as_chunks::<64>()
        .0
        .iter()
        .map(|b| b[1..].iter().filter(|&&c| c != 0).count() as f64)
        .collect();
    let (mut zblk, mut zcoef, mut hf, mut lf, mut absac, mut nac) =
        (0u64, 0u64, 0f64, 0f64, 0f64, 0u64);
    let mut dcs: Vec<f64> = Vec::with_capacity(nblocks);
    for b in coeffs[..nblocks * 64].as_chunks::<64>().0.iter() {
        dcs.push(b[0] as f64);
        if b[1..].iter().all(|&c| c == 0) {
            zblk += 1;
        }
        for (i, &c) in b.iter().enumerate().skip(1) {
            if c == 0 {
                zcoef += 1;
            } else {
                let e = (c as f64) * (c as f64);
                if i >= HF_START {
                    hf += e;
                } else {
                    lf += e;
                }
                absac += (c as f64).abs();
                nac += 1;
            }
        }
    }
    let n = nblocks as f64;
    let mean_dc = dcs.iter().sum::<f64>() / n;
    let var_dc = dcs.iter().map(|d| (d - mean_dc).powi(2)).sum::<f64>() / n;

    let mean_ac = accounts.iter().sum::<f64>() / n;
    let var_ac = accounts.iter().map(|a| (a - mean_ac).powi(2)).sum::<f64>() / n;
    let cut = mean_ac * 0.25;
    let gutted: Vec<bool> = accounts.iter().map(|&a| a <= cut).collect();
    let ngut = gutted.iter().filter(|&&g| g).count();
    let bw = blocks_wide.max(1);
    let mut clustered = 0usize;
    for (i, &g) in gutted.iter().enumerate() {
        if !g {
            continue;
        }
        let (x, y) = (i % bw, i / bw);
        let neigh = [
            (x > 0).then(|| i - 1),
            (x + 1 < bw).then(|| i + 1),
            (y > 0).then(|| i.wrapping_sub(bw)),
            (i + bw < gutted.len()).then(|| i + bw),
        ];
        if neigh.into_iter().flatten().any(|j| gutted[j]) {
            clustered += 1;
        }
    }

    Some(Feats {
        zero_ac_blocks: zblk as f64 / n,
        zero_ac_coefs: zcoef as f64 / (n * 63.0),
        hf_survival: if hf + lf > 0.0 { hf / (hf + lf) } else { 0.0 },
        mean_abs_ac: if nac > 0 { absac / nac as f64 } else { 0.0 },
        dc_spread: var_dc.sqrt(),
        ac_count_spread: var_ac.sqrt(),
        gutted_block_frac: ngut as f64 / n,
        gutted_clustering: if ngut > 0 {
            clustered as f64 / ngut as f64
        } else {
            0.0
        },
    })
}
