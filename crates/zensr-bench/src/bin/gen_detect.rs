//! JPEG multi-generation detector: dataset generator + feature extractor.
//!
//! Classifies a JPEG as one of:
//!   gen1  — encoded once (source may have been resampled BEFORE the first
//!           encode — that is the normal web flow and is still gen1)
//!   gen2a — decoded + re-encoded at identical dimensions (block grid kept)
//!   gen2r — decoded + resampled + re-encoded (block grid destroyed; CDN flow)
//!
//! Why: S10 quantization-consistency projection slack must scale with
//! generation count (pre-FDCT u8 rounding pushes truth outside the +-Q/2 box
//! more each generation). Detecting gen1 at decode time lets zensr tighten
//! the projection (~+0.45 ssim2 at q75) while staying safe on recompressed
//! input.
//!
//! This bin GENERATES a labelled dataset from the imazen-26 corpus (chains
//! via cjpeg + mozjpeg subprocesses) and emits one TSV row per generated
//! JPEG with detection features computed FROM THE FINAL JPEG BYTES ONLY.
//! Threshold fitting / classification lives in tools/gen_detect_train.py.
//!
//! Anti-leak design (v2, after the v1 postmortem):
//!   - ALL features are computed on a fixed centered WIN x WIN pixel window
//!     (32x32 luma blocks), so no feature can encode absolute image size.
//!     v1 leaked class through dc_expfill / dq_nbands (gen2r rows were
//!     smaller by construction and the model read block-count, not physics).
//!   - gen1 / gen2a chains include a pristine PRE-scale (sp) so the size
//!     distributions of all three classes overlap; "was it ever resampled"
//!     is deliberately NOT the discriminator (real gen1 is resampled too) —
//!     the discriminator must be "resampled from a COMPRESSED source".
//!
//! Feature groups:
//!   DQ (double-quantization comb) — luma + chroma per-band histograms of
//!     quantized coefficients; normalized chi-square vs a hole-punched local
//!     mean (period-agnostic comb detector); zero-gap counts; DC occupancy.
//!   Claimed-quality vs damage — libjpeg-style quality estimate from the
//!     stored tables vs nonzero fraction / energy per zigzag region / bpp.
//!   Pixel grid — 8x8 boundary/interior blockiness ratio; cross-axis
//!     "ghost" scale-match: spectral power of the gradient profile at the
//!     SAME old-grid period 8*s on BOTH axes (content periodicity is mostly
//!     axis-specific; a uniformly resampled old block grid is not).
//!
//! Usage: gen_detect <imazen26-root> <out.tsv>
//! Env:  ZENSR_GD_TRAIN_PER_SUB=12  sources per subcorpus for the train split
//!       ZENSR_GD_SPLIT=both|train|eval
//!       ZENSR_GD_THREADS=8
//!       ZENSR_GD_CAP=896           center-crop cap on the source
//!       ZENSR_GD_ENCODERS=turbo,mozjpeg  pool for enc1/enc2 (jpegli to probe)
//!       ZENSR_GD_KEEP=1            keep generated jpgs in scratch
//!       ZENSR_GD_DEBUG_HIST=<jpg>  dump per-band luma histograms and exit
//!       ZENSR_GD_DEBUG_FEAT=<jpg>  dump the feature row for one file and exit
//!
//! Eval split = the pinned eval_split/imazen26_eval_files.tsv files (never
//! train/tune on those); train = stride-sampled from the remaining files.

use std::collections::{HashSet, VecDeque};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::Instant;
use zensr_bench::*;
use zensr_micro::consist::ZIGZAG_TO_NATURAL;

const GEN1_QS: &[u32] = &[35, 55, 75, 92];
/// (q1, q2) pairs for the two-generation classes. q1<q2 (recompressed at
/// HIGHER quality) is the hard case production must catch — sampled densest.
const GEN2_PAIRS: &[(u32, u32)] = &[
    (35, 35),
    (35, 55),
    (35, 75),
    (35, 92),
    (55, 75),
    (55, 92),
    (75, 75),
    (75, 92),
    (92, 55),
    (92, 92),
];
/// Pristine pre-scale applied before the FIRST encode (all classes; real web
/// images are almost always resized from a larger original before encoding).
const PRESCALES: &[f64] = &[1.0, 0.85, 0.7, 0.6];
/// Second-generation resample factor (gen2r only).
const SCALES: &[f64] = &[0.55, 0.7, 0.85, 0.92];
const FILTERS: &[(&str, zenresize::Filter)] = &[
    ("lanczos", zenresize::Filter::Lanczos),
    ("catmullrom", zenresize::Filter::CatmullRom),
    ("mitchell", zenresize::Filter::Mitchell),
];
/// Fixed analysis window (pixels); 32x32 luma blocks. Chain min size is
/// cap*min(PRESCALES)*min(SCALES) = 896*0.6*0.55 = 295 > 256, so the window
/// never clamps and no feature can encode absolute image size.
const WIN: usize = 256;

/// libjpeg Annex K luma table (natural order) for claimed-quality estimation.
const STD_LUMA_Q: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69,
    56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81,
    104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

fn fnv(s: &str) -> u64 {
    // FNV-1a + murmur3 finalizer. Raw FNV's low bits are a near-linear
    // function of the LAST key byte (keys differing only in a trailing
    // '1'/'2' produced perfectly anti-correlated %2 picks — the v2 dataset
    // accidentally made EVERY gen2 chain cross-encoder, and a plain xor-fold
    // was still 99% anti-correlated). fmix64 fully scrambles; callers also
    // put the variant tag FIRST in the key.
    let mut h = 0xcbf29ce484222325u64;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^ (h >> 33)
}

fn encode(ppm: &Path, jpg: &Path, enc: &str, q: u32) -> bool {
    let home = std::env::var("HOME").unwrap();
    let st = match enc {
        "turbo" => Command::new("cjpeg")
            .args(["-quality", &q.to_string(), "-sample", "2x2", "-optimize", "-outfile"])
            .arg(jpg)
            .arg(ppm)
            .status(),
        "mozjpeg" => Command::new(format!("{home}/tmp/ati-bin/mozjpeg-cjpeg"))
            .env("LD_LIBRARY_PATH", format!("{home}/tmp/ati-bin/mozjpeg-lib64"))
            .args(["-quality", &q.to_string(), "-sample", "2x2", "-outfile"])
            .arg(jpg)
            .arg(ppm)
            .status(),
        "jpegli" => Command::new("cjpegli")
            .arg(ppm)
            .arg(jpg)
            .args(["-q", &q.to_string(), "--chroma_subsampling=420"])
            .status(),
        _ => panic!("unknown encoder {enc}"),
    };
    st.map(|s| s.success()).unwrap_or(false)
}

fn write_ppm(img: &Rgb8Img, path: &Path) {
    let mut buf = format!("P6\n{} {}\n255\n", img.w, img.h).into_bytes();
    buf.extend_from_slice(&img.px);
    std::fs::write(path, &buf).unwrap();
}

// ---------------- feature extraction (from JPEG bytes only) ----------------

#[derive(Default)]
struct Feats {
    w: usize,
    h: usize,
    bytes: usize,
    bpp: f64,
    q_claim: f64,
    qt_lo_mean: f64,
    // DQ comb (luma AC bands zz 1..=20, windowed blocks)
    dq_chi_max: f64,
    dq_chi_top3: f64,
    dq_chi_wmean: f64,
    dq_gapc: f64,
    dq_nbands: usize,
    // DC occupancy (luma, windowed)
    dc_fill: f64,
    dc_expfill: f64,
    dc_chi: f64,
    dc_width: usize,
    // chroma DQ (both chroma components pooled, bands zz 0..=5)
    c_chi_max: f64,
    c_gapc: f64,
    c_dc_fill: f64,
    // HF / claimed-quality-vs-damage stats (luma, windowed)
    nz_lo: f64,
    nz_mid: f64,
    nz_hi: f64,
    e_lo: f64,
    e_mid: f64,
    e_hi: f64,
    // pixel grid (windowed)
    blk_v: f64,
    blk_h: f64,
    ghost_v_snr: f64,
    ghost_v_per: f64,
    ghost_h_snr: f64,
    ghost_h_per: f64,
    ghost_match: f64,
    ghost_match_s: f64,
    t_coeff_ms: f64,
    t_pix_ms: f64,
    t_feat_ms: f64,
}

const FEAT_HEADER: &str = "w\th\tbytes\tbpp\tq_claim\tqt_lo_mean\tdq_chi_max\tdq_chi_top3\tdq_chi_wmean\tdq_gapc\tdq_nbands\tdc_fill\tdc_expfill\tdc_chi\tdc_width\tc_chi_max\tc_gapc\tc_dc_fill\tnz_lo\tnz_mid\tnz_hi\te_lo\te_mid\te_hi\tblk_v\tblk_h\tghost_v_snr\tghost_v_per\tghost_h_snr\tghost_h_per\tghost_match\tghost_match_s\tt_coeff_ms\tt_pix_ms\tt_feat_ms";

impl Feats {
    fn row(&self) -> String {
        format!(
            "{}\t{}\t{}\t{:.4}\t{:.1}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{:.4}\t{:.4}\t{:.4}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.5}\t{:.5}\t{:.5}\t{:.2}\t{:.3}\t{:.4}\t{:.4}\t{:.4}\t{:.3}\t{:.2}\t{:.3}\t{:.2}\t{:.3}\t{:.3}\t{:.2}\t{:.2}\t{:.2}",
            self.w, self.h, self.bytes, self.bpp, self.q_claim, self.qt_lo_mean,
            self.dq_chi_max, self.dq_chi_top3, self.dq_chi_wmean, self.dq_gapc, self.dq_nbands,
            self.dc_fill, self.dc_expfill, self.dc_chi, self.dc_width,
            self.c_chi_max, self.c_gapc, self.c_dc_fill,
            self.nz_lo, self.nz_mid, self.nz_hi, self.e_lo, self.e_mid, self.e_hi,
            self.blk_v, self.blk_h,
            self.ghost_v_snr, self.ghost_v_per, self.ghost_h_snr, self.ghost_h_per,
            self.ghost_match, self.ghost_match_s,
            self.t_coeff_ms, self.t_pix_ms, self.t_feat_ms
        )
    }
}

/// Normalized chi-square of a histogram vs a hole-punched local mean
/// (window +-3, center excluded). Smooth single-quantization histograms give
/// ~Poisson chi (~nbins/N, tiny); comb histograms (teeth + gaps) give
/// O(0.1..1.3) regardless of period, incl. split teeth and trellis encoders.
fn comb_chi(h: &[f64]) -> f64 {
    let n: f64 = h.iter().sum();
    if n <= 0.0 {
        return 0.0;
    }
    let b = h.len();
    let mut chi = 0.0f64;
    for c in 0..b {
        let (mut s, mut cnt) = (0.0f64, 0u32);
        for d in -3i64..=3 {
            if d == 0 {
                continue;
            }
            let j = c as i64 + d;
            if (0..b as i64).contains(&j) {
                s += h[j as usize];
                cnt += 1;
            }
        }
        let m = s / cnt as f64;
        chi += (h[c] - m) * (h[c] - m) / (h[c] + m + 2.0);
    }
    chi / n
}

/// Count of empty bins (1-indexed positions 1..=12) with real mass beyond
/// them — impossible under a single quantization.
fn zero_gaps(h: &[u32]) -> u32 {
    let b = h.len() - 1;
    let mut gaps = 0;
    for c in 1..=12.min(b) {
        let beyond: u64 = (c + 1..=b).map(|cc| h[cc] as u64).sum();
        if h[c] == 0 && beyond >= 8 {
            gaps += 1;
        }
    }
    gaps
}

/// DFT power at frequency f over an integer-indexed residual signal.
fn dft_pow(r: &[f64], f: f64) -> f64 {
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (x, v) in r.iter().enumerate() {
        let a = -2.0 * std::f64::consts::PI * f * (x as f64);
        re += v * a.cos();
        im += v * a.sin();
    }
    re * re + im * im
}

fn extract_features(data: &[u8]) -> Option<Feats> {
    let t0 = Instant::now();
    let mut f = Feats { bytes: data.len(), ..Default::default() };

    // ---- coefficient-domain ----
    let tc = Instant::now();
    let dc = zenjpeg::decoder::Decoder::new()
        .decode_coefficients(data, enough::Unstoppable)
        .ok()?;
    f.t_coeff_ms = tc.elapsed().as_secs_f64() * 1e3;
    let comp = &dc.components[0];
    let qt = dc.quant_tables[comp.quant_table_idx as usize]?;

    // claimed quality (libjpeg scaling of Annex K); median over entries
    let mut scales: Vec<f64> =
        (0..64).map(|i| qt[i] as f64 * 100.0 / STD_LUMA_Q[i] as f64).collect();
    scales.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sc = scales[32];
    f.q_claim = if sc <= 100.0 { (200.0 - sc) / 2.0 } else { 5000.0 / sc };
    f.qt_lo_mean = (1..10).map(|k| qt[ZIGZAG_TO_NATURAL[k]] as f64).sum::<f64>() / 9.0;

    // fixed centered block window (size-leak guard: all counts below are
    // computed over at most WBLK x WBLK blocks regardless of image size)
    const WBLK: usize = WIN / 8;
    let bw = comp.blocks_wide.min(WBLK);
    let bh = comp.blocks_high.min(WBLK);
    let bx0 = (comp.blocks_wide - bw) / 2;
    let by0 = (comp.blocks_high - bh) / 2;
    let nb = bw * bh;

    const B: usize = 32;
    const NBANDS: usize = 20;
    let mut hist = vec![[0u32; B + 1]; NBANDS];
    let mut dcv: Vec<i32> = Vec::with_capacity(nb);
    let mut nz = [0u64; 3];
    let mut ntot = [0u64; 3];
    let mut energy = [0f64; 3];
    for byi in 0..bh {
        for bxi in 0..bw {
            let b = (by0 + byi) * comp.blocks_wide + (bx0 + bxi);
            let blk = &comp.coeffs[b * 64..b * 64 + 64];
            dcv.push(blk[0] as i32);
            for k in 1..64 {
                let c = blk[k] as i64;
                let reg = if k <= 9 { 0 } else if k <= 27 { 1 } else { 2 };
                ntot[reg] += 1;
                if c != 0 {
                    nz[reg] += 1;
                    let q = qt[ZIGZAG_TO_NATURAL[k]] as i64;
                    energy[reg] += ((c * q) * (c * q)) as f64;
                }
                if k <= NBANDS {
                    let a = c.unsigned_abs() as usize;
                    if a <= B {
                        hist[k - 1][a] += 1;
                    }
                }
            }
        }
    }
    f.nz_lo = nz[0] as f64 / ntot[0].max(1) as f64;
    f.nz_mid = nz[1] as f64 / ntot[1].max(1) as f64;
    f.nz_hi = nz[2] as f64 / ntot[2].max(1) as f64;
    f.e_lo = energy[0] / nb.max(1) as f64;
    f.e_mid = energy[1] / nb.max(1) as f64;
    f.e_hi = energy[2] / nb.max(1) as f64;

    let mut chis: Vec<(f64, u64)> = Vec::new();
    let mut gapc = 0.0f64;
    for hk in hist.iter() {
        let n_ac: u64 = (1..=B).map(|c| hk[c] as u64).sum();
        if n_ac < 128 {
            continue;
        }
        let h: Vec<f64> = (1..=B).map(|c| hk[c] as f64).collect();
        chis.push((comb_chi(&h), n_ac));
        gapc += zero_gaps(hk) as f64;
    }
    chis.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    f.dq_nbands = chis.len();
    f.dq_chi_max = chis.first().map(|c| c.0).unwrap_or(0.0);
    f.dq_chi_top3 = if chis.is_empty() {
        0.0
    } else {
        chis.iter().take(3).map(|c| c.0).sum::<f64>() / chis.len().min(3) as f64
    };
    let wtot: u64 = chis.iter().map(|c| c.1).sum();
    f.dq_chi_wmean = if wtot > 0 {
        chis.iter().map(|c| c.0 * c.1 as f64).sum::<f64>() / wtot as f64
    } else {
        0.0
    };
    f.dq_gapc = gapc / (f.dq_nbands.max(1) as f64);

    // luma DC occupancy inside central value range
    dcv.sort_unstable();
    if dcv.len() >= 64 {
        let n = dcv.len();
        let p = |q: f64| dcv[((n - 1) as f64 * q) as usize];
        let (lo, hi) = (p(0.01), p(0.99));
        let width = ((hi - lo).unsigned_abs() as usize).min(900);
        f.dc_width = width;
        if width >= 16 {
            let mut cnt = vec![0.0f64; width + 1];
            let mut inside = 0u64;
            for &v in &dcv {
                if v >= lo && v <= lo + width as i32 {
                    cnt[(v - lo) as usize] += 1.0;
                    inside += 1;
                }
            }
            let filled = cnt.iter().filter(|v| **v > 0.0).count() as f64;
            f.dc_fill = filled / (width + 1) as f64;
            let lam = inside as f64 / (width + 1) as f64;
            f.dc_expfill = 1.0 - (-lam).exp();
            f.dc_chi = comb_chi(&cnt);
        }
    }

    // chroma DQ (4:2:0: chroma block window is half the luma window)
    let mut c_chis: Vec<f64> = Vec::new();
    let mut c_gaps = 0.0f64;
    let mut c_fill = (0.0f64, 0u32);
    for ci in 1..dc.components.len().min(3) {
        let cc = &dc.components[ci];
        let Some(cqt) = dc.quant_tables[cc.quant_table_idx as usize] else { continue };
        let _ = cqt;
        let cbw = cc.blocks_wide.min(WBLK / 2);
        let cbh = cc.blocks_high.min(WBLK / 2);
        let cx0 = (cc.blocks_wide - cbw) / 2;
        let cy0 = (cc.blocks_high - cbh) / 2;
        let mut ch = vec![[0u32; B + 1]; 6];
        let mut cdc: Vec<i32> = Vec::new();
        for byi in 0..cbh {
            for bxi in 0..cbw {
                let b = (cy0 + byi) * cc.blocks_wide + (cx0 + bxi);
                let blk = &cc.coeffs[b * 64..b * 64 + 64];
                cdc.push(blk[0] as i32);
                for k in 1..=6usize {
                    let a = (blk[k] as i64).unsigned_abs() as usize;
                    if a <= B {
                        ch[k - 1][a] += 1;
                    }
                }
            }
        }
        for hk in ch.iter() {
            let n_ac: u64 = (1..=B).map(|c| hk[c] as u64).sum();
            if n_ac < 64 {
                continue;
            }
            let h: Vec<f64> = (1..=B).map(|c| hk[c] as f64).collect();
            c_chis.push(comb_chi(&h));
            c_gaps += zero_gaps(hk) as f64;
        }
        cdc.sort_unstable();
        if cdc.len() >= 64 {
            let n = cdc.len();
            let p = |q: f64| cdc[((n - 1) as f64 * q) as usize];
            let width = ((p(0.99) - p(0.01)).unsigned_abs() as usize).min(900);
            if width >= 8 {
                let lo = p(0.01);
                let mut occ = vec![false; width + 1];
                for &v in &cdc {
                    if v >= lo && v <= lo + width as i32 {
                        occ[(v - lo) as usize] = true;
                    }
                }
                let fill = occ.iter().filter(|b| **b).count() as f64 / (width + 1) as f64;
                c_fill.0 += fill;
                c_fill.1 += 1;
            }
        }
    }
    c_chis.sort_by(|a, b| b.partial_cmp(a).unwrap());
    f.c_chi_max = c_chis.first().copied().unwrap_or(0.0);
    f.c_gapc = c_gaps / c_chis.len().max(1) as f64;
    f.c_dc_fill = if c_fill.1 > 0 { c_fill.0 / c_fill.1 as f64 } else { 1.0 };

    // ---- pixel-domain (windowed) ----
    let tp = Instant::now();
    let dec = zenjpeg::decoder::Decoder::new().decode(data, zenjpeg::encoder::Unstoppable).ok()?;
    let (w, h) = dec.dimensions();
    let (w, h) = (w as usize, h as usize);
    f.w = w;
    f.h = h;
    f.bpp = data.len() as f64 * 8.0 / (w * h) as f64;
    let px = dec.pixels_u8()?;
    f.t_pix_ms = tp.elapsed().as_secs_f64() * 1e3;
    let chans = px.len() / (w * h);
    if chans < 3 {
        return None;
    }
    // luma window, aligned to the CURRENT 8x8 grid (x0/y0 multiples of 8)
    let ww = w.min(WIN);
    let wh = h.min(WIN);
    let x0 = ((w - ww) / 2) & !7;
    let y0 = ((h - wh) / 2) & !7;
    let mut y = vec![0.0f32; ww * wh];
    for yy in 0..wh {
        for xx in 0..ww {
            let i = ((y0 + yy) * w + (x0 + xx)) * chans;
            y[yy * ww + xx] =
                0.299 * px[i] as f32 + 0.587 * px[i + 1] as f32 + 0.114 * px[i + 2] as f32;
        }
    }
    let mut dcol = vec![0.0f64; ww];
    for yy in 0..wh {
        let row = &y[yy * ww..(yy + 1) * ww];
        for x in 1..ww {
            dcol[x] += (row[x] - row[x - 1]).abs() as f64;
        }
    }
    for v in dcol.iter_mut() {
        *v /= wh as f64;
    }
    let mut drow = vec![0.0f64; wh];
    for yy in 1..wh {
        let (r0, r1) = (&y[(yy - 1) * ww..yy * ww], &y[yy * ww..(yy + 1) * ww]);
        let s: f32 = r0.iter().zip(r1.iter()).map(|(a, b)| (a - b).abs()).sum();
        drow[yy] = s as f64 / ww as f64;
    }
    // blockiness ratio + phase-mean-removed residual per axis
    let prep = |d: &[f64]| -> (f64, Vec<f64>, f64) {
        let (mut sb, mut nbn, mut si, mut ni) = (0.0, 0u32, 0.0, 0u32);
        for (x, &v) in d.iter().enumerate().skip(8) {
            if x % 8 == 0 {
                sb += v;
                nbn += 1;
            } else {
                si += v;
                ni += 1;
            }
        }
        let ratio =
            if ni > 0 && si > 0.0 { (sb / nbn.max(1) as f64) / (si / ni as f64) } else { 1.0 };
        let mut pm = [0.0f64; 8];
        let mut pn = [0u32; 8];
        for (x, &v) in d.iter().enumerate().skip(1) {
            pm[x % 8] += v;
            pn[x % 8] += 1;
        }
        for p in 0..8 {
            pm[p] /= pn[p].max(1) as f64;
        }
        let r: Vec<f64> = d.iter().enumerate().skip(1).map(|(x, &v)| v - pm[x % 8]).collect();
        let e: f64 = r.iter().map(|v| v * v).sum();
        (ratio, r, e)
    };
    let (bv, rv, ev_) = prep(&dcol);
    let (bh_, rh, eh) = prep(&drow);
    f.blk_v = bv;
    f.blk_h = bh_;
    // ghost scan restricted to plausible old-grid periods 8*s for
    // s in [0.50, 0.95] -> period in [4.0, 7.6]
    let s_lo = 0.50f64;
    let s_hi = 0.95f64;
    let nstep = 221;
    let mut best_v = (0.0f64, 0.0f64);
    let mut best_h = (0.0f64, 0.0f64);
    let mut best_match = (0.0f64, 0.0f64);
    let mut pv_all: Vec<f64> = Vec::with_capacity(nstep);
    let mut ph_all: Vec<f64> = Vec::with_capacity(nstep);
    let mut svals: Vec<f64> = Vec::with_capacity(nstep);
    for i in 0..nstep {
        let s = s_lo + (s_hi - s_lo) * i as f64 / (nstep - 1) as f64;
        let fr = 1.0 / (8.0 * s);
        // fundamental + 2nd harmonic: block-edge combs are step-like, and the
        // harmonic often carries MORE power than the fundamental after
        // resampling (v1 found period 3.0 for an s=0.75 chain = 2nd harmonic
        // of the old 6.0 grid). 2*fr stays below Nyquist for s >= 0.5.
        let pv = (dft_pow(&rv, fr) + dft_pow(&rv, 2.0 * fr)) / ev_.max(1e-9);
        let ph = (dft_pow(&rh, fr) + dft_pow(&rh, 2.0 * fr)) / eh.max(1e-9);
        pv_all.push(pv);
        ph_all.push(ph);
        svals.push(s);
        if pv > best_v.0 {
            best_v = (pv, 8.0 * s);
        }
        if ph > best_h.0 {
            best_h = (ph, 8.0 * s);
        }
    }
    // normalize each axis by its own median power (content/level calibration)
    let med = |v: &[f64]| -> f64 {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2].max(1e-9)
    };
    let (mv, mh) = (med(&pv_all), med(&ph_all));
    for i in 0..nstep {
        let m = (pv_all[i] / mv).min(ph_all[i] / mh);
        if m > best_match.0 {
            best_match = (m, svals[i]);
        }
    }
    f.ghost_v_snr = best_v.0;
    f.ghost_v_per = best_v.1;
    f.ghost_h_snr = best_h.0;
    f.ghost_h_per = best_h.1;
    f.ghost_match = best_match.0;
    f.ghost_match_s = best_match.1;
    f.t_feat_ms = t0.elapsed().as_secs_f64() * 1e3 - f.t_coeff_ms - f.t_pix_ms;
    Some(f)
}

// ---------------- box-violation physics (ZENSR_GD_PHYSICS mode) ----------------
// slack_probe's ZENSR_SLACK_RESIZE branch never produced data (it calls
// decode_any on a .ppm, which only handles png/jpg -> every image skips), so
// the resized-chain physics is measured here instead, with the resize done
// in-process. Chains at q1==q2 to match the calibrated aligned table.

/// JPEG-scaled 8x8 FDCT of one luma block (same basis as slack_probe /
/// consist::basis; tiny duplicate is fine for a calibration tool).
fn fdct_luma(y: &[f32], w: usize, h: usize, bx: usize, by: usize) -> [f32; 64] {
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

/// Per-coefficient excess = (|c_true - c_hat| - Q/2)/Q of a final JPEG's luma
/// coefficients vs the TRUE DCT of `truth` (same dimensions).
fn truth_excess(truth: &Rgb8Img, jpg: &[u8], out: &mut Vec<f32>, out_nz: &mut Vec<f32>) -> bool {
    use zensr_micro::consist::rgb_to_ycbcr_planes;
    let Ok(dc) = zenjpeg::decoder::Decoder::new().decode_coefficients(jpg, enough::Unstoppable)
    else {
        return false;
    };
    let comp = &dc.components[0];
    let Some(qt) = dc.quant_tables[comp.quant_table_idx as usize] else { return false };
    let plane = truth.w * truth.h;
    let mut rgbp = vec![0.0f32; 3 * plane];
    for i in 0..plane {
        for c in 0..3 {
            rgbp[c * plane + i] = truth.px[i * 3 + c] as f32 / 255.0;
        }
    }
    let (mut y, mut cb, mut cr) = (vec![0.0; plane], vec![0.0; plane], vec![0.0; plane]);
    rgb_to_ycbcr_planes(&rgbp, plane, &mut y, &mut cb, &mut cr);
    for by in 0..comp.blocks_high {
        for bx in 0..comp.blocks_wide {
            let f_true = fdct_luma(&y, truth.w, truth.h, bx, by);
            let blk = &comp.coeffs[(by * comp.blocks_wide + bx) * 64..][..64];
            for k in 0..64 {
                let nat = ZIGZAG_TO_NATURAL[k];
                let qv = qt[nat] as f32;
                let c_hat = blk[k] as f32 * qv;
                let e = ((f_true[nat] - c_hat).abs() - qv * 0.5) / qv;
                out.push(e);
                if blk[k] != 0 {
                    out_nz.push(e);
                }
            }
        }
    }
    true
}

fn physics(root: &Path, per_sub: usize, td: &Path) {
    std::fs::create_dir_all(td).unwrap();
    println!("# gen_detect physics per_sub={per_sub} chains=gen1,gen2a,gen2r-roundtrip,gen2r-cdn (q1==q2, resize=0.75 catmullrom)");
    println!("chain\tencoder\tq\tn\tp50\tp99\tp999\tmax\tviolation%\tn_nz\tp99_nz\tmax_nz\tviol%_nz");
    for enc in ["turbo", "mozjpeg"] {
        for q in [35u32, 75, 92] {
            let mut acc: std::collections::BTreeMap<&str, (Vec<f32>, Vec<f32>)> = Default::default();
            for (_, dir) in SUBCORPORA {
                let mut used = 0usize;
                for fpath in list_images(&root.join(dir)) {
                    if used >= per_sub {
                        break;
                    }
                    let Some(img) = decode_any(&fpath) else { continue };
                    let Some(hr) = center_crop(&img, 256) else { continue };
                    used += 1;
                    let ppm = td.join("p.ppm");
                    let jpg1 = td.join("p1.jpg");
                    let jpg2 = td.join("p2.jpg");
                    write_ppm(&hr, &ppm);
                    // gen1
                    if encode(&ppm, &jpg1, enc, q) {
                        let data = std::fs::read(&jpg1).unwrap();
                        let e = acc.entry("gen1").or_default();
                        truth_excess(&hr, &data, &mut e.0, &mut e.1);
                        // gen2a: decode + re-encode aligned
                        if let Some(d1) = decode_any(&jpg1) {
                            let pa = td.join("a.ppm");
                            write_ppm(&d1, &pa);
                            if encode(&pa, &jpg2, enc, q) {
                                let data2 = std::fs::read(&jpg2).unwrap();
                                let e = acc.entry("gen2a").or_default();
                                truth_excess(&hr, &data2, &mut e.0, &mut e.1);
                            }
                            // gen2r-roundtrip: resize 0.75 down + back up, re-encode
                            // at original dims; truth = pristine crop
                            let dn = resize_rgb8(
                                &d1,
                                d1.w * 3 / 4,
                                d1.h * 3 / 4,
                                zenresize::Filter::CatmullRom,
                            );
                            let up = resize_rgb8(&dn, d1.w, d1.h, zenresize::Filter::CatmullRom);
                            let pr = td.join("r.ppm");
                            write_ppm(&up, &pr);
                            if encode(&pr, &jpg2, enc, q) {
                                let data2 = std::fs::read(&jpg2).unwrap();
                                let e = acc.entry("gen2r-roundtrip").or_default();
                                truth_excess(&hr, &data2, &mut e.0, &mut e.1);
                            }
                            // gen2r-cdn: resize 0.75 down, re-encode at the SMALL
                            // dims; truth = pristine resized with the same kernel
                            let pc = td.join("c.ppm");
                            write_ppm(&dn, &pc);
                            if encode(&pc, &jpg2, enc, q) {
                                let truth_small = resize_rgb8(
                                    &hr,
                                    dn.w,
                                    dn.h,
                                    zenresize::Filter::CatmullRom,
                                );
                                let data2 = std::fs::read(&jpg2).unwrap();
                                let e = acc.entry("gen2r-cdn").or_default();
                                truth_excess(&truth_small, &data2, &mut e.0, &mut e.1);
                            }
                        }
                    }
                }
            }
            for (chain, (mut ex, mut exnz)) in acc {
                ex.sort_by(|a, b| a.partial_cmp(b).unwrap());
                exnz.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n = ex.len();
                if n == 0 {
                    continue;
                }
                let pct = |v: &Vec<f32>, p: f64| v[((v.len() as f64 - 1.0) * p) as usize];
                let viol = ex.iter().filter(|e| **e > 0.0).count() as f64 / n as f64 * 100.0;
                let nn = exnz.len();
                let violn = if nn > 0 {
                    exnz.iter().filter(|e| **e > 0.0).count() as f64 / nn as f64 * 100.0
                } else {
                    0.0
                };
                println!(
                    "{chain}\t{enc}\t{q}\t{n}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.2}\t{nn}\t{:.3}\t{:.3}\t{:.2}",
                    pct(&ex, 0.5),
                    pct(&ex, 0.99),
                    pct(&ex, 0.999),
                    ex[n - 1],
                    viol,
                    if nn > 0 { pct(&exnz, 0.99) } else { 0.0 },
                    if nn > 0 { exnz[nn - 1] } else { 0.0 },
                    violn
                );
            }
        }
    }
}

// ---------------- dataset generation ----------------

struct Task {
    split: &'static str,
    sub: &'static str,
    path: PathBuf,
}

fn main() {
    // Debug: dump per-band luma histograms for one file.
    if let Ok(p) = std::env::var("ZENSR_GD_DEBUG_HIST") {
        let data = std::fs::read(&p).expect("read jpg");
        let dc = zenjpeg::decoder::Decoder::new()
            .decode_coefficients(&data, enough::Unstoppable)
            .expect("coeffs");
        let comp = &dc.components[0];
        let qt = dc.quant_tables[comp.quant_table_idx as usize].expect("qt");
        let nb = comp.blocks_wide * comp.blocks_high;
        println!(
            "# {} blocks={} qt_zz[0..6]={:?}",
            p,
            nb,
            (0..6).map(|k| qt[ZIGZAG_TO_NATURAL[k]]).collect::<Vec<_>>()
        );
        for k in 1..=6usize {
            let mut h = [0u32; 33];
            for b in 0..nb {
                let c = comp.coeffs[b * 64 + k].unsigned_abs() as usize;
                if c <= 32 {
                    h[c] += 1;
                }
            }
            println!("zz{k} Q={}: {:?}", qt[ZIGZAG_TO_NATURAL[k]], &h[0..33]);
        }
        return;
    }
    if let Ok(p) = std::env::var("ZENSR_GD_DEBUG_FEAT") {
        let data = std::fs::read(&p).expect("read jpg");
        let f = extract_features(&data).expect("features");
        println!("{FEAT_HEADER}");
        println!("{}", f.row());
        return;
    }
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().expect("usage: gen_detect <imazen26-root> <out.tsv>"));
    if std::env::var("ZENSR_GD_PHYSICS").is_ok() {
        let per_sub: usize =
            std::env::var("ZENSR_GD_TRAIN_PER_SUB").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
        let home = std::env::var("HOME").unwrap();
        let td = PathBuf::from(&home)
            .join("tmp")
            .join(format!("zensr-gendet-phys-{}", std::process::id()));
        physics(&root, per_sub, &td);
        let _ = std::fs::remove_dir_all(&td);
        return;
    }
    let out_path = PathBuf::from(args.next().expect("out.tsv"));
    let per_sub: usize =
        std::env::var("ZENSR_GD_TRAIN_PER_SUB").ok().and_then(|v| v.parse().ok()).unwrap_or(12);
    let split_sel = std::env::var("ZENSR_GD_SPLIT").unwrap_or_else(|_| "both".into());
    let threads: usize =
        std::env::var("ZENSR_GD_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let cap: usize =
        std::env::var("ZENSR_GD_CAP").ok().and_then(|v| v.parse().ok()).unwrap_or(896);
    let encoders: Vec<String> = std::env::var("ZENSR_GD_ENCODERS")
        .unwrap_or_else(|_| "turbo,mozjpeg".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let keep = std::env::var("ZENSR_GD_KEEP").as_deref() == Ok("1");

    // pinned eval files
    let mut evset: HashSet<(String, String)> = HashSet::new();
    for line in std::fs::read_to_string("eval_split/imazen26_eval_files.tsv")
        .expect("run from zensr repo root (needs eval_split/imazen26_eval_files.tsv)")
        .lines()
    {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split('\t');
        if let (Some(d), Some(fname)) = (it.next(), it.next()) {
            evset.insert((d.to_string(), fname.to_string()));
        }
    }

    let mut tasks: VecDeque<Task> = VecDeque::new();
    for (_, dir) in SUBCORPORA {
        let files = list_images(&root.join(dir));
        let mut ev: Vec<PathBuf> = Vec::new();
        let mut tr: Vec<PathBuf> = Vec::new();
        for p in files {
            let fname = p.file_name().unwrap().to_string_lossy().to_string();
            if evset.contains(&(dir.to_string(), fname)) {
                ev.push(p);
            } else {
                tr.push(p);
            }
        }
        if split_sel != "train" {
            for p in ev {
                tasks.push_back(Task { split: "eval", sub: dir, path: p });
            }
        }
        if split_sel != "eval" {
            // stride-sample for diversity (nested dirs sort together)
            let n = tr.len().min(per_sub);
            if n > 0 {
                let stride = (tr.len() as f64 / n as f64).max(1.0);
                for i in 0..n {
                    let idx = ((i as f64 + 0.5) * stride) as usize;
                    tasks.push_back(Task {
                        split: "train",
                        sub: dir,
                        path: tr[idx.min(tr.len() - 1)].clone(),
                    });
                }
            }
        }
    }

    let commit = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let host = std::fs::read_to_string("/etc/hostname").unwrap_or_default();
    let out = std::fs::File::create(&out_path).expect("create out tsv");
    let mut wtr = std::io::BufWriter::new(out);
    writeln!(
        wtr,
        "# gen_detect commit={commit} host={} cap={cap} win={WIN} per_sub={per_sub} encoders={encoders:?} gen1_qs={GEN1_QS:?} gen2_pairs={GEN2_PAIRS:?} prescales={PRESCALES:?} scales={SCALES:?} n_sources={}",
        host.trim(),
        tasks.len()
    )
    .unwrap();
    writeln!(
        wtr,
        "split\tsub\tsrc\tsrcfmt\tcls\tsp\tenc1\tq1\tenc2\tq2\tscale\tfilt\t{FEAT_HEADER}"
    )
    .unwrap();
    let wtr = Mutex::new(wtr);
    let tasks = Mutex::new(tasks);
    let done = std::sync::atomic::AtomicUsize::new(0);

    let home = std::env::var("HOME").unwrap();
    let scratch =
        PathBuf::from(&home).join("tmp").join(format!("zensr-gendet-{}", std::process::id()));

    std::thread::scope(|s| {
        for tid in 0..threads {
            let tasks = &tasks;
            let wtr = &wtr;
            let done = &done;
            let encoders = &encoders;
            let td = scratch.join(format!("t{tid}"));
            s.spawn(move || {
                std::fs::create_dir_all(&td).unwrap();
                loop {
                    let task = { tasks.lock().unwrap().pop_front() };
                    let Some(task) = task else { break };
                    let src_rel = task.path.strip_prefix("/mnt/v/imazen-26").unwrap_or(&task.path);
                    let src_str = src_rel.to_string_lossy().to_string();
                    let srcfmt = task
                        .path
                        .extension()
                        .map(|e| e.to_string_lossy().to_ascii_lowercase())
                        .unwrap_or_default();
                    let srcfmt = if srcfmt == "jpeg" { "jpg".to_string() } else { srcfmt };
                    let Some(mut img) = decode_any(&task.path) else {
                        eprintln!("skip (decode): {src_str}");
                        continue;
                    };
                    // launder JPEG history out of jpg sources: non-integer
                    // downscale destroys the old grid + restores HF density
                    if srcfmt == "jpg" {
                        let (nw, nh) = (img.w * 5 / 8, img.h * 5 / 8);
                        if nw >= 64 && nh >= 64 {
                            img = resize_rgb8(&img, nw, nh, zenresize::Filter::CatmullRom);
                        }
                    }
                    let Some(base) = center_crop(&img, cap) else {
                        eprintln!("skip (small): {src_str}");
                        continue;
                    };
                    if base.w < cap || base.h < cap {
                        // undersized sources would reintroduce the size leak
                        // for gen1/gen2a; keep the size grid clean
                        eprintln!("skip (undersize {}x{}): {src_str}", base.w, base.h);
                        continue;
                    }
                    drop(img);
                    let pick = |tag: &str, arr_len: usize| -> usize {
                        // tag first: differences hit the hash state early
                        (fnv(&format!("{tag}|{src_str}")) % arr_len as u64) as usize
                    };
                    let mut rows: Vec<String> = Vec::new();
                    let mut emit =
                        |cls: &str, sp: f64, enc1: &str, q1: u32, enc2: &str, q2: u32,
                         scale: f64, filt: &str, jpg: &Path| {
                            let Ok(data) = std::fs::read(jpg) else { return };
                            let Some(f) = extract_features(&data) else {
                                eprintln!("skip (feat): {src_str} {cls} {q1}->{q2}");
                                return;
                            };
                            rows.push(format!(
                                "{}\t{}\t{}\t{}\t{cls}\t{sp:.2}\t{enc1}\t{q1}\t{enc2}\t{q2}\t{scale:.2}\t{filt}\t{}",
                                task.split, task.sub, src_str, srcfmt, f.row()
                            ));
                        };
                    // pristine pre-scaled bases (shared by all classes)
                    let mut bases: Vec<(f64, PathBuf)> = Vec::new();
                    for (si, &sp) in PRESCALES.iter().enumerate() {
                        let p = td.join(format!("b{si}.ppm"));
                        if sp >= 0.999 {
                            write_ppm(&base, &p);
                        } else {
                            let (fw, fh) = (
                                (base.w as f64 * sp) as usize,
                                (base.h as f64 * sp) as usize,
                            );
                            let (_, filt) = FILTERS[pick(&format!("pre{si}"), FILTERS.len())];
                            let r = resize_rgb8(&base, fw, fh, filt);
                            write_ppm(&r, &p);
                        }
                        bases.push((sp, p));
                    }
                    // gen1: pre-scale x q
                    for (i, &q) in GEN1_QS.iter().enumerate() {
                        let (sp, bp) = &bases[pick(&format!("g1sp{i}"), bases.len())];
                        let enc = &encoders[pick(&format!("g1e{i}"), encoders.len())];
                        let jpg = td.join("g1.jpg");
                        if encode(bp, &jpg, enc, q) {
                            emit("gen1", *sp, enc, q, "-", 0, 0.0, "-", &jpg);
                        }
                    }
                    // gen2 aligned + resized
                    for (i, &(q1, q2)) in GEN2_PAIRS.iter().enumerate() {
                        let (sp, bp) = &bases[pick(&format!("p{i}sp"), bases.len())];
                        let enc1 = &encoders[pick(&format!("p{i}e1"), encoders.len())];
                        let enc2 = &encoders[pick(&format!("p{i}e2"), encoders.len())];
                        let j1 = td.join("s1.jpg");
                        if !encode(bp, &j1, enc1, q1) {
                            continue;
                        }
                        let Some(d1) = decode_any(&j1) else { continue };
                        // aligned
                        let pa = td.join("a.ppm");
                        write_ppm(&d1, &pa);
                        let j2 = td.join("a.jpg");
                        if encode(&pa, &j2, enc2, q2) {
                            emit("gen2a", *sp, enc1, q1, enc2, q2, 1.0, "-", &j2);
                        }
                        // resized
                        let scale = SCALES[pick(&format!("p{i}s"), SCALES.len())];
                        let (fname, filt) = FILTERS[pick(&format!("p{i}f"), FILTERS.len())];
                        let (nw, nh) = (
                            ((d1.w as f64 * scale) as usize).max(64),
                            ((d1.h as f64 * scale) as usize).max(64),
                        );
                        let dr = resize_rgb8(&d1, nw, nh, filt);
                        let pr = td.join("r.ppm");
                        write_ppm(&dr, &pr);
                        let j3 = td.join("r.jpg");
                        if encode(&pr, &j3, enc2, q2) {
                            emit("gen2r", *sp, enc1, q1, enc2, q2, scale, fname, &j3);
                        }
                        if keep {
                            for (n, src) in [("a", &j2), ("r", &j3)] {
                                let _ = std::fs::copy(
                                    src,
                                    td.join(format!(
                                        "keep_{n}_{}_{q1}_{q2}.jpg",
                                        fnv(&src_str) % 10000
                                    )),
                                );
                            }
                        }
                    }
                    {
                        let mut wl = wtr.lock().unwrap();
                        for r in rows {
                            writeln!(wl, "{r}").unwrap();
                        }
                        wl.flush().unwrap();
                    }
                    let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    eprintln!("[{d}] {} {src_str}", task.split);
                }
            });
        }
    });
    if !keep {
        let _ = std::fs::remove_dir_all(&scratch);
    }
    eprintln!("done -> {}", out_path.display());
}
