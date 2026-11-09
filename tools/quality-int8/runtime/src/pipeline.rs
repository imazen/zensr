//! Shared, unchanged quality-study decoding, RGB8 conversion and projection.
use sha2::{Digest, Sha256};
use zensr_micro::{consist::*, guards::guarded_merge};
use zensr_zenjpeg::{ProjectionConfig, RestoreConfig};

pub(crate) fn hash(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}
pub(crate) fn decode(jpeg: &[u8], auto: bool) -> (Vec<f32>, usize, usize) {
    let mode = if auto {
        zenjpeg::decoder::DeblockMode::Auto
    } else {
        zenjpeg::decoder::DeblockMode::Off
    };
    let d = zenjpeg::decoder::Decoder::new()
        .deblock(mode)
        .decode(jpeg, enough::Unstoppable)
        .unwrap();
    let (w, h) = d.dimensions();
    let (w, h) = (w as usize, h as usize);
    let px = d.pixels_u8().unwrap();
    assert_eq!(px.len(), 3 * w * h);
    let n = w * h;
    let p = (0..3)
        .flat_map(|c| (0..n).map(move |i| px[3 * i + c] as f32 / 255.))
        .collect();
    (p, w, h)
}
// Same RGB guard/projection steps as zensr-zenjpeg at 3fbd9a5e. Native
// full-pipeline replay below checks the tool adapter before quality evaluation.
pub(crate) fn finish(
    mut out: Vec<f32>,
    input: &[f32],
    jpg: &[u8],
    probe: &zenjpeg::detect::JpegProbe,
    w: usize,
    h: usize,
) -> Vec<f32> {
    let n = w * h;
    guarded_merge(&mut out, input, h, w, 1, &RestoreConfig::default().guard);
    let (mut y, mut cb, mut cr) = (vec![0.; n], vec![0.; n], vec![0.; n]);
    rgb_to_ycbcr_planes(&out, n, &mut y, &mut cb, &mut cr);
    let mut ycc = y;
    ycc.extend(cb);
    ycc.extend(cr);
    let coeff = zenjpeg::decoder::Decoder::new()
        .decode_coefficients(jpg, enough::Unstoppable)
        .unwrap();
    let maxh = coeff.components.iter().map(|c| c.h_samp).max().unwrap();
    let maxv = coeff.components.iter().map(|c| c.v_samp).max().unwrap();
    let cfg = ProjectionConfig::with_slack_q(zensr_zenjpeg::slack_for(probe))
        .with_slack_abs(zensr_zenjpeg::slack_abs_for(probe));
    for (ci, c) in coeff.components.iter().enumerate().take(3) {
        let q = coeff.quant_tables[c.quant_table_idx as usize].unwrap();
        let view = CoeffView {
            coeffs: &c.coeffs,
            blocks_wide: c.blocks_wide,
            blocks_high: c.blocks_high,
            order: CoeffOrder::Zigzag,
            quant: &q,
        };
        let plane = &mut ycc[ci * n..(ci + 1) * n];
        if c.h_samp == maxh && c.v_samp == maxv {
            project_plane(plane, w, h, &view, &cfg);
        } else if c.h_samp * 2 == maxh && c.v_samp * 2 == maxv {
            project_chroma_420(plane, w, h, &view, &cfg);
        } else {
            panic!("unsupported study sampling");
        }
    }
    ycbcr_to_rgb_planes(&ycc[..n], &ycc[n..2 * n], &ycc[2 * n..], &mut out, n);
    out
}
pub(crate) fn metrics(reference: &[u8], pixels: &[u8], w: usize, h: usize) -> (f64, f64, f64) {
    use rgb::FromSlice;
    let a: Vec<[u8; 3]> = reference.as_chunks::<3>().0.to_vec();
    let b: Vec<[u8; 3]> = pixels.as_chunks::<3>().0.to_vec();
    let s = fast_ssim2::compute_ssimulacra2(
        imgref::Img::new(a, w, h).as_ref(),
        imgref::Img::new(b, w, h).as_ref(),
    )
    .unwrap();
    let d = butteraugli::butteraugli(
        imgref::Img::new(reference.as_rgb(), w, h),
        imgref::Img::new(pixels.as_rgb(), w, h),
        &butteraugli::ButteraugliParams::default(),
    )
    .unwrap();
    assert!(s.is_finite() && d.score.is_finite() && d.pnorm_3.is_finite());
    (s, d.score, d.pnorm_3)
}
pub(crate) fn rgb8(input: &[f32], w: usize, h: usize) -> Vec<u8> {
    let n = w * h;
    (0..n)
        .flat_map(|i| (0..3).map(move |c| (input[c * n + i] * 255. + 0.5).clamp(0., 255.) as u8))
        .collect()
}
