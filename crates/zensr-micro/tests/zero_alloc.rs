//! Bitwise parity and zero-allocation verification for `zensr-micro` workspaces.
#![forbid(unsafe_code)]

use std::sync::Mutex;
use zensr_micro::adopted::{AdoptedModel, ModelWorkspace};
use zensr_micro::consist::{
    project_chroma_420, project_chroma_420_with_scratch, rgb_to_ycbcr_planes,
    rgb_to_ycbcr_planes_inplace, ycbcr_to_rgb_planes, ycbcr_to_rgb_planes_inplace, ChromaScratch,
    CoeffOrder, CoeffView, ProjectionConfig,
};
use zensr_micro::guards::{guarded_merge, guarded_merge_with_scratch, GuardConfig, GuardScratch};

#[global_allocator]
static ALLOC: zenbench::HeapProfiler = zenbench::HeapProfiler::system();

/// `HeapPhase` is process-wide; serialize tests in this binary so concurrent
/// test threads never overlap a zero-allocation measurement window.
static SERIAL: Mutex<()> = Mutex::new(());

fn lcg_vec(n: usize, seed: u32, scale: f32, bias: f32) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let u = (s as f32) / (u32::MAX as f32);
            bias + (u * 2.0 - 1.0) * scale
        })
        .collect()
}

fn assert_bits_eq(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length mismatch");
    for (idx, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{context}: sample {idx} differed ({a} [{:#010x}] vs {e} [{:#010x}])",
            a.to_bits(),
            e.to_bits(),
        );
    }
}

fn assert_zero_alloc<R>(label: &str, f: impl FnOnce() -> R) -> R {
    let phase = zenbench::HeapPhase::begin().expect("HeapProfiler installed");
    let out = f();
    let stats = phase.finish().expect("HeapPhase finish");
    assert_eq!(
        (
            stats.allocations,
            stats.reallocations,
            stats.requested_bytes
        ),
        (0, 0, 0),
        "{label}: expected 0 heap allocations, got {stats:?}"
    );
    out
}

#[test]
fn compact_canon4_workspace_is_bit_exact_and_zero_alloc() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/adopted/dejpeg_rt24i_canon4");
    let weights =
        zensr_micro::decode_all_f16(&std::fs::read(root.join("weights_f16.raw")).unwrap());
    let model = AdoptedModel::load_compact(&weights, 24, 8, 1).unwrap();

    // Cold pre-sized workspace: 0 allocations even on the very first tile call.
    let (h, w) = (40usize, 36usize);
    let input = lcg_vec(3 * h * w, 101, 0.5, 0.5);
    let expected_whole = {
        let mut v = vec![0.0f32; 3 * h * w];
        model.forward(&input, h, w, &mut v);
        v
    };
    let mut presized = ModelWorkspace::with_tile_capacity(&model, h, w);
    let mut got = vec![0.0f32; 3 * h * w];
    assert_zero_alloc("presized forward_with_workspace", || {
        model.forward_with_workspace(&input, h, w, &mut got, &mut presized);
    });
    assert_bits_eq(&got, &expected_whole, "canon4 presized forward");

    // Multi-tile (tile = 32 on 77x83 -> 3x3 tiles), single-thread zero-alloc + 2-thread parity.
    let (mh, mw) = (77usize, 83usize);
    let big_in = lcg_vec(3 * mh * mw, 202, 0.5, 0.5);
    let expected_tiled = model.upscale_tiled(&big_in, mh, mw, 2, 32);
    let mut ws = ModelWorkspace::new();
    let mut tiled_out = vec![0.0f32; 3 * mh * mw];
    model.upscale_tiled_into(&big_in, mh, mw, 1, 32, &mut tiled_out, &mut ws);
    assert_bits_eq(&tiled_out, &expected_tiled, "canon4 warmup tiled");

    tiled_out.fill(-1.0);
    assert_zero_alloc("warm compact multi-tile upscale_tiled_into (1T)", || {
        model.upscale_tiled_into(&big_in, mh, mw, 1, 32, &mut tiled_out, &mut ws);
    });
    assert_bits_eq(&tiled_out, &expected_tiled, "canon4 warm 1T multi-tile");

    // Smaller single-tile image on the already-warmed workspace stays 0-alloc.
    let (sh, sw) = (17usize, 13usize);
    let small_in = lcg_vec(3 * sh * sw, 303, 0.5, 0.5);
    let expected_small = model.upscale_tiled(&small_in, sh, sw, 1, 0);
    let mut small_out = vec![0.0f32; 3 * sh * sw];
    assert_zero_alloc("warm compact single-tile upscale_tiled_into", || {
        model.upscale_tiled_into(&small_in, sh, sw, 1, 0, &mut small_out, &mut ws);
    });
    assert_bits_eq(&small_out, &expected_small, "canon4 warm 1T single-tile");

    // Multi-threaded workspace execution matches bit-for-bit as well.
    let mut mt_out = vec![0.0f32; 3 * mh * mw];
    model.upscale_tiled_into(&big_in, mh, mw, 2, 32, &mut mt_out, &mut ws);
    assert_bits_eq(&mt_out, &expected_tiled, "canon4 2T upscale_tiled_into");
}

#[test]
fn span48_workspace_is_bit_exact_and_zero_alloc() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let fc = 48usize;
    let scale = 2usize;
    let s2 = scale * scale;
    let n_weights = (3 * fc * 9 + fc)
        + 18 * (fc * fc * 9 + fc)
        + (fc * fc * 9 + fc)
        + (4 * fc * fc + fc)
        + (fc * 3 * s2 * 9 + 3 * s2);
    let raw = lcg_vec(n_weights, 404, 0.02, 0.0);
    let model = AdoptedModel::load_span48(&raw, scale).unwrap();

    let (h, w) = (41usize, 37usize);
    let input = lcg_vec(3 * h * w, 505, 0.45, 0.5);
    let expected = model.upscale_tiled(&input, h, w, 2, 32);

    let mut ws = ModelWorkspace::new();
    let mut out = vec![0.0f32; 3 * (h * scale) * (w * scale)];
    model.upscale_tiled_into(&input, h, w, 1, 32, &mut out, &mut ws);
    assert_bits_eq(&out, &expected, "span48 warmup multi-tile");

    out.fill(0.0);
    assert_zero_alloc("warm span48 multi-tile upscale_tiled_into", || {
        model.upscale_tiled_into(&input, h, w, 1, 32, &mut out, &mut ws);
    });
    assert_bits_eq(&out, &expected, "span48 warm multi-tile");
}

#[test]
fn guard_scratch_is_bit_exact_and_zero_alloc_including_fallback() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let (h, w, s) = (29usize, 31usize, 2usize);
    let lr = lcg_vec(3 * h * w, 606, 0.45, 0.5);
    let sr_init = lcg_vec(3 * h * w * s * s, 707, 0.45, 0.5);

    // 1. Standard guarded merge with texture gating active.
    let cfg = GuardConfig::default();
    let mut expected_sr = sr_init.clone();
    let expected_rep = guarded_merge(&mut expected_sr, &lr, h, w, s, &cfg);

    let mut scratch = GuardScratch::new();
    let mut actual_sr = sr_init.clone();
    let _ = guarded_merge_with_scratch(&mut actual_sr, &lr, h, w, s, &cfg, &mut scratch);

    actual_sr.copy_from_slice(&sr_init);
    let actual_rep = assert_zero_alloc("warm guarded_merge_with_scratch", || {
        guarded_merge_with_scratch(&mut actual_sr, &lr, h, w, s, &cfg, &mut scratch)
    });
    assert_bits_eq(&actual_sr, &expected_sr, "guarded_merge texture_gate=true");
    assert_eq!(
        actual_rep.clamped_frac.to_bits(),
        expected_rep.clamped_frac.to_bits()
    );
    assert_eq!(
        actual_rep.mean_alpha.to_bits(),
        expected_rep.mean_alpha.to_bits()
    );
    assert_eq!(
        actual_rep.roundtrip_mae.to_bits(),
        expected_rep.roundtrip_mae.to_bits()
    );
    assert_eq!(
        actual_rep.fallback_blend.to_bits(),
        expected_rep.fallback_blend.to_bits()
    );

    // 2. Adversarial SR output with texture_gate=false so both alpha reset and
    //    round-trip fallback blend execute with 0 allocations.
    let mut cfg_fb = GuardConfig::default();
    cfg_fb.residual_clamp = f32::INFINITY;
    cfg_fb.texture_gate = false;
    cfg_fb.roundtrip_mae_max = 0.05;

    let garbage = vec![0.95f32; 3 * h * w * s * s];
    let mut expected_fb = garbage.clone();
    let expected_fb_rep = guarded_merge(&mut expected_fb, &lr, h, w, s, &cfg_fb);
    assert!(expected_fb_rep.fallback_blend > 0.0);

    let mut actual_fb = garbage;
    let actual_fb_rep = assert_zero_alloc("warm guarded_merge fallback", || {
        guarded_merge_with_scratch(&mut actual_fb, &lr, h, w, s, &cfg_fb, &mut scratch)
    });
    assert_bits_eq(&actual_fb, &expected_fb, "guarded_merge fallback");
    assert_eq!(
        actual_fb_rep.fallback_blend.to_bits(),
        expected_fb_rep.fallback_blend.to_bits()
    );
    assert_eq!(actual_fb_rep.mean_alpha, 1.0);
}

#[test]
fn chroma_scratch_and_inplace_color_planes_are_bit_exact_and_zero_alloc() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let (h, w) = (19usize, 27usize);
    let plane = h * w;
    let rgb = lcg_vec(3 * plane, 808, 0.5, 0.5);

    // Out-of-place vs in-place RGB -> YCbCr -> RGB bit parity + 0 allocs.
    let mut y = vec![0.0f32; plane];
    let mut cb = vec![0.0f32; plane];
    let mut cr = vec![0.0f32; plane];
    rgb_to_ycbcr_planes(&rgb, plane, &mut y, &mut cb, &mut cr);

    let mut inplace = rgb.clone();
    assert_zero_alloc("rgb_to_ycbcr_planes_inplace", || {
        rgb_to_ycbcr_planes_inplace(&mut inplace, plane);
    });
    assert_bits_eq(&inplace[..plane], &y, "inplace Y");
    assert_bits_eq(&inplace[plane..2 * plane], &cb, "inplace Cb");
    assert_bits_eq(&inplace[2 * plane..3 * plane], &cr, "inplace Cr");

    let mut rgb_ref = vec![0.0f32; 3 * plane];
    ycbcr_to_rgb_planes(&y, &cb, &cr, &mut rgb_ref, plane);
    assert_zero_alloc("ycbcr_to_rgb_planes_inplace", || {
        ycbcr_to_rgb_planes_inplace(&mut inplace, plane);
    });
    assert_bits_eq(&inplace, &rgb_ref, "inplace RGB roundtrip");

    // 4:2:0 chroma projection bit parity + 0 warm allocs on odd dimensions.
    let (hw, hh) = (w.div_ceil(2), h.div_ceil(2));
    let (bw, bh) = (hw.div_ceil(8), hh.div_ceil(8));
    let coeffs = vec![0i16; bw * bh * 64];
    let quant = [16u16; 64];
    let cv = CoeffView {
        coeffs: &coeffs,
        blocks_wide: bw,
        blocks_high: bh,
        order: CoeffOrder::Zigzag,
        quant: &quant,
    };
    let cfg = ProjectionConfig::with_slack_q(0.1).with_slack_abs(1.0);

    let mut expected_cb = cb.clone();
    let expected_rep = project_chroma_420(&mut expected_cb, w, h, &cv, &cfg);

    let mut scratch = ChromaScratch::new();
    let mut actual_cb = cb.clone();
    let _ = project_chroma_420_with_scratch(&mut actual_cb, w, h, &cv, &cfg, &mut scratch);

    actual_cb.copy_from_slice(&cb);
    let actual_rep = assert_zero_alloc("warm project_chroma_420_with_scratch", || {
        project_chroma_420_with_scratch(&mut actual_cb, w, h, &cv, &cfg, &mut scratch)
    });
    assert_bits_eq(&actual_cb, &expected_cb, "project_chroma_420_with_scratch");
    assert_eq!(
        actual_rep.clamped_frac.to_bits(),
        expected_rep.clamped_frac.to_bits()
    );
    assert_eq!(
        actual_rep.mean_abs_change.to_bits(),
        expected_rep.mean_abs_change.to_bits()
    );
}
