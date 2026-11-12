//! Integration tests for `zensr` strided / `zenpixels` bridges, borrowed
//! `JpegDctBlocks` views, and zero-allocation `RepairWorkspace` execution.
#![forbid(unsafe_code)]

use std::sync::Mutex;
use zensr::api::{
    Budget, Consistency, Intent, NotCertified, Outcome, Provenance, RepairWorkspace, Restorer,
    Routing, SkipReason, WorkspaceOutcome,
};
use zensr::source::{
    Category, Coefficients, CoefficientsView, Component, DecodeMode, Dimensions, EncodedRgb8,
    JpegDctBlocks, JpegInfo, Needs, Quality, Source,
};
use zensr::{AdoptedModel, RestoreError};

#[global_allocator]
static ALLOC: zenbench::HeapProfiler = zenbench::HeapProfiler::system();

static SERIAL: Mutex<()> = Mutex::new(());

fn load_canon4() -> AdoptedModel {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/adopted/dejpeg_rt24i_canon4");
    let raw = std::fs::read(root.join("weights_f16.raw")).expect("canon4 weights present");
    let weights = zensr::decode_all_f16(&raw);
    AdoptedModel::load_compact(&weights, 24, 8, 1).expect("valid canon4 compact weights")
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

/// Synthetic zero-allocation `Source` exercising strided RGB8 input, both
/// `Plain` and `Deblocked` modes, 4:2:0 / 4:4:4 / 4:2:2 coefficient lattices,
/// and both `coefficients` (legacy `Vec`) and `coefficients_view` (zero-alloc).
struct MockJpegSource {
    info: JpegInfo,
    plain_strided: Vec<u8>,
    deblocked_strided: Vec<u8>,
    stride: usize,
    y_coeffs: Vec<i16>,
    cb_coeffs: Vec<i16>,
    cr_coeffs: Vec<i16>,
    y_bw: usize,
    y_bh: usize,
    c_bw: usize,
    c_bh: usize,
    y_samp: (u8, u8),
    c_samp: (u8, u8),
    quant_tables: [Option<[u16; 64]>; 2],
}

impl MockJpegSource {
    fn new(
        (width, height): (u32, u32),
        stride_pad_bytes: usize,
        encoder: &str,
        scale: &str,
        quality: f32,
        subsampling: &str,
        recorded: bool,
    ) -> Self {
        let (w, h) = (width as usize, height as usize);
        let stride = w * 3 + stride_pad_bytes;
        let mut plain_strided = vec![0xAAu8; h * stride];
        let mut deblocked_strided = vec![0x55u8; h * stride];
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 13 + y * 7 + 25) % 220 + 16) as u8;
                let g = ((x * 5 + y * 19 + 40) % 210 + 20) as u8;
                let b = ((x * 17 + y * 3 + 60) % 200 + 24) as u8;
                let off = y * stride + x * 3;
                plain_strided[off] = r;
                plain_strided[off + 1] = g;
                plain_strided[off + 2] = b;
                deblocked_strided[off] = r.saturating_add(1);
                deblocked_strided[off + 1] = g;
                deblocked_strided[off + 2] = b.saturating_sub(1);
            }
        }

        let ((y_samp, c_samp), (y_bw, y_bh), (c_bw, c_bh)) = match subsampling {
            "S444" => {
                let bw = w.div_ceil(8);
                let bh = h.div_ceil(8);
                (((1, 1), (1, 1)), (bw, bh), (bw, bh))
            }
            "S422" => {
                let c_bw = w.div_ceil(16);
                let c_bh = h.div_ceil(8);
                (((2, 1), (1, 1)), (c_bw * 2, c_bh), (c_bw, c_bh))
            }
            _ => {
                let c_bw = w.div_ceil(16);
                let c_bh = h.div_ceil(16);
                (((2, 2), (1, 1)), (c_bw * 2, c_bh * 2), (c_bw, c_bh))
            }
        };

        let mut y_coeffs = vec![0i16; y_bw * y_bh * 64];
        for (bi, blk) in y_coeffs.as_chunks_mut::<64>().0.iter_mut().enumerate() {
            blk[0] = ((bi as i16 % 17) - 8) * 3;
            blk[1] = (bi as i16 % 5) - 2;
            blk[2] = 1 - (bi as i16 % 3);
        }
        let mut cb_coeffs = vec![0i16; c_bw * c_bh * 64];
        let mut cr_coeffs = vec![0i16; c_bw * c_bh * 64];
        for (bi, blk) in cb_coeffs.as_chunks_mut::<64>().0.iter_mut().enumerate() {
            blk[0] = (bi as i16 % 7) - 3;
            blk[1] = 1;
        }
        for (bi, blk) in cr_coeffs.as_chunks_mut::<64>().0.iter_mut().enumerate() {
            blk[0] = 2 - (bi as i16 % 5);
            blk[2] = -1;
        }

        let mut qt_luma = [12u16; 64];
        let mut qt_chroma = [18u16; 64];
        for (i, q) in qt_luma.iter_mut().enumerate() {
            *q = (8 + (i % 24)) as u16;
        }
        for (i, q) in qt_chroma.iter_mut().enumerate() {
            *q = (12 + (i % 28)) as u16;
        }

        Self {
            info: JpegInfo {
                dimensions: Dimensions { width, height },
                encoder: Category(encoder.to_owned()),
                subsampling: Category(subsampling.to_owned()),
                quality: Quality {
                    value: quality,
                    scale: Category(scale.to_owned()),
                },
                num_components: 3,
                recorded,
                jpegli_aq_off: false,
                encoded_len: (w * h) / 2 + 128,
            },
            plain_strided,
            deblocked_strided,
            stride,
            y_coeffs,
            cb_coeffs,
            cr_coeffs,
            y_bw,
            y_bh,
            c_bw,
            c_bh,
            y_samp,
            c_samp,
            quant_tables: [Some(qt_luma), Some(qt_chroma)],
        }
    }

    fn component_array(&self) -> [Component<'_>; 3] {
        [
            Component {
                coeffs: &self.y_coeffs,
                blocks_wide: self.y_bw,
                blocks_high: self.y_bh,
                quant_table_idx: 0,
                h_samp: self.y_samp.0,
                v_samp: self.y_samp.1,
            },
            Component {
                coeffs: &self.cb_coeffs,
                blocks_wide: self.c_bw,
                blocks_high: self.c_bh,
                quant_table_idx: 1,
                h_samp: self.c_samp.0,
                v_samp: self.c_samp.1,
            },
            Component {
                coeffs: &self.cr_coeffs,
                blocks_wide: self.c_bw,
                blocks_high: self.c_bh,
                quant_table_idx: 1,
                h_samp: self.c_samp.0,
                v_samp: self.c_samp.1,
            },
        ]
    }
}

impl Source for MockJpegSource {
    fn info(&self) -> &JpegInfo {
        &self.info
    }

    fn prepare(&mut self, _needs: Needs) -> Result<(), RestoreError> {
        Ok(())
    }

    fn pixels(&self, mode: DecodeMode) -> Result<EncodedRgb8<'_>, RestoreError> {
        let data = match mode {
            DecodeMode::Plain => &self.plain_strided,
            DecodeMode::Deblocked => &self.deblocked_strided,
        };
        EncodedRgb8::new_strided(
            data,
            self.info.dimensions.width,
            self.info.dimensions.height,
            self.stride,
        )
        .map_err(RestoreError::UnsupportedPixels)
    }

    fn coefficients(&self) -> Result<Coefficients<'_>, RestoreError> {
        Ok(Coefficients {
            components: self.component_array().to_vec(),
            quant_tables: &self.quant_tables,
        })
    }

    fn coefficients_view(&self) -> Result<CoefficientsView<'_>, RestoreError> {
        let comps = self.component_array();
        Ok(CoefficientsView::new(&comps, &self.quant_tables))
    }
}

#[test]
fn jpeg_dct_blocks_and_zigzag_quant_tables_match_contract() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let src = MockJpegSource::new(
        (29, 19),
        11,
        "LibjpegTurbo",
        "IjgQuality",
        42.0,
        "S420",
        false,
    );
    let coeffs = src.coefficients().unwrap();
    let y = coeffs.components[0];
    let blocks = y.dct_blocks().unwrap();
    assert_eq!(blocks.blocks_wide, src.y_bw);
    assert_eq!(blocks.blocks_high, src.y_bh);
    assert_eq!(blocks.coeffs.len(), src.y_bw * src.y_bh * 64);
    assert!(blocks.quant_table.is_none());

    // Verify natural vs zigzag permutation at known indices:
    // zigzag[0]=0, zigzag[1]=1, zigzag[2]=8 (row 1, col 0), zigzag[3]=16 (row 2, col 0).
    let mut nat_qt = [0u16; 64];
    for (i, q) in nat_qt.iter_mut().enumerate() {
        *q = 100 + i as u16;
    }
    let qt_slice = [Some(nat_qt)];
    let c_view = CoefficientsView::new(&[y], &qt_slice);
    let zz = c_view.quant_table_zigzag(0).unwrap();
    assert_eq!(zz[0], 100);
    assert_eq!(zz[1], 101);
    assert_eq!(zz[2], 108);
    assert_eq!(zz[3], 116);
    assert_eq!(zz[63], 163);

    let blocks_q = blocks.with_quant_table(&zz);
    assert_eq!(blocks_q.quant_table, Some(&zz));
    let roundtrip = Component::from_dct_blocks(blocks_q, y.quant_table_idx, y.h_samp, y.v_samp);
    assert_eq!(roundtrip.blocks_wide, y.blocks_wide);
    assert_eq!(roundtrip.blocks_high, y.blocks_high);
    assert!(JpegDctBlocks::new(&y.coeffs[..63], 1, 1).is_err());
}

#[test]
fn encoded_rgb8_and_strided_writers_preserve_padding_and_match_packed() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let model = load_canon4();
    let restorer = Restorer::new().with_tier(Budget::Realtime, model);
    let mut src = MockJpegSource::new(
        (23, 17),
        13,
        "Mozjpeg",
        "MozjpegQuality",
        35.0,
        "S420",
        true,
    );
    let outcome = restorer
        .restore(&mut src)
        .routing(Routing::Always)
        .threads(1)
        .run()
        .unwrap();

    let px = outcome.pixels();
    let packed_u8 = px.to_rgb8();
    let packed_f32 = px.to_rgb_f32();

    // Write into padded strided buffers filled with sentinels; padding must stay untouched.
    let (w, h) = (px.width(), px.height());
    let stride_u8 = w * 3 + 19;
    let mut strided_u8 = vec![0xDEu8; h * stride_u8];
    assert_zero_alloc("write_rgb8_strided", || {
        px.write_rgb8_strided(&mut strided_u8, stride_u8).unwrap();
    });
    for y in 0..h {
        let row = &strided_u8[y * stride_u8..y * stride_u8 + w * 3];
        assert_eq!(row, &packed_u8[y * w * 3..(y + 1) * w * 3]);
        let pad = &strided_u8[y * stride_u8 + w * 3..(y + 1) * stride_u8];
        assert!(pad.iter().all(|&b| b == 0xDE));
    }

    let stride_f32 = w * 3 + 7;
    let sentinel = f32::from_bits(0x7fc0_1234);
    let mut strided_f32 = vec![sentinel; h * stride_f32];
    assert_zero_alloc("write_rgb_f32_strided", || {
        px.write_rgb_f32_strided(&mut strided_f32, stride_f32)
            .unwrap();
    });
    for y in 0..h {
        let row = &strided_f32[y * stride_f32..y * stride_f32 + w * 3];
        assert_bits_eq(row, &packed_f32[y * w * 3..(y + 1) * w * 3], "strided f32");
        for &p in &strided_f32[y * stride_f32 + w * 3..(y + 1) * stride_f32] {
            assert_eq!(p.to_bits(), sentinel.to_bits());
        }
    }

    assert!(px.write_rgb8_strided(&mut strided_u8, w * 3 - 1).is_err());
    assert!(EncodedRgb8::new_strided(&strided_u8, w as u32, h as u32, w * 3 - 1).is_err());
}

#[test]
fn repair_workspace_is_bit_exact_and_zero_alloc_across_outcomes() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let model = load_canon4();

    // Pre-size workspace for up to 96x96; even the FIRST ("cold") frame should allocate 0 bytes!
    let mut presized_ws = RepairWorkspace::with_capacity(96, 96, &model, 0);
    let restorer = Restorer::new().with_tier(Budget::Realtime, model);

    // Case 1: 4:2:0 odd partial-block dimensions (37x29), recorded metadata, Auto routing.
    let mut src_ref =
        MockJpegSource::new((37, 29), 9, "Mozjpeg", "MozjpegQuality", 32.0, "S420", true);
    // Independent split-buffer oracle using legacy allocating zensr-micro pipeline:
    let split_oracle = {
        let dec = src_ref.pixels(DecodeMode::Plain).unwrap();
        let (w, h) = (37usize, 29usize);
        let plane = w * h;
        let mut rgb = vec![0.0f32; 3 * plane];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    rgb[c * plane + y * w + x] =
                        dec.data[y * dec.stride + x * 3 + c] as f32 / 255.0;
                }
            }
        }
        let oracle_model = load_canon4();
        let mut sr = oracle_model.upscale_tiled(&rgb, h, w, 1, 0);
        zensr_micro::guards::guarded_merge(&mut sr, &rgb, h, w, 1, &zensr::GuardConfig::default());
        let (mut y_p, mut cb_p, mut cr_p) = (vec![0.0; plane], vec![0.0; plane], vec![0.0; plane]);
        zensr_micro::consist::rgb_to_ycbcr_planes(&sr, plane, &mut y_p, &mut cb_p, &mut cr_p);
        let coeffs = src_ref.coefficients().unwrap();
        let pcfg = zensr::ProjectionConfig::with_slack_q(zensr::slack_for(src_ref.info()))
            .with_slack_abs(zensr::slack_abs_for(src_ref.info()));
        let cv_y = zensr_micro::consist::CoeffView {
            coeffs: coeffs.components[0].coeffs,
            blocks_wide: coeffs.components[0].blocks_wide,
            blocks_high: coeffs.components[0].blocks_high,
            order: zensr_micro::consist::CoeffOrder::Zigzag,
            quant: coeffs.quant_table_natural(0).unwrap(),
        };
        let cv_cb = zensr_micro::consist::CoeffView {
            coeffs: coeffs.components[1].coeffs,
            blocks_wide: coeffs.components[1].blocks_wide,
            blocks_high: coeffs.components[1].blocks_high,
            order: zensr_micro::consist::CoeffOrder::Zigzag,
            quant: coeffs.quant_table_natural(1).unwrap(),
        };
        let cv_cr = zensr_micro::consist::CoeffView {
            coeffs: coeffs.components[2].coeffs,
            blocks_wide: coeffs.components[2].blocks_wide,
            blocks_high: coeffs.components[2].blocks_high,
            order: zensr_micro::consist::CoeffOrder::Zigzag,
            quant: coeffs.quant_table_natural(1).unwrap(),
        };
        zensr_micro::consist::project_plane(&mut y_p, w, h, &cv_y, &pcfg);
        zensr_micro::consist::project_chroma_420(&mut cb_p, w, h, &cv_cb, &pcfg);
        zensr_micro::consist::project_chroma_420(&mut cr_p, w, h, &cv_cr, &pcfg);
        let mut out_rgb = vec![0.0f32; 3 * plane];
        zensr_micro::consist::ycbcr_to_rgb_planes(&y_p, &cb_p, &cr_p, &mut out_rgb, plane);
        out_rgb
    };
    let ref_out = restorer
        .restore(&mut src_ref)
        .routing(Routing::Auto { min_gain: 0.25 })
        .threads(1)
        .run()
        .unwrap();
    assert_bits_eq(
        ref_out.pixels().planes(),
        &split_oracle,
        "Request::run() vs split-buffer zensr-micro oracle",
    );
    assert!(matches!(
        ref_out,
        Outcome::Restored {
            consistency: Consistency::Certified,
            ..
        }
    ));

    let mut src_ws =
        MockJpegSource::new((37, 29), 9, "Mozjpeg", "MozjpegQuality", 32.0, "S420", true);
    let mut caller_rgb8 = vec![0u8; 29 * (37 * 3 + 8)];
    let mut caller_f32 = vec![0.0f32; 29 * (37 * 3 + 4)];

    assert_zero_alloc(
        "cold presized 4:2:0 run_into_workspace + strided write",
        || {
            let ws_out = restorer
                .restore(&mut src_ws)
                .routing(Routing::Auto { min_gain: 0.25 })
                .threads(1)
                .run_into_workspace(&mut presized_ws)
                .unwrap();
            assert!(matches!(
                ws_out,
                WorkspaceOutcome::Restored {
                    consistency: Consistency::Certified,
                    ..
                }
            ));
            assert_bits_eq(
                ws_out.pixels().planes(),
                ref_out.pixels().planes(),
                "presized 4:2:0 planes",
            );
            assert_eq!(ws_out.report().encoder_family, "Mozjpeg (recorded)");
            assert_eq!(
                ws_out.report().clamped_fraction,
                ref_out.report().clamped_fraction
            );
            ws_out
                .pixels()
                .write_rgb8_strided(&mut caller_rgb8, 37 * 3 + 8)
                .unwrap();
            ws_out
                .pixels()
                .write_rgb_f32_strided(&mut caller_f32, 37 * 3 + 4)
                .unwrap();
        },
    );

    // Case 2: Warm lazy `RepairWorkspace::new()` across multi-tile 4:4:4 + Annex-K deblock q<=9.
    let mut lazy_ws = RepairWorkspace::new();
    let mut src_444_ref = MockJpegSource::new(
        (73, 67),
        5,
        "LibjpegTurbo",
        "IjgQuality",
        8.0,
        "S444",
        false,
    );
    let ref_444 = restorer
        .restore(&mut src_444_ref)
        .intent(Intent::Conservative)
        .threads(1)
        .run()
        .unwrap();
    assert!(ref_444.report().deblocked);

    let mut src_444 = MockJpegSource::new(
        (73, 67),
        5,
        "LibjpegTurbo",
        "IjgQuality",
        8.0,
        "S444",
        false,
    );
    // Warm-up run on lazy_ws:
    {
        let warm = restorer
            .restore(&mut src_444)
            .intent(Intent::Conservative)
            .threads(1)
            .run_into_workspace(&mut lazy_ws)
            .unwrap();
        assert_bits_eq(
            warm.pixels().planes(),
            ref_444.pixels().planes(),
            "4:4:4 warmup parity",
        );
    }
    // Second run on lazy_ws must allocate 0 bytes on the heap!
    let mut out_444_u8 = vec![0u8; 67 * 73 * 3];
    assert_zero_alloc("warm lazy_ws 4:4:4 Conservative deblocked", || {
        let out = restorer
            .restore(&mut src_444)
            .intent(Intent::Conservative)
            .threads(1)
            .run_into_workspace(&mut lazy_ws)
            .unwrap();
        assert!(out.applied());
        assert!(out.report().deblocked);
        assert_bits_eq(
            out.pixels().planes(),
            ref_444.pixels().planes(),
            "4:4:4 zero-alloc parity",
        );
        out.pixels()
            .write_rgb8_strided(&mut out_444_u8, 73 * 3)
            .unwrap();
    });
    assert_eq!(out_444_u8, ref_444.pixels().to_rgb8());

    // Also verify `Request::workspace(&mut lazy_ws).run()` matches bit-for-bit.
    let via_builder = restorer
        .restore(&mut src_444)
        .intent(Intent::Conservative)
        .threads(1)
        .workspace(&mut lazy_ws)
        .run()
        .unwrap();
    assert_bits_eq(
        via_builder.pixels().planes(),
        ref_444.pixels().planes(),
        "Request::workspace().run() parity",
    );

    // Case 3: Unchanged outcomes (`NearPristine`, `NotWorthIt`, `ConsistencyUnavailable`) are 0-alloc!
    let mut src_hq = MockJpegSource::new(
        (32, 24),
        0,
        "LibjpegTurbo",
        "IjgQuality",
        97.0,
        "S420",
        false,
    );
    assert_zero_alloc("warm NearPristine skip", || {
        let out = restorer
            .restore(&mut src_hq)
            .routing(Routing::Always)
            .provenance(Provenance::FreshEncode)
            .threads(1)
            .run_into_workspace(&mut lazy_ws)
            .unwrap();
        assert!(matches!(
            out,
            WorkspaceOutcome::Unchanged {
                why: SkipReason::NearPristine,
                ..
            }
        ));
    });

    assert_zero_alloc("warm Routing::Never skip", || {
        let out = restorer
            .restore(&mut src_444)
            .routing(Routing::Never)
            .threads(1)
            .run_into_workspace(&mut lazy_ws)
            .unwrap();
        assert!(matches!(
            out,
            WorkspaceOutcome::Unchanged {
                why: SkipReason::NotWorthIt { .. },
                ..
            }
        ));
    });

    let mut src_422 = MockJpegSource::new(
        (32, 24),
        0,
        "LibjpegTurbo",
        "IjgQuality",
        40.0,
        "S422",
        false,
    );
    assert_zero_alloc("warm ConsistencyUnavailable 4:2:2 skip", || {
        let out = restorer
            .restore(&mut src_422)
            .routing(Routing::Always)
            .require_consistency(true)
            .threads(1)
            .run_into_workspace(&mut lazy_ws)
            .unwrap();
        assert!(matches!(
            out,
            WorkspaceOutcome::Unchanged {
                why: SkipReason::ConsistencyUnavailable(NotCertified::ChromaGeometry),
                ..
            }
        ));
    });
}

#[cfg(feature = "zenpixels")]
#[test]
fn zenpixels_input_and_output_bridges_are_strided_and_zero_alloc() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let (w, h) = (25u32, 19u32);
    let stride = (w as usize * 3) + 12; // multiple of 3 for Rgb<u8> pixel alignment
    let mut raw = vec![0xCCu8; h as usize * stride];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let off = y * stride + x * 3;
            raw[off] = (x * 9 + 10) as u8;
            raw[off + 1] = (y * 11 + 20) as u8;
            raw[off + 2] = 140;
        }
    }

    // Input bridge: PixelSlice (typed and erased) <-> EncodedRgb8 using coding-space RGB8.
    let slice = zenpixels::PixelSlice::new(&raw, w, h, stride, zenpixels::PixelDescriptor::RGB8)
        .expect("valid strided RGB8 slice");
    let typed = slice.try_typed::<rgb::Rgb<u8>>().unwrap();
    let enc = EncodedRgb8::from(typed);
    assert_eq!(enc.dimensions(), (w, h));
    assert_eq!(enc.stride, stride);

    let back_slice = zenpixels::PixelSlice::try_from(enc).unwrap();
    assert_eq!(back_slice.descriptor(), zenpixels::PixelDescriptor::RGB8);
    assert_eq!(back_slice.stride(), stride);
    let enc2 = EncodedRgb8::try_from(back_slice).unwrap();
    assert_eq!(enc2.stride, stride);

    // Output bridge: zero-alloc write_pixel_slice (U8 + F32) and owned to_pixel_buffer_rgb8/f32.
    let model = load_canon4();
    let mut ws = RepairWorkspace::with_capacity(32, 32, &model, 0);
    let restorer = Restorer::new().with_tier(Budget::Realtime, model);
    let mut src = MockJpegSource::new(
        (w, h),
        6,
        "Cjpegli",
        "ButteraugliDistance",
        2.5,
        "S420",
        false,
    );

    let mut dst_u8_backing = vec![0xEEu8; h as usize * stride];
    let mut dst_u8_slice = zenpixels::PixelSliceMut::new(
        &mut dst_u8_backing,
        w,
        h,
        stride,
        zenpixels::PixelDescriptor::RGB8,
    )
    .unwrap();

    let mut dst_f32_buf = zenpixels::PixelBuffer::<rgb::Rgb<f32>>::try_new_typed(w, h).unwrap();

    assert_zero_alloc("zenpixels write_pixel_slice U8 + F32", || {
        let out = restorer
            .restore(&mut src)
            .routing(Routing::Always)
            .threads(1)
            .run_into_workspace(&mut ws)
            .unwrap();
        out.pixels().write_pixel_slice(&mut dst_u8_slice).unwrap();
        let mut f32_slice = dst_f32_buf.as_slice_mut();
        out.pixels().write_pixel_slice(&mut f32_slice).unwrap();
    });

    // Verify padding bytes in strided dst_u8_backing stayed 0xEE and pixels match to_pixel_buffer_rgb8.
    let full_out = restorer
        .restore(&mut src)
        .routing(Routing::Always)
        .threads(1)
        .run()
        .unwrap();
    let owned_u8 = full_out.pixels().to_pixel_buffer_rgb8().unwrap();
    let owned_f32 = full_out.pixels().to_pixel_buffer_rgb_f32().unwrap();
    assert_eq!(owned_u8.descriptor(), zenpixels::PixelDescriptor::RGB8);
    assert_eq!(owned_f32.descriptor(), zenpixels::PixelDescriptor::RGBF32);

    let owned_u8_view = owned_u8.rows(0, h);
    for y in 0..h as usize {
        assert_eq!(
            &dst_u8_backing[y * stride..y * stride + w as usize * 3],
            owned_u8_view.row(y as u32),
        );
        assert!(
            dst_u8_backing[y * stride + w as usize * 3..(y + 1) * stride]
                .iter()
                .all(|&b| b == 0xEE)
        );
    }
    let got_f32_view = dst_f32_buf.rows(0, h);
    let exp_f32_view = owned_f32.rows(0, h);
    for y in 0..h {
        assert_eq!(got_f32_view.row(y), exp_f32_view.row(y));
    }
}
