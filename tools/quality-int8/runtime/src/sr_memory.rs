//! Private output-storage experiment using the existing SPANF tile kernel.
use sha2::{Digest, Sha256};
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::{fs, hint::black_box, time::Instant};
use tract_onnx::prelude::TractResult;
use tract_onnx::tract_core::internal::{anyhow, ensure};
use zensr_micro::{SpanfModel, spanf_x4_tiled};

fn store(value: f32, dst: &mut [u8], bits: usize) {
    let max = if bits == 8 { 255.0 } else { 65535.0 };
    let value = (value * max + 0.5).clamp(0.0, max) as u16;
    if bits == 8 {
        dst[0] = value as u8;
    } else {
        dst[..2].copy_from_slice(&value.to_le_bytes());
    }
}

fn convert(input: &[f32], h: usize, w: usize, bits: usize) -> Vec<u8> {
    let bytes = bits / 8;
    let plane = h * w;
    let mut out = vec![0; plane * 3 * bytes];
    for i in 0..plane {
        for c in 0..3 {
            store(input[c * plane + i], &mut out[(i * 3 + c) * bytes..], bits);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn compact(
    model: &SpanfModel,
    input: &[f32],
    h: usize,
    w: usize,
    stride: usize,
    threads: usize,
    tile: usize,
    bits: usize,
) -> Vec<u8> {
    assert!(matches!(bits, 8 | 16) && stride >= w && tile >= 32);
    assert_eq!(input.len(), 3 * h * stride);
    let (oh, ow, bytes) = (h * 4, w * 4, bits / 8);
    let (nx, ny) = (w.div_ceil(tile), h.div_ceil(tile));
    let out = Mutex::new(vec![0; oh * ow * 3 * bytes]);
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1).min(nx * ny) {
            scope.spawn(|| {
                let mut extended = Vec::new();
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= nx * ny {
                        break;
                    }
                    let (x0, y0) = ((index % nx) * tile, (index / nx) * tile);
                    let (x1, y1) = ((x0 + tile).min(w), (y0 + tile).min(h));
                    let (ex0, ey0) = (x0.saturating_sub(16), y0.saturating_sub(16));
                    let (ex1, ey1) = ((x1 + 16).min(w), (y1 + 16).min(h));
                    let (eh, ew) = (ey1 - ey0, ex1 - ex0);
                    extended.resize(3 * eh * ew, 0.0);
                    for c in 0..3 {
                        for y in 0..eh {
                            let source = (c * h + ey0 + y) * stride + ex0;
                            extended[(c * eh + y) * ew..][..ew]
                                .copy_from_slice(&input[source..][..ew]);
                        }
                    }
                    // Reuse the existing forward implementation. The extended
                    // image is one tile; the outer crop supplies the real halo.
                    let float = spanf_x4_tiled(model, &extended, eh, ew, 1, eh.max(ew).max(32));
                    assert!(float.iter().all(|v| v.is_finite()));
                    let (cx, cy) = ((x0 - ex0) * 4, (y0 - ey0) * 4);
                    let (cw, ch) = ((x1 - x0) * 4, (y1 - y0) * 4);
                    let mut output = out.lock().unwrap();
                    for y in 0..ch {
                        for x in 0..cw {
                            let dst = ((y0 * 4 + y) * ow + x0 * 4 + x) * 3 * bytes;
                            for c in 0..3 {
                                let src = c * eh * ew * 16 + (cy + y) * ew * 4 + cx + x;
                                store(float[src], &mut output[dst + c * bytes..], bits);
                            }
                        }
                    }
                    // Only this small tile remains f32; completed output is integer.
                }
            });
        }
    });
    out.into_inner().unwrap()
}

pub(crate) fn run(args: &[String]) -> TractResult<()> {
    ensure!(
        args.len() == 8,
        "cpu_sr kind model input h w threads samples"
    );
    let (h, w, threads, samples) = (
        args[4].parse::<usize>()?,
        args[5].parse::<usize>()?,
        args[6].parse::<usize>()?,
        args[7].parse::<usize>()?,
    );
    ensure!(
        h > 0 && w > 0 && (1..=8).contains(&threads) && (1..=10).contains(&samples),
        "invalid budget"
    );
    let bits = if args[1].starts_with("spanf-rgb8") {
        8
    } else {
        16
    };
    let reference = args[1].ends_with("-reference");
    let model = SpanfModel::from_f16_bytes(&fs::read(&args[2])?).map_err(|e| anyhow!(e))?;
    let raw = fs::read(&args[3])?;
    ensure!(raw.len() == 3 * h * w * 4, "input dimensions differ");
    let input_sha = super::hex(&Sha256::digest(&raw));
    let input: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    drop(raw);
    ensure!(input.iter().all(|x| x.is_finite()), "nonfinite input");
    let infer = || {
        if reference {
            let float = spanf_x4_tiled(&model, &input, h, w, threads, 0);
            assert!(float.iter().all(|v| v.is_finite()));
            convert(&float, h * 4, w * 4, bits)
        } else {
            compact(&model, &input, h, w, w, threads, 128, bits)
        }
    };
    // Keep a digest, never multiple full outputs, during validation.
    let expected = super::hex(&Sha256::digest(infer()));
    for _ in 0..samples {
        let start = Instant::now();
        let output = infer();
        let elapsed = start.elapsed().as_nanos();
        ensure!(
            super::hex(&Sha256::digest(&output)) == expected,
            "repeat differs"
        );
        black_box(&output);
        println!("latency_ns={elapsed}");
    }
    println!(
        "IDENTITY {}",
        serde_json::json!({"input_sha256":input_sha,
        "shape":[1,3,h*4,w*4], "bits":bits, "output_sha256":expected,
        "kind":args[1], "compute":"FP32", "output":"interleaved integer, little-endian u16",
        "conversion":"clamp(value * max + 0.5, 0, max), truncate"})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn integer_boundaries() {
        let values = [-1.0, 0.0, 0.5 / 255.0, 0.5, 1.0, 2.0];
        let mut out = [0; 6];
        for (i, value) in values.into_iter().enumerate() {
            store(value, &mut out[i..], 8);
        }
        assert_eq!(out, [0, 0, 1, 128, 255, 255]);
        let mut word = [0; 2];
        store(0.5, &mut word, 16);
        assert_eq!(u16::from_le_bytes(word), 32768);
    }
    #[test]
    fn completed_tiles_match_float_conversion_with_stride_and_odd_edges() {
        let raw: Vec<f32> = (0..zensr_micro::TOTAL_FLOATS)
            .map(|i| ((i * 97 % 251) as f32 / 251.0 - 0.5) * 0.015)
            .collect();
        let model = SpanfModel::new(raw).unwrap();
        for (h, w) in [(1, 1), (5, 39), (65, 33), (35, 67)] {
            let input: Vec<f32> = (0..3 * h * w)
                .map(|i| (i * 127 % 251) as f32 / 251.0)
                .collect();
            let float = spanf_x4_tiled(&model, &input, h, w, 1, 32);
            let stride = w + 7;
            let mut padded = vec![f32::NAN; 3 * h * stride];
            for c in 0..3 {
                for y in 0..h {
                    padded[(c * h + y) * stride..][..w]
                        .copy_from_slice(&input[(c * h + y) * w..][..w]);
                }
            }
            for threads in [1, 3] {
                for bits in [8, 16] {
                    let actual = compact(&model, &padded, h, w, stride, threads, 32, bits);
                    assert_eq!(actual, convert(&float, h * 4, w * 4, bits));
                    if bits == 8 {
                        assert_eq!(actual, crate::pipeline::rgb8(&float, w * 4, h * 4));
                    }
                }
            }
        }
    }
}
