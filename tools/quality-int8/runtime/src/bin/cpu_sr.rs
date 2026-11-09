//! Decoded-pixel SR timing worker for zenbench's external-process protocol.
#![forbid(unsafe_code)]
use sha2::{Digest, Sha256};
use std::{fs, hint::black_box, path::Path, time::Instant};
use tract_onnx::prelude::*;
use tract_onnx::tract_core::internal::{anyhow, bail, ensure};
use zensr_micro::{SpanfModel, adopted::AdoptedModel};
#[path = "../sr_memory.rs"]
mod memory;
#[cfg(test)]
#[allow(dead_code)]
#[path = "../pipeline.rs"]
mod pipeline;

fn digest(data: &[f32]) -> String {
    let mut hash = Sha256::new();
    for x in data {
        hash.update(x.to_le_bytes());
    }
    hex(&hash.finalize())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

enum Model {
    Adopted(AdoptedModel),
    Spanf(Box<SpanfModel>),
    Tract(std::sync::Arc<TypedRunnableModel>),
}

impl Model {
    fn run(&self, input: &TValue, h: usize, w: usize, threads: usize) -> TractResult<Vec<f32>> {
        match self {
            Self::Adopted(m) => Ok(m.upscale_tiled(
                input.try_as_plain_ram()?.as_slice::<f32>()?,
                h,
                w,
                threads,
                0,
            )),
            Self::Spanf(m) => Ok(zensr_micro::spanf_x4_tiled(
                m,
                input.try_as_plain_ram()?.as_slice::<f32>()?,
                h,
                w,
                threads,
                0,
            )),
            Self::Tract(m) => {
                let out = m.run(tvec!(input.clone()))?;
                ensure!(
                    out.len() == 1 && out[0].shape() == [1, 3, h * 4, w * 4],
                    "tract output shape differs"
                );
                Ok(out[0].try_as_plain_ram()?.as_slice::<f32>()?.to_vec())
            }
        }
    }
}

fn main() -> TractResult<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_some_and(|s| {
        matches!(
            s.as_str(),
            "spanf-rgb8" | "spanf-rgb16" | "spanf-rgb8-reference" | "spanf-rgb16-reference"
        )
    }) {
        return memory::run(&args);
    }
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
        (1..=8).contains(&threads) && (1..=10).contains(&samples),
        "invalid work budget"
    );
    let path = Path::new(&args[2]);
    let model = match args[1].as_str() {
        "adopted" => {
            let meta: serde_json::Value =
                serde_json::from_slice(&fs::read(path.join("meta.json"))?)?;
            ensure!(meta["scale"] == 4, "expected a x4 model");
            let raw = zensr_micro::decode_all_f16(&fs::read(path.join("weights_f16.raw"))?);
            let model = match meta["arch"].as_str() {
                Some("compact") => AdoptedModel::load_compact(
                    &raw,
                    meta["nf"].as_u64().unwrap() as usize,
                    meta["nc"].as_u64().unwrap() as usize,
                    4,
                ),
                Some("span48") => AdoptedModel::load_span48(&raw, 4),
                _ => bail!("unsupported native architecture"),
            }
            .map_err(|s| anyhow!(s))?;
            Model::Adopted(model)
        }
        "spanf" => Model::Spanf(Box::new(
            SpanfModel::from_f16_bytes(&fs::read(path)?).map_err(|s| anyhow!(s))?,
        )),
        "tract" => {
            let model = tract_onnx::onnx()
                .model_for_path(path)?
                .with_input_fact(0, f32::fact([1, 3, h, w]).into())?
                .into_optimized()?;
            let options = tract_onnx::tract_core::runtime::RunOptions {
                executor: if threads > 1 {
                    Some(tract_linalg::multithread::Executor::multithread(threads))
                } else {
                    None
                },
                ..Default::default()
            };
            Model::Tract(model.into_runnable_with_options(&options)?)
        }
        _ => bail!("unsupported model kind"),
    };
    let bytes = fs::read(&args[3])?;
    ensure!(bytes.len() == 3 * h * w * 4, "input size differs");
    let values: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    ensure!(values.iter().all(|v| v.is_finite()), "nonfinite input");
    let input: TValue = Tensor::from_shape(&[1, 3, h, w], &values)?.into();
    let first = model.run(&input, h, w, threads)?;
    ensure!(
        first.len() == 3 * h * w * 16 && first.iter().all(|v| v.is_finite()),
        "invalid output"
    );
    let expected = digest(&first);
    let repeat = model.run(&input, h, w, threads)?;
    ensure!(
        digest(&repeat) == expected,
        "repeated native outputs differ"
    );
    // Our native tiled implementations promise exact thread-count invariance.
    if !matches!(model, Model::Tract(_)) {
        ensure!(
            digest(&model.run(&input, h, w, 1)?) == expected,
            "native thread-count pixels differ"
        );
    }
    println!(
        "IDENTITY {}",
        serde_json::json!({"kind":args[1],"requested_threads":threads,"precision":"f32 compute; native f16 weights expanded before timing",
        "input_sha256":hex(&Sha256::digest(&bytes)),"output_sha256":expected,"shape":[1,3,h*4,w*4],
        "spanf_default_core_tile":128,"codecs_included":false,"iqa_included":false})
    );
    drop(first);
    drop(repeat);
    for _ in 0..samples {
        let started = Instant::now();
        let output = model.run(black_box(&input), h, w, threads)?;
        let elapsed = started.elapsed().as_nanos();
        ensure!(
            output.iter().all(|v| v.is_finite()) && digest(&output) == expected,
            "timed output differs"
        );
        black_box(&output);
        println!("latency_ns={elapsed}");
    }
    Ok(())
}
