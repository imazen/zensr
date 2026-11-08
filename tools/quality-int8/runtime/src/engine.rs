use std::{fs, path::Path};
use tract_onnx::prelude::*;
use zensr_micro::adopted::AdoptedModel;
pub(crate) const ARMS: [&str; 6] = [
    "native",
    "float",
    "weight-int8",
    "int8",
    "final-f32",
    "final2-f32",
];
pub(crate) fn floats(path: &Path) -> Vec<f32> {
    let b = fs::read(path).unwrap();
    assert!(b.len().is_multiple_of(4));
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}
pub(crate) enum Engine {
    Native(AdoptedModel),
    Tract(TypedRunnableModel),
}
impl Engine {
    pub(crate) fn load(root: &Path, name: &str, h: usize, w: usize) -> TractResult<Self> {
        if name == "native" {
            return Ok(Self::Native(
                AdoptedModel::load_compact(&floats(&root.join("weights.raw")), 64, 16, 1)
                    .map_err(|s| anyhow!(s))?,
            ));
        }
        let model = tract_onnx::onnx()
            .model_for_path(root.join(format!("{name}.onnx")))?
            .with_input_fact(0, f32::fact([1, 3, h, w]).into())?
            .into_optimized()?;
        Ok(Self::Tract(model.into_runnable()?))
    }
    pub(crate) fn inspect(&self) -> String {
        match self {
            Self::Native(_) => "native f32 SIMD kernels".into(),
            Self::Tract(p) => p
                .model()
                .nodes()
                .iter()
                .map(|n| {
                    format!(
                        "{} {} {:?} {:?}",
                        n.name,
                        n.op.name(),
                        n.op.info(),
                        n.outputs
                            .iter()
                            .map(|o| o.fact.datum_type)
                            .collect::<Vec<_>>()
                    )
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
    pub(crate) fn run(
        &self,
        input: &TValue,
        h: usize,
        w: usize,
        tiled: bool,
    ) -> TractResult<Vec<f32>> {
        match self {
            Self::Native(m) => {
                let data = input.as_slice::<f32>()?;
                if tiled {
                    Ok(m.upscale_tiled(data, h, w, 1, 0))
                } else {
                    let mut out = vec![0.; 3 * h * w];
                    m.forward(data, h, w, &mut out);
                    Ok(out)
                }
            }
            Self::Tract(p) => {
                let out = p.run(tvec!(input.clone()))?;
                Ok(out[0].as_slice::<f32>()?.to_vec())
            }
        }
    }
}
pub(crate) fn tensor(input: &[f32], h: usize, w: usize) -> TValue {
    Tensor::from_shape(&[1, 3, h, w], input).unwrap().into()
}
pub(crate) fn rgb8(input: &[f32], w: usize, h: usize) -> Vec<u8> {
    let n = w * h;
    (0..n)
        .flat_map(|i| (0..3).map(move |c| (input[c * n + i] * 255. + 0.5).clamp(0., 255.) as u8))
        .collect()
}
