#![forbid(unsafe_code)]
mod engine;
use engine::*;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::Mutex,
};
use tract_onnx::prelude::*;
use tract_onnx::tract_core::internal::{bail, ensure};
use zensr_micro::{consist::*, guards::guarded_merge};
use zensr_zenjpeg::{ProjectionConfig, RestoreConfig};
#[path = "../../../../crates/zensr-zenjpeg/src/record/probe.rs"]
mod record_probe;
fn hash(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}
fn decode(jpeg: &[u8], auto: bool) -> (Vec<f32>, usize, usize) {
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
fn finish(
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
fn metrics(reference: &[u8], pixels: &[u8], w: usize, h: usize) -> (f64, f64, f64) {
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
fn errors(a: &[f32], b: &[f32], w: usize, h: usize) -> (f64, f32, usize, usize, usize, usize) {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(b).all(|x| x.is_finite()));
    let sum = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() as f64)
        .sum::<f64>()
        / a.len() as f64;
    let max = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let aa = rgb8(a, w, h);
    let bb = rgb8(b, w, h);
    let mut counts = [0usize; 4];
    for (&x, y) in aa.iter().zip(bb) {
        let d = x.abs_diff(y) as usize;
        counts[0] += usize::from(d > 0);
        counts[1] += usize::from(d > 1);
        counts[2] += usize::from(d > 2);
        counts[3] = counts[3].max(d);
    }
    (sum, max, counts[0], counts[1], counts[2], counts[3])
}
fn evaluate(exports: &Path, inputs: &Path, out: &Path, jobs: usize, limit: Option<usize>) {
    fs::create_dir(out).unwrap();
    fs::create_dir(out.join("cells")).unwrap();
    let text = fs::read_to_string(inputs).unwrap();
    let mut lines = text.lines();
    let header: Vec<_> = lines.next().unwrap().split('\t').collect();
    let at = |s: &str| header.iter().position(|v| *v == s).unwrap();
    let mut rows: Vec<Vec<&str>> = lines.map(|l| l.split('\t').collect()).collect();
    if let Some(n) = limit {
        rows.truncate(n);
    }
    let required = rows.len();
    assert!(required > 0 && jobs > 0);
    let done = Mutex::new(0usize);
    let columns = "id\tsource_id\torigin\tcontent\tcohort\tsize\tencoder\tss\tq\tarm\twidth\theight\tidentity_ssim2\traw_ssim2\tprojected_ssim2\traw_butter_max\tprojected_butter_max\traw_butter_pnorm3\tprojected_butter_pnorm3\traw_mae\traw_max\traw_rgb8_changed\traw_rgb8_gt1\traw_rgb8_gt2\traw_rgb8_max\tprojected_mae\tprojected_max\tprojected_rgb8_changed\tprojected_rgb8_gt1\tprojected_rgb8_gt2\tprojected_rgb8_max\tvalues\tinput_rgb8_roundtrip_max\traw_outside01\tprojected_outside01\tactivation_range_exceedances\tactivation_values\traw_rgb8_sha256\tprojected_rgb8_sha256\traw_f32_sha256\tprojected_f32_sha256\n";
    let plan_hash = hash(&fs::read(exports.join("PLAN.json")).unwrap());
    let input_hash = hash(text.as_bytes());
    let progress = |stage: &str, n: usize| {
        let record = serde_json::json!({
            "stage": stage,
            "pid": std::process::id(),
            "complete_inputs": n,
            "required_inputs": required,
            "arms": ARMS,
            "exports_plan_sha256": plan_hash,
            "inputs_sha256": input_hash,
        });
        fs::write(
            out.join("PROGRESS.json"),
            serde_json::to_vec_pretty(&record).unwrap(),
        )
        .unwrap();
    };
    let identity = serde_json::json!({
        "binary_sha256": hash(&fs::read(std::env::current_exe().unwrap()).unwrap()),
        "inputs_sha256": input_hash,
        "plan_sha256": plan_hash,
        "joint_sha256": hash(&fs::read(exports.join("joint.onnx")).unwrap()),
        "required_inputs": required,
        "arms": ARMS,
    });
    fs::write(
        out.join("IDENTITY.json"),
        serde_json::to_vec_pretty(&identity).unwrap(),
    )
    .unwrap();
    progress("running", 0);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .unwrap();

    pool.install(|| {
        rows.par_chunks(required.div_ceil(jobs)).for_each(|chunk| {
            let mut cache: HashMap<(usize, usize), (Vec<Engine>, Engine)> = HashMap::new();
            let layout: serde_json::Value =
                serde_json::from_slice(&fs::read(exports.join("joint-map.json")).unwrap()).unwrap();
            for (rowindex, r) in chunk.iter().enumerate() {
                let jpg = fs::read(r[at("jpeg")]).unwrap();
                assert_eq!(hash(&jpg), r[at("jpeg_sha256")]);
                let gt = fs::read(r[at("reference_rgb8")]).unwrap();
                assert_eq!(hash(&gt), r[at("reference_rgb8_sha256")]);
                let mut probe = zenjpeg::detect::probe(&jpg).unwrap();
                record_probe::apply_record(&mut probe);
                let (input, w, h) = decode(&jpg, zensr_zenjpeg::policy_wants_auto(&probe));
                assert_eq!(
                    (w, h),
                    (
                        r[at("width")].parse::<usize>().unwrap(),
                        r[at("height")].parse::<usize>().unwrap()
                    ),
                    "manifest geometry differs from JPEG",
                );
                let (identity, iw, ih) = decode(&jpg, false);
                assert_eq!((w, h), (iw, ih));
                assert_eq!(gt.len(), 3 * w * h);
                let idscore = metrics(&gt, &rgb8(&identity, w, h), w, h).0;
                assert!(w * h <= i32::MAX as usize / 64, "counter domain exceeded");
                let (engines, joint) = cache.entry((h, w)).or_insert_with(|| {
                    let engines = ARMS
                        .iter()
                        .map(|arm| Engine::load_audited(exports, arm, h, w).unwrap())
                        .collect();
                    (engines, Engine::load(exports, "joint", h, w).unwrap())
                });
                let x = tensor(&input, h, w);
                let joint_values = match joint {
                    Engine::Tract(p) => p.run(tvec!(x.clone())).unwrap(),
                    Engine::Native(_) => unreachable!(),
                };
                let baseline = engines[0].run(&x, h, w, false).unwrap();
                let finished = finish(baseline.clone(), &input, &jpg, &probe, w, h);
                if rowindex == 0 {
                    let model = zensr_micro::adopted::AdoptedModel::load_compact(
                        &floats(&exports.join("weights.raw")),
                        64,
                        16,
                        1,
                    )
                    .unwrap();
                    let expected = zensr_zenjpeg::restore_jpeg(
                        &jpg,
                        &model,
                        &RestoreConfig::default().with_high_q_identity(false),
                    )
                    .unwrap();
                    assert_eq!(
                        rgb8(&finished, w, h),
                        expected.to_rgb8(),
                        "native projection adapter disagrees with serving pipeline"
                    );
                }
                let mut cell = String::from(columns);
                for (arm, engine) in ARMS.iter().zip(engines.iter()) {
                    let (raw, range_counts) = if *arm == "native" {
                        (baseline.clone(), vec![])
                    } else {
                        let mapping = &layout[*arm];
                        let pixels = joint_values[mapping["pixels"].as_u64().unwrap() as usize]
                            .try_as_plain_ram()
                            .unwrap()
                            .as_slice::<f32>()
                            .unwrap()
                            .to_vec();
                        let counts = mapping["counts"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| {
                                joint_values[v.as_u64().unwrap() as usize]
                                    .cast_to_scalar::<i64>()
                                    .unwrap()
                            })
                            .collect::<Vec<_>>();
                        (pixels, counts)
                    };
                    if rowindex == 0 {
                        let plain = Engine::load(exports, arm, h, w)
                            .unwrap()
                            .run(&x, h, w, false)
                            .unwrap();
                        assert_eq!(
                            raw.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            plain.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "shared graph changed full-size output"
                        );
                        let separate = engine.run_audited(&x, h, w).unwrap();
                        assert_eq!(
                            range_counts, separate.1,
                            "shared graph changed range counters"
                        );
                    }
                    let projected = if *arm == "native" {
                        finished.clone()
                    } else {
                        finish(raw.clone(), &input, &jpg, &probe, w, h)
                    };
                    let rm = metrics(&gt, &rgb8(&raw, w, h), w, h);
                    let pm = metrics(&gt, &rgb8(&projected, w, h), w, h);
                    let re = errors(&baseline, &raw, w, h);
                    let pe = errors(&finished, &projected, w, h);
                    let roundtrip = input
                        .iter()
                        .map(|v| (v - (v * 255.).round() / 255.).abs())
                        .fold(0f32, f32::max);
                    let mut fields = [
                        "id",
                        "source_id",
                        "origin",
                        "content",
                        "cohort",
                        "size",
                        "encoder",
                        "ss",
                        "q",
                    ]
                    .map(|name| r[at(name)].to_owned())
                    .to_vec();
                    fields.extend([
                        arm.to_string(),
                        w.to_string(),
                        h.to_string(),
                        format!("{idscore:.17}"),
                    ]);
                    fields.extend([rm.0, pm.0, rm.1, pm.1, rm.2, pm.2].map(|v| format!("{v:.17}")));
                    fields.extend([
                        re.0.to_string(),
                        re.1.to_string(),
                        re.2.to_string(),
                        re.3.to_string(),
                        re.4.to_string(),
                        re.5.to_string(),
                        pe.0.to_string(),
                        pe.1.to_string(),
                        pe.2.to_string(),
                        pe.3.to_string(),
                        pe.4.to_string(),
                        pe.5.to_string(),
                        raw.len().to_string(),
                        roundtrip.to_string(),
                        raw.iter()
                            .filter(|x| **x < 0. || **x > 1.)
                            .count()
                            .to_string(),
                        projected
                            .iter()
                            .filter(|x| **x < 0. || **x > 1.)
                            .count()
                            .to_string(),
                        range_counts.iter().sum::<i64>().to_string(),
                        (if range_counts.is_empty() {
                            0
                        } else {
                            (3 + 64 * (range_counts.len() - 1)) * w * h
                        })
                        .to_string(),
                        hash(&rgb8(&raw, w, h)),
                        hash(&rgb8(&projected, w, h)),
                        hash(&raw.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()),
                        hash(
                            &projected
                                .iter()
                                .flat_map(|v| v.to_le_bytes())
                                .collect::<Vec<_>>(),
                        ),
                    ]);
                    cell.push_str(&fields.join("\t"));
                    cell.push('\n');
                }
                let p = out.join("cells").join(format!("{}.tsv", r[at("id")]));
                fs::write(p, cell).unwrap();
                let mut n = done.lock().unwrap();
                *n += 1;
                if n.is_multiple_of(8) || *n == required {
                    progress("running", *n);
                    println!("completed {}/{}", *n, required);
                }
            }
        });
    });
    let mut file = fs::File::create_new(out.join("results.tsv")).unwrap();
    file.write_all(columns.as_bytes()).unwrap();
    for r in &rows {
        let cell =
            fs::read_to_string(out.join("cells").join(format!("{}.tsv", r[at("id")]))).unwrap();
        for line in cell.lines().skip(1) {
            writeln!(file, "{line}").unwrap();
        }
    }
    file.sync_all().unwrap();
    fs::write(
        out.join("COMPLETE"),
        hash(&fs::read(out.join("results.tsv")).unwrap()),
    )
    .unwrap();
    progress("complete", required);
}
fn main() -> TractResult<()> {
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("inspect") => {
            let root = Path::new(&args[2]);
            let h = args[3].parse()?;
            let w = args[4].parse()?;
            let output = Path::new(&args[5]);
            fs::create_dir(output)?;
            let x = tensor(&floats(&root.join("gold-input.raw")), 17, 13);
            let gold = floats(&root.join("gold-float.raw"));
            for arm in ARMS {
                let e = Engine::load(root, arm, h, w)?;
                fs::write(output.join(format!("{arm}.txt")), e.inspect())?;
                let e = Engine::load(root, arm, 17, 13)?;
                let y = e.run(&x, 17, 13, false)?;
                let error = gold
                    .iter()
                    .zip(&y)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                if arm == "native" || arm == "float" {
                    ensure!(
                        error < 5e-4,
                        "unchanged adopted-model golden ceiling: {arm} {error}"
                    );
                }
                let audited = Engine::load_audited(root, arm, 17, 13)?;
                let (audit_pixels, counts) = audited.run_audited(&x, 17, 13)?;
                assert_eq!(
                    y.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    audit_pixels.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "audit instrumentation changed primary output"
                );
                assert!(counts.iter().all(|n| *n >= 0));
                fs::write(
                    output.join(format!("{arm}-gold.raw")),
                    y.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
                )?;
                println!("{arm} golden max error {error}");
            }
        }
        Some("rgb-grid") => {
            let root = Path::new(&args[2]);
            let output = Path::new(&args[3]);
            let model_path = root.join("int8.onnx");
            let model = tract_onnx::onnx()
                .model_for_path(&model_path)?
                .with_input_fact(0, f32::fact([1, 3, 1, 256]).into())?
                .with_outputs_by_name(["c0qx"])?
                .into_optimized()?
                .into_runnable()?;
            let expected: Vec<u8> = (0..768).map(|v| (v % 256) as u8).collect();
            let input: Vec<f32> = expected.iter().map(|v| *v as f32 / 255.).collect();
            let result = model.run(tvec!(tensor(&input, 1, 256)))?;
            let view = result[0].try_as_plain_ram()?;
            let codes = view.as_slice::<u8>()?;
            ensure!(
                codes == expected,
                "RGB8 input quantizer changed a code value"
            );
            let record = serde_json::json!({
                "binary_sha256": hash(&fs::read(std::env::current_exe()?)?),
                "model_sha256": hash(&fs::read(model_path)?),
                "selected_node": "c0qx",
                "values_per_channel": 256,
                "channels": 3,
                "checked_values": codes.len(),
                "exact": true,
                "output_sha256": hash(codes),
                "scope": "First input quantizer from the frozen integer graph; internal activations/weights and final RGB errors measured separately",
            });
            let mut file = fs::File::create_new(output)?;
            file.write_all(&serde_json::to_vec_pretty(&record)?)?;
            file.sync_all()?;
            println!("RGB8 input quantizer preserves all 256 codes in each channel");
        }
        Some("probe") => {
            let root = Path::new(&args[2]);
            let arm = &args[3];
            let h = args[4].parse()?;
            let w = args[5].parse()?;
            let count: usize = args[6].parse()?;
            let e = Engine::load(root, arm, h, w)?;
            let values: Vec<f32> = (0..3 * h * w)
                .map(|i| ((i * 13 + i / 17) % 251) as f32 / 255.)
                .collect();
            let x = tensor(&values, h, w);
            println!("probe pid {} arm {}", std::process::id(), arm);
            for _ in 0..count {
                std::hint::black_box(e.run(&x, h, w, false)?);
            }
        }
        Some("evaluate") => {
            evaluate(
                &PathBuf::from(&args[2]),
                &PathBuf::from(&args[3]),
                &PathBuf::from(&args[4]),
                args[5].parse()?,
                args.get(6).map(|s| s.parse().unwrap()),
            );
        }
        _ => bail!(
            "inspect EXPORTS H W OUTPUT | rgb-grid EXPORTS OUTPUT.json | probe EXPORTS ARM H W COUNT | evaluate EXPORTS INPUTS.tsv OUTPUT JOBS [LIMIT]"
        ),
    };
    Ok(())
}

impl Engine {
    fn inspect(&self) -> String {
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
}
fn rgb8(input: &[f32], w: usize, h: usize) -> Vec<u8> {
    let n = w * h;
    (0..n)
        .flat_map(|i| (0..3).map(move |c| (input[c * n + i] * 255. + 0.5).clamp(0., 255.) as u8))
        .collect()
}

impl Engine {
    fn load_audited(root: &Path, name: &str, h: usize, w: usize) -> TractResult<Self> {
        Self::load_impl(root, name, h, w, 1, true)
    }
    fn run_audited(&self, input: &TValue, h: usize, w: usize) -> TractResult<(Vec<f32>, Vec<i64>)> {
        match self {
            Self::Native(_) => Ok((self.run(input, h, w, false)?, vec![])),
            Self::Tract(p) => {
                let values = p.run(tvec!(input.clone()))?;
                let pixels = values[0].try_as_plain_ram()?.as_slice::<f32>()?.to_vec();
                let counts = values
                    .iter()
                    .skip(1)
                    .map(|v| v.cast_to_scalar::<i64>())
                    .collect::<TractResult<Vec<_>>>()?;
                Ok((pixels, counts))
            }
        }
    }
}
