//! Streaming external-model scoring; output RGB and diagnostic maps never hit disk.
#![forbid(unsafe_code)]
#[path = "../pipeline.rs"]
mod pipeline;
#[path = "../../../../../crates/zensr-zenjpeg/src/record/probe.rs"]
mod record_probe;
use pipeline::{decode, finish, hash, metrics, rgb8};
use serde_json::{Value, json};
use std::{
    fs,
    io::{self, BufRead, Read, Write},
};
use zensr_micro::adopted::AdoptedModel;
use zensr_zenjpeg::RestoreConfig;

struct Context {
    id: String,
    jpeg: Vec<u8>,
    reference: Vec<u8>,
    input: Vec<f32>,
    auto: Vec<f32>,
    probe: zenjpeg::detect::JpegProbe,
    w: usize,
    h: usize,
}

fn text<'a>(v: &'a Value, key: &str) -> &'a str {
    v[key].as_str().expect("required string field")
}

fn measure(context: &Context, raw: Vec<f32>, input: &[f32]) -> Value {
    assert_eq!(raw.len(), 3 * context.w * context.h);
    assert!(raw.iter().all(|v| v.is_finite()));
    let projected = finish(
        raw.clone(),
        input,
        &context.jpeg,
        &context.probe,
        context.w,
        context.h,
    );
    let mut result = serde_json::Map::new();
    result.insert("id".into(), json!(context.id));
    for (stage, planes) in [("raw", raw), ("projected", projected)] {
        let pixels = rgb8(&planes, context.w, context.h);
        let (ssim, max, pnorm) = metrics(&context.reference, &pixels, context.w, context.h);
        result.insert(
            stage.into(),
            json!({"ssim2":format!("{ssim:.17}"),"butter_max":format!("{max:.17}"),
            "butter_pnorm3":format!("{pnorm:.17}"),"rgb8_sha256":hash(&pixels)}),
        );
    }
    Value::Object(result)
}

fn reply(out: &mut impl Write, value: &Value) {
    serde_json::to_writer(&mut *out, value).unwrap();
    out.write_all(b"\n").unwrap();
    out.flush().unwrap();
}

fn main() {
    let weights = fs::read(std::env::args().nth(1).expect("quality-model f16 path")).unwrap();
    let expanded = zensr_micro::decode_all_f16(&weights);
    let expanded_bytes: Vec<u8> = expanded.iter().flat_map(|v| v.to_le_bytes()).collect();
    let model = AdoptedModel::load_compact(&expanded, 64, 16, 1).unwrap();
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());
    let mut current: Option<Context> = None;
    reply(
        &mut output,
        &json!({"ready":true,"weights_sha256":hash(&weights),
            "expanded_weights_sha256":hash(&expanded_bytes)}),
    );
    loop {
        let mut line = String::new();
        if input.read_line(&mut line).unwrap() == 0 {
            break;
        }
        let request: Value = serde_json::from_str(&line).unwrap();
        match text(&request, "action") {
            "prepare" => {
                let jpeg = fs::read(text(&request, "jpeg")).unwrap();
                let reference = fs::read(text(&request, "reference_rgb8")).unwrap();
                assert_eq!(hash(&jpeg), text(&request, "jpeg_sha256"));
                assert_eq!(hash(&reference), text(&request, "reference_rgb8_sha256"));
                let mut probe = zenjpeg::detect::probe(&jpeg).unwrap();
                record_probe::apply_record(&mut probe);
                let (plain, w, h) = decode(&jpeg, false);
                assert_eq!(request["width"].as_u64().unwrap() as usize, w);
                assert_eq!(request["height"].as_u64().unwrap() as usize, h);
                assert!(w.checked_mul(h).unwrap() <= 16_777_216);
                assert_eq!(reference.len(), 3 * w * h);
                let (auto, aw, ah) = decode(&jpeg, zensr_zenjpeg::policy_wants_auto(&probe));
                assert_eq!((aw, ah), (w, h));
                let (ssim, max, pnorm) = metrics(&reference, &rgb8(&plain, w, h), w, h);
                reply(
                    &mut output,
                    &json!({"id":request["id"],"width":w,"height":h,
                    "input_bytes":plain.len()*4,"identity_ssim2":format!("{ssim:.17}"),
                    "identity_butter_max":format!("{max:.17}"),"identity_butter_pnorm3":format!("{pnorm:.17}")}),
                );
                for value in &plain {
                    output.write_all(&value.to_le_bytes()).unwrap();
                }
                output.flush().unwrap();
                current = Some(Context {
                    id: text(&request, "id").into(),
                    jpeg,
                    reference,
                    input: plain,
                    auto,
                    probe,
                    w,
                    h,
                });
            }
            "score" => {
                let context = current.as_ref().expect("prepare before scoring");
                assert_eq!(text(&request, "id"), context.id);
                let bytes = request["bytes"].as_u64().unwrap() as usize;
                assert_eq!(bytes, 3 * context.w * context.h * 4);
                let mut payload = vec![0u8; bytes];
                input.read_exact(&mut payload).unwrap();
                let planes = payload
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| f32::from_le_bytes(*v))
                    .collect();
                reply(&mut output, &measure(context, planes, &context.input));
            }
            "native-control" => {
                let context = current.as_ref().expect("prepare before scoring");
                assert_eq!(text(&request, "id"), context.id);
                let mut planes = vec![0.; 3 * context.w * context.h];
                model.forward(&context.auto, context.h, context.w, &mut planes);
                let actual = measure(context, planes, &context.auto);
                let expected = zensr_zenjpeg::restore_jpeg(
                    &context.jpeg,
                    &model,
                    &RestoreConfig::default()
                        .with_threads(1)
                        .with_high_q_identity(false),
                )
                .unwrap();
                assert_eq!(
                    actual["projected"]["rgb8_sha256"].as_str().unwrap(),
                    hash(&expected.to_rgb8()),
                    "shared projection adapter differs from serving pipeline"
                );
                reply(&mut output, &actual);
            }
            "quit" => break,
            _ => panic!("unknown external-scoring action"),
        }
    }
}
