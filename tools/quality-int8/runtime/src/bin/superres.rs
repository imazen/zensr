//! Fixed ×4 corpus preparation, scalar IQA and lossless scored-image retention.
#![forbid(unsafe_code)]
#[path = "../../../../../crates/zensr-bench/src/bin/routing_study/storage.rs"]
mod artifact_storage;
#[path = "../cascade.rs"]
mod cascade;
#[allow(dead_code)]
#[path = "../pipeline.rs"]
mod pipeline;
use pipeline::{decode, hash, metrics, rgb8};
use serde_json::{Value, json};
use std::{
    fs,
    io::{self, BufRead, Read, Write},
    path::Path,
};
use zensr_micro::adopted::AdoptedModel;

fn text<'a>(v: &'a Value, key: &str) -> &'a str {
    v[key].as_str().unwrap()
}
fn resize(p: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Vec<u8> {
    let config = zenresize::ResizeConfig::builder(w as u32, h as u32, ow as u32, oh as u32)
        .filter(zenresize::Filter::CatmullRom)
        .format(zenresize::PixelDescriptor::RGB8_SRGB)
        .build();
    zenresize::Resizer::new(&config).resize(p)
}
fn write_new(path: &Path, data: &[u8]) {
    if path.exists() {
        assert_eq!(fs::read(path).unwrap(), data);
    } else {
        fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)
            .unwrap()
            .write_all(data)
            .unwrap();
    }
}
fn prepare(manifest: &str, output: &str) {
    let root = Path::new(output);
    fs::create_dir_all(root).unwrap();
    let mut rows = Vec::new();
    for line in fs::read_to_string(manifest).unwrap().lines() {
        let source: Value = serde_json::from_str(line).unwrap();
        let original = fs::read(text(&source, "reference_rgb8")).unwrap();
        assert_eq!(hash(&original), text(&source, "reference_rgb8_sha256"));
        let sw = source["width"].as_str().unwrap().parse::<usize>().unwrap();
        let sh = source["height"].as_str().unwrap().parse::<usize>().unwrap();
        assert_eq!(original.len(), sw * sh * 3);
        let (w, h) = (sw / 4 * 4, sh / 4 * 4);
        assert!(w >= 64 && h >= 64);
        let mut reference = Vec::new();
        for y in 0..h {
            reference.extend_from_slice(&original[y * sw * 3..y * sw * 3 + w * 3]);
        }
        let low = resize(&reference, w, h, w / 4, h / 4);
        let reference_path = root.join(format!("{}-reference.rgb", text(&source, "source_id")));
        write_new(&reference_path, &reference);
        for q in (0..=100).step_by(5) {
            for ss in if q == 0 {
                vec!["clean"]
            } else {
                vec!["420", "444"]
            } {
                let sampling = if ss == "420" {
                    zenjpeg::encoder::ChromaSubsampling::Quarter
                } else {
                    zenjpeg::encoder::ChromaSubsampling::None
                };
                let bytes = if q == 0 {
                    low.clone()
                } else {
                    let pixels: Vec<rgb::Rgb<u8>> = low
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .map(|p| rgb::Rgb::new(p[0], p[1], p[2]))
                        .collect();
                    zenjpeg::encoder::EncoderConfig::ycbcr(q as f32, sampling)
                        .encode(&pixels, (w / 4) as u32, (h / 4) as u32)
                        .unwrap()
                };
                let path = root.join(format!(
                    "{}-{q}-{ss}.{}",
                    text(&source, "source_id"),
                    if q == 0 { "rgb" } else { "jpg" }
                ));
                write_new(&path, &bytes);
                let id = hash(
                    format!(
                        "{}:{}:{q}:{ss}:{}",
                        hash(&reference),
                        hash(&bytes),
                        hash(&low)
                    )
                    .as_bytes(),
                );
                rows.push(json!({"id":id,"source_id":source["source_id"],"origin":source["origin"],"content":source["content"],"q":q,"ss":ss,
                    "input":path,"input_sha256":hash(&bytes),"reference":reference_path,"reference_sha256":hash(&reference),
                    "width":w,"height":h,"low_width":w/4,"low_height":h/4,"source":source,
                    "degradation":"crop right/bottom to multiple 4; zenresize CatmullRom linear-light RGB8; zenjpeg default JPEG"}));
            }
        }
    }
    assert_eq!(rows.len(), 697);
    let bytes = rows
        .iter()
        .map(|v| serde_json::to_string(v).unwrap() + "\n")
        .collect::<String>();
    write_new(&root.join("inputs.jsonl"), bytes.as_bytes());
}
fn retain_png(image: imgref::ImgRef<rgb::Rgb<u8>>, root: &Path) -> Value {
    let config = zenpng::EncodeConfig::default()
        .with_compression(zenpng::Compression::Fast)
        .with_parallel(false)
        .with_srgb_intent(Some(0));
    let bytes = zenpng::encode_rgb8(
        image,
        None,
        &config,
        &enough::Unstoppable,
        &enough::Unstoppable,
    )
    .unwrap();
    let digest = hash(&bytes);
    fs::create_dir_all(root).unwrap();
    let path = root.join(format!("{digest}.png"));
    artifact_storage::publish(&path, &bytes);
    json!({"path":path,"sha256":digest,"bytes":bytes.len(),"width":image.width(),"height":image.height(),"format":"PNG RGB8 sRGB lossless"})
}
fn measure(reference: &[u8], planes: &[f32], w: usize, h: usize, artifacts: &Path) -> Value {
    use rgb::FromSlice;
    assert_eq!(planes.len(), 3 * w * h);
    assert!(planes.iter().all(|v| v.is_finite()));
    let pixels = rgb8(planes, w, h);
    let (ssim, max, pnorm) = metrics(reference, &pixels, w, h);
    let png = retain_png(imgref::ImgRef::new(pixels.as_rgb(), w, h), artifacts);
    json!({"ssim2":format!("{ssim:.17}"),"butter_max":format!("{max:.17}"),"butter_pnorm3":format!("{pnorm:.17}"),"rgb8_sha256":hash(&pixels),"png":png})
}
fn planes(p: &[u8], w: usize, h: usize) -> Vec<f32> {
    let mut result = vec![0.; p.len()];
    for i in 0..w * h {
        for c in 0..3 {
            result[c * w * h + i] = f32::from(p[i * 3 + c]) / 255.;
        }
    }
    result
}
fn reply(out: &mut impl Write, value: &Value) {
    serde_json::to_writer(&mut *out, value).unwrap();
    out.write_all(b"\n").unwrap();
    out.flush().unwrap();
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args[1] == "render-rgb" {
        assert_eq!(
            args.len(),
            7,
            "render-rgb input width height stride artifacts"
        );
        let data = fs::read(&args[2]).unwrap();
        let w = args[3].parse::<usize>().unwrap();
        let h = args[4].parse::<usize>().unwrap();
        let stride = args[5].parse::<usize>().unwrap();
        assert!(w > 0 && h > 0 && stride >= w);
        assert!(data.len() >= ((h - 1) * stride + w) * 3 && data.len().is_multiple_of(3));
        use rgb::FromSlice;
        println!(
            "{}",
            retain_png(
                imgref::ImgRef::new_stride(data.as_rgb(), w, h, stride),
                Path::new(&args[6])
            )
        );
        return;
    }
    if args[1] == "cascade" {
        cascade::run(&args);
        return;
    }
    if args[1] == "prepare" {
        prepare(&args[2], &args[3]);
        return;
    }
    assert_eq!(args[1], "score");
    assert_eq!(
        args.len(),
        4,
        "score requires weights and retained-image directory"
    );
    let artifacts = Path::new(&args[3]);
    let span_bytes = fs::read(&args[2]).unwrap();
    let span = AdoptedModel::load_span48(&zensr_micro::decode_all_f16(&span_bytes), 4).unwrap();
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());
    reply(
        &mut output,
        &json!({"ready":true,"span_sha256":hash(&span_bytes)}),
    );
    let mut context: Option<(Value, Vec<u8>)> = None;
    loop {
        let mut line = String::new();
        if input.read_line(&mut line).unwrap() == 0 {
            break;
        }
        let request: Value = serde_json::from_str(&line).unwrap();
        match text(&request, "action") {
            "prepare" => {
                let row = &request["row"];
                let bytes = fs::read(text(row, "input")).unwrap();
                assert_eq!(hash(&bytes), text(row, "input_sha256"));
                let reference = fs::read(text(row, "reference")).unwrap();
                assert_eq!(hash(&reference), text(row, "reference_sha256"));
                let w = row["width"].as_u64().unwrap() as usize;
                let h = row["height"].as_u64().unwrap() as usize;
                assert!(w * h <= 1_048_576);
                assert_eq!(reference.len(), 3 * w * h);
                let low = if row["q"] == 0 {
                    assert_eq!(bytes.len(), 3 * w * h / 16);
                    planes(&bytes, w / 4, h / 4)
                } else {
                    let (p, lw, lh) = decode(&bytes, false);
                    assert_eq!((lw, lh), (w / 4, h / 4));
                    p
                };
                let baseline = planes(&resize(&rgb8(&low, w / 4, h / 4), w / 4, h / 4, w, h), w, h);
                let mut native = vec![0.; w * h * 3];
                span.forward(&low, h / 4, w / 4, &mut native);
                reply(
                    &mut output,
                    &json!({"id":row["id"],"input_bytes":low.len()*4,"baseline":measure(&reference,&baseline,w,h,artifacts),"native_span":measure(&reference,&native,w,h,artifacts)}),
                );
                for v in low {
                    output.write_all(&v.to_le_bytes()).unwrap();
                }
                output.flush().unwrap();
                context = Some((row.clone(), reference));
            }
            "score" => {
                let (row, reference) = context.as_ref().unwrap();
                assert_eq!(request["id"], row["id"]);
                let w = row["width"].as_u64().unwrap() as usize;
                let h = row["height"].as_u64().unwrap() as usize;
                let n = request["bytes"].as_u64().unwrap() as usize;
                assert_eq!(n, 12 * w * h);
                let mut bytes = vec![0; n];
                input.read_exact(&mut bytes).unwrap();
                let planes: Vec<f32> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| f32::from_le_bytes(*v))
                    .collect();
                reply(
                    &mut output,
                    &json!({"id":row["id"],"score":measure(reference,&planes,w,h,artifacts)}),
                );
            }
            "quit" => break,
            _ => panic!("unknown action"),
        }
    }
}

#[cfg(test)]
mod retained_image_tests {
    use super::*;

    #[test]
    fn stride_padding_never_enters_retained_pixels() {
        let scratch = std::env::var_os("TMPDIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(std::env::var_os("HOME").unwrap()).join("tmp")
            });
        let root = scratch.join(format!("zensr-png-test-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let pixels: Vec<_> = (0..15u8)
            .map(|i| rgb::Rgb::new(i.wrapping_mul(71), i.wrapping_mul(43), i.wrapping_mul(193)))
            .collect();
        let mut padded = vec![rgb::Rgb::new(255, 0, 255); 7 * 3];
        for y in 0..3 {
            padded[y * 7..y * 7 + 5].copy_from_slice(&pixels[y * 5..y * 5 + 5]);
        }
        let tight = retain_png(imgref::ImgRef::new(&pixels, 5, 3), &root);
        let strided = retain_png(imgref::ImgRef::new_stride(&padded, 5, 3, 7), &root);
        assert_eq!(tight, strided);
        let path = Path::new(tight["path"].as_str().unwrap());
        assert!(fs::read(path).unwrap().starts_with(b"\x89PNG\r\n\x1a\n"));
        fs::remove_dir_all(root).unwrap();
    }
}
