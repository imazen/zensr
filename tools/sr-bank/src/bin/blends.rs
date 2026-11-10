//! Resident filter/blend evaluation from the lossless candidate bank.
#![forbid(unsafe_code)]
#[allow(dead_code, clippy::too_many_arguments, clippy::bool_assert_comparison)]
#[path = "../../../sr-blend/src/vendor/classical.rs"]
mod classical;
#[cfg(test)]
#[path = "../../../sr-blend/src/vendor/studio601_gray.rs"]
mod metrics;
#[path = "../../../sr-blend/src/pixel_ops.rs"]
mod pixel_ops;
mod decode {
    pub(crate) struct Rgb8Image {
        pub width: u32,
        pub height: u32,
        pub pixels: Vec<u8>,
    }
}
use decode::Rgb8Image;
use imij::{Channels, Cicp, SampleType, planes::SeekReader};
use imij_set::Reader;
use pixel_ops::{View, blend, filter, float_resize, floats, rgb_view};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs::File,
    io::{BufWriter, Write},
};
use zensim::{RgbSlice, Zensim, ZensimProfile, ZensimScratch};
type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
fn number(v: &Value, k: &str) -> usize {
    v[k].as_u64().unwrap() as usize
}
fn text<'a>(v: &'a Value, k: &str) -> &'a str {
    v[k].as_str().unwrap()
}
fn hash(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}
fn image(reader: &mut Reader<SeekReader<File>>, record: &Value) -> Result<Vec<u8>> {
    let e = reader
        .get(record["key"].as_u64().unwrap())?
        .ok_or("Missing retained image")?;
    let mut bytes = vec![];
    reader.copy_to(&e, &mut bytes)?;
    if hash(&bytes) != text(record, "encoded_sha256") {
        return Err("Retained IMI identity changed".into());
    }
    let (info, p) = imij::decode(&bytes)?;
    if info.width as usize != number(record, "width")
        || info.height as usize != number(record, "height")
        || info.channels != Channels::Rgb
        || info.sample != SampleType::U8
        || info.cicp != Cicp::SRGB
        || hash(&p) != text(record, "pixels_sha256")
    {
        return Err("Retained pixel identity changed".into());
    }
    Ok(p)
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 3 {
        return Err("usage: blends SHARD_JSON NEW_OUTPUT_JSONL".into());
    }
    let record: Value = serde_json::from_slice(&std::fs::read(&a[1])?)?;
    let mut reader = Reader::open(SeekReader::new(File::open(text(&record, "archive"))?))?;
    let file = if a[2] == "-" {
        None
    } else {
        Some(File::create_new(&a[2])?)
    };
    let sink: Box<dyn Write> = if let Some(f) = &file {
        Box::new(f.try_clone()?)
    } else {
        Box::new(std::io::stdout())
    };
    let mut output = BufWriter::new(sink);
    let source = &record["source"];
    let z = Zensim::new(ZensimProfile::A).with_parallel(false);
    let mut references = std::collections::HashMap::new();
    for r in record["references"].as_array().unwrap() {
        references.insert(
            text(&r["hr"], "label").to_string(),
            image(&mut reader, &r["hr"])?,
        );
    }
    let mut rows = 0u64;
    for cell in record["cells"].as_array().unwrap() {
        let edge = number(cell, "edge");
        let w = number(cell, "width");
        let h = number(cell, "height");
        let reference = &references[&format!("e{edge}/hr")];
        let mut inputs = vec![("direct", image(&mut reader, &cell["decoded"])?)];
        if number(cell, "q") != 0 {
            for (name, label) in [("half", "half-repair"), ("repaired", "project-default")] {
                let r = cell["repair"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|r| text(r, "label").ends_with(&format!("/{label}")))
                    .ok_or("Missing repair input")?;
                inputs.push((name, image(&mut reader, r)?));
            }
        }
        let mut sr = vec![];
        for r in cell["sr"].as_array().unwrap() {
            let p = image(&mut reader, r)?;
            sr.push((
                text(r, "model"),
                text(r, "repair"),
                floats(rgb_view(&p, w * 4, h * 4))?,
            ));
        }
        for scale in [2, 4] {
            let (ow, oh) = (w * scale, h * scale);
            let reference_pixels = if scale == 4 {
                reference.clone()
            } else {
                let c = zenresize::ResizeConfig::builder(
                    (w * 4) as u32,
                    (h * 4) as u32,
                    ow as u32,
                    oh as u32,
                )
                .format(zenresize::PixelDescriptor::RGB8_SRGB)
                .filter(zenresize::Filter::CatmullRom)
                .build();
                zenresize::Resizer::new(&c).resize(reference)
            };
            let reference = Rgb8Image {
                width: ow as u32,
                height: oh as u32,
                pixels: reference_pixels,
            };
            let prepared = z.precompute_reference(&RgbSlice::new(
                reference.pixels.as_chunks::<3>().0,
                ow,
                oh,
            ))?;
            let mut scratch = ZensimScratch::new();
            let mut scaled = vec![];
            for (model, stage, p) in &sr {
                let pixels = if scale == 4 {
                    p.clone()
                } else {
                    float_resize(
                        View {
                            data: p,
                            width: w * 4,
                            height: h * 4,
                            stride: w * 12,
                        },
                        ow,
                        oh,
                        zenresize::Filter::Box,
                    )?
                };
                scaled.push((*model, *stage, pixels));
            }
            let mut candidate = Rgb8Image {
                width: ow as u32,
                height: oh as u32,
                pixels: Vec::with_capacity(ow * oh * 3),
            };
            for (stage, low) in &inputs {
                let low = floats(rgb_view(low, w, h))?;
                for kernel in ["triangle", "mitchell", "catmull_rom", "lanczos2"] {
                    let base = float_resize(
                        View {
                            data: &low,
                            width: w,
                            height: h,
                            stride: w * 3,
                        },
                        ow,
                        oh,
                        filter(kernel)?,
                    )?;
                    for (model, sr_stage, p) in &scaled {
                        if sr_stage != stage {
                            continue;
                        }
                        for quarter in 0..=4 {
                            blend(
                                View {
                                    data: &base,
                                    width: ow,
                                    height: oh,
                                    stride: ow * 3,
                                },
                                View {
                                    data: p,
                                    width: ow,
                                    height: oh,
                                    stride: ow * 3,
                                },
                                quarter,
                                &mut candidate.pixels,
                            )?;
                            let psnr =
                                classical::score(classical::Kind::Psnr, &reference, &candidate)?;
                            let similarity = z
                                .compute_with_ref_into(
                                    &prepared,
                                    &RgbSlice::new(candidate.pixels.as_chunks::<3>().0, ow, oh),
                                    &mut scratch,
                                )?
                                .score();
                            if !similarity.is_finite() {
                                return Err("Nonfinite zensim-A".into());
                            }
                            serde_json::to_writer(
                                &mut output,
                                &json!({"source_id":source["id"],"split":source["split"],"category":source["category"],"id":cell["id"],"q":cell["q"],"sampling":cell["sampling"],"edge":edge,"model":model,"repair":stage,"scale":scale,"filter":kernel,"intensity_quarter":quarter,"psnr":if psnr.is_infinite(){json!("inf")}else{json!(psnr)},"zensim_a":similarity,"rgb8_sha256":hash(&candidate.pixels)}),
                            )?;
                            output.write_all(b"\n")?;
                            rows += 1;
                        }
                    }
                }
            }
        }
    }
    output.flush()?;
    if let Some(f) = file {
        f.sync_all()?;
    }
    eprintln!("{rows} scored mixtures");
    Ok(())
}
