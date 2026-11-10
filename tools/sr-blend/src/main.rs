//! Score retained SR/filter mixtures without repeating inference or image I/O.
#![forbid(unsafe_code)]
// Keep the copied zenmetrics module and its tests unchanged; preserve its lint context.
#[allow(dead_code, clippy::too_many_arguments, clippy::bool_assert_comparison)]
#[path = "vendor/classical.rs"]
mod classical;
#[cfg(test)]
#[path = "vendor/studio601_gray.rs"]
mod metrics;
mod decode {
    pub(crate) struct Rgb8Image {
        pub width: u32,
        pub height: u32,
        pub pixels: Vec<u8>,
    }
}
mod pixel_ops;
use decode::Rgb8Image;
use pixel_ops::{View, blend, filter, float_resize, floats, rgb_view};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs,
    io::{BufWriter, Write},
    path::Path,
    sync::Mutex,
};
use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer};
use zensim::{RgbSlice, Zensim, ZensimProfile, ZensimScratch};
type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
#[derive(Deserialize)]
struct Artifact {
    path: String,
    sha256: String,
}
#[derive(Deserialize)]
struct Sr {
    model: String,
    png: Artifact,
    rgb8_sha256: String,
}
#[derive(Deserialize)]
struct Case {
    id: String,
    source_id: String,
    origin: String,
    category: String,
    q: u32,
    sampling: String,
    width: usize,
    height: usize,
    low_width: usize,
    low_height: usize,
    original: Artifact,
    low: Artifact,
    sr: Vec<Sr>,
}
#[derive(Deserialize)]
struct Manifest {
    scales: Vec<usize>,
    filters: Vec<String>,
    intensity_quarters: Vec<u8>,
    cases: Vec<Case>,
}
fn hash(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn read(a: &Artifact) -> Result<Vec<u8>> {
    let bytes = fs::read(&a.path)?;
    if hash(&bytes) != a.sha256 {
        return Err(format!("Artifact hash differs: {}", a.path).into());
    }
    Ok(bytes)
}
fn png(a: &Artifact, w: usize, h: usize, pixel_hash: &str) -> Result<Vec<u8>> {
    let bytes = read(a)?;
    let info = zenpng::probe(&bytes)?;
    if (info.width as usize, info.height as usize) != (w, h) {
        return Err("SR PNG shape differs".into());
    }
    let output = zenpng::decode(
        &bytes,
        &zenpng::PngDecodeConfig::strict(),
        &enough::Unstoppable,
    )?;
    let view = output.pixels.as_slice();
    let mut pixels = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        let row = view.row(y as u32);
        if row.len() != w * 3 {
            return Err("SR PNG must decode as RGB8".into());
        }
        pixels.extend_from_slice(row);
    }
    if hash(&pixels) != pixel_hash {
        return Err("SR decoded pixels differ from retained score identity".into());
    }
    Ok(pixels)
}
fn evaluate(case: &Case, m: &Manifest) -> Result<Vec<Value>> {
    if case.width * case.height > 1_048_576
        || case.width != case.low_width * 4
        || case.height != case.low_height * 4
    {
        return Err("Case exceeds admitted geometry".into());
    }
    let original = read(&case.original)?;
    if original.len() != case.width * case.height * 3 {
        return Err("Original RGB8 shape differs".into());
    }
    let encoded = read(&case.low)?;
    let low = if case.q == 0 {
        encoded
    } else {
        let image = zenjpeg::decoder::Decoder::new()
            .num_threads(1)
            .auto_orient(false)
            .decode(&encoded, enough::Unstoppable)?;
        if (image.width() as usize, image.height() as usize) != (case.low_width, case.low_height) {
            return Err("Low image shape differs".into());
        }
        image.into_pixels_u8().ok_or("Low image is not RGB8")?
    };
    let low = floats(rgb_view(&low, case.low_width, case.low_height))?;
    let mut outputs = Vec::new();
    for model in &case.sr {
        outputs.push((
            model,
            png(&model.png, case.width, case.height, &model.rgb8_sha256)?,
        ));
    }
    let z = Zensim::new(ZensimProfile::A).with_parallel(false);
    let mut scores = Vec::new();
    for &scale in &m.scales {
        let (w, h) = (case.low_width * scale, case.low_height * scale);
        let reference_pixels = if scale == 4 {
            original.clone()
        } else {
            let config =
                ResizeConfig::builder(case.width as u32, case.height as u32, w as u32, h as u32)
                    .format(PixelDescriptor::RGB8_SRGB)
                    .filter(Filter::CatmullRom)
                    .build();
            Resizer::new(&config).resize(&original)
        };
        let reference = Rgb8Image {
            width: w as u32,
            height: h as u32,
            pixels: reference_pixels,
        };
        let prepared =
            z.precompute_reference(&RgbSlice::new(reference.pixels.as_chunks::<3>().0, w, h))?;
        let mut scratch = ZensimScratch::new();
        // Resize each filter once; both models reuse the same baseline samples.
        let mut bases = Vec::new();
        for name in &m.filters {
            bases.push((
                name,
                float_resize(
                    View {
                        data: &low,
                        width: case.low_width,
                        height: case.low_height,
                        stride: case.low_width * 3,
                    },
                    w,
                    h,
                    filter(name)?,
                )?,
            ));
        }
        for (model, pixels) in &outputs {
            let sr = floats(rgb_view(pixels, case.width, case.height))?;
            let sr = if scale == 4 {
                sr
            } else {
                float_resize(
                    View {
                        data: &sr,
                        width: case.width,
                        height: case.height,
                        stride: case.width * 3,
                    },
                    w,
                    h,
                    Filter::Box,
                )?
            };
            let mut candidate = Rgb8Image {
                width: w as u32,
                height: h as u32,
                pixels: Vec::with_capacity(w * h * 3),
            };
            for (name, base) in &bases {
                for &q in &m.intensity_quarters {
                    blend(
                        View {
                            data: base,
                            width: w,
                            height: h,
                            stride: w * 3,
                        },
                        View {
                            data: &sr,
                            width: w,
                            height: h,
                            stride: w * 3,
                        },
                        q,
                        &mut candidate.pixels,
                    )?;
                    let psnr = classical::score(classical::Kind::Psnr, &reference, &candidate)?;
                    let similarity = z
                        .compute_with_ref_into(
                            &prepared,
                            &RgbSlice::new(candidate.pixels.as_chunks::<3>().0, w, h),
                            &mut scratch,
                        )?
                        .score();
                    if !similarity.is_finite() {
                        return Err("Nonfinite zensim-A score".into());
                    }
                    scores.push(json!({"id":case.id,"source_id":case.source_id,"origin":case.origin,"category":case.category,"q":case.q,"sampling":case.sampling,"model":model.model,"scale":scale,"filter":name,"intensity_quarter":q,"width":w,"height":h,"psnr":if psnr.is_infinite(){json!("inf")}else{json!(psnr)},"zensim_a":similarity,"rgb8_sha256":hash(&candidate.pixels)}));
                }
            }
        }
    }
    Ok(scores)
}
struct Sink {
    writer: BufWriter<fs::File>,
    cases: usize,
    rows: usize,
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("zensr-blend-eval inputs.json new-output-directory".into());
    }
    let bytes = fs::read(&args[1])?;
    let m: Manifest = serde_json::from_slice(&bytes)?;
    if m.scales != [2, 4]
        || m.filters != ["triangle", "mitchell", "catmull_rom", "lanczos2"]
        || m.intensity_quarters != [0, 1, 2, 3, 4]
        || m.cases.is_empty()
    {
        return Err("Frozen experiment grid differs".into());
    }
    let mut ids = std::collections::BTreeSet::new();
    for case in &m.cases {
        if !ids.insert(&case.id)
            || case.sr.iter().map(|s| s.model.as_str()).collect::<Vec<_>>() != ["spanf", "dscf"]
        {
            return Err("Duplicate case or incomplete model pair".into());
        }
    }
    let root = Path::new(&args[2]);
    fs::create_dir(root)?;
    let required = m
        .cases
        .iter()
        .map(|c| c.sr.len() * m.scales.len() * m.filters.len() * m.intensity_quarters.len())
        .sum::<usize>();
    let identity = json!({"manifest_sha256":hash(&bytes),"profile":"zensim-a","zensim_revision":"6060039d8c0ace71d47c722ea76233bb2d745a22","psnr_source":"imazen/zenmetrics@a3329089d95c1d0bd546dfc63213b220c03c6e82","binary_sha256":hash(&fs::read(std::env::current_exe()?)?),"required_cases":m.cases.len(),"required_rows":required,"maps":false,"mode":"resident original/LR/two SR images per case; resident reference and reusable scratch; ephemeral filter/intensity variants","threads":rayon::current_num_threads()});
    fs::write(
        root.join("IDENTITY.json"),
        serde_json::to_vec_pretty(&identity)?,
    )?;
    let sink = Mutex::new(Sink {
        writer: BufWriter::new(fs::File::create(root.join("scores.jsonl"))?),
        cases: 0,
        rows: 0,
    });
    let result=m.cases.par_iter().try_for_each(|case|->Result<()> {
        let scores=evaluate(case,&m)?;
        let mut sink=sink.lock().map_err(|_|"Score writer lock poisoned")?;
        for row in &scores {serde_json::to_writer(&mut sink.writer,row)?;sink.writer.write_all(b"\n")?;}
        sink.writer.flush()?;sink.rows+=scores.len();sink.cases+=1;
        fs::write(root.join("PROGRESS.json"),serde_json::to_vec_pretty(&json!({"pid":std::process::id(),"stage":"scoring","completed_cases":sink.cases,"required_cases":m.cases.len(),"completed_rows":sink.rows,"required_rows":required,"last_check_unix":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs()}))?)?;
        println!("{}/{} cases, {}/{} scores",sink.cases,m.cases.len(),sink.rows,required);
        Ok(())
    });
    if let Err(error) = result {
        fs::write(root.join("FAILED.txt"), error.to_string())?;
        return Err(error);
    }
    let mut sink = sink
        .into_inner()
        .map_err(|_| "Score writer lock poisoned")?;
    sink.writer.flush()?;
    sink.writer.get_ref().sync_all()?;
    if sink.rows != required || sink.cases != m.cases.len() {
        return Err("Incomplete score coverage".into());
    }
    fs::write(
        root.join("COMPLETE.json"),
        serde_json::to_vec_pretty(
            &json!({"cases":sink.cases,"rows":sink.rows,"scores_sha256":hash(&fs::read(root.join("scores.jsonl"))?),"identity_sha256":hash(&fs::read(root.join("IDENTITY.json"))?)}),
        )?,
    )?;
    fs::write(
        root.join("PROGRESS.json"),
        serde_json::to_vec_pretty(&json!({
            "pid":null,"stage":"completed","completed_cases":sink.cases,"required_cases":m.cases.len(),
            "completed_rows":sink.rows,"required_rows":required,
            "last_check_unix":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs()
        }))?,
    )?;
    Ok(())
}
fn main() -> Result<()> {
    run()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn blend_endpoints_and_strides_preserve_pixels() {
        let base = [0.0, 0.5, 1.0, f32::NAN, 0.2, 0.4, 0.6];
        let sr = [1.0, 0.0, 0.5, f32::NAN, 0.8, 0.6, 0.4];
        let a = View {
            data: &base,
            width: 1,
            height: 2,
            stride: 4,
        };
        let b = View {
            data: &sr,
            width: 1,
            height: 2,
            stride: 4,
        };
        let mut out = Vec::new();
        blend(a, b, 0, &mut out).unwrap();
        assert_eq!(out, [0, 128, 255, 51, 102, 153]);
        blend(a, b, 4, &mut out).unwrap();
        assert_eq!(out, [255, 0, 128, 204, 153, 102]);
        blend(a, b, 2, &mut out).unwrap();
        assert_eq!(out, [128, 64, 191, 128, 128, 128]);
        assert!(blend(a, b, 5, &mut out).is_err());
    }
    #[test]
    fn resident_reference_matches_cold_zensim_a() {
        for (w, h) in [(19, 23), (512, 384)] {
            let pixels: Vec<[u8; 3]> = (0..w * h)
                .map(|i| [(i * 71) as u8, (i * 13) as u8, (i * 193) as u8])
                .collect();
            let mut changed = pixels.clone();
            changed[81] = [30, 40, 50];
            let z = Zensim::new(ZensimProfile::A).with_parallel(false);
            let source = RgbSlice::new(&pixels, w, h);
            let candidate = RgbSlice::new(&changed, w, h);
            let prepared = z.precompute_reference(&source).unwrap();
            let mut scratch = ZensimScratch::new();
            assert_eq!(z.compute(&source, &source).unwrap().score(), 100.0);
            assert_eq!(
                z.compute_with_ref_into(&prepared, &source, &mut scratch)
                    .unwrap()
                    .score(),
                100.0
            );
            assert_eq!(
                z.compute(&source, &candidate).unwrap().score(),
                z.compute_with_ref_into(&prepared, &candidate, &mut scratch)
                    .unwrap()
                    .score()
            );
        }
    }
    #[test]
    fn filter_resize_ignores_stride_padding() {
        let tight: Vec<f32> = (0..27).map(|x| x as f32 / 27.0).collect();
        let mut padded = vec![f32::NAN; 12 * 3];
        for y in 0..3 {
            padded[y * 12..y * 12 + 9].copy_from_slice(&tight[y * 9..y * 9 + 9]);
        }
        for name in ["triangle", "mitchell", "catmull_rom", "lanczos2"] {
            let a = float_resize(
                View {
                    data: &tight,
                    width: 3,
                    height: 3,
                    stride: 9,
                },
                12,
                12,
                filter(name).unwrap(),
            )
            .unwrap();
            let b = float_resize(
                View {
                    data: &padded,
                    width: 3,
                    height: 3,
                    stride: 12,
                },
                12,
                12,
                filter(name).unwrap(),
            )
            .unwrap();
            assert_eq!(a, b);
            assert!(a.iter().all(|v| v.is_finite()));
        }
    }
    #[test]
    fn psnr_identity_and_known_error_use_zenmetrics() {
        let a = Rgb8Image {
            width: 8,
            height: 8,
            pixels: vec![0; 8 * 8 * 3],
        };
        let b = Rgb8Image {
            width: 8,
            height: 8,
            pixels: vec![1; 8 * 8 * 3],
        };
        assert!(
            classical::score(classical::Kind::Psnr, &a, &a)
                .unwrap()
                .is_infinite()
        );
        assert_eq!(
            classical::score(classical::Kind::Psnr, &a, &b).unwrap(),
            10.0f64 * (255.0f64 * 255.0).log10()
        );
    }
}
