//! Offline candidate-bank worker. Pixel payloads stay in pipes and lossless archives.
#![forbid(unsafe_code)]
#[allow(dead_code, clippy::too_many_arguments, clippy::bool_assert_comparison)]
#[path = "../../sr-blend/src/vendor/classical.rs"]
mod classical;
#[cfg(test)]
#[path = "../../sr-blend/src/vendor/studio601_gray.rs"]
mod metrics;
mod decode {
    pub(crate) struct Rgb8Image {
        pub width: u32,
        pub height: u32,
        pub pixels: Vec<u8>,
    }
}
use imij::{Channels, Cicp, ImageInfo, SampleType};
use imij_set::{EntryMetadata, Reader, Summary, Writer};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs::File,
    io::{self, BufRead, BufWriter, Read, Write},
};
use zenanalyze_api::{NamedFeature, Request, Select};
use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer};
use zensr_micro::{adopted::AdoptedModel, consist::*, guards::guarded_merge};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
type ArchiveWriter = Writer<BufWriter<File>>;

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn number(v: &Value, name: &str) -> usize {
    v[name].as_u64().expect("integer argument") as usize
}
fn text<'a>(v: &'a Value, name: &str) -> &'a str {
    v[name].as_str().expect("string argument")
}
fn reply(out: &mut impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *out, value)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}
fn shape(w: usize, h: usize) -> Result<usize> {
    w.checked_mul(h)
        .filter(|&n| w > 0 && h > 0 && n <= 64 * 1024 * 1024)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| "Image exceeds worker geometry budget".into())
}
fn fit(w: usize, h: usize, edge: usize) -> Result<(usize, usize)> {
    shape(w, h)?;
    if !(16..=256).contains(&edge) {
        return Err("LR edge outside study bounds".into());
    }
    let long = w.max(h);
    Ok((
        ((w * edge + long / 2) / long).max(1),
        ((h * edge + long / 2) / long).max(1),
    ))
}
fn resize(p: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Vec<u8> {
    let c = ResizeConfig::builder(w as u32, h as u32, ow as u32, oh as u32)
        .format(PixelDescriptor::RGB8_SRGB)
        .filter(Filter::CatmullRom)
        .build();
    Resizer::new(&c).resize(p)
}
fn planes(p: &[u8]) -> Vec<f32> {
    (0..3)
        .flat_map(|c| {
            p.as_chunks::<3>()
                .0
                .iter()
                .map(move |v| f32::from(v[c]) / 255.)
        })
        .collect()
}
fn rgb8(p: &[f32]) -> Result<Vec<u8>> {
    if !p.len().is_multiple_of(3) || p.iter().any(|x| !x.is_finite()) {
        return Err("Invalid planar RGB".into());
    }
    let n = p.len() / 3;
    Ok((0..n)
        .flat_map(|i| (0..3).map(move |c| (p[c * n + i] * 255. + 0.5).clamp(0., 255.) as u8))
        .collect())
}
fn features(p: &[u8], w: usize, h: usize) -> Result<Value> {
    let offer = zenanalyze::offer_for_request(p, w as u32, h as u32, &Request::new(Select::All))?;
    let names: Vec<_> = offer
        .features()
        .iter()
        .map(|f| NamedFeature::parse(f.qualified_name()).unwrap())
        .collect();
    let ask = Request::new(Select::Features(&names));
    if !offer.satisfies(&ask) {
        return Err("Extracted offer does not satisfy its qualified request".into());
    }
    let reused = offer
        .reuse_for(&ask)
        .ok_or("Qualified feature reuse failed")?;
    let own: Vec<_> = offer
        .features()
        .iter()
        .map(|f| f.value().to_f32())
        .collect();
    if reused != own || own.iter().any(|x| !x.is_finite()) {
        return Err("Invalid reused feature vector".into());
    }
    let provenance = offer.provenance();
    Ok(
        json!({"schema_hash":offer.schema_hash(),"config_hash":provenance.config_hash(),
        "descriptor_hash":provenance.descriptor_hash(),"analyzer_version":provenance.analyzer_version(),
        "names":names.iter().map(|n|n.qualified_name()).collect::<Vec<_>>(),"values":own,
        "pixels_sha256":hash(p),"width":w,"height":h,"stage":"decoded-model-input-rgb8"}),
    )
}
fn score(a: &[u8], b: &[u8], w: usize, h: usize, artifact: bool) -> Result<Value> {
    if a.len() != shape(w, h)? || b.len() != a.len() {
        return Err("Score geometry differs".into());
    }
    let psnr = classical::score(
        classical::Kind::Psnr,
        &decode::Rgb8Image {
            width: w as u32,
            height: h as u32,
            pixels: a.to_vec(),
        },
        &decode::Rgb8Image {
            width: w as u32,
            height: h as u32,
            pixels: b.to_vec(),
        },
    )?;
    let metric = zensim::Zensim::new(zensim::ZensimProfile::A).with_parallel(false);
    let za = metric
        .compute(
            &zensim::RgbSlice::new(a.as_chunks::<3>().0, w, h),
            &zensim::RgbSlice::new(b.as_chunks::<3>().0, w, h),
        )?
        .score();
    let ssim = if artifact {
        let aa: Vec<[u8; 3]> = a.as_chunks::<3>().0.to_vec();
        let bb: Vec<[u8; 3]> = b.as_chunks::<3>().0.to_vec();
        Some(fast_ssim2::compute_ssimulacra2(
            imgref::Img::new(aa, w, h).as_ref(),
            imgref::Img::new(bb, w, h).as_ref(),
        )?)
    } else {
        None
    };
    Ok(
        json!({"psnr":if psnr.is_infinite(){json!("inf")}else{json!(psnr)},"zensim_a":za,"ssim2":ssim}),
    )
}

struct State {
    source: Vec<u8>,
    sw: usize,
    sh: usize,
    edge: usize,
    reference: Vec<u8>,
    clean: Vec<u8>,
    input: Vec<u8>,
    jpeg: Vec<u8>,
    w: usize,
    h: usize,
    writer: Option<ArchiveWriter>,
    key: u64,
}
impl State {
    fn new() -> Self {
        Self {
            source: vec![],
            sw: 0,
            sh: 0,
            edge: 0,
            reference: vec![],
            clean: vec![],
            input: vec![],
            jpeg: vec![],
            w: 0,
            h: 0,
            writer: None,
            key: 0,
        }
    }
    fn jpeg_record(&mut self, label: &str) -> Result<Value> {
        let mut meta = EntryMetadata::default();
        meta.filename = Some(format!("{label}.jpg"));
        meta.attributes
            .insert("sha256".into(), imij_set::Value::Text(hash(&self.jpeg)));
        self.writer
            .as_mut()
            .ok_or("Archive not open")?
            .insert_with_metadata(
                self.key,
                self.jpeg.len() as u64,
                0,
                Summary::default(),
                &meta,
                &mut &self.jpeg[..],
            )?;
        let record =
            json!({"key":self.key,"label":label,"bytes":self.jpeg.len(),"sha256":hash(&self.jpeg)});
        self.key += 1;
        Ok(record)
    }
    fn put(
        &mut self,
        p: &[u8],
        w: usize,
        h: usize,
        sample: SampleType,
        label: &str,
    ) -> Result<Value> {
        if p.len() != shape(w, h)? * sample.bytes() {
            return Err("Archive pixel payload differs".into());
        }
        let mut info = ImageInfo::new(w as u32, h as u32, Channels::Rgb, sample);
        info.cicp = Cicp::SRGB;
        let mut config = imij::EncodeConfig::default();
        config.effort = 1.;
        let encoded = imij::encode(
            p,
            info.row_bytes(),
            &info,
            &imij::Metadata::default(),
            &config,
        )?;
        let (decoded, pixels) = imij::decode(&encoded)?;
        if decoded != info || pixels != p {
            return Err("Lossless storage round trip changed pixels".into());
        }
        let mut meta = EntryMetadata::default();
        meta.filename = Some(format!("{label}.imi"));
        meta.attributes
            .insert("pixels_sha256".into(), imij_set::Value::Text(hash(p)));
        let mut summary = Summary::default();
        summary.width = w as u32;
        summary.height = h as u32;
        summary.channels = Channels::Rgb as u8;
        summary.sample = sample as u8;
        summary.cicp = Some(Cicp::SRGB);
        summary.image_count = Some(1);
        self.writer
            .as_mut()
            .ok_or("Archive not open")?
            .insert_with_metadata(
                self.key,
                encoded.len() as u64,
                0,
                summary,
                &meta,
                &mut &encoded[..],
            )?;
        let record = json!({"key":self.key,"label":label,"width":w,"height":h,"sample":format!("{sample:?}"),"pixel_bytes":p.len(),"encoded_bytes":encoded.len(),"pixels_sha256":hash(p),"encoded_sha256":hash(&encoded)});
        self.key += 1;
        Ok(record)
    }
    fn projected(&self, raw: &[f32], slack: f32) -> Result<Vec<f32>> {
        if self.jpeg.is_empty() {
            return Err("Projection requires encoded JPEG coefficients".into());
        }
        let w = self.w;
        let h = self.h;
        let n = w * h;
        let probe = zenjpeg::detect::probe(&self.jpeg)?;
        let mut out = raw.to_vec();
        guarded_merge(
            &mut out,
            &planes(&self.input),
            h,
            w,
            1,
            &zensr_zenjpeg::RestoreConfig::default().guard,
        );
        if slack == 0. {
            return Ok(out);
        }
        let (mut y, mut cb, mut cr) = (vec![0.; n], vec![0.; n], vec![0.; n]);
        rgb_to_ycbcr_planes(&out, n, &mut y, &mut cb, &mut cr);
        let coeff = zenjpeg::decoder::Decoder::new()
            .decode_coefficients(&self.jpeg, enough::Unstoppable)?;
        let maxh = coeff
            .components
            .iter()
            .map(|c| c.h_samp)
            .max()
            .ok_or("Missing component")?;
        let maxv = coeff
            .components
            .iter()
            .map(|c| c.v_samp)
            .max()
            .ok_or("Missing component")?;
        let cfg = ProjectionConfig::with_slack_q(zensr_zenjpeg::slack_for(&probe) * slack)
            .with_slack_abs(zensr_zenjpeg::slack_abs_for(&probe) * slack);
        for (ci, c) in coeff.components.iter().enumerate().take(3) {
            let q = coeff.quant_tables[c.quant_table_idx as usize]
                .ok_or("Missing quantization table")?;
            let view = CoeffView {
                coeffs: &c.coeffs,
                blocks_wide: c.blocks_wide,
                blocks_high: c.blocks_high,
                order: CoeffOrder::Zigzag,
                quant: &q,
            };
            let p = match ci {
                0 => &mut y,
                1 => &mut cb,
                _ => &mut cr,
            };
            if c.h_samp == maxh && c.v_samp == maxv {
                project_plane(p, w, h, &view, &cfg);
            } else if c.h_samp * 2 == maxh && c.v_samp * 2 == maxv {
                project_chroma_420(p, w, h, &view, &cfg);
            } else {
                return Err("Study requires 444 or 420 JPEG sampling".into());
            }
        }
        ycbcr_to_rgb_planes(&y, &cb, &cr, &mut out, n);
        Ok(out)
    }
}

fn main() -> Result<()> {
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let mut state = State::new();
    reply(&mut out, &json!({"ready":true}))?;
    loop {
        let mut line = String::new();
        if input.read_line(&mut line)? == 0 {
            break;
        }
        let v: Value = serde_json::from_str(&line)?;
        match text(&v, "action") {
            "catalog" => {
                let mut reader = Reader::open(imij::planes::SeekReader::new(File::open(text(
                    &v, "archive",
                ))?))?;
                let mut listing = reader.listing()?;
                let mut entries = vec![];
                while let Some(e) = listing.next(&mut reader)? {
                    let s = e.summary();
                    entries.push(json!({"key":e.key(),"bytes":e.len(),"width":s.width,"height":s.height,"channels":s.channels,"sample":s.sample,"filename":e.metadata().filename}));
                }
                reply(&mut out, &json!({"entries":entries}))?;
            }
            "source" => {
                if state.writer.is_some() {
                    return Err("Finish the previous archive first".into());
                }
                let mut reader = Reader::open(imij::planes::SeekReader::new(File::open(text(
                    &v, "archive",
                ))?))?;
                let entry = reader
                    .get(v["key"].as_u64().ok_or("Missing source key")?)?
                    .ok_or("Source entry missing")?;
                if entry.len() > 512 * 1024 * 1024 {
                    return Err("Source entry exceeds byte budget".into());
                }
                let mut blob = vec![];
                reader.copy_to(&entry, &mut blob)?;
                let (info, _) = imij::probe(&blob)?;
                shape(info.width as usize, info.height as usize)?;
                if !matches!(info.channels, Channels::Rgb | Channels::Gray)
                    || info.sample != SampleType::U8
                    || info.cicp != Cicp::SRGB
                {
                    return Err("Cohort requires rendered sRGB RGB8 sources".into());
                }
                let (info, p) = imij::decode(&blob)?;
                state = State::new();
                state.source = if info.channels == Channels::Gray {
                    p.iter().flat_map(|&v| [v, v, v]).collect()
                } else {
                    p
                };
                state.sw = info.width as usize;
                state.sh = info.height as usize;
                state.writer = Some(Writer::new(BufWriter::new(File::create_new(text(
                    &v, "output",
                ))?)));
                reply(
                    &mut out,
                    &json!({"source_sha256":hash(&blob),"source_pixels_sha256":hash(&state.source),"width":state.sw,"height":state.sh}),
                )?;
            }
            "prepare" => {
                let edge = number(&v, "edge");
                let (w, h) = fit(state.sw, state.sh, edge)?;
                if edge != state.edge {
                    state.reference = resize(&state.source, state.sw, state.sh, w * 4, h * 4);
                    state.clean = resize(&state.reference, w * 4, h * 4, w, h);
                    state.edge = edge;
                    state.w = w;
                    state.h = h;
                }
                let q = number(&v, "q");
                state.jpeg = if q == 0 {
                    vec![]
                } else {
                    let ss = match text(&v, "sampling") {
                        "420" => zenjpeg::encoder::ChromaSubsampling::Quarter,
                        "444" => zenjpeg::encoder::ChromaSubsampling::None,
                        _ => return Err("Unknown JPEG sampling".into()),
                    };
                    let rgb: Vec<rgb::Rgb<u8>> = state
                        .clean
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .map(|p| rgb::Rgb::new(p[0], p[1], p[2]))
                        .collect();
                    zenjpeg::encoder::EncoderConfig::ycbcr(q as f32, ss)
                        .encode(&rgb, w as u32, h as u32)?
                };
                state.input = if q == 0 {
                    state.clean.clone()
                } else {
                    zenjpeg::decoder::Decoder::new()
                        .num_threads(1)
                        .auto_orient(false)
                        .decode(&state.jpeg, enough::Unstoppable)?
                        .into_pixels_u8()
                        .ok_or("Expected RGB8 JPEG decode")?
                };
                let f = features(&state.input, w, h)?;
                let encoded = if q == 0 {
                    Value::Null
                } else {
                    state.jpeg_record(text(&v, "id"))?
                };
                reply(
                    &mut out,
                    &json!({"width":w,"height":h,"bytes":state.input.len(),"features":f,"jpeg":encoded,"baseline":score(&state.clean,&state.input,w,h,true)?}),
                )?;
                out.write_all(&state.input)?;
                out.flush()?;
            }
            "raw" => {
                let n = state.w * state.h * 3;
                let mut b = vec![0; n * 4];
                input.read_exact(&mut b)?;
                let raw: Vec<f32> = b
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|a| f32::from_le_bytes(*a))
                    .collect();
                rgb8(&raw)?;
                let mut raw_interleaved = Vec::with_capacity(n * 4);
                for i in 0..n / 3 {
                    for c in 0..3 {
                        raw_interleaved.extend_from_slice(&raw[c * (n / 3) + i].to_le_bytes());
                    }
                }
                let raw_record = state.put(
                    &raw_interleaved,
                    state.w,
                    state.h,
                    SampleType::F32,
                    &format!("{}/raw-canon4-f32", text(&v, "id")),
                )?;
                let mut records = vec![raw_record];
                let mut selected = vec![];
                for (name, slack) in [
                    ("guard", 0.),
                    ("project-half", 0.5),
                    ("project-default", 1.),
                    ("project-double", 2.),
                ] {
                    let image = rgb8(&state.projected(&raw, slack)?)?;
                    let mut r = state.put(
                        &image,
                        state.w,
                        state.h,
                        SampleType::U8,
                        &format!("{}/{name}", text(&v, "id")),
                    )?;
                    r["scores"] = score(&state.clean, &image, state.w, state.h, true)?;
                    records.push(r);
                    if slack == 1. {
                        selected = image;
                    }
                }
                reply(&mut out, &json!({"records":records,"bytes":selected.len()}))?;
                out.write_all(&selected)?;
                out.flush()?;
            }
            "put" => {
                let w = number(&v, "width");
                let h = number(&v, "height");
                let n = shape(w, h)?;
                let mut b = vec![0; n];
                input.read_exact(&mut b)?;
                let mut r = state.put(&b, w, h, SampleType::U8, text(&v, "label"))?;
                if v["score"].as_bool() == Some(true) {
                    let reference = if w == state.w && h == state.h {
                        &state.clean
                    } else if w == state.w * 4 && h == state.h * 4 {
                        &state.reference
                    } else {
                        return Err("Unknown scoring scale".into());
                    };
                    r["scores"] = score(reference, &b, w, h, w == state.w)?;
                }
                reply(&mut out, &r)?;
            }
            "references" => {
                let hr = state.reference.clone();
                let lr = state.clean.clone();
                let a = state.put(
                    &hr,
                    state.w * 4,
                    state.h * 4,
                    SampleType::U8,
                    text(&v, "hr_label"),
                )?;
                let b = state.put(&lr, state.w, state.h, SampleType::U8, text(&v, "lr_label"))?;
                reply(&mut out, &json!({"hr":a,"lr":b}))?;
            }
            "native" => {
                let weights = zensr_micro::decode_all_f16(&std::fs::read(text(&v, "weights"))?);
                let m = AdoptedModel::load_compact(&weights, 24, 8, 1)?;
                let mut p = vec![0.; state.input.len()];
                m.forward(&planes(&state.input), state.h, state.w, &mut p);
                let b: Vec<u8> = p.iter().flat_map(|x| x.to_le_bytes()).collect();
                reply(&mut out, &json!({"bytes":b.len()}))?;
                out.write_all(&b)?;
                out.flush()?;
            }
            "compare" => {
                let w = number(&v, "width");
                let h = number(&v, "height");
                let n = shape(w, h)?;
                let mut a = vec![0; n];
                let mut b = vec![0; n];
                input.read_exact(&mut a)?;
                input.read_exact(&mut b)?;
                reply(&mut out, &score(&a, &b, w, h, true)?)?;
            }
            "finish" => {
                let writer = state.writer.take().ok_or("Archive not open")?;
                let (sink, _) = writer.finish()?;
                sink.into_inner()?.sync_all()?;
                reply(&mut out, &json!({"entries":state.key}))?;
            }
            _ => return Err("Unknown action".into()),
        }
    }
    if state.writer.is_some() {
        return Err("Unfinished archive on end of input".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fit_preserves_complete_frame_and_orientation() {
        for (w, h) in [(4000, 3000), (3000, 4000), (16001, 701), (19, 23)] {
            for e in [32, 64, 128] {
                let (a, b) = fit(w, h, e).unwrap();
                assert_eq!(a.max(b), e);
                assert!(
                    (a as f64 / b as f64 - w as f64 / h as f64).abs()
                        <= w as f64 / h as f64 / (b as f64) + 1. / b as f64
                );
            }
        }
    }
    #[test]
    fn existing_offer_contract_reuses_qualified_values() {
        let p: Vec<u8> = (0..64 * 48 * 3).map(|i| (i * 37 % 251) as u8).collect();
        let f = features(&p, 64, 48).unwrap();
        assert!(!f["names"].as_array().unwrap().is_empty());
        assert_eq!(f["pixels_sha256"], hash(&p));
    }
    #[test]
    fn undefined_flat_spectrum_is_missing_and_common_requests_still_reuse() {
        use zenanalyze_api::{OwnedFeatureResult, OwnedOffer, Provenance};
        let p: Vec<u8> = (0..32 * 32 * 3).map(|i| (i * 37 % 251) as u8).collect();
        let full = features(&p, 32, 32).unwrap();
        let flat = features(&vec![128; 32 * 32 * 3], 32, 32).unwrap();
        let owned = |v: &Value| {
            OwnedOffer::new(
                v["names"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(v["values"].as_array().unwrap())
                    .map(|(n, x)| {
                        OwnedFeatureResult::new(n.as_str().unwrap(), x.as_f64().unwrap() as f32)
                    })
                    .collect(),
                Provenance::new("test"),
            )
        };
        let full_offer = owned(&full);
        let flat_offer = owned(&flat);
        let spectral = full_offer
            .features()
            .iter()
            .find(|f| f.name() == "spectral_slope_y")
            .unwrap();
        let spectral_want = [NamedFeature::parse(spectral.qualified_name()).unwrap()];
        assert!(!flat_offer.satisfies(&Request::new(Select::Features(&spectral_want))));
        let common: Vec<_> = flat_offer
            .features()
            .iter()
            .map(|f| NamedFeature::parse(f.qualified_name()).unwrap())
            .collect();
        let request = Request::new(Select::Features(&common));
        assert!(full_offer.satisfies(&request));
        assert!(flat_offer.satisfies(&request));
        assert_eq!(flat_offer.reuse_for(&request).unwrap().len(), common.len());
    }
    #[test]
    fn storage_is_exact_for_float_samples() {
        let p: Vec<u8> = [0.0f32, -0.0, 0.000001, 1.0, -0.4, 1.5]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let info = ImageInfo::new(2, 1, Channels::Rgb, SampleType::F32);
        let b = imij::encode(
            &p,
            info.row_bytes(),
            &info,
            &imij::Metadata::default(),
            &imij::EncodeConfig::default(),
        )
        .unwrap();
        assert_eq!(imij::decode(&b).unwrap().1, p);
    }
}
