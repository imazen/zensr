//! Read-only partial-block padding counterfactual; runtime projection is unchanged.
#![forbid(unsafe_code)]
#[path = "../retained.rs"]
mod retained;
use imij::{Channels, Cicp, ImageInfo, SampleType, planes::SeekReader};
use imij_set::Reader;
use retained::{hash, image, number, text};
use serde_json::{Value, json};
use std::{
    error::Error,
    fs::{self, File},
    path::Path,
};
use zensr_micro::{consist::*, guards::guarded_merge};
type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn payload(reader: &mut Reader<SeekReader<File>>, r: &Value) -> Result<Vec<u8>> {
    let e = reader
        .get(r["key"].as_u64().ok_or("Missing key")?)?
        .ok_or("Missing entry")?;
    let mut p = vec![];
    reader.copy_to(&e, &mut p)?;
    let expected = r
        .get("encoded_sha256")
        .or_else(|| r.get("sha256"))
        .ok_or("Missing hash")?;
    if hash(&p) != expected.as_str().ok_or("Invalid hash")? {
        return Err("Entry bytes changed".into());
    }
    Ok(p)
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
fn rgb8(p: &[f32]) -> Vec<u8> {
    let n = p.len() / 3;
    (0..n)
        .flat_map(|i| (0..3).map(move |c| (p[c * n + i] * 255. + 0.5).clamp(0., 255.) as u8))
        .collect()
}
fn stats(p: &[u8]) -> Value {
    let spread: Vec<_> = p
        .as_chunks::<3>()
        .0
        .iter()
        .map(|v| u32::from(v.iter().max().unwrap() - v.iter().min().unwrap()))
        .collect();
    json!({"max_spread":spread.iter().max(),"mean_spread":spread.iter().sum::<u32>() as f64/spread.len() as f64})
}
fn retain(root: &Path, id: &str, p: &[u8], w: usize, h: usize) -> Result<Value> {
    let mut info = ImageInfo::new(w as u32, h as u32, Channels::Rgb, SampleType::U8);
    info.cicp = Cicp::SRGB;
    let bytes = imij::encode(
        p,
        w * 3,
        &info,
        &imij::Metadata::default(),
        &imij::EncodeConfig::default(),
    )?;
    let (decoded, rgb) = imij::decode(&bytes)?;
    if decoded != info || rgb != p {
        return Err("IMI roundtrip changed projection witness".into());
    }
    let path = format!("{id}.imi");
    fs::write(root.join(&path), &bytes)?;
    Ok(json!({"path":path,"pixels_sha256":hash(p),"encoded_sha256":hash(&bytes)}))
}

// Reconstruct the bitstream's complete blocks, including samples outside the
// visible frame. This supplies a measured counterfactual to edge replication.
fn decoded_blocks(cv: &CoeffView<'_>) -> Vec<f32> {
    let mut basis = [[0.; 8]; 8];
    for (u, row) in basis.iter_mut().enumerate() {
        for (x, v) in row.iter_mut().enumerate() {
            *v = 0.5
                * if u == 0 { 0.5f32.sqrt() } else { 1. }
                * (((2 * x + 1) * u) as f32 * std::f32::consts::PI / 16.).cos();
        }
    }
    let width = cv.blocks_wide * 8;
    let mut out = vec![0.; width * cv.blocks_high * 8];
    for by in 0..cv.blocks_high {
        for bx in 0..cv.blocks_wide {
            let block = &cv.coeffs[(by * cv.blocks_wide + bx) * 64..][..64];
            for y in 0..8 {
                for x in 0..8 {
                    let mut sum = 0.;
                    for (k, &v) in block.iter().enumerate() {
                        let nat = if cv.order == CoeffOrder::Zigzag {
                            ZIGZAG_TO_NATURAL[k]
                        } else {
                            k
                        };
                        sum += f32::from(v)
                            * f32::from(cv.quant[nat])
                            * basis[nat / 8][y]
                            * basis[nat % 8][x];
                    }
                    out[(by * 8 + y) * width + bx * 8 + x] = (sum + 128.) / 255.;
                }
            }
        }
    }
    out
}
fn encoded_padding(
    p: &mut [f32],
    w: usize,
    h: usize,
    cv: &CoeffView<'_>,
    cfg: &ProjectionConfig,
) -> Value {
    let (pw, ph) = (cv.blocks_wide * 8, cv.blocks_high * 8);
    let mut extended = decoded_blocks(cv);
    let row_means: Vec<_> = (0..ph.min(8))
        .map(|y| extended[y * pw..y * pw + w].iter().sum::<f32>() / w as f32 * 255.)
        .collect();
    for y in 0..h {
        extended[y * pw..y * pw + w].copy_from_slice(&p[y * w..(y + 1) * w]);
    }
    project_plane(&mut extended, pw, ph, cv, cfg);
    for y in 0..h {
        p[y * w..(y + 1) * w].copy_from_slice(&extended[y * pw..y * pw + w]);
    }
    json!({"visible_width":w,"visible_height":h,"block_row_means":row_means})
}
fn encoded_chroma(
    p: &mut [f32],
    w: usize,
    h: usize,
    cv: &CoeffView<'_>,
    cfg: &ProjectionConfig,
) -> Value {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut down = vec![0.; cw * ch];
    // Same box decimation and replicated correction as project_chroma_420.
    for y in 0..ch {
        for x in 0..cw {
            down[y * cw + x] = 0.25
                * (p[2 * y * w + 2 * x]
                    + p[2 * y * w + (2 * x + 1).min(w - 1)]
                    + p[(2 * y + 1).min(h - 1) * w + 2 * x]
                    + p[(2 * y + 1).min(h - 1) * w + (2 * x + 1).min(w - 1)]);
        }
    }
    let before = down.clone();
    let detail = encoded_padding(&mut down, cw, ch, cv, cfg);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let j = (y / 2) * cw + x / 2;
            p[i] = (p[i] + down[j] - before[j]).clamp(0., 1.);
        }
    }
    detail
}

fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 3 {
        return Err("usage: projection_edges SHARD_JSON NEW_OUTPUT".into());
    }
    let bank: Value = serde_json::from_slice(&fs::read(&a[1])?)?;
    let root = Path::new(&a[2]);
    fs::create_dir(root)?;
    let mut reader = Reader::open(SeekReader::new(File::open(text(&bank, "archive"))?))?;
    let mut rows = vec![];
    for cell in bank["cells"].as_array().ok_or("Missing cells")? {
        if number(cell, "edge") != 32
            || ![40, 50].contains(&number(cell, "q"))
            || text(cell, "sampling") != "420"
        {
            continue;
        }
        let (w, h) = (number(cell, "width"), number(cell, "height"));
        let n = w * h;
        let original = image(&mut reader, &cell["decoded"])?;
        let stages = cell["repair"].as_array().ok_or("Missing stages")?;
        let raw = stages
            .iter()
            .find(|r| text(r, "label").ends_with("/raw-canon4-f32"))
            .ok_or("Missing raw F32")?;
        let blob = payload(&mut reader, raw)?;
        let (info, values) = imij::decode(&blob)?;
        if info.sample != SampleType::F32 || hash(&values) != text(raw, "pixels_sha256") {
            return Err("Invalid raw witness".into());
        }
        let values: Vec<_> = values
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let mut guarded: Vec<_> = (0..3)
            .flat_map(|c| values.as_chunks::<3>().0.iter().map(move |v| v[c]))
            .collect();
        guarded_merge(
            &mut guarded,
            &planes(&original),
            h,
            w,
            1,
            &zensr_zenjpeg::RestoreConfig::default().guard,
        );
        let guard = stages
            .iter()
            .find(|r| text(r, "label").ends_with("/guard"))
            .ok_or("Missing guard")?;
        if hash(&rgb8(&guarded)) != text(guard, "pixels_sha256") {
            return Err("Guard replay differs".into());
        }
        let jpeg = payload(&mut reader, &cell["jpeg"])?;
        let probe = zenjpeg::detect::probe(&jpeg)?;
        let cfg = ProjectionConfig::with_slack_q(zensr_zenjpeg::slack_for(&probe))
            .with_slack_abs(zensr_zenjpeg::slack_abs_for(&probe));
        let coeff =
            zenjpeg::decoder::Decoder::new().decode_coefficients(&jpeg, enough::Unstoppable)?;
        for encoded in [false, true] {
            let (mut y, mut cb, mut cr) = (vec![0.; n], vec![0.; n], vec![0.; n]);
            rgb_to_ycbcr_planes(&guarded, n, &mut y, &mut cb, &mut cr);
            let mut details = vec![];
            for (ci, c) in coeff.components.iter().enumerate().take(3) {
                let quant =
                    coeff.quant_tables[c.quant_table_idx as usize].ok_or("Missing quant")?;
                let cv = CoeffView {
                    coeffs: &c.coeffs,
                    blocks_wide: c.blocks_wide,
                    blocks_high: c.blocks_high,
                    order: CoeffOrder::Zigzag,
                    quant: &quant,
                };
                let p = match ci {
                    0 => &mut y,
                    1 => &mut cb,
                    _ => &mut cr,
                };
                if encoded {
                    details.push(if ci == 0 {
                        encoded_padding(p, w, h, &cv, &cfg)
                    } else {
                        encoded_chroma(p, w, h, &cv, &cfg)
                    });
                } else if ci == 0 {
                    project_plane(p, w, h, &cv, &cfg);
                } else {
                    project_chroma_420(p, w, h, &cv, &cfg);
                }
            }
            let mut output = vec![0.; n * 3];
            ycbcr_to_rgb_planes(&y, &cb, &cr, &mut output, n);
            let pixels = rgb8(&output);
            if !encoded {
                let expected = stages
                    .iter()
                    .find(|r| text(r, "label").ends_with("/project-default"))
                    .ok_or("Missing projected stage")?;
                if hash(&pixels) != text(expected, "pixels_sha256") {
                    return Err("Default projection replay differs".into());
                }
            }
            let clean = bank["references"]
                .as_array()
                .ok_or("Missing references")?
                .iter()
                .find(|r| text(&r["lr"], "label") == "e32/clean")
                .ok_or("Missing clean")?;
            let clean = image(&mut reader, &clean["lr"])?;
            let z = zensim::Zensim::new(zensim::ZensimProfile::A).with_parallel(false);
            let score = z
                .compute(
                    &zensim::RgbSlice::new(clean.as_chunks::<3>().0, w, h),
                    &zensim::RgbSlice::new(pixels.as_chunks::<3>().0, w, h),
                )?
                .score();
            let mode = if encoded {
                "encoded-padding"
            } else {
                "replicated-padding"
            };
            rows.push(json!({"id":cell["id"],"padding":mode,"chroma":stats(&pixels),"zensim_a":score,
                "artifact":retain(root,&format!("q{}-{mode}",number(cell,"q")),&pixels,w,h)?,"details":details}));
        }
    }
    fs::write(
        root.join("RESULT.json"),
        serde_json::to_vec_pretty(
            &json!({"rows":rows,"source_record_sha256":hash(&fs::read(&a[1])?),"runtime_changed":false}),
        )?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stored_dc_reconstructs_full_block_without_clipping_hidden_values() {
        let q = [1u16; 64];
        for dc in [-2048i16, 0, 800] {
            let mut coefficients = [0i16; 64];
            coefficients[0] = dc;
            let cv = CoeffView {
                coeffs: &coefficients,
                blocks_wide: 1,
                blocks_high: 1,
                order: CoeffOrder::Natural,
                quant: &q,
            };
            let expected = (f32::from(dc) / 8. + 128.) / 255.;
            for v in decoded_blocks(&cv) {
                assert!((v - expected).abs() < 1e-6, "{dc}: {v} vs {expected}");
            }
        }
    }
}
