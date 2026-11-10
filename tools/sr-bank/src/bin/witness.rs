//! Export lossless, hash-checked bank witnesses for the existing browser viewer.
#![forbid(unsafe_code)]
#[allow(dead_code)]
#[path = "../../../sr-blend/src/pixel_ops.rs"]
mod pixel_ops;
#[path = "../retained.rs"]
mod retained;
use imij::planes::SeekReader;
use imij_set::Reader;
use pixel_ops::{View, blend, filter, float_resize, floats, rgb_view};
use retained::{hash, image, number, text};
use rgb::FromSlice;
use serde_json::{Value, json};
use std::{
    error::Error,
    fs::{self, File},
    io::Write,
    path::Path,
};
type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn png(root: &Path, pixels: &[u8], w: usize, h: usize) -> Result<Value> {
    let config = zenpng::EncodeConfig::default()
        .with_compression(zenpng::Compression::Fast)
        .with_parallel(false)
        .with_srgb_intent(Some(0));
    let bytes = zenpng::encode_rgb8(
        imgref::ImgRef::new(pixels.as_rgb(), w, h),
        None,
        &config,
        &enough::Unstoppable,
        &enough::Unstoppable,
    )?;
    let decoded = zenpng::decode(
        &bytes,
        &zenpng::PngDecodeConfig::strict(),
        &enough::Unstoppable,
    )?;
    if (decoded.info.width as usize, decoded.info.height as usize) != (w, h) {
        return Err("Witness PNG changed dimensions".into());
    }
    let view = decoded.pixels.as_slice();
    for y in 0..h {
        let row = view.row(y as u32);
        let expected = &pixels[y * w * 3..(y + 1) * w * 3];
        // The pinned encoder can losslessly reduce RGB to Gray. Use the same
        // exact channel replication as the bank's grayscale source admission.
        let equal = if decoded.info.color_type == 0 && row.len() == w {
            row.iter()
                .flat_map(|&v| [v, v, v])
                .eq(expected.iter().copied())
        } else {
            row == expected
        };
        if !equal {
            return Err("Witness PNG changed pixels".into());
        }
    }
    let path = format!("images/{}.png", hash(&bytes));
    let destination = root.join(&path);
    if destination.exists() {
        if fs::read(&destination)? != bytes {
            return Err("Witness PNG identity conflict".into());
        }
    } else {
        let mut file = File::create_new(&destination)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
    }
    Ok(
        json!({"path":path,"width":w,"height":h,"png_sha256":hash(&bytes),"pixels_sha256":hash(pixels)}),
    )
}

fn export(reader: &mut Reader<SeekReader<File>>, r: &Value, root: &Path) -> Result<Value> {
    let p = image(reader, r)?;
    png(root, &p, number(r, "width"), number(r, "height"))
}

fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 3 {
        return Err("usage: witness SHARD_JSON NEW_OUTPUT_DIRECTORY".into());
    }
    let record: Value = serde_json::from_slice(&fs::read(&a[1])?)?;
    let root = Path::new(&a[2]);
    fs::create_dir(root)?;
    fs::create_dir(root.join("images"))?;
    fs::create_dir(root.join("jpegs"))?;
    let mut reader = Reader::open(SeekReader::new(File::open(text(&record, "archive"))?))?;
    let mut cases = vec![];
    for cell in record["cells"].as_array().ok_or("Missing cells")? {
        if ![40, 50].contains(&number(cell, "q")) || text(cell, "sampling") != "420" {
            continue;
        }
        let jpeg = &cell["jpeg"];
        let entry = reader
            .get(jpeg["key"].as_u64().ok_or("Missing JPEG key")?)?
            .ok_or("Missing retained JPEG")?;
        let mut jpeg_bytes = vec![];
        reader.copy_to(&entry, &mut jpeg_bytes)?;
        if hash(&jpeg_bytes) != text(jpeg, "sha256") {
            return Err("Retained JPEG identity changed".into());
        }
        let jpeg_path = format!("jpegs/{}.jpg", hash(&jpeg_bytes));
        let mut file = File::create_new(root.join(&jpeg_path))?;
        file.write_all(&jpeg_bytes)?;
        file.sync_all()?;
        let (w, h) = (number(cell, "width"), number(cell, "height"));
        let reference = record["references"]
            .as_array()
            .ok_or("Missing references")?
            .iter()
            .find(|r| text(&r["hr"], "label") == format!("e{}/hr", number(cell, "edge")))
            .ok_or("Missing HR reference")?;
        let mut row = json!({"id":cell["id"],"edge":cell["edge"],"q":cell["q"],"sampling":cell["sampling"],
            "jpeg":jpeg_path,"reference":export(&mut reader,&reference["hr"],root)?,
            "clean_lr":export(&mut reader,&reference["lr"],root)?,
            "decoded_lr":export(&mut reader,&cell["decoded"],root)?,"sr":[],"filters":[]});
        let repaired = cell["repair"]
            .as_array()
            .ok_or("Missing repair")?
            .iter()
            .find(|r| text(r, "label").ends_with("/project-default"))
            .ok_or("Missing full repair")?;
        row["repaired_lr"] = export(&mut reader, repaired, root)?;
        for r in cell["sr"].as_array().ok_or("Missing SR")? {
            if !["direct", "repaired"].contains(&text(r, "repair")) {
                continue;
            }
            let mut v = export(&mut reader, r, root)?;
            v["model"] = r["model"].clone();
            v["repair"] = r["repair"].clone();
            v["scores"] = r["scores"].clone();
            row["sr"].as_array_mut().unwrap().push(v);
        }
        let p = image(&mut reader, &cell["decoded"])?;
        let f = floats(rgb_view(&p, w, h))?;
        for name in ["triangle", "mitchell", "catmull_rom", "lanczos2"] {
            let resized = float_resize(
                View {
                    data: &f,
                    width: w,
                    height: h,
                    stride: w * 3,
                },
                w * 4,
                h * 4,
                filter(name)?,
            )?;
            let v = View {
                data: &resized,
                width: w * 4,
                height: h * 4,
                stride: w * 12,
            };
            let mut pixels = vec![];
            blend(v, v, 0, &mut pixels)?;
            let mut entry = png(root, &pixels, w * 4, h * 4)?;
            entry["filter"] = json!(name);
            row["filters"].as_array_mut().unwrap().push(entry);
        }
        cases.push(row);
    }
    let output = json!({"source_record_sha256":hash(&fs::read(&a[1])?),"source":record["source"],"cases":cases});
    fs::write(
        root.join("WITNESSES.json"),
        serde_json::to_vec_pretty(&output)?,
    )?;
    Ok(())
}
