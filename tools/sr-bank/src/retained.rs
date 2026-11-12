//! Hash-checked access to retained RGB8 bank images.
use imij::{Channels, Cicp, SampleType, planes::SeekReader};
use imij_set::Reader;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{error::Error, fs::File};
type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
pub(crate) fn number(v: &Value, k: &str) -> usize {
    v[k].as_u64().unwrap() as usize
}
pub(crate) fn text<'a>(v: &'a Value, k: &str) -> &'a str {
    v[k].as_str().unwrap()
}
pub(crate) fn hash(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}
pub(crate) fn image(reader: &mut Reader<SeekReader<File>>, record: &Value) -> Result<Vec<u8>> {
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
