//! Offline bake/replay adapter using the existing zenpredict and feature contracts.
#![forbid(unsafe_code)]
use serde::Deserialize;
use std::{
    error::Error,
    fs,
    io::{BufWriter, Write},
};
use zenanalyze_api::{FeatureResult, NamedFeature, Offer, Provenance, Request, Select};
use zenpredict::{Model, Predictor};
#[derive(Deserialize)]
struct Vectors {
    schema_hash: u64,
    names: Vec<String>,
    rows: Vec<Vec<f32>>,
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if (args.get(1).map(String::as_str) == Some("bake") && args.len() != 4)
        || (args.get(1).map(String::as_str) != Some("bake") && args.len() != 5)
    {
        return Err("usage: selector bake JSON BIN | selector replay BIN VECTORS OUTPUT".into());
    }
    match args[1].as_str() {
        "bake" => {
            let bytes = zenpredict_bake::bake_from_json_str(&fs::read_to_string(&args[2])?)?;
            let model = Model::from_bytes(&bytes)?;
            let mut out = fs::File::create_new(&args[3])?;
            out.write_all(&bytes)?;
            out.sync_all()?;
            println!(
                "{} inputs, {} outputs, {} bytes",
                model.n_inputs(),
                model.n_outputs(),
                bytes.len()
            );
        }
        "replay" => {
            let data: Vectors = serde_json::from_slice(&fs::read(&args[3])?)?;
            let bytes = fs::read(&args[2])?;
            let model = Model::from_bytes(&bytes)?;
            if model.schema_hash() != data.schema_hash {
                return Err("Baked schema differs".into());
            }
            let wants: Vec<_> = model
                .feature_columns()
                .map(|s| NamedFeature::parse(s).ok_or("Baked column is not qualified"))
                .collect::<Result<_, _>>()?;
            if wants.len() != model.n_inputs() || data.names.len() != model.n_inputs() {
                return Err("Feature-column count differs".into());
            }
            let names: Vec<_> = data
                .names
                .iter()
                .map(|s| NamedFeature::parse(s).ok_or("Unqualified offered column"))
                .collect::<Result<_, _>>()?;
            let request = Request::new(Select::Features(&wants));
            let mut predictor = Predictor::new(&model);
            let mut out = BufWriter::new(fs::File::create_new(&args[4])?);
            for row in data.rows {
                if row.len() != names.len() || row.iter().any(|x| !x.is_finite()) {
                    return Err("Invalid offered vector".into());
                }
                let entries: Vec<_> = names
                    .iter()
                    .zip(row)
                    .map(|(n, v)| FeatureResult::new(*n, v))
                    .collect();
                let offer = Offer::new(&entries, Provenance::new("retained-bank"));
                if !offer.satisfies(&request) {
                    return Err("Existing offer cannot satisfy model request".into());
                }
                let values = offer.reuse_for(&request).ok_or("Offer reuse failed")?;
                let prediction = predictor.predict(&values)?;
                if prediction.iter().any(|x| !x.is_finite()) {
                    return Err("Invalid predictor output".into());
                }
                serde_json::to_writer(&mut out, prediction)?;
                out.write_all(b"\n")?;
            }
            out.flush()?;
            out.into_inner()?.sync_all()?;
        }
        _ => return Err("Unknown selector command".into()),
    }
    Ok(())
}
