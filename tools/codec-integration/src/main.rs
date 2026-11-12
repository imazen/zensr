//! Compare the native decoder integration against the frozen public restoration path.
use std::{error::Error, fs, path::Path, sync::Arc};
use zenjpeg::decoder::Decoder;
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("codec-integration input-manifest.json output.json".into());
    }
    let rows: Vec<serde_json::Value> = serde_json::from_slice(&fs::read(&args[1])?)?;
    let weights = zensr_micro::decode_all_f16(zensr_micro::canon4::WEIGHTS_F16);
    let old = legacy::api::Restorer::new().with_tier(
        legacy::api::Budget::Realtime,
        legacy::AdoptedModel::load_compact(&weights, 24, 8, 1)?,
    );
    let restorer = Arc::new(zensr::api::Restorer::new().with_tier(
        zensr::api::Budget::Realtime,
        zensr::AdoptedModel::load_compact(&weights, 24, 8, 1)?,
    ));
    let decoder = Decoder::new()
        .artifact_removal(restorer)
        .auto_orient(false)
        .num_threads(1);
    let mut records = Vec::new();
    let mut applied = 0;
    for row in rows {
        let path = row["input"].as_str().ok_or("missing input")?;
        let bytes = fs::read(path)?;
        let reference = old.restore(&bytes).threads(1).run()?;
        let actual = decoder.decode(&bytes, enough::Unstoppable)?;
        let expected = reference.pixels().to_rgb8();
        if actual.pixels_u8() != Some(expected.as_slice()) {
            return Err(format!("native RGB8 differs: {path}").into());
        }
        let summary = actual
            .artifact_removal_report()
            .ok_or("missing native report")?;
        if summary.applied != matches!(reference, legacy::api::Outcome::Restored { .. }) {
            return Err(format!("native route differs: {path}").into());
        }
        applied += usize::from(summary.applied);
        records.push(serde_json::json!({"input":path,"width":actual.width(),"height":actual.height(),"applied":summary.applied,"values":expected.len(),"rgb8_exact":true}));
        fs::write(Path::new(&args[2]), serde_json::to_vec_pretty(&records)?)?;
    }
    println!(
        "{} cases; {applied} restored; native RGB8 and routing exact",
        records.len()
    );
    Ok(())
}
