//! Run from the workspace: cargo run --release --locked -p zensr-zenjpeg
//! --example repair -- input.jpg output.rgb8
use std::{env, error::Error, fs, io, path::Path};
use zensr_micro::{adopted::AdoptedModel, decode_all_f16};
use zensr_zenjpeg::api::{Budget, Restorer};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args_os().collect();
    if args.len() != 3 {
        return Err("usage: repair input.jpg output.rgb8".into());
    }
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/adopted/dejpeg_rt24i_canon4/weights_f16.raw");
    let weights = decode_all_f16(&fs::read(path)?);
    let model = AdoptedModel::load_compact(&weights, 24, 8, 1)
        .map_err(|message| io::Error::new(io::ErrorKind::InvalidData, message))?;
    let restorer = Restorer::new().with_tier(Budget::Realtime, model);
    let jpeg = fs::read(&args[1])?;
    let outcome = restorer.restore(&jpeg).threads(1).run()?;
    let pixels = outcome.pixels();
    fs::write(&args[2], pixels.to_rgb8())?;
    println!(
        "{} × {} RGB8 pixels written",
        pixels.width(),
        pixels.height()
    );
    Ok(())
}
