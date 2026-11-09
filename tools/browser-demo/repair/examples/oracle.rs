//! Independent native public-API oracle for the browser glue's upright JPEG case.
use std::{
    error::Error,
    fs::{self, OpenOptions},
    io::Write,
};
use zensr_micro::{adopted::AdoptedModel, decode_all_f16};
use zensr_zenjpeg::api::{Budget, Restorer};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("oracle weights input.jpg output.rgb8".into());
    }
    let model = AdoptedModel::load_compact(&decode_all_f16(&fs::read(&args[1])?), 24, 8, 1)?;
    let restorer = Restorer::new().with_tier(Budget::Realtime, model);
    let jpeg = fs::read(&args[2])?;
    let result = restorer.restore(&jpeg).threads(1).run()?;
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[3])?
        .write_all(&result.pixels().to_rgb8())?;
    Ok(())
}
