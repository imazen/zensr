//! Native consumer of the same codec/runtime integration as the WASM bindings.
use std::{
    error::Error,
    fs::{self, OpenOptions},
    io::Write,
    sync::Arc,
};
use zensr::{
    AdoptedModel,
    api::{Budget, Restorer},
    decode_all_f16,
};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("oracle weights input.jpg output.rgb8".into());
    }
    let model = AdoptedModel::load_compact(&decode_all_f16(&fs::read(&args[1])?), 24, 8, 1)?;
    let restorer = Arc::new(Restorer::new().with_tier(Budget::Realtime, model));
    let result = zenjpeg::decoder::Decoder::new()
        .num_threads(1)
        .auto_orient(false)
        .artifact_removal(restorer)
        .decode(&fs::read(&args[2])?, enough::Unstoppable)?;
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[3])?
        .write_all(result.pixels_u8().ok_or("not RGB8")?)?;
    Ok(())
}
