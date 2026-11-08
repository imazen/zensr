//! Run with `repair input.jpg output.rgb8 [--model path.zensr]`.
use std::{env, error::Error, fs, io, path::Path};
use zensr_micro::{adopted::AdoptedModel, decode_all_f16};
use zensr_zenjpeg::{
    api::{Budget, Restorer, Routing},
    model_file::ModelBundle,
};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let usage = "usage: repair input.jpg output.rgb8 [--model path.zensr] [--routing auto|always|never] [--threads N]";
    if args.len() < 2 {
        return Err(usage.into());
    }
    let mut model_file = None;
    let mut routing = None;
    let mut threads = 1;
    let (options, remainder) = args[2..].as_chunks::<2>();
    for pair in options {
        match pair[0].to_str() {
            Some("--model") if model_file.is_none() => model_file = Some(&pair[1]),
            Some("--routing") => {
                routing = match pair[1].to_str() {
                    Some("auto") => None,
                    Some("always") => Some(Routing::Always),
                    Some("never") => Some(Routing::Never),
                    _ => return Err(usage.into()),
                }
            }
            Some("--threads") => threads = pair[1].to_str().ok_or(usage)?.parse()?,
            _ => return Err(usage.into()),
        }
    }
    if !remainder.is_empty() {
        return Err(usage.into());
    }
    let restorer = if let Some(path) = model_file {
        ModelBundle::from_file(path)
            .and_then(ModelBundle::into_restorer)
            .map_err(|message| io::Error::new(io::ErrorKind::InvalidData, message))?
    } else {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/adopted/dejpeg_rt24i_canon4/weights_f16.raw");
        let weights = decode_all_f16(&fs::read(path)?);
        let model = AdoptedModel::load_compact(&weights, 24, 8, 1)
            .map_err(|message| io::Error::new(io::ErrorKind::InvalidData, message))?;
        Restorer::new().with_tier(Budget::Realtime, model)
    };
    let jpeg = fs::read(&args[0])?;
    let mut request = restorer.restore(&jpeg).threads(threads);
    if let Some(routing) = routing {
        request = request.routing(routing);
    }
    let outcome = request.run()?;
    let pixels = outcome.pixels();
    fs::write(&args[1], pixels.to_rgb8())?;
    println!(
        "{} × {} RGB8 pixels written",
        pixels.width(),
        pixels.height()
    );
    Ok(())
}
