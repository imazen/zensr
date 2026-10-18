//! Profile unchanged compact-model inference on deterministic synthetic input.
#![forbid(unsafe_code)]

use std::hint::black_box;
use std::io::{BufWriter, Write};
use zensr_micro::adopted::AdoptedModel;

fn number(text: Option<String>, name: &str) -> usize {
    text.unwrap_or_else(|| panic!("missing {name}"))
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("invalid {name}"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // See coz-rs: Rust's default alternate signal stack can be too small.
    coz::thread_init();
    let mut args = std::env::args().skip(1);
    let weights = args.next().expect("weights_f16.raw path");
    let nf = number(args.next(), "nf");
    let nc = number(args.next(), "nc");
    let scale = number(args.next(), "scale");
    let height = number(args.next(), "height");
    let width = number(args.next(), "width");
    let iterations = number(args.next(), "iterations");
    let threads = number(args.next(), "threads (0 = whole-image forward)");
    let dump = args.next();
    assert!(args.next().is_none(), "unexpected extra argument");
    assert!(height > 0 && width > 0 && iterations > 0 && scale > 0);
    let count = height
        .checked_mul(width)
        .and_then(|n| n.checked_mul(3))
        .expect("input size overflow");
    let output_count = count
        .checked_mul(scale)
        .and_then(|n| n.checked_mul(scale))
        .expect("output size overflow");
    let bytes = std::fs::read(weights)?;
    assert!(bytes.len().is_multiple_of(2), "truncated f16 weights");
    let raw = zensr_micro::decode_all_f16(&bytes);
    let model = AdoptedModel::load_compact(&raw, nf, nc, scale)?;
    let mut state = 0x7b19_e493u32;
    let input: Vec<f32> = (0..count)
        .map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // CHW RGB in [0, 1], seeded noise with alternating coarse blocks.
            let base = if (i / 8) % 2 == 0 { 0.125 } else { 0.625 };
            base + (state >> 24) as f32 / 1024.0
        })
        .collect();
    let mut output = vec![0.0f32; output_count];
    eprintln!(
        "compact nf={nf} nc={nc} scale={scale} input={width}x{height} threads={threads} iterations={iterations} synthetic-seed=7b19e493"
    );
    for _ in 0..iterations {
        if threads == 0 {
            model.forward(black_box(&input), height, width, &mut output);
        } else {
            output = model.upscale_tiled(black_box(&input), height, width, threads, 128);
        }
        black_box(output.as_slice());
        coz::progress!("compact-inference");
    }
    assert_eq!(output.len(), output_count);
    assert!(
        output.iter().all(|x| x.is_finite()),
        "nonfinite inference output"
    );
    let fingerprint = output.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, value| {
        (hash ^ u64::from(value.to_bits())).wrapping_mul(0x100_0000_01b3)
    });
    println!(
        "output-elements={} fingerprint={fingerprint:016x}",
        output.len()
    );
    if let Some(path) = dump {
        // Exact byte comparison uses these raw outputs and an external SHA256.
        // Refuse replacement of an existing measurement artifact.
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        let mut writer = BufWriter::new(file);
        for value in output {
            writer.write_all(&value.to_le_bytes())?;
        }
        writer.flush()?;
    }
    Ok(())
}
