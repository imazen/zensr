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
    // Optional dispatch-tier forcing for cross-tier bit-exactness runs:
    // ZENSR_PROFILE_TIER=auto (default) | v3 | scalar. Process-wide token
    // disable before model load; runtime source is not modified.
    #[cfg(target_arch = "x86_64")]
    {
        match std::env::var("ZENSR_PROFILE_TIER").as_deref() {
            Ok("v3") => {
                archmage::X64V4xToken::dangerously_disable_token_process_wide(true).ok();
            }
            Ok("scalar") => {
                archmage::X64V4xToken::dangerously_disable_token_process_wide(true).ok();
                archmage::X64V3Token::dangerously_disable_token_process_wide(true).ok();
            }
            Ok("auto") | Err(_) => {}
            Ok(other) => panic!("unknown ZENSR_PROFILE_TIER={other}"),
        }
    }
    let mut args = std::env::args().skip(1);
    let first = args.next().expect("weights_f16.raw path or 'spec'");
    let (weights, nf, nc, scale, height, width, iterations, threads, dump);
    let span: bool;
    if first == "spec" {
        // spec-file mode: `BIN spec <file> [dump]`; file is whitespace-
        // separated `weights nf nc scale h w iters threads` for zenbench
        // pair case-input hashing.
        let spec = args.next().expect("spec file path");
        let text = std::fs::read_to_string(&spec).expect("read spec");
        let t: Vec<&str> = text.split_whitespace().collect();
        let num = |i: usize, n: &str| number(t.get(i).map(|x| x.to_string()), n);
        weights = t.first().expect("spec missing weights").to_string();
        if t.get(1) == Some(&"span48") {
            // span48 arch: `weights span48 scale h w iters threads`
            nf = 0;
            nc = 0;
            span = true;
            scale = num(2, "scale");
            height = num(3, "height");
            width = num(4, "width");
            iterations = num(5, "iterations");
            threads = num(6, "threads");
        } else {
            nf = num(1, "nf");
            nc = num(2, "nc");
            span = false;
            scale = num(3, "scale");
            height = num(4, "height");
            width = num(5, "width");
            iterations = num(6, "iterations");
            threads = num(7, "threads");
        }
        dump = args.next();
        assert!(args.next().is_none(), "unexpected extra argument");
    } else {
        weights = first;
        span = false;
        nf = number(args.next(), "nf");
        nc = number(args.next(), "nc");
        scale = number(args.next(), "scale");
        height = number(args.next(), "height");
        width = number(args.next(), "width");
        iterations = number(args.next(), "iterations");
        threads = number(args.next(), "threads (0 = whole-image forward)");
        dump = args.next();
        assert!(args.next().is_none(), "unexpected extra argument");
    }
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
    let model = if span {
        AdoptedModel::load_span48(&raw, scale)?
    } else {
        AdoptedModel::load_compact(&raw, nf, nc, scale)?
    };
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
    // Per-iteration wall time on stdout for zenbench pair `elapsed_ns`
    // metric extraction (rows count == iterations). Off unless
    // ZENSR_PROFILE_TIMING=1 so plain/profile runs stay quiet.
    let timed = std::env::var("ZENSR_PROFILE_TIMING").as_deref() == Ok("1");
    let mut t_iter = std::time::Instant::now();
    for _ in 0..iterations {
        if timed {
            t_iter = std::time::Instant::now();
        }
        if threads == 0 {
            model.forward(black_box(&input), height, width, &mut output);
        } else {
            output = model.upscale_tiled(black_box(&input), height, width, threads, 128);
        }
        black_box(output.as_slice());
        coz::progress!("compact-inference");
        if timed {
            println!("elapsed_ns={}", t_iter.elapsed().as_nanos());
        }
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
