//! Native x1 dejpeg rows in the systems_eval TSV schema (x1 track protocol:
//! degrade the full-res 512 crop, restore at same size, score vs original).
//! Append to the day TSV and summarize to compare against identity/C_repair.
//!
//! Usage: x1_eval <imazen26-root> <out-tsv> [per-sub=8] [threads=12] [model-dir=dejpeg_1x] [label=S6_dejpeg]

use std::fmt::Write as _;
use std::path::PathBuf;
use zensr_bench::*;

fn main() {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().expect("corpus root"));
    let out_path = PathBuf::from(args.next().expect("out tsv"));
    let per_sub: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(8);
    let threads: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(12);
    let model_dir = args.next().unwrap_or_else(|| "dejpeg_1x".into());
    let label = args.next().unwrap_or_else(|| "S6_dejpeg".into());
    let m = load_adopted(&model_dir).expect("x1 model (train first)");
    assert_eq!(m.scale, 1, "x1_eval requires a scale-1 model");

    let mut tsv = String::new();
    for (sub, dir) in SUBCORPORA {
        let files = list_images(&root.join(dir));
        let mut used = 0usize;
        for f in files {
            if used >= per_sub {
                break;
            }
            let Some(img) = decode_any(&f) else { continue };
            let Some(hr) = center_crop(&img, 512) else {
                continue;
            };
            let fname = f.file_name().unwrap().to_string_lossy().to_string();
            used += 1;
            for deg in ["q75", "q50", "q35"] {
                let lr = match deg {
                    "q75" => turbo_jpeg(&hr, 75),
                    "q50" => turbo_jpeg(&hr, 50),
                    _ => turbo_jpeg(&hr, 35),
                };
                let o = run_guarded(&m, &lr, threads, true);
                let s = score(&hr, &o);
                let _ = writeln!(
                    tsv,
                    "{sub}\t{fname}\tx1\t{deg}\t{label}\t{:.3}\t{:.3}\t{:.4}",
                    s.psnr, s.ssim2, s.butter
                );
            }
            eprintln!("done {sub}/{fname}");
        }
    }
    std::fs::write(&out_path, &tsv).expect("write tsv");
    eprintln!("wrote {}", out_path.display());
}
