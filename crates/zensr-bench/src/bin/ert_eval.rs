//! E_rt (rt_distill_2x) x2-track rows in the systems_eval TSV schema.
//! Run after training exports; append output to the day's main TSV, then
//! `systems_eval summarize <tsv>` for the combined table.
//!
//! Usage: ert_eval <corpus-root> <out-tsv> [per-sub=8] [threads=12] [model-dir=rt_distill_2x] [label=E_rt]

use std::fmt::Write as _;
use std::path::PathBuf;
use zensr_bench::*;

fn main() {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().expect("corpus root"));
    let out_path = PathBuf::from(args.next().expect("out tsv"));
    let per_sub: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(8);
    let threads: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(12);
    let model_dir = args.next().unwrap_or_else(|| "rt_distill_2x".into());
    let label = args.next().unwrap_or_else(|| "E_rt".into());
    let rt = load_adopted(&model_dir).expect("student model dir (train first)");
    assert_eq!(rt.scale, 2);

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
            let half = resize_rgb8(&hr, hr.w / 2, hr.h / 2, zenresize::Filter::CatmullRom);
            for deg in ["clean", "q75", "q50", "q35"] {
                let lr = match deg {
                    "clean" => Rgb8Img {
                        px: half.px.clone(),
                        w: half.w,
                        h: half.h,
                    },
                    "q75" => turbo_jpeg(&half, 75),
                    "q50" => turbo_jpeg(&half, 50),
                    _ => turbo_jpeg(&half, 35),
                };
                let o = run_guarded(&rt, &lr, threads, true);
                let s = score(&hr, &o);
                let _ = writeln!(
                    tsv,
                    "{sub}\t{fname}\tx2\t{deg}\t{label}\t{:.3}\t{:.3}\t{:.4}",
                    s.psnr, s.ssim2, s.butter
                );
            }
            eprintln!("done {sub}/{fname}");
        }
    }
    std::fs::write(&out_path, &tsv).expect("write tsv");
    eprintln!("wrote {}", out_path.display());
}
