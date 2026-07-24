//! One-shot TEST-slice scoring: Lanczos/CatmullRom baselines + fixed model list
//! over a flat dir of people photos (virgin-shard test corpus). x2 protocol
//! identical to systems_eval's x2 track. Touch the test slice ONCE per
//! milestone — model selection happens on the dev slices, never here.
//!
//! Usage: people_test_eval <flat-image-dir> <out-tsv> [threads=12]

use std::fmt::Write as _;
use std::path::PathBuf;
use zensr_bench::*;

const MODELS: &[(&str, &str)] = &[
    ("nomosuni_compact_2x", "A2c"),
    ("people_rtc_2x", "P_rtc"),
    ("people_a2c_2x", "P_a2c"),
];

fn main() {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().expect("image dir"));
    let out_path = PathBuf::from(args.next().expect("out tsv"));
    let threads: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(12);
    let models: Vec<_> = MODELS
        .iter()
        .map(|(d, l)| (load_adopted(d).unwrap_or_else(|| panic!("{d}")), *l))
        .collect();

    let mut tsv = String::from("sub\tfile\ttrack\tdeg\tsystem\tpsnr\tssim2\tbutter_n3\n");
    let mut n = 0usize;
    for f in list_images(&root) {
        let Some(img) = decode_any(&f) else { continue };
        let Some(hr) = center_crop(&img, 512) else { continue };
        let fname = f.file_name().unwrap().to_string_lossy().to_string();
        n += 1;
        let half = resize_rgb8(&hr, hr.w / 2, hr.h / 2, zenresize::Filter::CatmullRom);
        for deg in ["clean", "q75", "q50", "q35"] {
            let lr = match deg {
                "clean" => Rgb8Img { px: half.px.clone(), w: half.w, h: half.h },
                "q75" => turbo_jpeg(&half, 75),
                "q50" => turbo_jpeg(&half, 50),
                _ => turbo_jpeg(&half, 35),
            };
            let mut outs: Vec<(String, Rgb8Img)> = vec![
                ("lanczos".into(), resize_rgb8(&lr, hr.w, hr.h, zenresize::Filter::Lanczos)),
                ("catmullrom".into(), resize_rgb8(&lr, hr.w, hr.h, zenresize::Filter::CatmullRom)),
            ];
            for (m, label) in &models {
                outs.push(((*label).into(), run_guarded(m, &lr, threads, true)));
            }
            for (name, o) in &outs {
                let s = score(&hr, o);
                let _ = writeln!(
                    tsv,
                    "people-test\t{fname}\tx2\t{deg}\t{name}\t{:.3}\t{:.3}\t{:.4}",
                    s.psnr, s.ssim2, s.butter
                );
            }
        }
        eprintln!("done {fname}");
    }
    std::fs::write(&out_path, &tsv).expect("write tsv");
    eprintln!("scored {n} images -> {}", out_path.display());
}
