//! Render gallery panels for the report: per scene (name, path, deg) from a
//! scenes.tsv, write gt + degraded-upscaled baselines + model outputs as PNGs
//! (512x512, x2 protocol identical to systems_eval). Output dir per scene.
//!
//! Usage: gallery_dump <scenes.tsv> <out-root> [threads=12]

use std::path::PathBuf;
use zensr_bench::*;

const MODELS: &[(&str, &str)] = &[
    ("nomosuni_compact_2x", "A2c"),
    ("nomosuni_span_2x", "A2span"),
    ("people_a2c_2x", "P_a2c"),
    ("people_rtc_2x", "P_rtc"),
    ("rtc_distill_2x", "E_rtc"),
];

fn save_png(img: &Rgb8Img, path: &PathBuf) {
    let px: &[rgb::Rgb<u8>] = bytemuck::cast_slice(&img.px);
    let r = imgref::ImgRef::new(px, img.w, img.h);
    let data = zenpng::encode_rgb8(
        r,
        None,
        &zenpng::EncodeConfig::default(),
        &enough::Unstoppable,
        &enough::Unstoppable,
    )
    .expect("png encode");
    std::fs::write(path, data).expect("write png");
}

fn main() {
    let mut args = std::env::args().skip(1);
    let scenes = std::fs::read_to_string(args.next().expect("scenes.tsv")).expect("read scenes");
    let out_root = PathBuf::from(args.next().expect("out root"));
    let threads: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(12);
    let models: Vec<_> = MODELS
        .iter()
        .map(|(d, l)| (load_adopted(d).unwrap_or_else(|| panic!("{d}")), *l))
        .collect();

    for line in scenes.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        if c.len() != 3 {
            continue;
        }
        let (name, path, deg) = (c[0], PathBuf::from(c[1]), c[2]);
        let img = decode_any(&path).expect("decode scene");
        let hr = center_crop(&img, 512).expect("crop 512");
        let d = out_root.join(name);
        std::fs::create_dir_all(&d).unwrap();
        save_png(&hr, &d.join("gt.png"));
        let half = resize_rgb8(&hr, hr.w / 2, hr.h / 2, zenresize::Filter::CatmullRom);
        let lr = match deg {
            "clean" => half,
            "q75" => turbo_jpeg(&half, 75),
            "q50" => turbo_jpeg(&half, 50),
            _ => turbo_jpeg(&half, 35),
        };
        save_png(
            &resize_rgb8(&lr, hr.w, hr.h, zenresize::Filter::Lanczos),
            &d.join("lanczos.png"),
        );
        for (m, label) in &models {
            save_png(&run_guarded(m, &lr, threads, true), &d.join(format!("{label}.png")));
        }
        eprintln!("done {name}");
    }
}
