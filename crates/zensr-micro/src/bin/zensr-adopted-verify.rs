//! Golden verification for adopted models (dirs under models/adopted/).
//! Checks whole-image forward AND the tiled path (seams) vs torch goldens.

use zensr_micro::adopted::AdoptedModel;

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn meta_field(meta: &str, key: &str) -> String {
    let pat = format!("\"{key}\":");
    let i = meta.find(&pat).unwrap_or_else(|| panic!("meta missing {key}"));
    let rest = &meta[i + pat.len()..];
    rest.trim_start()
        .trim_start_matches('"')
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

fn main() {
    let root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/adopted".into());
    let mut fails = 0usize;
    let mut dirs: Vec<_> = std::fs::read_dir(&root)
        .expect("adopted dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    for d in dirs {
        let name = d.file_name().unwrap().to_string_lossy().to_string();
        let meta = std::fs::read_to_string(d.join("meta.json")).expect("meta");
        let arch = meta_field(&meta, "arch");
        let scale: usize = meta_field(&meta, "scale").parse().unwrap();
        let nf: usize = meta_field(&meta, "nf").parse().unwrap();
        let nc: usize = meta_field(&meta, "nc").parse().unwrap();
        let raw = read_f32(&d.join("weights.raw"));
        let model = match arch.as_str() {
            "compact" => AdoptedModel::load_compact(&raw, nf, nc, scale),
            "span48" => AdoptedModel::load_span48(&raw, scale),
            other => panic!("unknown arch {other}"),
        }
        .unwrap_or_else(|e| panic!("{name}: {e}"));

        for (h, w) in [(40usize, 36usize), (17, 13)] {
            let gi = read_f32(&d.join(format!("gold_in_{h}x{w}.raw")));
            let go = read_f32(&d.join(format!("gold_out_{h}x{w}.raw")));
            let mut out = vec![0.0f32; 3 * h * w * scale * scale];
            model.forward(&gi, h, w, &mut out);
            let tiled = model.upscale_tiled(&gi, h, w, 2, 32);
            let stat = |a: &[f32]| {
                let mut mx = 0.0f32;
                let mut bad = 0usize;
                for (x, y) in a.iter().zip(go.iter()) {
                    if !x.is_finite() {
                        bad += 1;
                    } else {
                        mx = mx.max((x - y).abs());
                    }
                }
                (mx, bad)
            };
            let (mf, bf) = stat(&out);
            let (mt, bt) = stat(&tiled);
            let ok = mf < 5e-4 && mt < 5e-4 && bf == 0 && bt == 0;
            if !ok {
                fails += 1;
            }
            println!(
                "{name} {h}x{w}: forward={mf:.2e} tiled={mt:.2e} nonfinite={}/{} {}",
                bf,
                bt,
                if ok { "OK" } else { "FAIL" }
            );
        }
        // f16 ship-file check (loose tolerance; f16 measured transparent)
        if let Ok(fb) = std::fs::read(d.join("weights_f16.raw")) {
            let raw16 = zensr_micro::decode_all_f16(&fb);
            let m16 = match arch.as_str() {
                "compact" => AdoptedModel::load_compact(&raw16, nf, nc, scale),
                _ => AdoptedModel::load_span48(&raw16, scale),
            }
            .unwrap();
            let (h, w) = (40usize, 36usize);
            let gi = read_f32(&d.join(format!("gold_in_{h}x{w}.raw")));
            let go = read_f32(&d.join(format!("gold_out_{h}x{w}.raw")));
            let mut out = vec![0.0f32; 3 * h * w * scale * scale];
            m16.forward(&gi, h, w, &mut out);
            let mut mx = 0.0f32;
            for (x, y) in out.iter().zip(go.iter()) {
                assert!(x.is_finite());
                mx = mx.max((x - y).abs());
            }
            let ok16 = mx < 0.05;
            if !ok16 {
                fails += 1;
            }
            println!("{name} f16: max={mx:.2e} {}", if ok16 { "OK" } else { "FAIL" });
        }
    }
    if fails > 0 {
        eprintln!("{fails} golden checks FAILED");
        std::process::exit(1);
    }
    println!("ALL ADOPTED GOLDENS PASS");
}
