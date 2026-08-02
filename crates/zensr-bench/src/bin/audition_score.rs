//! Score teacher-audition outputs (~/tmp/zensr-audition) vs GT crops.
//! Emits per-image TSV + per-(sub,deg,variant) medians and wins-vs-lanczos.
//!
//! Usage: audition_score [root=~/tmp/zensr-audition] [out-tsv]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use zensr_bench::*;

fn main() {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(
        args.next()
            .unwrap_or_else(|| format!("{}/tmp/zensr-audition", std::env::var("HOME").unwrap())),
    );
    let out_tsv = args
        .next()
        .unwrap_or_else(|| "benchmarks/teacher_audition_2026-07-23.tsv".into());

    let variants: Vec<String> = std::fs::read_dir(&root)
        .expect("audition root")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n != "gt" && !n.starts_with('_'))
        .collect();

    // gt/<sub>__<stem>.png ; variant/<sub>__<stem>__<deg>.png
    let mut tsv = String::from("sub\tstem\tdeg\tvariant\tpsnr\tssim2\tbutter_n3\n");
    // (sub,deg,variant) -> per-stem ssim2 ; plus lanczos reference per (sub,deg,stem)
    let mut per: BTreeMap<(String, String, String), Vec<(String, f64, f64)>> = BTreeMap::new();
    for gte in std::fs::read_dir(root.join("gt")).expect("gt dir") {
        let gtp = gte.unwrap().path();
        let full = gtp.file_stem().unwrap().to_string_lossy().to_string();
        let (sub, stem) = full.split_once("__").expect("sub__stem");
        let gt = decode_any(&gtp).expect("gt decode");
        for v in &variants {
            for deg in ["clean", "q75", "q50", "q35"] {
                let p = root.join(v).join(format!("{full}__{deg}.png"));
                let Some(img) = decode_any(&p) else { continue };
                let s = score(&gt, &img);
                let _ = writeln!(
                    tsv,
                    "{sub}\t{stem}\t{deg}\t{v}\t{:.3}\t{:.3}\t{:.4}",
                    s.psnr, s.ssim2, s.butter
                );
                per.entry((sub.into(), deg.into(), v.clone()))
                    .or_default()
                    .push((stem.to_string(), s.ssim2, s.butter));
            }
        }
        eprintln!("scored {full}");
    }
    std::fs::write(&out_tsv, &tsv).expect("write tsv");

    let med = |xs: &mut Vec<f64>| -> f64 {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs[xs.len() / 2]
    };
    println!("sub\tdeg\tvariant\tn\tssim2_med\tbutter_med\twins_vs_lanczos");
    for ((sub, deg, v), rows) in &per {
        if v == "lanczos" {
            continue;
        }
        let lan: BTreeMap<&str, f64> = per[&(sub.clone(), deg.clone(), "lanczos".into())]
            .iter()
            .map(|(s, ss, _)| (s.as_str(), *ss))
            .collect();
        let wins = rows
            .iter()
            .filter(|(s, ss, _)| *ss > lan[s.as_str()])
            .count();
        let mut ss: Vec<f64> = rows.iter().map(|r| r.1).collect();
        let mut bb: Vec<f64> = rows.iter().map(|r| r.2).collect();
        println!(
            "{sub}\t{deg}\t{v}\t{}\t{:.2}\t{:.3}\t{}/{}",
            rows.len(),
            med(&mut ss),
            med(&mut bb),
            wins,
            rows.len()
        );
    }
    // lanczos medians for reference
    for ((sub, deg, v), rows) in &per {
        if v != "lanczos" {
            continue;
        }
        let mut ss: Vec<f64> = rows.iter().map(|r| r.1).collect();
        let mut bb: Vec<f64> = rows.iter().map(|r| r.2).collect();
        println!(
            "{sub}\t{deg}\tlanczos\t{}\t{:.2}\t{:.3}\t-",
            rows.len(),
            med(&mut ss),
            med(&mut bb)
        );
    }
}
