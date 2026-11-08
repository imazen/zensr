//! Frozen-manifest, inference-input-only routing experiment. Results and content
//! addressed artifacts are retained per cell; interrupted runs can resume.
//! Usage: routing_study manifest output role sizes qualities encoders jobs
//! Sizes are maximum edges; aspect ratio is preserved and inputs never upscaled.
use rayon::prelude::*;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use zensr_bench::*;

#[path = "coef_features/metrics.rs"]
mod coefficients;
#[path = "dejpeg_eval/encoding.rs"]
mod encoding;
#[path = "routing_study/storage.rs"]
mod storage;

const COEF_NAMES: &[&str] = &[
    "zero_ac_blocks",
    "zero_ac_coefs",
    "hf_survival",
    "mean_abs_ac",
    "dc_spread",
    "ac_count_spread",
    "gutted_block_frac",
    "gutted_clustering",
];
const PIXEL_NAMES: &[&str] = &[
    "edge_slope_stdev",
    "patch_fraction_fast",
    "aq_map_std",
    "variance",
    "colourfulness",
    "laplacian_variance",
    "variance_spread",
    "distinct_color_bins",
    "palette_density",
    "luma_histogram_entropy",
    "luma_kurtosis",
    "skin_tone_fraction",
    "chroma_luma_covariance_cb",
    "chroma_luma_covariance_cr",
    "cr_peak_sharpness",
    "orientation_energy_ratio",
    "aq_map_p90",
    "noise_floor_y_p25",
    "noise_floor_y_p90",
    "noise_floor_uv_p50",
    "noise_floor_uv_p75",
    "noise_floor_uv_p90",
    "quant_survival_y",
    "xyb444_color_loss",
];

fn sha(bytes: &[u8]) -> String {
    let mut child = Command::new("sha256sum")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("sha256sum");
    child.stdin.take().unwrap().write_all(bytes).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    let value = String::from_utf8(out.stdout).unwrap();
    let hash = value.split_whitespace().next().unwrap();
    assert_eq!(hash.len(), 64);
    hash.to_string()
}

fn encoder_identity(encoders: &[&str]) -> String {
    let home = PathBuf::from(std::env::var("HOME").unwrap());
    let mut libraries = std::collections::BTreeMap::new();
    for encoder in encoders {
        let program = match *encoder {
            "turbo" | "jpegli" => {
                let command = if *encoder == "turbo" {
                    "cjpeg"
                } else {
                    "cjpegli"
                };
                let out = Command::new("which").arg(command).output().unwrap();
                assert!(out.status.success(), "missing encoder {command}");
                PathBuf::from(String::from_utf8(out.stdout).unwrap().trim())
            }
            "mozjpeg" => home.join("tmp/ati-bin/mozjpeg-cjpeg"),
            name if name == "zenjpeg" || name.starts_with("zenjpeg:") => std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .join("zjtool"),
            _ => panic!("unknown encoder {encoder}"),
        };
        libraries.insert(
            program.display().to_string(),
            sha(&fs::read(&program).unwrap()),
        );
        let mut ldd = Command::new("ldd");
        ldd.arg(&program);
        if *encoder == "mozjpeg" {
            ldd.env("LD_LIBRARY_PATH", home.join("tmp/ati-bin/mozjpeg-lib64"));
        }
        let out = ldd.output().unwrap();
        let text = String::from_utf8(out.stdout).unwrap();
        assert!(
            !text.contains("not found"),
            "missing encoder shared library"
        );
        for path in text.split_whitespace().filter(|v| v.starts_with('/')) {
            libraries.insert(path.to_string(), sha(&fs::read(path).unwrap()));
        }
    }
    libraries
        .into_iter()
        .map(|(p, h)| format!("{p}\t{h}\n"))
        .collect()
}

fn artifact(root: &Path, bytes: &[u8], suffix: &str) -> String {
    // Retain exact bytes while avoiding uncompressed RGB/map storage blowup.
    let compressed;
    let suffix_owned;
    let (bytes, suffix) = if suffix == "jpg" {
        (bytes, suffix)
    } else {
        let mut child = Command::new("zstd")
            .args(["-q", "-3", "-T1", "-c"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        // Drain stdout concurrently with stdin: large maps exceed pipe capacity.
        let mut stdin = child.stdin.take().unwrap();
        let writer = std::thread::scope(|scope| {
            let writing = scope.spawn(move || {
                let result = stdin.write_all(bytes);
                drop(stdin);
                result
            });
            let output = child.wait_with_output().unwrap();
            writing.join().unwrap().unwrap();
            output
        });
        assert!(writer.status.success(), "artifact compression");
        compressed = writer.stdout;
        suffix_owned = format!("{suffix}.zst");
        (compressed.as_slice(), suffix_owned.as_str())
    };
    let hash = sha(bytes);
    let dir = root.join("artifacts").join(&hash[..2]);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{hash}.{suffix}"));
    storage::publish(&path, bytes);
    format!("artifacts/{}/{hash}.{suffix}", &hash[..2])
}

fn finite(v: f64) -> String {
    assert!(v.is_finite(), "non-finite measured feature/metric");
    format!("{v:.12}")
}

fn metric_columns(reference: &Rgb8Img, image: &Rgb8Img) -> Vec<String> {
    use rgb::FromSlice;
    let result = butteraugli::butteraugli(
        imgref::Img::new(reference.px.as_rgb(), reference.w, reference.h),
        imgref::Img::new(image.px.as_rgb(), image.w, image.h),
        &butteraugli::ButteraugliParams::default().with_compute_diffmap(false),
    )
    .expect("butteraugli");
    assert!(
        result.diffmap.is_none(),
        "score-only metric must not return a map"
    );
    let mut out = vec![
        finite(ssim2(reference, image)),
        finite(psnr_rgb8(reference, image)),
        finite(result.score),
        finite(result.pnorm_3),
        String::new(), // No diagnostic map is generated or retained.
    ];
    // Preserve per-channel squared-error sums, not only aggregate PSNR.
    for c in 0..3 {
        let sum = reference
            .px
            .iter()
            .skip(c)
            .step_by(3)
            .zip(image.px.iter().skip(c).step_by(3))
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum::<f64>();
        out.push(finite(sum));
    }
    out
}

fn header() -> String {
    let mut names: Vec<String> = [
        "file",
        "content",
        "origin",
        "role",
        "ref_kind",
        "reference_sha256",
        "width",
        "height",
        "size",
        "encoder",
        "ss",
        "nominal_q",
        "jpeg_artifact",
        "reference_artifact",
        "decoded_artifact",
        "restored_artifact",
        "probe_family",
        "probe_scale",
        "probe_value",
        "estimated_gain",
        "gate",
        "conservative_gate",
        "bpp",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    for n in COEF_NAMES {
        names.push(format!("coef_{n}"));
    }
    for n in PIXEL_NAMES {
        names.push(format!("pixel_{n}"));
    }
    names.push("chooser_p_graphics".into());
    for n in [
        "clamped_frac",
        "mean_alpha",
        "roundtrip_mae",
        "fallback_blend",
        "projection_y",
        "projection_cb",
        "projection_cr",
    ] {
        names.push(format!("post_{n}"));
    }
    for arm in ["identity", "restored"] {
        for field in [
            "ssim2",
            "psnr",
            "butter_max",
            "butter_pnorm3",
            "butter_map",
            "sse_r",
            "sse_g",
            "sse_b",
        ] {
            names.push(format!("{arm}_{field}"));
        }
    }
    names.join("\t") + "\n"
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    assert_eq!(
        a.len(),
        8,
        "manifest output role sizes qualities encoders jobs exclusions"
    );
    let manifest = fs::read_to_string(&a[0]).unwrap();
    let root = PathBuf::from(&a[1]);
    let role = &a[2];
    assert_ne!(role, "exclude_model_training_origin");
    if role == "locked_test" {
        assert!(
            root.parent()
                .unwrap()
                .join("selection.freeze.json")
                .is_file(),
            "freeze candidate selection before opening reserved test"
        );
    }
    let sizes: Vec<usize> = a[3].split(',').map(|v| v.parse().unwrap()).collect();
    let qualities: Vec<u32> = a[4].split(',').map(|v| v.parse().unwrap()).collect();
    let encoders: Vec<&str> = a[5].split(',').collect();
    let jobs: usize = a[6].parse().unwrap();
    let exclusions: Vec<(&str, u32)> = a[7]
        .split(',')
        .filter(|v| !v.is_empty())
        .map(|v| {
            let (enc, q) = v.split_once('=').expect("exclusion encoder=quality");
            assert!(encoders.contains(&enc));
            (enc, q.parse().unwrap())
        })
        .collect();
    assert!(sizes.iter().all(|&v| v >= 32) && qualities.iter().all(|&v| v <= 100));
    fs::create_dir_all(root.join("cells")).unwrap();
    fs::create_dir_all(root.join("scratch")).unwrap();
    // Never silently mix outputs built from a different executable or model.
    let binary = sha(&fs::read(std::env::current_exe().unwrap()).unwrap());
    let weights = fs::read("models/adopted/dejpeg_rt24i_canon4/weights_f16.raw").unwrap();
    let encoder_versions = encoder_identity(&encoders);
    let identity = format!("binary_sha256\t{binary}\nweights_sha256\t{}\nmanifest_sha256\t{}\nrole\t{role}\nsizes\t{}\nqualities\t{}\nencoders\t{}\n", sha(&weights), sha(manifest.as_bytes()), a[3], a[4], a[5]);
    let identity = identity
        + &format!(
            "encoder_dependencies_sha256\t{}\n",
            sha(encoder_versions.as_bytes())
        );
    let identity = identity + &format!("exclusions\t{}\nretention\tscores-only-v1\n", a[7]);
    let run_id = root.join("IDENTITY.tsv");
    if run_id.exists() {
        assert_eq!(fs::read_to_string(&run_id).unwrap(), identity);
    } else {
        fs::write(&run_id, identity).unwrap();
    }
    fs::write(root.join("ENCODERS.tsv"), encoder_versions).unwrap();
    let versions = zenanalyze::versioning::feature_qualified_names();
    let mut version_text = String::new();
    for name in PIXEL_NAMES {
        let (_, qualified) = versions
            .iter()
            .find(|(n, _)| n == name)
            .expect("unversioned feature");
        version_text.push_str(&format!("{name}\t{qualified}\n"));
    }
    fs::write(root.join("FEATURES.tsv"), version_text).unwrap();
    let weights = zensr_micro::decode_all_f16(&weights);
    let model = zensr_micro::adopted::AdoptedModel::load_compact(&weights, 24, 8, 1).unwrap();
    let restorer = zensr_zenjpeg::api::Restorer::new().with_tier(
        zensr_zenjpeg::api::Budget::Quality,
        zensr_micro::adopted::AdoptedModel::load_compact(&weights, 24, 8, 1).unwrap(),
    );
    let rows: Vec<Vec<&str>> = manifest
        .lines()
        .skip(1)
        .map(|l| l.split('\t').collect())
        .filter(|r: &Vec<&str>| r[4] == role)
        .collect();
    assert!(!rows.is_empty(), "no references for role");
    let mut coverage = String::from("file\tsize\tencoder\tss\tq\tstatus\n");
    let mut required = 0usize;
    for r in &rows {
        let sw: usize = r[7].parse().unwrap();
        let sh: usize = r[8].parse().unwrap();
        for &size in &sizes {
            for &encoder in &encoders {
                for ss in ["420", "444"] {
                    for &q in &qualities {
                        let ratio = size as f64 / sw.max(sh) as f64;
                        let w = (sw as f64 * ratio).round() as usize;
                        let h = (sh as f64 * ratio).round() as usize;
                        let status = if size > sw.max(sh) {
                            "excluded_no_upscale"
                        } else if w < 8 || h < 8 {
                            "excluded_metric_aspect_domain"
                        } else if exclusions.contains(&(encoder, q)) {
                            "excluded_by_caller_unsupported"
                        } else {
                            required += 1;
                            "required"
                        };
                        coverage.push_str(&format!(
                            "{}\t{size}\t{encoder}\t{ss}\t{q}\t{status}\n",
                            r[1]
                        ));
                    }
                }
            }
        }
    }
    fs::write(root.join("COVERAGE.tsv"), coverage).unwrap();
    eprintln!("required cells {required}; exclusions recorded in COVERAGE.tsv");
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .unwrap();
    pool.install(|| rows.par_iter().enumerate().for_each(|(index,r)| {
        assert_eq!(r.len(), 10, "manifest schema");
        let bytes=fs::read(r[0]).unwrap();
        assert_eq!(sha(&bytes), r[9], "reference bytes changed");
        let source=decode_any(Path::new(r[0])).expect("reference decode");
        let work=root.join("scratch").join(index.to_string());
        fs::create_dir_all(&work).unwrap();
        for &size in &sizes {
            // Native size is an explicit support limit, never an upscale.
            if size > source.w.max(source.h) { continue; }
            let scale=size as f64/source.w.max(source.h) as f64;
            let w=(source.w as f64*scale).round().max(1.) as usize;
            let h=(source.h as f64*scale).round().max(1.) as usize;
            if w<8 || h<8 { continue; } // Explicitly recorded in COVERAGE.tsv.
            let reference=if (w,h)==(source.w,source.h) { Rgb8Img {px:source.px.clone(),w,h} }
                else { resize_rgb8(&source,w,h,zenresize::Filter::Mitchell) };
            let ref_art=artifact(&root,&reference.px,"rgb8");
            let ppm=work.join("reference.ppm");
            encoding::write_ppm(&reference,&ppm);
            for &encoder in &encoders { for ss in ["420","444"] { for &q in &qualities {
                if exclusions.contains(&(encoder,q)) { continue; } // Caller-declared and recorded.
                let key=sha(format!("{}\t{size}\t{encoder}\t{ss}\t{q}",r[9]).as_bytes());
                let cell=root.join("cells").join(format!("{key}.tsv"));
                if cell.exists() {
                    let existing=fs::read_to_string(&cell).unwrap();
                    assert_eq!(existing.lines().count(),2,"incomplete committed cell");
                    assert!(existing.starts_with(&header()),"changed cell schema");
                    continue;
                }
                let jpg=work.join("input.jpg");
                assert!(encoding::encode(&ppm,&jpg,encoder,q,ss),"encode failed");
                let data=fs::read(&jpg).unwrap();
                let jpg_art=artifact(&root,&data,"jpg");
                let decoded=encoding::zj_decode(&data,false);
                let decode_art=String::new(); // Pixels live only until scoring finishes.
                let coef=zenjpeg::decoder::Decoder::new().decode_coefficients(&data,enough::Unstoppable).unwrap();
                let plane=&coef.components[0];
                let cf=coefficients::features(&plane.coeffs,plane.coeffs.len()/64,plane.blocks_wide).unwrap();
                let (crop,cw,ch)=zensr_zenjpeg::chooser::center_crop_rgb8(&decoded.px,w,h);
                let offer=zenanalyze::offer_for_request(&crop,cw,ch,&zensr_zenjpeg::chooser::chooser_request()).unwrap();
                let chooser=zensr_zenjpeg::chooser::classify_from_owned_offer(&offer);
                let estimate=restorer.restore(&data).budget(zensr_zenjpeg::api::Budget::Quality)
                    .routing(zensr_zenjpeg::api::Routing::Auto{min_gain:f32::INFINITY}).run().unwrap();
                let predicted=match &estimate {
                    zensr_zenjpeg::api::Outcome::Unchanged{why:zensr_zenjpeg::api::SkipReason::NotWorthIt{estimated_gain},..}=>*estimated_gain,
                    _=>panic!("metadata must not restore"),
                };
                let info=estimate.report();
                let probe=zenjpeg::detect::probe(&data).unwrap();
                let restored=zensr_zenjpeg::restore_jpeg(&data,&model,&zensr_zenjpeg::RestoreConfig::default()
                    .with_threads(1).with_high_q_identity(false)).unwrap();
                let output=Rgb8Img{px:restored.to_rgb8(),w,h};
                let out_art=String::new(); // Retain scalar results, not per-case RGB buffers.
                let mut values=vec![r[1].into(),r[2].into(),r[3].into(),r[4].into(),r[6].into(),r[9].into(),
                    w.to_string(),h.to_string(),size.to_string(),encoder.into(),ss.into(),q.to_string(),jpg_art,
                    ref_art.clone(),decode_art,out_art,info.encoder_family.clone(),info.quality_scale.clone(),
                    finite(info.quality as f64),finite(predicted as f64),
                    zensr_zenjpeg::policy_high_q_identity(&probe).to_string(),
                    zensr_zenjpeg::policy_high_q_identity_with_margin(&probe,6.).to_string(),finite(data.len() as f64*8./(w*h) as f64)];
                for v in [cf.zero_ac_blocks,cf.zero_ac_coefs,cf.hf_survival,cf.mean_abs_ac,cf.dc_spread,
                    cf.ac_count_spread,cf.gutted_block_frac,cf.gutted_clustering] {values.push(finite(v));}
                for name in PIXEL_NAMES { values.push(finite(offer.get(name).expect("missing required feature").float() as f64)); }
                values.push(finite(chooser.p_graphics as f64));
                let g=&restored.report.guard;
                for v in [g.clamped_frac,g.mean_alpha,g.roundtrip_mae,g.fallback_blend] {values.push(finite(v as f64));}
                assert_eq!(restored.report.projection.len(),3,"incomplete projection");
                for p in &restored.report.projection {values.push(finite(p.clamped_frac as f64));}
                values.extend(metric_columns(&reference,&decoded));
                values.extend(metric_columns(&reference,&output));
                let head=header();assert_eq!(values.len(),head.trim_end().split('\t').count());
                let partial=cell.with_extension("partial");
                fs::write(&partial,head+&values.join("\t")+"\n").unwrap();
                fs::rename(partial,cell).unwrap();
            }}}
        }
        eprintln!("completed reference {}/{} {}",index+1,rows.len(),r[1]);
    }));
    let completed = fs::read_dir(root.join("cells"))
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .path()
                .extension()
                .is_some_and(|x| x == "tsv")
        })
        .count();
    assert_eq!(completed, required, "incomplete grid");
    fs::write(
        root.join("COMPLETE"),
        format!("{completed}/{required} required cells completed\n"),
    )
    .unwrap();
}

#[cfg(test)]
mod scoring_tests {
    use super::*;

    #[test]
    fn identity_scores_are_retained_without_a_diagnostic_map() {
        let image = Rgb8Img {
            px: vec![128; 32 * 32 * 3],
            w: 32,
            h: 32,
        };
        let columns = metric_columns(&image, &image);
        assert_eq!(columns.len(), 8);
        assert_eq!(columns[0].parse::<f64>().unwrap(), 100.0);
        assert_eq!(columns[2].parse::<f64>().unwrap(), 0.0);
        assert_eq!(columns[3].parse::<f64>().unwrap(), 0.0);
        assert!(columns[4].is_empty());
        assert!(columns[5..]
            .iter()
            .all(|v| v.parse::<f64>().unwrap() == 0.0));
    }
}
