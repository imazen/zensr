//! Canon4 QAT: exact packed-weight loading and paired full-pipeline IQA.
//! Usage: qat_eval verify model-dir | qat_eval inputs.tsv training-root output jobs
#![forbid(unsafe_code)]
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path, sync::Mutex};
// Reuse the bench RGB/SSIM2 adapter from src/lib.rs at edfd6097; this
// isolated tool omits the unrelated codec/ONNX dependency graph.
struct Rgb8Img {
    px: Vec<u8>,
    w: usize,
    h: usize,
}
fn ssim2(a: &Rgb8Img, b: &Rgb8Img) -> f64 {
    let av: Vec<[u8; 3]> = a.px.as_chunks::<3>().0.to_vec();
    let bv: Vec<[u8; 3]> = b.px.as_chunks::<3>().0.to_vec();
    fast_ssim2::compute_ssimulacra2(
        imgref::Img::new(av, a.w, a.h).as_ref(),
        imgref::Img::new(bv, b.w, b.h).as_ref(),
    )
    .expect("SSIMULACRA2")
}
use zensr_micro::adopted::AdoptedModel;
use zensr_zenjpeg::{
    RestoreConfig,
    api::{Budget, Outcome, Restorer},
};
#[path = "../../../../crates/zensr-bench/src/bin/routing_study/storage.rs"]
mod storage;

fn hash(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(64);
    for b in Sha256::digest(bytes) {
        text.push(HEX[(b >> 4) as usize] as char);
        text.push(HEX[(b & 15) as usize] as char);
    }
    text
}
fn floats(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len().is_multiple_of(4));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}
fn unpack(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() < 24
        || &bytes[..8] != b"ZSRQ8PC\0"
        || floats(&bytes[8..24])
            .iter()
            .enumerate()
            .any(|(i, v)| v.to_bits() != [1, 24, 8, 1][i])
    {
        return Err("unsupported Q8PC header".into());
    }
    let mut rest = &bytes[24..];
    fn take<'a>(r: &mut &'a [u8], n: usize) -> Result<&'a [u8], String> {
        let (a, b) = r.split_at_checked(n).ok_or("truncated Q8PC weights")?;
        *r = b;
        Ok(a)
    }
    let mut out = Vec::with_capacity(43203);
    for i in 0..10 {
        let (cin, cout) = (if i == 0 { 3 } else { 24 }, if i == 9 { 3 } else { 24 });
        let scales = floats(take(&mut rest, cout * 4)?);
        if scales.iter().any(|s| !s.is_finite() || *s <= 0.) {
            return Err("invalid Q8PC scale".into());
        }
        let integer = take(&mut rest, cout * cin * 9)?;
        for (&scale, channel) in scales.iter().zip(integer.chunks_exact(cin * 9)) {
            for &b in channel {
                let v = b as i8;
                if v == i8::MIN {
                    return Err("out of range Q8PC code".into());
                }
                out.push(f32::from(v) * scale);
            }
        }
        out.extend(floats(take(&mut rest, cout * 4)?));
        if i != 9 {
            out.extend(floats(take(&mut rest, cout * 4)?));
        }
    }
    if !rest.is_empty() || out.len() != 43203 || out.iter().any(|v| !v.is_finite()) {
        return Err("invalid Q8PC payload".into());
    }
    Ok(out)
}
fn load(dir: &Path) -> (AdoptedModel, String) {
    let raw = fs::read(dir.join("weights.raw")).expect("f32 reference weights");
    let expected = floats(&raw);
    let packed = dir.join("weights_q8pc.bin");
    let (values, digest) = if packed.exists() {
        let bytes = fs::read(packed).unwrap();
        let values = unpack(&bytes).unwrap();
        assert!(
            values
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits())
                && values.len() == expected.len(),
            "packed weights differ from Python export"
        );
        (values, hash(&bytes))
    } else {
        (expected, hash(&raw))
    };
    let model = AdoptedModel::load_compact(&values, 24, 8, 1).unwrap();
    let input = floats(&fs::read(dir.join("gold_in_17x13.raw")).unwrap());
    let expected = floats(&fs::read(dir.join("gold_out_17x13.raw")).unwrap());
    let result = model.upscale_tiled(&input, 17, 13, 1, 0);
    assert_eq!(result.len(), expected.len());
    assert!(result.iter().all(|v| v.is_finite()));
    let max = result
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max < 5e-4,
        "existing adopted-model golden tolerance failed: {max}"
    );
    (model, format!("{}\t{}\t{max:.12}", hash(&raw), digest))
}
fn artifact(root: &Path, bytes: &[u8], suffix: &str) -> String {
    // Reuse routing_study's bounded zstd pipe, draining output concurrently.
    use std::process::{Command, Stdio};
    let mut child = Command::new("zstd")
        .args(["-q", "-3", "-T1", "-c"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let result = std::thread::scope(|scope| {
        let writing = scope.spawn(move || {
            let result = stdin.write_all(bytes);
            drop(stdin);
            result
        });
        let result = child.wait_with_output().unwrap();
        writing.join().unwrap().unwrap();
        result
    });
    assert!(result.status.success(), "artifact compression");
    let bytes = &result.stdout;
    let digest = hash(bytes);
    let dir = root.join("artifacts").join(&digest[..2]);
    fs::create_dir_all(&dir).unwrap();
    let relative = format!("artifacts/{}/{digest}.{suffix}.zst", &digest[..2]);
    storage::publish(&root.join(&relative), bytes);
    relative
}

struct Measures {
    ssim: f64,
    max: f64,
    pnorm: f64,
    pixels: String,
    map: String,
}
fn measure(root: &Path, reference: &Rgb8Img, image: &Rgb8Img) -> Measures {
    use rgb::FromSlice;
    assert_eq!((reference.w, reference.h), (image.w, image.h));
    let ssim = ssim2(reference, image);
    assert!(ssim.is_finite());
    let result = butteraugli::butteraugli(
        imgref::Img::new(reference.px.as_rgb(), reference.w, reference.h),
        imgref::Img::new(image.px.as_rgb(), image.w, image.h),
        &butteraugli::ButteraugliParams::default().with_compute_diffmap(true),
    )
    .unwrap();
    assert!(result.score.is_finite() && result.pnorm_3.is_finite());
    let map: Vec<u8> = result
        .diffmap
        .as_ref()
        .unwrap()
        .buf()
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    Measures {
        ssim,
        max: result.score,
        pnorm: result.pnorm_3,
        pixels: artifact(root, &image.px, "rgb8"),
        map: artifact(root, &map, "butter-f32le"),
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 3 && args[1] == "verify" {
        let (_, identity) = load(Path::new(&args[2]));
        println!("{identity}");
        return;
    }
    assert!(
        args.len() == 5 || (args.len() == 7 && args[5] == "--only-model" && args[6] == "ptq8-gpu"),
        "qat_eval inputs.tsv training-root output jobs"
    );
    let training = Path::new(&args[2]);
    assert!(
        training.join("COMPLETE").is_file(),
        "freeze all selected checkpoints before evaluation"
    );
    let output = Path::new(&args[3]);
    fs::create_dir(output).expect("new output directory required");
    fs::create_dir(output.join("cells")).unwrap();
    let mut names = vec!["baseline".to_owned(), "ptq8".to_owned()];
    for seed in [7, 17, 29] {
        for arm in ["float", "qat8"] {
            names.push(format!("{arm}-s{seed}-selected"));
        }
    }
    if args.len() == 7 {
        names = vec!["baseline".to_owned(), "ptq8-gpu".to_owned()];
    }
    let mut identity = format!(
        "binary_sha256\t{}\ninputs_sha256\t{}\ntraining_plan_sha256\t{}\n",
        hash(&fs::read(std::env::current_exe().unwrap()).unwrap()),
        hash(&fs::read(&args[1]).unwrap()),
        hash(&fs::read(training.join("PLAN.json")).unwrap())
    );
    let models: Vec<_> = names
        .iter()
        .map(|name| {
            let (m, id) = load(&training.join(name));
            identity += &format!("{name}\t{id}\n");
            m
        })
        .collect();
    fs::write(output.join("IDENTITY.tsv"), identity).unwrap();
    let (base, _) = load(&training.join("baseline"));
    let restorer = Restorer::new().with_tier(Budget::Realtime, base);
    let manifest = fs::read_to_string(&args[1]).unwrap();
    fs::write(output.join("inputs.tsv"), &manifest).unwrap();
    let mut lines = manifest.lines();
    let columns: Vec<_> = lines.next().unwrap().split('\t').collect();
    let rows: Vec<Vec<_>> = lines.map(|l| l.split('\t').collect()).collect();
    assert!(!rows.is_empty());
    let col = |name: &str| {
        columns
            .iter()
            .position(|s| *s == name)
            .expect("manifest column")
    };
    let reference_i = col("reference_rgb8");
    let refsha_i = col("reference_rgb8_sha256");
    let w_i = col("width");
    let h_i = col("height");
    let mut references = std::collections::HashMap::new();
    for r in &rows {
        assert_eq!(r.len(), columns.len());
        references.entry(r[reference_i]).or_insert_with(|| {
            let bytes = fs::read(r[reference_i]).unwrap();
            assert_eq!(hash(&bytes), r[refsha_i]);
            let w = r[w_i].parse::<usize>().unwrap();
            let h = r[h_i].parse::<usize>().unwrap();
            assert_eq!(bytes.len(), w * h * 3);
            Rgb8Img { px: bytes, w, h }
        });
    }
    let header = "id\tsource_id\torigin\tcontent\tcohort\tsize\tencoder\tss\tq\tarm\tdefault_restore\tidentity_ssim2\tforced_ssim2\trouted_ssim2\tidentity_butter_max\tforced_butter_max\tidentity_butter_pnorm3\tforced_butter_pnorm3\tforced_artifact\trouted_artifact\tidentity_artifact\tforced_butter_map\tidentity_butter_map\n";
    let mut count = Mutex::new(0usize);
    let progress = |stage: &str, done: usize| {
        let path = output.join("PROGRESS.tsv");
        let tmp = output.join("PROGRESS.new");
        fs::write(
            &tmp,
            format!(
                "pid\t{}\nstage\t{stage}\ncompleted\t{done}\nrequired\t{}\n",
                std::process::id(),
                rows.len()
            ),
        )
        .unwrap();
        fs::rename(tmp, path).unwrap();
    };
    progress("running", 0);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args[4].parse().unwrap())
        .build()
        .unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.install(||rows.par_iter().for_each(|r|{
        let data=fs::read(r[col("jpeg")]).unwrap();assert_eq!(hash(&data),r[col("jpeg_sha256")]);
        let reference=&references[r[reference_i]];
        let dec=zenjpeg::decoder::Decoder::new().deblock(zenjpeg::decoder::DeblockMode::Off).decode(&data,enough::Unstoppable).unwrap();
        let (w,h)=dec.dimensions();assert_eq!((w as usize,h as usize),(reference.w,reference.h));
        let plain=Rgb8Img{px:dec.pixels_u8().unwrap().to_vec(),w:w as usize,h:h as usize};
        let identity=measure(output,reference,&plain);
        let routed=restorer.restore(&data).threads(1).run().unwrap();let restore=matches!(routed,Outcome::Restored{..});
        if !restore {assert_eq!(routed.pixels().to_rgb8(),plain.px);}
        let mut cell=header.to_owned();
        for (name,model) in names.iter().zip(&models) {
            let cfg=RestoreConfig::default().with_threads(1).with_high_q_identity(false);
            let restored=zensr_zenjpeg::restore_jpeg(&data,model,&cfg).unwrap();assert!(restored.planes.iter().all(|v|v.is_finite()));
            let pixels=Rgb8Img{px:restored.to_rgb8(),w:restored.width,h:restored.height};
            if name=="baseline" && restore {assert_eq!(pixels.px,routed.pixels().to_rgb8(),"frozen default pipeline differs from forced baseline");}
            let forced=measure(output,reference,&pixels);let routed_metric=if restore {&forced}else{&identity};
            let metadata=["id","source_id","origin","content","cohort","size","encoder","ss","q"].map(|s|r[col(s)]).join("\t");
            cell+=&format!("{metadata}\t{name}\t{}\t{:.17}\t{:.17}\t{:.17}\t{:.17}\t{:.17}\t{:.17}\t{:.17}\t{}\t{}\t{}\t{}\t{}\n",u8::from(restore),identity.ssim,forced.ssim,routed_metric.ssim,identity.max,forced.max,identity.pnorm,forced.pnorm,forced.pixels,routed_metric.pixels,identity.pixels,forced.map,identity.map);
        }
        storage::publish(&output.join("cells").join(format!("{}.tsv",r[col("id")])),cell.as_bytes());
        let mut done=count.lock().unwrap();*done+=1;
        if done.is_multiple_of(16) || *done==rows.len() {progress("running",*done);println!("completed {}/{} input cells",*done,rows.len());}
    }))
    }));
    if let Err(error) = result {
        progress("failed", *count.get_mut().unwrap());
        std::panic::resume_unwind(error);
    }
    progress("complete", rows.len());
    let mut report = fs::File::create_new(output.join("results.tsv")).unwrap();
    report.write_all(header.as_bytes()).unwrap();
    for r in &rows {
        let cell =
            fs::read_to_string(output.join("cells").join(format!("{}.tsv", r[col("id")]))).unwrap();
        let mut rows = cell.lines();
        assert_eq!(rows.next(), Some(header.trim_end()));
        for row in rows {
            writeln!(report, "{row}").unwrap();
        }
    }
    report.sync_all().unwrap();
    fs::write(
        output.join("COMPLETE"),
        hash(&fs::read(output.join("results.tsv")).unwrap()),
    )
    .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Vec<u8> {
        let mut b = b"ZSRQ8PC\0".to_vec();
        for v in [1u32, 24, 8, 1] {
            b.extend(v.to_le_bytes());
        }
        for i in 0..10 {
            let (cin, cout) = (if i == 0 { 3 } else { 24 }, if i == 9 { 3 } else { 24 });
            for _ in 0..cout {
                b.extend(0.125f32.to_le_bytes());
            }
            b.extend(vec![3u8; cout * cin * 9]);
            for _ in 0..cout {
                b.extend(0.25f32.to_le_bytes());
            }
            if i != 9 {
                for _ in 0..cout {
                    b.extend(0.5f32.to_le_bytes());
                }
            }
        }
        b
    }
    #[test]
    fn compact_per_channel_layout_and_bounds() {
        let bytes = fixture();
        let raw = unpack(&bytes).unwrap();
        assert_eq!(raw.len(), 43203);
        assert!(raw[..648].iter().all(|v| *v == 0.375));
        assert!(raw[648..672].iter().all(|v| *v == 0.25));
        assert!(raw[672..696].iter().all(|v| *v == 0.5));
        for end in [0, 23, bytes.len() - 1] {
            assert!(unpack(&bytes[..end]).is_err());
        }
        let mut b = bytes.clone();
        b.push(0);
        assert!(unpack(&b).is_err());
        let mut b = bytes.clone();
        b[24..28].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(unpack(&b).is_err());
        let mut b = bytes;
        b[24 + 24 * 4] = 128;
        assert!(unpack(&b).is_err());
    }
}
