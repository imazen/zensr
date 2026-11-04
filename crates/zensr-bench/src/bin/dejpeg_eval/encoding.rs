use std::path::PathBuf;
use std::process::Command;
use zensr_bench::Rgb8Img;

pub(super) fn write_ppm(img: &Rgb8Img, p: &PathBuf) {
    let mut buf = format!("P6\n{} {}\n255\n", img.w, img.h).into_bytes();
    buf.extend_from_slice(&img.px);
    std::fs::write(p, buf).unwrap();
}

pub(super) fn encode(ppm: &PathBuf, jpg: &PathBuf, enc: &str, q: u32, ss: &str) -> bool {
    let home = std::env::var("HOME").unwrap();
    let st = match enc {
        "turbo" => Command::new("cjpeg")
            .args([
                "-quality",
                &q.to_string(),
                "-sample",
                if ss == "420" { "2x2" } else { "1x1" },
                "-optimize",
                "-outfile",
            ])
            .arg(jpg)
            .arg(ppm)
            .status(),
        "mozjpeg" => Command::new(format!("{home}/tmp/ati-bin/mozjpeg-cjpeg"))
            .env(
                "LD_LIBRARY_PATH",
                format!("{home}/tmp/ati-bin/mozjpeg-lib64"),
            )
            .args([
                "-quality",
                &q.to_string(),
                "-sample",
                if ss == "420" { "2x2" } else { "1x1" },
                "-outfile",
            ])
            .arg(jpg)
            .arg(ppm)
            .status(),
        "jpegli" => Command::new("cjpegli")
            .arg(ppm)
            .arg(jpg)
            .args(["-q", &q.to_string(), &format!("--chroma_subsampling={ss}")])
            .status(),
        _ => {
            // "zenjpeg" or "zenjpeg:<variant>" (knob settings: zjtool.rs)
            let variant = enc.split_once(':').map(|(_, v)| v).unwrap_or("default");
            let me = std::env::current_exe().unwrap();
            Command::new(me.parent().unwrap().join("zjtool"))
                .arg("enc")
                .arg(ppm)
                .arg(jpg)
                .args([&q.to_string(), ss, variant])
                .status()
        }
    };
    st.map(|s| s.success()).unwrap_or(false)
}

pub(super) fn zj_decode(data: &[u8], deblock_auto: bool) -> Rgb8Img {
    let mode = if deblock_auto {
        zenjpeg::decoder::DeblockMode::Auto
    } else {
        zenjpeg::decoder::DeblockMode::Off
    };
    let r = zenjpeg::decoder::Decoder::new()
        .deblock(mode)
        .decode(data, enough::Unstoppable)
        .expect("zenjpeg decode");
    let (w, h) = r.dimensions();
    Rgb8Img {
        px: r.pixels_u8().expect("u8").to_vec(),
        w: w as usize,
        h: h as usize,
    }
}
