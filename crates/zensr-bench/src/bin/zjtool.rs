//! zenjpeg encode/decode/probe CLI for the dejpeg-v2 data pipeline.
//! The DEPLOYMENT decoder is zenjpeg — all training pairs run through here.
//!
//!   zjtool enc <in.ppm> <out.jpg> <q> <420|444>       zenjpeg encoder
//!   zjtool dec <in.jpg> <out.ppm> <off|auto>          zenjpeg decoder (deblock mode)
//!   zjtool probe <in.jpg>                             fingerprint -> one TSV line

use zensr_bench::Rgb8Img;

fn read_ppm(p: &str) -> Rgb8Img {
    let d = std::fs::read(p).expect("read ppm");
    // byte-wise P6 header parse: magic, then 3 ints (w, h, maxval), each
    // preceded by whitespace/comments, followed by ONE whitespace byte.
    assert_eq!(&d[..2], b"P6", "not P6");
    let mut pos = 2usize;
    let mut vals = [0usize; 3];
    for v in vals.iter_mut() {
        while d[pos].is_ascii_whitespace() {
            pos += 1;
        }
        while d[pos] == b'#' {
            while d[pos] != b'\n' {
                pos += 1;
            }
            while d[pos].is_ascii_whitespace() {
                pos += 1;
            }
        }
        let mut n = 0usize;
        while d[pos].is_ascii_digit() {
            n = n * 10 + (d[pos] - b'0') as usize;
            pos += 1;
        }
        *v = n;
    }
    pos += 1; // single whitespace after maxval
    let (w, h, maxv) = (vals[0], vals[1], vals[2]);
    assert_eq!(maxv, 255, "maxval");
    Rgb8Img {
        px: d[pos..pos + 3 * w * h].to_vec(),
        w,
        h,
    }
}

fn write_ppm(img: &Rgb8Img, p: &str) {
    let mut buf = format!("P6\n{} {}\n255\n", img.w, img.h).into_bytes();
    buf.extend_from_slice(&img.px);
    std::fs::write(p, buf).expect("write ppm");
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a[0].as_str() {
        "enc" => {
            let img = read_ppm(&a[1]);
            let q: f32 = a[3].parse().unwrap();
            let ss = match a[4].as_str() {
                "420" => zenjpeg::encoder::ChromaSubsampling::Quarter,
                "444" => zenjpeg::encoder::ChromaSubsampling::None,
                other => panic!("subsampling {other}"),
            };
            let cfg = zenjpeg::encoder::EncoderConfig::ycbcr(q, ss);
            let px: &[rgb::Rgb<u8>] = bytemuck::cast_slice(&img.px);
            let jpeg = cfg
                .encode(px, img.w as u32, img.h as u32)
                .expect("zenjpeg encode");
            std::fs::write(&a[2], jpeg).expect("write jpg");
        }
        "dec" => {
            let data = std::fs::read(&a[1]).expect("read jpg");
            let mode = match a[3].as_str() {
                "off" => zenjpeg::decoder::DeblockMode::Off,
                "auto" => zenjpeg::decoder::DeblockMode::Auto,
                other => panic!("deblock {other}"),
            };
            let r = zenjpeg::decoder::Decoder::new()
                .deblock(mode)
                .decode(&data, enough::Unstoppable)
                .expect("zenjpeg decode");
            let (w, h) = r.dimensions();
            let px = r.pixels_u8().expect("u8 pixels").to_vec();
            assert_eq!(px.len(), 3 * w as usize * h as usize, "expect RGB8");
            write_ppm(
                &Rgb8Img {
                    px,
                    w: w as usize,
                    h: h as usize,
                },
                &a[2],
            );
        }
        // dmap <in.jpg> <out.pgm16> : per-8x8-luma-block damage map (S10 math)
        //   damage = sum over bands of Q[u,v]^2/12 where coeff==0 (erased detail
        //   uncertainty), log1p-scaled to u16. Dims = blocks_wide x blocks_high.
        "dmap" => {
            let data = std::fs::read(&a[1]).expect("read jpg");
            let dc = zenjpeg::decoder::Decoder::new()
                .decode_coefficients(&data, enough::Unstoppable)
                .expect("coeffs");
            let y = &dc.components[0];
            let qt = dc.quant_tables[y.quant_table_idx as usize].expect("luma quant table");
            let (bw, bh) = (y.blocks_wide, y.blocks_high);
            let mut out = vec![0u16; bw * bh];
            for b in 0..bw * bh {
                let blk = &y.coeffs[b * 64..(b + 1) * 64];
                let mut dmg = 0.0f64;
                for k in 1..64 {
                    if blk[k] == 0 {
                        let q = qt[k] as f64;
                        dmg += q * q / 12.0;
                    }
                }
                // log1p scale: q=255 worst case sum ~ 3.4e5 -> ln ~ 12.7; map to u16
                out[b] = ((dmg.ln_1p() / 13.0).min(1.0) * 65535.0) as u16;
            }
            let mut buf = format!("P5\n{bw} {bh}\n65535\n").into_bytes();
            for v in &out {
                buf.extend_from_slice(&v.to_be_bytes());
            }
            std::fs::write(&a[2], buf).expect("write pgm");
        }
        "probe" => {
            let data = std::fs::read(&a[1]).expect("read jpg");
            match zenjpeg::detect::probe(&data) {
                Ok(p) => println!(
                    "{:?}\t{:.1}\t{:?}\t{:?}\t{:?}",
                    p.encoder, p.quality.value, p.quality.scale, p.subsampling, p.mode
                ),
                Err(e) => println!("ERR\t{e:?}"),
            }
        }
        other => panic!("unknown subcommand {other}"),
    }
}
