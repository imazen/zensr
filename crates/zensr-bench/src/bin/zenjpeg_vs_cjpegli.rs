//! Can we tell zenjpeg apart from cjpegli, and how good is the quality estimate?
//!
//! zenjpeg inherits jpegli's adaptive-quantization lineage, so both plausibly
//! present the same marker signature. If the probe cannot separate them, they
//! share a projection slack whether or not their quantisers agree.
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig};

fn dqt_luma(data: &[u8]) -> Option<[u16; 64]> {
    const ZZ: [usize; 64] = [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
        20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    ];
    let mut i = 2usize;
    while i + 3 < data.len() {
        if data[i] != 0xFF {
            i += 1;
            continue;
        }
        let m = data[i + 1];
        if m == 0xD8 || m == 0x01 || (0xD0..=0xD7).contains(&m) {
            i += 2;
            continue;
        }
        if m == 0xD9 {
            break;
        }
        let len = ((data[i + 2] as usize) << 8) | data[i + 3] as usize;
        if i + 2 + len > data.len() {
            break;
        }
        if m == 0xDB {
            let seg = &data[i + 4..i + 2 + len];
            let (pq, tq) = (seg[0] >> 4, seg[0] & 15);
            if tq == 0 && pq == 0 && seg.len() >= 65 {
                let mut out = [0u16; 64];
                for k in 0..64 {
                    out[ZZ[k]] = seg[1 + k] as u16;
                }
                return Some(out);
            }
        }
        if m == 0xDA {
            break;
        }
        i += 2 + len;
    }
    None
}

fn main() {
    let (w, h) = (128usize, 128usize);
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            rgb[i] = ((x * 7 + y * 3) % 256) as u8;
            rgb[i + 1] = ((x * 2 + y * 11) % 256) as u8;
            rgb[i + 2] = ((x * 13 + (y * y) % 97) % 256) as u8;
        }
    }
    let px: &[rgb::Rgb<u8>] = bytemuck::cast_slice(&rgb);
    println!("q\tfamily\tdistance\tvalues");
    for qi in 1..=100u32 {
        let q = qi as f32;
        let jpg = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
            .encode(px, w as u32, h as u32)
            .expect("encode");
        let p = zenjpeg::detect::probe(&jpg).expect("probe");
        let t = dqt_luma(&jpg).expect("dqt");
        let id = zensr_zenjpeg::qtables::identify_luma(&t, 1);
        let _ = id;
        let vals: Vec<String> = t.iter().map(|v| v.to_string()).collect();
        println!(
            "{qi}\t{:?}\t{:.3}\t{}",
            p.encoder,
            p.quality.value,
            vals.join(",")
        );
    }
}
