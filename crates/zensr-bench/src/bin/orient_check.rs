//! Does the pinned zenjpeg's default auto-orient agree with a reference?
//!
//! zenjpeg#149 was a real defect exactly here — native `auto_orient(true)` (the
//! default) was wrong on EXIF-rotated 4:2:0 JPEGs, 36% of one image more than 2%
//! off against ImageMagick `-auto-orient`, and ~134 rotated JPEGs corpus-wide
//! were affected. It is recorded as fixed, but every clean reference this corpus
//! builds decodes through that path, and a note is not a measurement.
//!
//! Usage: orient_check <src.jpg> <reference.png>
//! Reference: `convert src.jpg -auto-orient -colorspace sRGB PNG24:ref.png`
use std::path::PathBuf;

fn main() {
    let mut a = std::env::args().skip(1);
    let src = PathBuf::from(a.next().expect("src image"));
    let refp = PathBuf::from(a.next().expect("reference png"));
    let got = zensr_bench::decode_any(&src).expect("decode src");
    let want = zensr_bench::decode_any(&refp).expect("decode reference");
    if (got.w, got.h) != (want.w, want.h) {
        println!(
            "DIMS DIFFER  got {}x{}  ref {}x{}  <- orientation not applied the same way",
            got.w, got.h, want.w, want.h
        );
        std::process::exit(1);
    }
    let n = got.px.len().min(want.px.len());
    let (mut max, mut over2, mut sum) = (0u8, 0usize, 0f64);
    for i in 0..n {
        let d = got.px[i].abs_diff(want.px[i]);
        max = max.max(d);
        // ">2%" is the threshold the original defect was reported at: 2% of
        // full scale is ~5 levels.
        if d > 5 {
            over2 += 1;
        }
        sum += f64::from(d);
    }
    println!(
        "{}x{}  max {max}  mean {:.3}  >2%: {over2}/{n} ({:.3}%)",
        got.w,
        got.h,
        sum / n as f64,
        100.0 * over2 as f64 / n as f64
    );
}
