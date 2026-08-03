//! Do two JPEGs carry IDENTICAL quantized coefficients?
//!
//! Comparing decoded pixels conflates three things: quantization decisions,
//! DCT implementation (islow vs float vs ifast), and colour conversion
//! rounding. Only the first says whether a quantization table is enough to
//! reproduce an encoder. This compares the coefficients themselves.
fn main() {
    let a = std::env::args().nth(1).expect("a.jpg");
    let b = std::env::args().nth(2).expect("b.jpg");
    let load = |p: &str| {
        zenjpeg::decoder::Decoder::new()
            .decode_coefficients(&std::fs::read(p).unwrap(), enough::Unstoppable)
            .expect("coefficients")
    };
    let (ca, cb) = (load(&a), load(&b));
    if ca.components.len() != cb.components.len() {
        println!("component count differs");
        return;
    }
    let (mut total, mut diff, mut maxd) = (0u64, 0u64, 0i32);
    for (x, y) in ca.components.iter().zip(cb.components.iter()) {
        if x.coeffs.len() != y.coeffs.len() {
            println!("coefficient count differs");
            return;
        }
        for (p, q) in x.coeffs.iter().zip(y.coeffs.iter()) {
            total += 1;
            let d = (*p as i32 - *q as i32).abs();
            if d != 0 {
                diff += 1;
                maxd = maxd.max(d);
            }
        }
    }
    println!(
        "{:.4}% of coefficients differ (max |delta| {maxd} quantizer steps, n={total})",
        100.0 * diff as f64 / total as f64
    );
}
