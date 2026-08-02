fn main() {
    let path = std::env::args().nth(1).unwrap();
    let data = std::fs::read(&path).unwrap();
    match zenpng::decode(
        &data,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    ) {
        Ok(o) => {
            let v = o.pixels.as_slice();
            println!(
                "OK {}x{} desc={:?} bpp={}",
                v.width(),
                v.rows(),
                v.descriptor(),
                v.descriptor().bytes_per_pixel()
            );
        }
        Err(e) => println!("ERR {e}"),
    }
}
