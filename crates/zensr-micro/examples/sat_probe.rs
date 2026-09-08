//! What does this machine actually saturate at, and how long does asking cost?
//!
//! `upscale_tiled` plans its tiling for the threads the box DELIVERS, not the
//! ones it is asked for, because chasing one tile per requested thread made the
//! realtime model slower at 28 threads than at 12 (see `crate::scaling`). That
//! number is measured once per process; this prints it, so a surprising tile
//! choice on an unfamiliar box can be explained rather than guessed at.
//!
//! `ZENSR_THREAD_SATURATION=N` pins it — use that to reproduce a measurement,
//! or to check what the uncapped rule would have picked (`=4096`).
fn main() {
    let t = std::time::Instant::now();
    let sat = zensr_micro::thread_saturation();
    let probe_ms = t.elapsed().as_secs_f64() * 1e3;
    let hw = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let pinned = std::env::var("ZENSR_THREAD_SATURATION").ok();
    println!(
        "hardware {hw} threads, saturation {sat}{}, probe {probe_ms:.1} ms",
        match &pinned {
            Some(v) => format!(" (PINNED by ZENSR_THREAD_SATURATION={v})"),
            None => String::new(),
        }
    );
    println!("{:>10}  {:>9}", "requested", "effective");
    for r in [1usize, 2, 4, 8, 12, 16, 28, 32, 64] {
        println!("{r:>10}  {:>9}", zensr_micro::effective_threads(r));
    }
}
