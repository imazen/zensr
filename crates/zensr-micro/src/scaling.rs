//! How many threads this machine actually delivers on the conv kernel.
//!
//! # Why this exists
//!
//! `upscale_tiled` chooses its tile so that there is at least one tile per
//! thread. That is right when the threads are real. MEASURED 2026-09-08 on a
//! 7950X (16C/32T), realtime model, 384px: chasing one tile per thread at 28
//! threads produces 36 tiles of 76px and runs in **28.8 ms**, while 16 tiles of
//! 108px runs in **16.4 ms** — 1.76x faster with twelve threads left idle. The
//! same default is slower at 28 threads than it is at 12.
//!
//! The kernel is not thread-bound there, it is bandwidth-bound, so the extra
//! threads buy nothing while the extra tiles each pay the halo border again.
//! Every closed-form tile rule assumes a speedup of `min(tiles, threads)` and
//! therefore picks too small; the missing term is a property of the MACHINE,
//! which is not derivable from the image, the model or the thread count.
//! `benchmarks/tile_ladder_2026-09-08.md` has the full accounting.
//!
//! So measure it. That is what this module does — and **measuring it as one
//! machine-wide number does not work.** The tiler does not use it.
//!
//! # FALSIFIED as a tiling input (2026-09-08)
//!
//! Wired into `upscale_tiled`'s starvation target and A/B'd against the
//! uncapped rule — same binary both arms, `ZENSR_THREAD_SATURATION` pinning the
//! cap so only the planned-for number changed, 3 paired reps, probe amortised:
//!
//! | cell | uncapped | capped at 13 | |
//! |---|---|---|---|
//! | realtime 384px 28T | 24.3 ms | 20.3 ms | **+15.1%** |
//! | realtime 768px 28T | 53.5 ms | 51.8 ms | +2.8% |
//! | realtime 512px 28T | 25.9 ms | 31.5 ms | **−29.6%** |
//! | realtime 384px 12T | 17.5 ms | 22.4 ms | **−28.0%** |
//! | realtime 192px 28T | 6.4 ms | 8.9 ms | **−39.1%** |
//! | quality 384px 28T | 239.3 ms | 334.2 ms | **−39.7%** |
//!
//! **Net median −2.6%.** Only the cell it was designed for improves.
//!
//! The reason is that saturation is not a property of the machine alone: it
//! moves with the per-tile working set. This box probes at 13 on a 96px, 24
//! channel workload, but at 512px the tiled run productively uses far more than
//! 13 threads — the measured optimum there is 49 tiles — so capping to 13 throws
//! away real parallelism. A probe shaped like the real tile would be
//! size-dependent and per-model, which is a probe per call, not per process.
//!
//! # What it is still for
//!
//! A diagnostic. `examples/sat_probe` prints it, and it answers "does this box
//! scale on this kernel at all" — which is worth knowing before blaming a tile
//! rule for a machine's memory bandwidth. `ZENSR_THREAD_SATURATION` pins it.

use std::sync::OnceLock;
use std::time::Instant;

/// Aggregate conv throughput scales this far and no further, in threads.
///
/// Measured once per process and cached (about 40 ms on a 32-thread box).
/// `ZENSR_THREAD_SATURATION=N` overrides the probe entirely (0 or unparseable
/// is ignored).
///
/// **Diagnostic only — the tiler does not consult this.** See the module docs
/// for the A/B that rejected it as a tiling input.
pub fn thread_saturation() -> usize {
    static SAT: OnceLock<usize> = OnceLock::new();
    *SAT.get_or_init(|| {
        if let Some(n) = std::env::var("ZENSR_THREAD_SATURATION")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|n| *n > 0)
        {
            return n;
        }
        let hw = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        probe(hw)
    })
}

/// `requested`, capped by the measured saturation. Kept because it is the
/// natural way to ask the question and because the rejection is worth being
/// able to re-run — NOT used by `upscale_tiled`, which plans for the requested
/// thread count. See the module docs.
pub fn effective_threads(requested: usize) -> usize {
    requested.max(1).min(thread_saturation().max(1))
}

/// One conv3x3 pass over a small plane. Shaped like the real tile work — same
/// kernel, same channel count as the realtime model — because the thing being
/// measured is bandwidth contention between concurrent tiles, which a
/// cache-resident toy loop would not show.
fn unit(inp: &[f32], wts: &[f32], bias: &[f32], out: &mut [f32], reps: usize) {
    for _ in 0..reps {
        crate::simd::conv3x3_dispatch(inp, CH, wts, bias, out, CH, SIDE, SIDE);
        // Defeat any attempt to hoist the call out of the loop.
        std::hint::black_box(&out[0]);
    }
}

const SIDE: usize = 96;
const CH: usize = 24;

/// Aggregate speedup of `n` concurrent conv streams over one, rounded.
///
/// Not `n * t1 / tn` on a single timing pair — the two phases are interleaved
/// and repeated, and the MEDIAN of the per-round ratios is taken, because a
/// single pair on a shared box is exactly the measurement that reports whatever
/// else was running at the time.
fn probe(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mk = |seed: u32, len: usize| -> Vec<f32> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32 / 8_388_608.0) - 1.0
            })
            .collect()
    };
    let inp = mk(7, CH * SIDE * SIDE);
    let wts = mk(11, CH * CH * 9);
    let bias = mk(13, CH);

    // Warm the caches and size the round so one phase is a few milliseconds:
    // long enough to swamp thread spawn, short enough that the whole probe
    // stays inside a few tens of ms.
    let mut out = vec![0f32; CH * SIDE * SIDE];
    unit(&inp, &wts, &bias, &mut out, 1);
    let t = Instant::now();
    unit(&inp, &wts, &bias, &mut out, 2);
    let per_pass = t.elapsed().as_secs_f64() / 2.0;
    let reps = ((0.003 / per_pass.max(1e-9)).ceil() as usize).clamp(1, 512);

    let mut ratios = Vec::with_capacity(3);
    for _ in 0..3 {
        let t1 = {
            let mut o = vec![0f32; CH * SIDE * SIDE];
            let t = Instant::now();
            unit(&inp, &wts, &bias, &mut o, reps);
            t.elapsed().as_secs_f64()
        };
        let tn = {
            let t = Instant::now();
            std::thread::scope(|sc| {
                for _ in 0..n {
                    sc.spawn(|| {
                        let mut o = vec![0f32; CH * SIDE * SIDE];
                        unit(&inp, &wts, &bias, &mut o, reps);
                    });
                }
            });
            t.elapsed().as_secs_f64()
        };
        if t1 > 0.0 && tn > 0.0 {
            ratios.push(n as f64 * t1 / tn);
        }
    }
    if ratios.is_empty() {
        return n;
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let s = ratios[ratios.len() / 2];
    // Round to a whole thread, and never report more than were asked to run —
    // a ratio above n means the single-thread phase was disturbed, not that the
    // box has more cores than it has.
    (s.round() as usize).clamp(1, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The probe has to return something usable on any machine, including a
    /// single core and including one under load: never zero, never more than
    /// the hardware has.
    #[test]
    fn saturation_is_in_range() {
        let hw = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let sat = thread_saturation();
        assert!(sat >= 1, "saturation {sat} below 1");
        assert!(sat <= hw.max(1), "saturation {sat} above hardware {hw}");
    }

    /// Asking for fewer threads than the machine saturates at must return the
    /// request unchanged — the cap only ever removes threads the box cannot
    /// deliver.
    #[test]
    fn effective_never_exceeds_request() {
        for req in [1usize, 2, 3, 4, 8, 12, 28, 256] {
            let eff = effective_threads(req);
            assert!(eff >= 1 && eff <= req.max(1), "req {req} -> eff {eff}");
        }
        assert_eq!(effective_threads(0), 1, "zero threads must mean one");
    }
}
