//! GPU spike (task #15): can wgpu/CUDA beat the CPU tiers END-TO-END
//! (upload + compute + readback) at web-image sizes?
//!
//! One deliberately naive conv3x3+PReLU kernel (thread-per-output-pixel,
//! no shared-memory tiling) chained into the two production topologies.
//! This measures the FLOOR of GPU perf; a tiled kernel only improves it.
//!
//! Usage: gpu_spike            (runs 0.25/1/4 MP on the compiled backend)
//! Build: --features cuda | --features wgpu

use cubecl::prelude::*;
use std::time::Instant;

#[cube(launch_unchecked)]
fn conv3x3_prelu(
    inp: &Array<f32>,
    wts: &Array<f32>,
    bias: &Array<f32>,
    slopes: &Array<f32>,
    out: &mut Array<f32>,
    #[comptime] cin: u32,
    #[comptime] cout: u32,
    #[comptime] h: u32,
    #[comptime] w: u32,
    #[comptime] apply_act: u32,
) {
    let idx = ABSOLUTE_POS;
    let cin = cin as usize;
    let cout = cout as usize;
    let h = h as usize;
    let w = w as usize;
    let plane = h * w;
    if idx >= cout * plane {
        terminate!();
    }
    let oc = idx / plane;
    let rem = idx % plane;
    let y = rem / w;
    let x = rem % w;
    let mut acc = bias[oc];
    for ic in 0..cin {
        let kbase = (oc * cin + ic) * 9;
        let ibase = ic * plane;
        for ky in 0..3usize {
            let yy = y + ky;
            if yy >= 1 && yy <= h {
                let row = ibase + (yy - 1) * w;
                for kx in 0..3usize {
                    let xx = x + kx;
                    if xx >= 1 && xx <= w {
                        acc += wts[kbase + ky * 3 + kx] * inp[row + (xx - 1)];
                    }
                }
            }
        }
    }
    if apply_act == 1 {
        let s = slopes[oc];
        if acc < 0.0 {
            acc = s * acc;
        }
    }
    out[idx] = acc;
}

fn lcg(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

fn cpu_ref(
    inp: &[f32],
    wts: &[f32],
    bias: &[f32],
    slopes: &[f32],
    cin: usize,
    cout: usize,
    h: usize,
    w: usize,
    act: bool,
) -> Vec<f32> {
    let plane = h * w;
    let mut out = vec![0.0f32; cout * plane];
    for oc in 0..cout {
        for y in 0..h {
            for x in 0..w {
                let mut acc = bias[oc];
                for ic in 0..cin {
                    for ky in 0..3isize {
                        let yy = y as isize + ky - 1;
                        if yy < 0 || yy >= h as isize {
                            continue;
                        }
                        for kx in 0..3isize {
                            let xx = x as isize + kx - 1;
                            if xx < 0 || xx >= w as isize {
                                continue;
                            }
                            acc += wts[(oc * cin + ic) * 9 + (ky * 3 + kx) as usize]
                                * inp[ic * plane + yy as usize * w + xx as usize];
                        }
                    }
                }
                if act && acc < 0.0 {
                    acc *= slopes[oc];
                }
                out[oc * plane + y * w + x] = acc;
            }
        }
    }
    out
}

fn run<R: Runtime>(name: &str) {
    let client = R::client(&Default::default());
    // correctness gate vs CPU reference (small case)
    {
        let (cin, cout, h, w) = (8usize, 8usize, 13usize, 11usize);
        let plane = h * w;
        let inp = lcg(cin * plane, 7);
        let wts = lcg(cout * cin * 9, 11);
        let bias = lcg(cout, 13);
        let slopes = lcg(cout, 17);
        let hi = client.create_from_slice(f32::as_bytes(&inp));
        let hw = client.create_from_slice(f32::as_bytes(&wts));
        let hb = client.create_from_slice(f32::as_bytes(&bias));
        let hs = client.create_from_slice(f32::as_bytes(&slopes));
        let ho = client.create_from_slice(f32::as_bytes(&vec![0.0f32; cout * plane]));
        let n = cout * plane;
        unsafe {
            conv3x3_prelu::launch_unchecked::<R>(
                &client,
                {
                    let g = (n as u32).div_ceil(256);
                    let gx = g.min(65535);
                    CubeCount::Static(gx, g.div_ceil(gx), 1)
                },
                CubeDim::new_1d(256),
                ArrayArg::from_raw_parts(hi.clone(), cin * plane),
                ArrayArg::from_raw_parts(hw.clone(), wts.len()),
                ArrayArg::from_raw_parts(hb.clone(), cout),
                ArrayArg::from_raw_parts(hs.clone(), cout),
                ArrayArg::from_raw_parts(ho.clone(), n),
                cin as u32,
                cout as u32,
                h as u32,
                w as u32,
                1u32,
            );
        }
        let got = f32::from_bytes(&client.read_one(ho.clone()).unwrap()).to_vec();
        let want = cpu_ref(&inp, &wts, &bias, &slopes, cin, cout, h, w, true);
        let mx = got
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(mx < 1e-4, "{name}: GPU/CPU mismatch {mx}");
        println!("{name}: correctness OK (max diff {mx:.2e})");
    }
    // topology timings
    for (label, nf, nc) in [("quality nf64 nc16", 64usize, 16usize), ("rt nf24 nc8", 24, 8)] {
        for side in [512usize, 1024, 2048] {
            let (h, w) = (side, side);
            let plane = h * w;
            let mp = plane as f64 / 1e6;
            // weights per layer (first 3->nf, mids nf->nf, last nf->3)
            let w0 = lcg(nf * 3 * 9, 3);
            let wm = lcg(nf * nf * 9, 5);
            let wl = lcg(3 * nf * 9, 7);
            let bias_nf = lcg(nf, 9);
            let bias3 = lcg(3, 11);
            let slp = lcg(nf, 15);
            let inp = lcg(3 * plane, 21);
            let t0 = Instant::now();
            let h_in = client.create_from_slice(f32::as_bytes(&inp));
            let h_w0 = client.create_from_slice(f32::as_bytes(&w0));
            let h_wm = client.create_from_slice(f32::as_bytes(&wm));
            let h_wl = client.create_from_slice(f32::as_bytes(&wl));
            let h_bnf = client.create_from_slice(f32::as_bytes(&bias_nf));
            let h_b3 = client.create_from_slice(f32::as_bytes(&bias3));
            let h_slp = client.create_from_slice(f32::as_bytes(&slp));
            let h_a = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nf * plane]));
            let h_b = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nf * plane]));
            let h_out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; 3 * plane]));
            client.sync();
            let t_up = t0.elapsed().as_secs_f64() * 1e3;
            let launch = |hin: &cubecl::server::Handle,
                          hw_: &cubecl::server::Handle,
                          hb_: &cubecl::server::Handle,
                          hout: &cubecl::server::Handle,
                          cin: usize,
                          cout: usize,
                          act: u32| {
                let n = cout * plane;
                unsafe {
                    conv3x3_prelu::launch_unchecked::<R>(
                        &client,
                        {
                    let g = (n as u32).div_ceil(256);
                    let gx = g.min(65535);
                    CubeCount::Static(gx, g.div_ceil(gx), 1)
                },
                        CubeDim::new_1d(256),
                        ArrayArg::from_raw_parts(hin.clone(), cin * plane),
                        ArrayArg::from_raw_parts(hw_.clone(), cout * cin * 9),
                        ArrayArg::from_raw_parts(hb_.clone(), cout),
                        ArrayArg::from_raw_parts(h_slp.clone(), nf),
                        ArrayArg::from_raw_parts(hout.clone(), n),
                        cin as u32,
                        cout as u32,
                        h as u32,
                        w as u32,
                        act,
                    );
                }
            };
            let t1 = Instant::now();
            launch(&h_in, &h_w0, &h_bnf, &h_a, 3, nf, 1);
            for i in 0..nc {
                let (src, dst) = if i % 2 == 0 { (&h_a, &h_b) } else { (&h_b, &h_a) };
                launch(src, &h_wm, &h_bnf, dst, nf, nf, 1);
            }
            let last_in = if nc % 2 == 0 { &h_a } else { &h_b };
            launch(last_in, &h_wl, &h_b3, &h_out, nf, 3, 0);
            client.sync();
            let t_k = t1.elapsed().as_secs_f64() * 1e3;
            let t2 = Instant::now();
            let _res = client.read_one(h_out.clone()).unwrap();
            let t_dn = t2.elapsed().as_secs_f64() * 1e3;
            let total = t_up + t_k + t_dn;
            println!(
                "{name} {label} {side}x{side} ({mp:.2}MP): up {t_up:.1}ms kernels {t_k:.1}ms read {t_dn:.1}ms total {total:.1}ms = {:.2} MP/s ({:.3} s/MP)",
                mp / (total / 1e3), total / 1e3 / mp
            );
        }
    }
}

fn main() {
    #[cfg(feature = "cuda")]
    run::<cubecl::cuda::CudaRuntime>("cuda");
    #[cfg(feature = "wgpu")]
    run::<cubecl::wgpu::WgpuRuntime>("wgpu");
    #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
    eprintln!("build with --features cuda or --features wgpu");
}
