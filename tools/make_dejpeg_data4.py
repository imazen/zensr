#!/usr/bin/env python3
"""Dejpeg-v4: policy-mix decode + conditioning channels (S2a/S7/S10 ablation).

Per pair (crop 128, imazen-26 pinned-exclusion, image-level val):
  encoder x ss x qU(5,96) + 5% clean -> jpg
  lr      = zenjpeg decode, policy mix (auto iff enc in {turbo,mozjpeg} & q<=9)
  scalar  = probe-derived severity in [0,1] (Ijg/Moz: (100-q)/100; jpegli d/15)
  dmap    = zjtool dmap (16x16 u16, per-block erased-band uncertainty)

Out: ZENSR_DATA(~/tmp/zensr-dejpeg-v4)/{lr_u8,hr_u8,cond_scalar_f32,dmap_u16}.npy
     + pairs.tsv + meta.json.   Usage: make_dejpeg_data4.py [n=24000] [workers=8]
"""
import json
import os
import random
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_distill_data import SUBS, list_train_files  # noqa: E402
from make_dejpeg_data2 import (ENCODERS, encode, read_ppm, write_ppm, run,  # noqa: E402
                               gen_crops, ZJTOOL, CROP, VAL_TAIL)

OUT = os.path.expanduser(os.environ.get("ZENSR_DATA", "~/tmp/zensr-dejpeg-v4"))
SEED = 20260728


def probe_severity(jpg):
    r = subprocess.run([ZJTOOL, "probe", jpg], capture_output=True, text=True)
    c = r.stdout.strip().split("\t")
    if len(c) < 4 or c[0] == "ERR":
        return 0.5
    val, scale = float(c[1]), c[2]
    if scale in ("IjgQuality", "MozjpegQuality"):
        return max(0.0, min(1.0, (100.0 - val) / 100.0))
    return max(0.0, min(1.0, val / 15.0))  # ButteraugliDistance


def read_dmap(path):
    with open(path, "rb") as f:
        assert f.read(2) == b"P5"
        vals = []
        while len(vals) < 3:
            tok = b""
            c = f.read(1)
            while c.isspace():
                c = f.read(1)
            while c and not c.isspace():
                tok += c
                c = f.read(1)
            vals.append(int(tok))
        w, h, _ = vals
        return np.frombuffer(f.read(2 * w * h), dtype=">u2").astype(np.uint16).reshape(h, w)


def make_pair(args):
    # gens: list of (enc, ss, q) — len 1 = single generation (the default);
    # len 2/3 = generation-loss chains (decode of gen k re-encoded as k+1).
    # The FINAL generation's file drives the deblock policy / dmap / probe —
    # exactly what the runtime pipeline will see. GT stays the pristine crop.
    idx, crop_rgb, gens, clean = args
    enc, ss, q = gens[-1]
    nb = CROP // 8
    if clean:
        return idx, crop_rgb, crop_rgb, 0.0, np.zeros((nb, nb), np.uint16), enc, ss, q, clean, len(gens)
    with tempfile.TemporaryDirectory(dir=os.path.expanduser("~/tmp")) as td:
        ppm, jpg = os.path.join(td, "c.ppm"), os.path.join(td, "c.jpg")
        write_ppm(ppm, cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
        for gi, (genc, gss, gq) in enumerate(gens):
            encode(ppm, jpg, genc, gq, gss)
            if gi + 1 < len(gens):
                # decode pixel-exact and feed the next generation
                mid = os.path.join(td, f"m{gi}.ppm")
                run([ZJTOOL, "dec", jpg, mid, "off"])
                ppm = mid
        mode = "auto" if (enc in ("turbo", "mozjpeg") and q <= 9) else "off"
        dec = os.path.join(td, "d.ppm")
        run([ZJTOOL, "dec", jpg, dec, mode])
        dm = os.path.join(td, "d.pgm")
        run([ZJTOOL, "dmap", jpg, dm])
        sev = probe_severity(jpg)
        return idx, crop_rgb, read_ppm(dec).copy(), sev, read_dmap(dm).copy(), enc, ss, q, clean, len(gens)


def main():
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 24000
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    os.makedirs(OUT, exist_ok=True)
    pool_files = []
    for s in SUBS:
        pool_files += list_train_files(s)
    rng = random.Random(SEED)
    rng.shuffle(pool_files)
    n_val = max(16, len(pool_files) // 20)
    val_f, train_f = pool_files[-n_val:], pool_files[:-n_val]
    print(f"train files {len(train_f)} / val files {n_val}", flush=True)
    crops = gen_crops(train_f, n_pairs - VAL_TAIL, rng) + gen_crops(val_f, VAL_TAIL, rng)
    print(f"{len(crops)} crops ready", flush=True)
    # Generation-loss augmentation (SYSTEMS.md "generation loss"): fractions
    # of pairs re-encoded 2x/3x with independent random enc/ss/q per
    # generation. Default OFF — enable per measured gen_eval verdict.
    p_gen2 = float(os.environ.get("ZENSR_GEN2", "0"))
    p_gen3 = float(os.environ.get("ZENSR_GEN3", "0"))
    jobs = []
    for i, c in enumerate(crops):
        clean = rng.random() < 0.05
        def step():
            return (rng.choice(ENCODERS), "420" if rng.random() < 0.6 else "444",
                    rng.randrange(5, 97))
        r = rng.random()
        ngens = 3 if r < p_gen3 else 2 if r < p_gen3 + p_gen2 else 1
        jobs.append((i, c, [step() for _ in range(ngens)], clean))
    n = len(jobs)
    nb = CROP // 8
    hr = np.zeros((n, CROP, CROP, 3), np.uint8)
    lr = np.zeros((n, CROP, CROP, 3), np.uint8)
    sc = np.zeros((n,), np.float32)
    dm = np.zeros((n, nb, nb), np.uint16)
    meta_rows = [None] * n
    done = 0
    with Pool(workers) as p:
        for idx, h, l, s_, d_, enc, ss, q, clean, gens in p.imap_unordered(
            make_pair, jobs, chunksize=16
        ):
            hr[idx], lr[idx], sc[idx], dm[idx] = h, l, s_, d_
            meta_rows[idx] = f"{idx}\t{enc}\t{ss}\t{q}\t{int(clean)}\t{s_:.3f}\t{gens}"
            done += 1
            if done % 2000 == 0:
                print(f"{done}/{n}", flush=True)
    np.save(os.path.join(OUT, "lr_u8.npy"), lr)
    np.save(os.path.join(OUT, "hr_u8.npy"), hr)
    np.save(os.path.join(OUT, "cond_scalar_f32.npy"), sc)
    np.save(os.path.join(OUT, "dmap_u16.npy"), dm)
    json.dump({"n": n, "val_tail": VAL_TAIL, "seed": SEED, "crop": CROP,
               "task": "v4 conditioning ablation (policy-mix decode)",
               "q": "U(5,96)+5% clean", "encoders": ENCODERS,
               "subs": SUBS, "gen2": p_gen2, "gen3": p_gen3,
               "cond": "scalar=probe severity; dmap=per-block erased-band Q^2/12 log-scaled",
               "val_split": "image-level"},
              open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    with open(os.path.join(OUT, "pairs.tsv"), "w") as f:
        f.write("idx\tencoder\tss\tq\tclean\tseverity\tgens\n")
        f.write("\n".join(meta_rows) + "\n")
    print(f"DONE {n} pairs ({(hr.nbytes + lr.nbytes + dm.nbytes)/1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
