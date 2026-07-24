#!/usr/bin/env python3
"""Dejpeg-v2 GT pairs: 4 real encoders x {420,444} x q~U(10,96) + 5% clean,
ALL decoded by zenjpeg (deployment decoder) in BOTH deblock arms.

Per pair: crop 128 from imazen-26 train files (pinned eval exclusion,
image-level val) -> encoder in {turbo, mozjpeg, jpegli, zenjpeg} ->
zjtool dec off + auto. Outputs:
  ~/tmp/zensr-dejpeg-v2/off/{lr_u8.npy, hr_u8.npy, meta.json}
  ~/tmp/zensr-dejpeg-v2/auto/{lr_u8.npy, hr_u8.npy, meta.json}
  ~/tmp/zensr-dejpeg-v2/pairs.tsv   (idx, encoder, ss, q, clean)

Usage: make_dejpeg_data2.py [n_pairs=24000] [workers=8]
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

OUT = os.path.expanduser(os.environ.get("ZENSR_DATA", "~/tmp/zensr-dejpeg-v2"))
ZJTOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "target", "release-fast", "zjtool")
MOZ = os.path.expanduser("~/tmp/ati-bin/mozjpeg-cjpeg")
MOZ_LIB = os.path.expanduser("~/tmp/ati-bin/mozjpeg-lib64")
CROP = 128
SEED = 20260727
VAL_TAIL = 512
ENCODERS = ["turbo", "mozjpeg", "jpegli", "zenjpeg"]


def write_ppm(path, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with open(path, "wb") as f:
        f.write(f"P6\n{bgr.shape[1]} {bgr.shape[0]}\n255\n".encode())
        f.write(rgb.tobytes())


def read_ppm(path):
    with open(path, "rb") as f:
        assert f.read(2) == b"P6"
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
        return np.frombuffer(f.read(3 * w * h), dtype=np.uint8).reshape(h, w, 3)


def run(cmd, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    subprocess.run(cmd, check=True, capture_output=True, env=e)


def encode(ppm, jpg, enc, q, ss):
    if enc == "turbo":
        samp = "2x2" if ss == "420" else "1x1"
        run(["cjpeg", "-quality", str(q), "-sample", samp, "-optimize", "-outfile", jpg, ppm])
    elif enc == "mozjpeg":
        samp = "2x2" if ss == "420" else "1x1"
        run([MOZ, "-quality", str(q), "-sample", samp, "-outfile", jpg, ppm],
            env={"LD_LIBRARY_PATH": MOZ_LIB})
    elif enc == "jpegli":
        run(["cjpegli", ppm, jpg, "-q", str(q), f"--chroma_subsampling={ss}"])
    else:  # zenjpeg
        run([ZJTOOL, "enc", ppm, jpg, str(q), ss])


def make_pair(args):
    idx, crop_rgb, enc, ss, q, clean = args
    if clean:
        return idx, crop_rgb, crop_rgb, crop_rgb, enc, ss, q, clean
    with tempfile.TemporaryDirectory(dir=os.path.expanduser("~/tmp")) as td:
        ppm = os.path.join(td, "c.ppm")
        jpg = os.path.join(td, "c.jpg")
        write_ppm(ppm, cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
        encode(ppm, jpg, enc, q, ss)
        off = os.path.join(td, "off.ppm")
        auto = os.path.join(td, "auto.ppm")
        run([ZJTOOL, "dec", jpg, off, "off"])
        run([ZJTOOL, "dec", jpg, auto, "auto"])
        return idx, crop_rgb, read_ppm(off).copy(), read_ppm(auto).copy(), enc, ss, q, clean


def gen_crops(files, want, rng):
    out = []
    fi = 0
    while len(out) < want:
        f = files[fi % len(files)]
        fi += 1
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None or min(img.shape[:2]) < CROP:
            continue
        for _ in range(6):
            if len(out) >= want:
                break
            y = rng.randrange(0, img.shape[0] - CROP + 1)
            x = rng.randrange(0, img.shape[1] - CROP + 1)
            hr = img[y:y + CROP, x:x + CROP]
            if cv2.cvtColor(hr, cv2.COLOR_BGR2GRAY).std() < 8.0:
                continue
            out.append(cv2.cvtColor(hr, cv2.COLOR_BGR2RGB))
    return out


def main():
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 24000
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    for arm in ("off", "auto"):
        os.makedirs(os.path.join(OUT, arm), exist_ok=True)
    pool_files = []
    for s in SUBS:
        fs = list_train_files(s)
        pool_files += fs
    rng = random.Random(SEED)
    rng.shuffle(pool_files)
    n_val = max(16, len(pool_files) // 20)
    val_f, train_f = pool_files[-n_val:], pool_files[:-n_val]
    print(f"train files {len(train_f)} / val files {n_val}", flush=True)

    crops = gen_crops(train_f, n_pairs - VAL_TAIL, rng) + gen_crops(val_f, VAL_TAIL, rng)
    print(f"{len(crops)} crops ready", flush=True)
    jobs = []
    for i, c in enumerate(crops):
        clean = rng.random() < 0.05
        enc = rng.choice(ENCODERS)
        ss = "420" if rng.random() < 0.6 else "444"
        q = rng.randrange(10, 97)
        jobs.append((i, c, enc, ss, q, clean))

    n = len(jobs)
    hr = np.zeros((n, CROP, CROP, 3), np.uint8)
    lr_off = np.zeros((n, CROP, CROP, 3), np.uint8)
    lr_auto = np.zeros((n, CROP, CROP, 3), np.uint8)
    meta_rows = [None] * n
    done = 0
    with Pool(workers) as p:
        for idx, h, lo, la, enc, ss, q, clean in p.imap_unordered(make_pair, jobs, chunksize=16):
            hr[idx] = h
            lr_off[idx] = lo
            lr_auto[idx] = la
            meta_rows[idx] = f"{idx}\t{enc}\t{ss}\t{q}\t{int(clean)}"
            done += 1
            if done % 2000 == 0:
                print(f"{done}/{n}", flush=True)

    for arm, lr in (("off", lr_off), ("auto", lr_auto)):
        d = os.path.join(OUT, arm)
        np.save(os.path.join(d, "lr_u8.npy"), lr)
        np.save(os.path.join(d, "hr_u8.npy"), hr)
        json.dump({"n": n, "val_tail": VAL_TAIL, "seed": SEED, "crop": CROP,
                   "arm": arm, "task": "x1 dejpeg v2 (4 encoders, zenjpeg decode)",
                   "encoders": ENCODERS, "ss": "420 60% / 444 40%", "q": "U(10,96) + 5% clean",
                   "source": "imazen-26 train files (pinned exclusion)",
                   "val_split": "image-level"},
                  open(os.path.join(d, "meta.json"), "w"), indent=1)
    with open(os.path.join(OUT, "pairs.tsv"), "w") as f:
        f.write("idx\tencoder\tss\tq\tclean\n")
        f.write("\n".join(meta_rows) + "\n")
    print(f"DONE {n} pairs x 2 arms ({(hr.nbytes*3)/1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
