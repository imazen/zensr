#!/usr/bin/env python3
"""Take the first N training pairs of a dataset, keeping its val tail intact.

Why this exists: comparing "24k crops" against "100k crops" using two
separately-built datasets is not a test of dataset SIZE — the 2026-08-02
attempt found the two sets differed in target representation (u8 vs f16
teacher targets) on top of size, which is a second variable. Slicing one
dataset gives arms that share generation process, teacher, target dtype, crop
size and val split exactly, so size is the only thing that moves.

The val tail (last 512 rows) is copied verbatim, so both arms are scored
against an identical validation set.

pairs.tsv is rewritten with renumbered indices, because the trainer's qboost
reads column 0 as a row index into the arrays.

Usage: make_data_subset.py <src-dir> <dst-dir> <n-train> [val-tail=512]
"""
import json
import os
import shutil
import sys

import numpy as np


def main():
    src, dst = sys.argv[1], sys.argv[2]
    n_train = int(sys.argv[3])
    val_tail = int(sys.argv[4]) if len(sys.argv) > 4 else 512
    os.makedirs(dst, exist_ok=True)

    names = [f for f in ("lr_u8.npy", "hr_f16.npy", "hr_u8.npy",
                         "cond_scalar_f32.npy", "dmap_u16.npy")
             if os.path.exists(os.path.join(src, f))]
    if not any(n.startswith("lr_") for n in names):
        sys.exit(f"{src}: no lr_*.npy — wrong directory?")

    total = None
    for n in names:
        a = np.load(os.path.join(src, n), mmap_mode="r")
        if total is None:
            total = a.shape[0]
            if n_train + val_tail > total:
                sys.exit(f"asked for {n_train}+{val_tail} but {n} has {total} rows")
        elif a.shape[0] != total:
            sys.exit(f"{n} has {a.shape[0]} rows, expected {total} — arrays are misaligned")
        out = np.concatenate([np.asarray(a[:n_train]), np.asarray(a[total - val_tail:])])
        np.save(os.path.join(dst, n), out)
        print(f"{n}: {total} -> {out.shape[0]} rows ({out.nbytes / 1e9:.2f} GB)", flush=True)

    # qboost indexes the arrays by pairs.tsv column 0, so renumber.
    ptsv = os.path.join(src, "pairs.tsv")
    if os.path.exists(ptsv):
        with open(ptsv) as f:
            head, *rows = f.read().splitlines()
        keep = rows[:n_train] + rows[total - val_tail:]
        with open(os.path.join(dst, "pairs.tsv"), "w") as f:
            f.write(head + "\n")
            for i, r in enumerate(keep):
                c = r.split("\t")
                c[0] = str(i)
                f.write("\t".join(c) + "\n")
        print(f"pairs.tsv: {len(rows)} -> {len(keep)} rows, indices renumbered")

    meta = {}
    mp = os.path.join(src, "meta.json")
    if os.path.exists(mp):
        try:
            meta = json.load(open(mp))
        except (ValueError, OSError):
            meta = {}
    meta.update({"n": n_train + val_tail, "val_tail": val_tail,
                 "subset_of": os.path.abspath(src), "subset_n_train": n_train,
                 "note": "first n_train rows + the ORIGINAL val tail, so a "
                         "size A/B shares generation, teacher, target dtype "
                         "and validation set exactly"})
    json.dump(meta, open(os.path.join(dst, "meta.json"), "w"), indent=1)
    for extra in ("PRISTINE_MANIFEST.tsv",):
        if os.path.exists(os.path.join(src, extra)):
            shutil.copy(os.path.join(src, extra), dst)
    print("DONE", dst)


if __name__ == "__main__":
    main()
