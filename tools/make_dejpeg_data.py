#!/usr/bin/env python3
"""GT pairs for native x1 JPEG-artifact inversion (S6).

HR = 128px crops of imazen-26 TRAIN files (pinned eval exclusion — same
listing as make_distill_data); LR = cv2/turbo JPEG q35-95 of the SAME crop
(no scaling). Image-level val split. Out: ZENSR_DATA (default
~/tmp/zensr-dejpeg)/{lr_u8.npy, hr_u8.npy, meta.json}; val = last 512.

Usage: make_dejpeg_data.py [n_pairs=24000]
"""
import json
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_distill_data import SUBS, list_train_files  # noqa: E402  (pinned exclusion)

OUT = os.path.expanduser(os.environ.get("ZENSR_DATA", "~/tmp/zensr-dejpeg"))
CROP = 128
SEED = 20260726
VAL_TAIL = 512


def main():
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 24000
    os.makedirs(OUT, exist_ok=True)
    pool = []
    for s in SUBS:
        fs = list_train_files(s)
        pool += fs
        print(f"{s}: {len(fs)} train files", flush=True)
    rng = random.Random(SEED)
    rng.shuffle(pool)
    n_val = max(16, len(pool) // 20)
    val_pool, train_pool = pool[-n_val:], pool[:-n_val]
    print(f"train files {len(train_pool)} / val files {n_val}", flush=True)

    def fill(files, want):
        lr_l, hr_l = [], []
        fi = 0
        while len(hr_l) < want:
            f = files[fi % len(files)]
            fi += 1
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            if img is None or min(img.shape[:2]) < CROP:
                continue
            for _ in range(6):
                if len(hr_l) >= want:
                    break
                y = rng.randrange(0, img.shape[0] - CROP + 1)
                x = rng.randrange(0, img.shape[1] - CROP + 1)
                hr = img[y:y + CROP, x:x + CROP]
                if cv2.cvtColor(hr, cv2.COLOR_BGR2GRAY).std() < 8.0:
                    continue
                q = rng.randrange(35, 96)
                ok, enc = cv2.imencode(".jpg", hr, [cv2.IMWRITE_JPEG_QUALITY, q])
                if not ok:
                    continue
                lr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                hr_l.append(cv2.cvtColor(hr, cv2.COLOR_BGR2RGB))
                lr_l.append(cv2.cvtColor(lr, cv2.COLOR_BGR2RGB))
            if len(hr_l) % 2048 < 6:
                print(f"{len(hr_l)}/{want}", flush=True)
        return lr_l, hr_l

    lr_t, hr_t = fill(train_pool, n_pairs - VAL_TAIL)
    lr_v, hr_v = fill(val_pool, VAL_TAIL)
    lr = np.stack(lr_t + lr_v)
    hr = np.stack(hr_t + hr_v)
    np.save(os.path.join(OUT, "lr_u8.npy"), lr)
    np.save(os.path.join(OUT, "hr_u8.npy"), hr)
    json.dump({"n": int(lr.shape[0]), "val_tail": VAL_TAIL, "seed": SEED, "crop": CROP,
               "task": "x1 dejpeg inversion", "degrade": "cv2 jpeg q35-95, same-size",
               "source": "imazen-26 train files (pinned eval exclusion)",
               "val_split": "image-level"},
              open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    print(f"DONE {lr.shape[0]} pairs ({(lr.nbytes+hr.nbytes)/1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
