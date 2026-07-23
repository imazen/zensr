#!/usr/bin/env python3
"""Ground-truth training pairs from the zensr-people-v1 pool (P2-mini).

HR = random 192px crops of pool photos (eval_ids EXCLUDED, luma-std filtered);
LR = INTER_AREA down 2x + JPEG q40-95 (cv2/libjpeg-turbo, 4:2:0 default) —
matches the distill-data recipe; the EVAL degradation stays system cjpeg.

Out: ZENSR_DATA (default ~/tmp/zensr-people-gt)/{lr_u8.npy (N,96,96,3),
hr_u8.npy (N,192,192,3), meta.json}. Val = last 512 pairs (by source split:
last 5% of shuffled images feed only the val tail — no crop-level leakage).

Usage: make_people_gt_data.py [n_pairs=24000] [crops_per_img=10]
"""
import json
import os
import random
import sys

import cv2
import numpy as np

POOL = os.path.join(
    os.environ.get("ZENSR_PEOPLE_OUT", "/mnt/tower/input/zensr-people-v1"), "pool"
)
EVAL_IDS = os.path.join(
    os.environ.get("ZENSR_PEOPLE_OUT", "/mnt/tower/input/zensr-people-v1"), "eval_ids.txt"
)
OUT = os.path.expanduser(os.environ.get("ZENSR_DATA", "~/tmp/zensr-people-gt"))
SEED = 20260724
VAL_TAIL = 512


def main():
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 24000
    per_img = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    os.makedirs(OUT, exist_ok=True)
    excl = set()
    if os.path.exists(EVAL_IDS):
        excl = {l.strip() for l in open(EVAL_IDS) if l.strip()}
    files = sorted(
        f for f in os.listdir(POOL)
        if f.endswith(".jpg") and os.path.splitext(f)[0] not in excl
    )
    rng = random.Random(SEED)
    rng.shuffle(files)
    n_val_imgs = max(8, len(files) // 20)
    train_files, val_files = files[:-n_val_imgs], files[-n_val_imgs:]
    print(f"pool {len(files)} imgs (excluded {len(excl)} eval); "
          f"train {len(train_files)} / val-imgs {n_val_imgs}", flush=True)

    def crops_from(flist, want):
        lr_l, hr_l = [], []
        fi = 0
        while len(hr_l) < want and fi < len(flist) * 4:
            fn = flist[fi % len(flist)]
            fi += 1
            img = cv2.imread(os.path.join(POOL, fn), cv2.IMREAD_COLOR)
            if img is None or min(img.shape[:2]) < 400:
                continue
            h, w = img.shape[:2]
            got = 0
            for _ in range(per_img * 3):
                if got >= per_img or len(hr_l) >= want:
                    break
                y = rng.randint(0, h - 192)
                x = rng.randint(0, w - 192)
                hr = img[y:y + 192, x:x + 192]
                luma = cv2.cvtColor(hr, cv2.COLOR_BGR2GRAY)
                if luma.std() < 10.0:  # skip flat sky/wall crops
                    continue
                lr = cv2.resize(hr, (96, 96), interpolation=cv2.INTER_AREA)
                q = rng.randint(40, 95)
                ok, enc = cv2.imencode(".jpg", lr, [cv2.IMWRITE_JPEG_QUALITY, q])
                if not ok:
                    continue
                lr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                hr_l.append(cv2.cvtColor(hr, cv2.COLOR_BGR2RGB))
                lr_l.append(cv2.cvtColor(lr, cv2.COLOR_BGR2RGB))
                got += 1
            if fi % 200 == 0:
                print(f"{len(hr_l)}/{want}", flush=True)
        return lr_l, hr_l

    lr_t, hr_t = crops_from(train_files, n_pairs - VAL_TAIL)
    lr_v, hr_v = crops_from(val_files, VAL_TAIL)
    lr = np.stack(lr_t + lr_v)
    hr = np.stack(hr_t + hr_v)
    np.save(os.path.join(OUT, "lr_u8.npy"), lr)
    np.save(os.path.join(OUT, "hr_u8.npy"), hr)
    json.dump(
        {"n": int(lr.shape[0]), "val_tail": VAL_TAIL, "seed": SEED,
         "source": "zensr-people-v1 pool (pxhere CC0), eval_ids excluded",
         "degrade": "area-down2x + cv2 jpeg q40-95",
         "val_split": "image-level (last 5% of shuffled images)"},
        open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    print(f"DONE {lr.shape[0]} pairs -> {OUT} "
          f"({lr.nbytes / 1e9:.3f} + {hr.nbytes / 1e9:.3f} GB)", flush=True)


if __name__ == "__main__":
    main()
