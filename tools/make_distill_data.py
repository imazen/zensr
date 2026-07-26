#!/usr/bin/env python3
"""Generate distillation pairs for the realtime-2x student (S-E pilot).

Input crops: HR from imazen-26 (SKIPPING each dir's first 8 sorted files —
those are the frozen eval split), downscaled 2x (area) then JPEG-degraded via
cv2 (libjpeg-turbo lineage) at q in [40,90], 4:2:0.
Target: 2xNomosUni_span_multijpg (teacher) output on the degraded LR, computed
on GPU with the same functional forward as dump_adopted.py (merged Conv3XC).

Output shards: ~/tmp/zensr-distill/{lr_u8.npy, teacher_f16.npy, meta.json}
(lr 96x96 u8 HWC, teacher 192x192 f16 CHW). Val split = last 512 pairs.
"""
import json
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_adopted import compact_forward, load_sd, prepare_span_sd, span_forward  # noqa: E402

ROOT = os.environ.get("ZENSR_ROOT", "/mnt/v/imazen-26")
SUBS = ["lilith", "unsplash-people", "screen", "internet-archive-scans",
        "national-park-service", "unsplash-renders", "unsplash-textures", "office-documents"]
# ZENSR_SUBS=screen,office-documents,... restricts sources (class-specialist
# datasets, e.g. the graphics model). Exclusion discipline is unchanged.
if os.environ.get("ZENSR_SUBS"):
    SUBS = [s.strip() for s in os.environ["ZENSR_SUBS"].split(",") if s.strip()]
OUT = os.path.expanduser(os.environ.get("ZENSR_DATA", "~/tmp/zensr-distill"))
N_PAIRS = 14000
CROP = 192  # HR crop; LR = 96
SEED = 20260723

EVAL_PIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "eval_split", "imazen26_eval_files.tsv")


def eval_pinned(sub):
    out = set()
    if os.path.exists(EVAL_PIN):
        for line in open(EVAL_PIN):
            c = line.rstrip("\n").split("\t")
            if len(c) == 2 and c[0] == sub and not c[0].startswith("#"):
                out.add(c[1])
    return out


def list_train_files(sub):
    d = os.path.join(ROOT, sub)
    files = []
    for base, _, names in os.walk(d):
        for n in sorted(names):
            if n.lower().endswith((".png", ".jpg", ".jpeg")):
                files.append(os.path.join(base, n))
    files.sort()
    # Frozen eval = first-8-sorted UNION the pinned actually-evaluated list.
    # Runtime "first 8 usable" slides past decode-skipped files (teresa leak,
    # 2026-07-24 postmortem) — the pinned file is authoritative.
    pinned = eval_pinned(sub)
    return [f for f in files[8:] if os.path.basename(f) not in pinned]


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Teacher selection (ZENSR_TEACHER): "span" = 2xNomosUni_span (SSIM2 king at
    # q<=50), "compact" = 2xNomosUni_compact (q75 + butteraugli + worse-rate king).
    # prepare_span_sd merges Conv3XC branches; span_forward normalizes input
    # itself ((x-mean)*255, official). The first 14k-pair run predated the norm
    # + inplace-SiLU concat fixes -> constant-gray teacher, fully discarded.
    W = "/mnt/tower/output/zensr-training/adopted-weights"
    teacher = os.environ.get("ZENSR_TEACHER", "span")
    if teacher == "compact":
        sd = load_sd(os.path.join(W, "2xNomosUni_compact_multijpg_ldl.pth"))
        fwd = lambda t: compact_forward(sd, t, 2)
    else:
        sd, _ = prepare_span_sd(os.path.join(W, "2xNomosUni_span_multijpg.pth"))
        fwd = lambda t: span_forward(sd, t, 2)
    sd = {k: v.to(dev) for k, v in sd.items()}

    pool = []
    for s in SUBS:
        fs = list_train_files(s)
        pool += fs
        print(f"{s}: {len(fs)} train files")
    rng.shuffle(pool)
    # image-level val: last 512 pairs come ONLY from val-reserved files
    n_val_files = max(16, len(pool) // 20)
    val_pool, train_pool = pool[-n_val_files:], pool[:-n_val_files]
    TRAIN_TARGET = N_PAIRS - 512
    print(f"train files {len(train_pool)} / val files {n_val_files}", flush=True)

    lr_all = np.zeros((N_PAIRS, 96, 96, 3), dtype=np.uint8)
    tg_all = np.zeros((N_PAIRS, 3, 192, 192), dtype=np.float16)
    made = 0
    fi = 0
    batch_lr = []
    while made < N_PAIRS:
        phase_target = TRAIN_TARGET if made < TRAIN_TARGET else N_PAIRS
        src = train_pool if made < TRAIN_TARGET else val_pool
        f = src[fi % len(src)]
        fi += 1
        img = cv2.imread(f, cv2.IMREAD_COLOR)  # BGR
        if img is None or img.shape[0] < CROP or img.shape[1] < CROP:
            continue
        for _ in range(min(4, 1 + img.shape[0] * img.shape[1] // (CROP * CROP * 4))):
            if made + len(batch_lr) >= phase_target:
                break
            y = rng.randrange(0, img.shape[0] - CROP + 1)
            x = rng.randrange(0, img.shape[1] - CROP + 1)
            hr = img[y:y + CROP, x:x + CROP]
            lr = cv2.resize(hr, (96, 96), interpolation=cv2.INTER_AREA)
            q = rng.randrange(40, 91)
            ok, enc = cv2.imencode(".jpg", lr, [cv2.IMWRITE_JPEG_QUALITY, q])
            if not ok:
                continue
            lr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            batch_lr.append(lr)
        if len(batch_lr) >= 32 or (made + len(batch_lr) >= phase_target and batch_lr):
            arr = np.stack(batch_lr)  # B,96,96,3 BGR u8
            rgb = arr[..., ::-1].astype(np.float32) / 255.0
            t = torch.from_numpy(rgb.transpose(0, 3, 1, 2).copy()).to(dev)
            with torch.no_grad():
                out = fwd(t).clamp(0, 1)
            n = len(batch_lr)
            lr_all[made:made + n] = arr[..., ::-1]  # store RGB u8
            tg_all[made:made + n] = out.cpu().numpy().astype(np.float16)
            made += n
            batch_lr = []
            if made % 1024 < 32:
                print(f"{made}/{N_PAIRS}")
    np.save(os.path.join(OUT, "lr_u8.npy"), lr_all)
    np.save(os.path.join(OUT, "teacher_f16.npy"), tg_all)
    json.dump({"n": N_PAIRS, "val_tail": 512, "teacher": "2xNomosUni_span_multijpg",
               "degrade": "area-down2x + cv2 jpeg q40-90", "seed": SEED,
               "eval_split_excluded": "first-8-sorted UNION pinned eval_split/imazen26_eval_files.tsv", "val_split": "image-level (last 5% of shuffled files)"},
              open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    print("DONE", lr_all.nbytes / 1e9, "GB +", tg_all.nbytes / 1e9, "GB")


if __name__ == "__main__":
    main()
