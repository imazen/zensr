#!/usr/bin/env python3
"""Build a distillation dataset for the x1 realtime tier: replace hr targets
with the QUALITY-tier teacher's outputs (S9 recipe — E_rtc showed training on
teacher outputs dominates GT for small students).

Reads  ZENSR_SRC  (dir with lr_u8.npy/hr_u8.npy/pairs.tsv — dejpeg-v4 layout)
       ZENSR_TEACHER_CKPT (torch .pth for the nf64/nc16 x1 teacher)
Writes ZENSR_DATA (lr_u8.npy hardlink/copy, hr_u8.npy = teacher(lr) as u8,
       pairs.tsv copy). u8 target quantization is a known rung-1 shortcut
       (f16 targets are the follow-up if the rung looks promising).

Usage: ZENSR_SRC=... ZENSR_TEACHER_CKPT=... ZENSR_DATA=... make_teacher_targets.py
"""
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ZENSR_SCALE", "1")
os.environ.setdefault("ZENSR_NF", "64")
os.environ.setdefault("ZENSR_NC", "16")
import train_people as tp  # noqa: E402  (Student arch + norm conventions)

SRC = os.path.expanduser(os.environ["ZENSR_SRC"])
CKPT = os.path.expanduser(os.environ["ZENSR_TEACHER_CKPT"])
OUT = os.path.expanduser(os.environ["ZENSR_DATA"])


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    m = tp.Student().to(dev).eval()
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    for k in ("sd", "model", "params", "params_ema", "state_dict"):
        if k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=True)
    lr_all = np.load(os.path.join(SRC, "lr_u8.npy"), mmap_mode="r")
    n = lr_all.shape[0]
    out = np.zeros_like(lr_all)
    bs = 256
    with torch.no_grad():
        for i in range(0, n, bs):
            x = torch.from_numpy(lr_all[i:i + bs].copy()).to(dev).permute(0, 3, 1, 2).float().div_(255)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev != "cpu"):
                y = m(x)
            y8 = (y.float().clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            out[i:i + bs] = y8
            if (i // bs) % 20 == 0:
                print(f"{i}/{n}", flush=True)
    np.save(os.path.join(OUT, "hr_u8.npy"), out)
    # lr + pairs pass through unchanged
    if not os.path.exists(os.path.join(OUT, "lr_u8.npy")):
        shutil.copy(os.path.join(SRC, "lr_u8.npy"), os.path.join(OUT, "lr_u8.npy"))
    shutil.copy(os.path.join(SRC, "pairs.tsv"), os.path.join(OUT, "pairs.tsv"))
    print(f"DONE teacher targets {n} pairs -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
