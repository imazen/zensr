#!/usr/bin/env python3
"""Teacher audition for the people/texture gap (S9 rung 4 gating experiment).

Question: does ANY off-the-shelf heavyweight beat Lanczos (and the incumbent
A2c teacher) on people/textures/art-scans at web-JPEG degradations, at the
x2-target protocol we would actually distill with?

Protocol (matches systems_eval x2 track in spirit; self-contained kernels):
  HR = center-crop 512 of each frozen-eval file (first 8 sorted per subdir)
  LR = INTER_AREA down to 256, degraded clean/q75/q50/q35 via SYSTEM cjpeg
  x4 teachers: out = AREA-down(model(LR) [1024]) -> 512
  x2 teachers: out = model(LR) -> 512
  baseline: cv2 LANCZOS4 up -> 512
Outputs PNGs to ~/tmp/zensr-audition/{gt,<variant>}/... ; score with
`audition_score` (Rust, ssim2+butteraugli+psnr) -> benchmarks TSV.
"""
import os
import subprocess
import sys

import cv2
import numpy as np
import torch
from spandrel import ModelLoader

W = "/mnt/tower/output/zensr-training/adopted-weights"
SRC = "/mnt/v/imazen-26"
OUT = os.path.expanduser("~/tmp/zensr-audition")
SUBS = [("people", "unsplash-people"), ("textures", "unsplash-textures"),
        ("art-scans", "internet-archive-scans"), ("photos", "lilith")]
TEACHERS = [
    # (name, file, scale)
    ("realesrnet_x4", "RealESRNet_x4plus.pth", 4),
    ("realesrgan_x4", "RealESRGAN_x4plus.pth", 4),
    ("nomoswebphoto_plksr_x4", "4xNomosWebPhoto_RealPLKSR.safetensors", 4),
    ("faceupdat_x4", "4xFaceUpDAT.safetensors", 4),
    ("a2c_compact_x2", "2xNomosUni_compact_multijpg_ldl.pth", 2),  # incumbent
]
DEGS = [("clean", 0), ("q75", 75), ("q50", 50), ("q35", 35)]


def eval_files(d):
    # recursive, sorted by relative path — matches zensr-bench list_images +
    # the frozen first-8 eval split (lilith / art-scans use subdirectories)
    fs = []
    for root, _, files in os.walk(d):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                fs.append(os.path.relpath(os.path.join(root, f), d))
    return sorted(fs)[:8]


def cjpeg_roundtrip(img_bgr, q):
    d = os.path.join(OUT, "_tmp")
    os.makedirs(d, exist_ok=True)
    ppm, jpg = os.path.join(d, "t.ppm"), os.path.join(d, "t.jpg")
    cv2.imwrite(ppm, img_bgr)
    subprocess.run(["cjpeg", "-quality", str(q), "-sample", "2x2", "-optimize",
                    "-outfile", jpg, ppm], check=True, capture_output=True)
    out = cv2.imread(jpg, cv2.IMREAD_COLOR)
    assert out is not None
    return out


def main():
    dev = "cuda"
    os.makedirs(os.path.join(OUT, "gt"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "lanczos"), exist_ok=True)
    models = []
    for name, fn, scale in TEACHERS:
        m = ModelLoader().load_from_file(os.path.join(W, fn))
        assert m.scale == scale, (name, m.scale)
        models.append((name, m.model.eval().to(dev), scale))
        os.makedirs(os.path.join(OUT, name), exist_ok=True)
        print(f"loaded {name} ({m.architecture.name}, x{m.scale})", flush=True)

    for sub, d in SUBS:
        for fn in eval_files(os.path.join(SRC, d)):
            img = cv2.imread(os.path.join(SRC, d, fn), cv2.IMREAD_COLOR)
            if img is None or img.shape[0] < 512 or img.shape[1] < 512:
                continue
            y0 = (img.shape[0] - 512) // 2
            x0 = (img.shape[1] - 512) // 2
            hr = img[y0:y0 + 512, x0:x0 + 512]
            stem = f"{sub}__{os.path.splitext(fn)[0].replace(os.sep, '-')}"
            cv2.imwrite(os.path.join(OUT, "gt", f"{stem}.png"), hr)
            lr0 = cv2.resize(hr, (256, 256), interpolation=cv2.INTER_AREA)
            for deg, q in DEGS:
                lr = lr0 if q == 0 else cjpeg_roundtrip(lr0, q)
                cv2.imwrite(os.path.join(OUT, "lanczos", f"{stem}__{deg}.png"),
                            cv2.resize(lr, (512, 512), interpolation=cv2.INTER_LANCZOS4))
                x = torch.from_numpy(
                    cv2.cvtColor(lr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                ).permute(2, 0, 1)[None].to(dev)
                for name, net, scale in models:
                    with torch.no_grad():
                        y = net(x).clamp(0, 1)
                    y = (y[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
                    y = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
                    if y.shape[0] != 512:
                        y = cv2.resize(y, (512, 512), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(OUT, name, f"{stem}__{deg}.png"), y)
            print(f"done {stem}", flush=True)
    print("AUDITION DONE", flush=True)


if __name__ == "__main__":
    main()
