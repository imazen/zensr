#!/usr/bin/env python3
"""Teacher audition for the people/texture gap (S9 rung 4 gating experiment).

Question: does ANY off-the-shelf heavyweight beat Lanczos (and the incumbent
A2c teacher) on people/textures/art-scans at web-JPEG degradations, at the
x2-target protocol we would actually distill with?

Protocol (matches systems_eval x2 track in spirit; self-contained kernels):
  HR = center-crop 512 of each held-out file (test bucket, 8 per folder)
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_split import split_map  # noqa: E402
from imazen26_canonical import CANONICAL_ROOT, canonical_for  # noqa: E402

SPLIT = split_map()

W = "/mnt/tower/output/zensr-training/adopted-weights"
# Repointed 2026-09-07 to the canonical corpus (docs/CORPUS-REPOINT-HANDOFF.md).
SRC = CANONICAL_ROOT
# The four content classes this audition is about. `photos` spans four canonical
# folders because the old flat `lilith` did — see tools/imazen26_canonical.py.
SUBS = [("people", ["2000-unsplash-people"]),
        ("textures", ["2400-unsplash-textures"]),
        ("art-scans", ["6600-ia-scans-manuscript-illustrations",
                       "6800-ia-scans-manuscript-text"]),
        ("photos", canonical_for("lilith"))]
N_PER_CLASS = 8
TEACHERS = [
    # (name, file, scale)
    ("realesrnet_x4", "RealESRNet_x4plus.pth", 4),
    ("realesrgan_x4", "RealESRGAN_x4plus.pth", 4),
    ("nomoswebphoto_plksr_x4", "4xNomosWebPhoto_RealPLKSR.safetensors", 4),
    ("faceupdat_x4", "4xFaceUpDAT.safetensors", 4),
    ("a2c_compact_x2", "2xNomosUni_compact_multijpg_ldl.pth", 2),  # incumbent
]
DEGS = [("clean", 0), ("q75", 75), ("q50", 50), ("q35", 35)]


def eval_files(folders):
    """Held-out files for a content class: the TEST bucket of the origin split,
    first N_PER_CLASS by path.

    Was "first 8 sorted per subdir", which is not a split — it admitted training
    images whenever a file ahead of it failed to decode. The bucket cannot slide.
    """
    fs = [p for p, b in SPLIT.items()
          if b == "test" and p.split("/", 1)[0] in set(folders)]
    return sorted(fs)[:N_PER_CLASS]


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

    for sub, folders in SUBS:
        for fn in eval_files(folders):
            img = cv2.imread(os.path.join(SRC, fn), cv2.IMREAD_COLOR)
            if img is None or img.shape[0] < 512 or img.shape[1] < 512:
                continue
            y0 = (img.shape[0] - 512) // 2
            x0 = (img.shape[1] - 512) // 2
            hr = img[y0:y0 + 512, x0:x0 + 512]
            stem = f"{sub}__{os.path.splitext(fn)[0].replace(os.sep, '-')}"
            # Reference kind travels with the crop: a JPEG-sourced HR is itself
            # compressed, and the scores have to be read split by that.
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
