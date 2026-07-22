#!/usr/bin/env python3
"""Dump SPANF (NTIRE25 team24, fc=32, x4) weights + a golden input/output pair
as raw little-endian f32 for the zensr-micro runtime.

Fixed tensor order (zensr-micro hardcodes shapes/offsets):
  conv_near.weight                        [48,1,3,3]   grouped, no bias
  block_{1..5}. c1/c2/c3 .eval_conv       weight+bias  (b1: 3->32,32->32,32->32; b2-5: 32ch)
  conv_cat.weight [32,112,1,1], .bias [32]
  conv_2.eval_conv.weight [48,32,3,3], .bias [48]

Golden: input ramp (i%251)/251 over [1,3,64,64] -> model output [1,3,256,256].
Outputs to ../models/: spanf_weights.raw, spanf_in_64.raw, spanf_gold_256.raw
"""
import os
import sys

import numpy as np
import torch

NTIRE = "/home/lilith/work/superrez/NTIRE2025_ESR"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
sys.path.insert(0, NTIRE)


def main():
    from models.team24_SPANF import SPANF
    m = SPANF(3, 3, upscale=4, feature_channels=32)
    sd = torch.load(os.path.join(NTIRE, "model_zoo", "team24_spanf.pth"),
                    map_location="cpu", weights_only=True)
    m.load_state_dict(sd, strict=True)
    m = m.eval().cpu()

    order = ["conv_near.weight"]
    for b in range(1, 6):
        for c in ("c1_r", "c2_r", "c3_r"):
            order += [f"block_{b}.{c}.eval_conv.weight",
                      f"block_{b}.{c}.eval_conv.bias"]
    order += ["conv_cat.weight", "conv_cat.bias",
              "conv_2.eval_conv.weight", "conv_2.eval_conv.bias"]

    sd = m.state_dict()
    blobs, total = [], 0
    for name in order:
        t = sd[name].detach().cpu().numpy().astype("<f4")
        print(f"{name}\t{tuple(t.shape)}\t{t.size}")
        blobs.append(t.tobytes())
        total += t.size
    os.makedirs(OUT, exist_ok=True)
    wpath = os.path.join(OUT, "spanf_weights.raw")
    with open(wpath, "wb") as f:
        f.write(b"".join(blobs))
    print(f"total_floats={total} bytes={total * 4} -> {wpath}")

    n = 3 * 64 * 64
    inp = (np.arange(n, dtype=np.int64) % 251).astype("<f4") / 251.0
    x = torch.from_numpy(inp.reshape(1, 3, 64, 64).copy())
    with torch.no_grad():
        y = m(x)
    y = y.numpy().astype("<f4")
    inp.tofile(os.path.join(OUT, "spanf_in_64.raw"))
    y.tofile(os.path.join(OUT, "spanf_gold_256.raw"))
    print(f"golden: in {x.shape} -> out {y.shape}, "
          f"range [{y.min():.4f},{y.max():.4f}]")


if __name__ == "__main__":
    main()
