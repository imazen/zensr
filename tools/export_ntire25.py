#!/usr/bin/env python3
"""Export selected NTIRE2025_ESR model_zoo checkpoints to fixed-shape ONNX for tract.

Mirrors the instantiation lines in NTIRE2025_ESR/test_demo.py exactly.
Usage: python3 export_ntire25.py [HxW ...]   (default 256x256)
Outputs: ../models/<name>_x4_<H>x<W>.onnx
"""
import os
import sys

NTIRE = "/home/lilith/work/superrez/NTIRE2025_ESR"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
sys.path.insert(0, NTIRE)

import torch  # noqa: E402


def build(model_id: int):
    zoo = os.path.join(NTIRE, "model_zoo")
    if model_id == 0:
        from models.team00_EFDN import EFDN
        m = EFDN()
        sd = torch.load(os.path.join(zoo, "team00_EFDN.pth"), map_location="cpu", weights_only=True)
        m.load_state_dict(sd)
        return "EFDN_baseline", m
    if model_id == 7:
        from models.team07_NanoSR import NanoSR_inference
        m = NanoSR_inference(3, 3)
        sd = torch.load(os.path.join(zoo, "team07_NanoSR.pth"), map_location="cpu", weights_only=True)
        m.load_state_dict(sd)
        return "NanoSR", m
    if model_id == 24:
        from models.team24_SPANF import SPANF
        m = SPANF(3, 3, upscale=4, feature_channels=32)
        sd = torch.load(os.path.join(zoo, "team24_spanf.pth"), map_location="cpu", weights_only=True)
        m.load_state_dict(sd, strict=True)
        return "SPANF", m
    if model_id == 31:
        from models.team31_TSR import TSR
        m = TSR()
        sd = torch.load(os.path.join(zoo, "team31_TSR.pth"), map_location="cpu", weights_only=True)
        m.load_state_dict(sd)
        return "TSR", m
    raise SystemExit(f"unknown model id {model_id}")


def main():
    sizes = [tuple(int(x) for x in a.split("x")) for a in sys.argv[1:]] or [(256, 256)]
    os.makedirs(OUT, exist_ok=True)
    for mid in (0, 7, 24, 31):
        name, m = build(mid)
        m = m.eval().cpu()
        n_params = sum(p.numel() for p in m.parameters())
        for h, w in sizes:
            dummy = torch.randn(1, 3, h, w)
            path = os.path.join(OUT, f"{name}_x4_{h}x{w}.onnx")
            torch.onnx.export(
                m, (dummy,), path,
                input_names=["lr"], output_names=["hr"],
                opset_version=17, do_constant_folding=True, dynamo=False,
            )
            sz = os.path.getsize(path)
            with torch.no_grad():
                out = m(dummy)
            print(f"{name}\tparams={n_params}\t{h}x{w}->{tuple(out.shape)}\t"
                  f"onnx={sz}B\t{path}")


if __name__ == "__main__":
    main()
