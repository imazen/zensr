#!/usr/bin/env python3
"""Weight-compression accuracy study for SPANF (fc=32, x4).

Variants: f16, int8 per-tensor, int8/int6/int4 per-output-channel (symmetric),
k-means 256-entry codebook (8-bit indices, non-uniform). Conv weights only;
biases stay fp32 (they're 592 floats total — compressing them is pointless).

Reference = fp32-weight model output. Metrics on [0,1]-clipped outputs:
max_abs, RMSE, PSNR, and %% of pixels whose u8 rounding changes.

Also dumps Rust-loadable f16/int8pc weight files (fixed dump order) to models/.
Outputs a TSV to benchmarks/.
"""
import copy
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from export_ntire25 import build  # noqa: E402

OUT = os.path.join(HERE, "..", "models")
BENCH = os.path.join(HERE, "..", "benchmarks")


def quantize(model, mode):
    m = copy.deepcopy(model)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.dim() != 4 or "weight" not in name:
                continue
            w = p.data
            if mode == "f16":
                p.data = w.half().float()
            elif mode == "int8_pt":
                s = w.abs().max().clamp_min(1e-12) / 127.0
                p.data = torch.round(w / s).clamp(-127, 127) * s
            elif mode in ("int8_pc", "int6_pc", "int4_pc"):
                lv = {"int8_pc": 127.0, "int6_pc": 31.0, "int4_pc": 7.0}[mode]
                s = w.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12) / lv
                p.data = torch.round(w / s).clamp(-lv, lv) * s
            elif mode == "int8_pc_nogate":
                # per-channel int8 EXCEPT the gate-feeding c3 convs of the
                # gated blocks (2..5) — locates the sensitivity.
                if any(f"block_{b}.c3_r" in name for b in (2, 3, 4, 5)):
                    continue
                s_ = w.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12) / 127.0
                p.data = torch.round(w / s_).clamp(-127, 127) * s_
            elif mode == "k256":
                flat = w.flatten()
                # 1-D k-means, quantile init, 12 Lloyd iterations.
                qs = torch.quantile(flat, torch.linspace(0, 1, 256))
                c = qs.clone()
                for _ in range(12):
                    idx = torch.bucketize(flat, (c[:-1] + c[1:]) / 2)
                    for k in range(256):
                        sel = flat[idx == k]
                        if sel.numel() > 0:
                            c[k] = sel.mean()
                idx = torch.bucketize(flat, (c[:-1] + c[1:]) / 2)
                p.data = c[idx].reshape(w.shape)
            else:
                raise SystemExit(f"unknown mode {mode}")
    return m


def load_inputs():
    inputs = {}
    n = 3 * 64 * 64
    ramp = (np.arange(n, dtype=np.int64) % 251).astype(np.float32) / 251.0
    inputs["ramp64"] = torch.from_numpy(ramp.reshape(1, 3, 64, 64).copy())
    try:
        from PIL import Image

        cands = ["/home/lilith/work/imageflow/examples/encode_avif/waterhouse.jpg"] + sorted(
            glob.glob("/mnt/v/collections/art-cc0/*.jpg")
            + glob.glob("/mnt/v/collections/art-cc0/*.png")
        )
        for path in cands:
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            if img.width < 200 or img.height < 200:
                continue
            a = np.asarray(img, dtype=np.float32) / 255.0
            # two 96x96 crops: center + top-left detail
            ch, cw = a.shape[0] // 2, a.shape[1] // 2
            for tag, (y, x) in {"c": (ch - 48, cw - 48), "tl": (8, 8)}.items():
                t = a[y : y + 96, x : x + 96, :].transpose(2, 0, 1)[None]
                inputs[f"{os.path.basename(path)}:{tag}"] = torch.from_numpy(
                    t.copy()
                )
            break
        else:
            print("NOTE: no usable real image found; ramp-only study")
    except ImportError:
        print("NOTE: PIL unavailable; ramp-only study")
    return inputs


def psnr(ref, x):
    mse = float(((ref - x) ** 2).mean())
    return 99.0 if mse == 0 else 10.0 * np.log10(1.0 / mse)


def dump_rust_variants(model):
    """Dump f16 and int8-per-channel files in the fixed dump order."""
    order = ["conv_near.weight"]
    for b in range(1, 6):
        for c in ("c1_r", "c2_r", "c3_r"):
            order += [f"block_{b}.{c}.eval_conv.weight", f"block_{b}.{c}.eval_conv.bias"]
    order += ["conv_cat.weight", "conv_cat.bias", "conv_2.eval_conv.weight", "conv_2.eval_conv.bias"]
    sd = model.state_dict()
    f16 = bytearray()
    i8 = bytearray()
    for name in order:
        t = sd[name].detach().cpu()
        if t.dim() == 4:
            f16 += t.numpy().astype("<f2").tobytes()
            lv = 127.0
            s = t.abs().amax(dim=(1, 2, 3)).clamp_min(1e-12) / lv
            q = torch.round(t / s[:, None, None, None]).clamp(-lv, lv).to(torch.int8)
            i8 += s.numpy().astype("<f4").tobytes()
            i8 += q.numpy().astype("i1").tobytes()
        else:
            f16 += t.numpy().astype("<f4").tobytes()  # biases stay f32 in both
            i8 += t.numpy().astype("<f4").tobytes()
    with open(os.path.join(OUT, "spanf_weights_f16.raw"), "wb") as f:
        f.write(f16)
    with open(os.path.join(OUT, "spanf_weights_int8pc.raw"), "wb") as f:
        f.write(i8)
    print(f"dumped f16 ({len(f16)}B) + int8pc ({len(i8)}B) variants")


def main():
    torch.manual_seed(0)
    _, model = build(24)
    model = model.eval().cpu()
    inputs = load_inputs()
    refs = {}
    with torch.no_grad():
        for k, x in inputs.items():
            refs[k] = model(x).clamp(0, 1)

    modes = ["f16", "int8_pt", "int8_pc", "int8_pc_nogate", "int6_pc", "int4_pc", "k256"]
    os.makedirs(BENCH, exist_ok=True)
    tsv_path = os.path.join(BENCH, "quant_accuracy_2026-07-22.tsv")
    rows = ["mode\tinput\tmax_abs\trmse\tpsnr_db\tu8_mismatch_pct"]
    for mode in modes:
        qm = quantize(model, mode).eval()
        with torch.no_grad():
            for k, x in inputs.items():
                y = qm(x).clamp(0, 1)
                r = refs[k]
                d = (y - r).abs()
                mm = (
                    (torch.round(y * 255) != torch.round(r * 255))
                    .float()
                    .mean()
                    .item()
                    * 100
                )
                rows.append(
                    f"{mode}\t{k}\t{d.max().item():.4e}\t"
                    f"{float((d ** 2).mean()) ** 0.5:.4e}\t"
                    f"{psnr(r, y):.2f}\t{mm:.3f}"
                )
                print(rows[-1])
    with open(tsv_path, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"wrote {tsv_path}")
    dump_rust_variants(model)


if __name__ == "__main__":
    main()
