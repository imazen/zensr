#!/usr/bin/env python3
"""Dump adopted model weights + torch goldens for zensr-micro.

Functional forwards built directly from state-dict tensors (no basicsr/neosr
imports). For SPAN's Conv3XC, eval_conv freshness is VERIFIED against the
training branches on random input; mismatch > 1e-4 aborts that model.

Per model -> models/adopted/<name>/{meta.json, weights.raw, gold_in_*.raw, gold_out_*.raw}
Weight order (f32 LE):
  compact: [conv w,b, prelu a] x (nc+1 blocks: first 3->nf then nc x nf->nf) + final conv w,b
  span48:  conv_1 w,b | per block b1..b6: c1 w,b c2 w,b c3 w,b | conv_2 w,b | conv_cat w,b | ups w,b
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

W = "/mnt/tower/output/zensr-training/adopted-weights"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "adopted")

MODELS = [
    # (file, arch, scale, name)
    ("realesr-general-x4v3.pth", "compact", 4, "general_x4v3"),
    ("realesr-general-wdn-x4v3.pth", "compact", 4, "general_wdn_x4v3"),
    ("realesr-animevideov3.pth", "compact", 4, "animevideo_x4v3"),
    ("2xNomosUni_compact_multijpg_ldl.pth", "compact", 2, "nomosuni_compact_2x"),
    ("2xHFA2k_compact_multijpg.pth", "compact", 2, "hfa2k_compact_2x"),
    ("2xNomosUni_span_multijpg.pth", "span48", 2, "nomosuni_span_2x"),
    ("4xNomosUni_span_multijpg.pth", "span48", 4, "nomosuni_span_4x"),
]


def load_sd(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("params", "params_ema", "state_dict"):
        if k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    return {k: v.float() for k, v in sd.items() if isinstance(v, torch.Tensor)}


def compact_forward(sd, x, scale):
    # body.0 conv, body.1 prelu, body.2 conv, ... body.{2nc+2} final conv; then shuffle + nearest add
    i = 0
    out = x
    convs = sorted(
        {int(k.split(".")[1]) for k in sd if k.startswith("body.") and k.endswith(".weight")}
    )
    last = max(convs)
    while i <= last:
        wk, bk = f"body.{i}.weight", f"body.{i}.bias"
        if wk in sd and sd[wk].dim() == 4:
            out = F.conv2d(out, sd[wk], sd[bk], padding=1)
        elif wk in sd and sd[wk].dim() == 1:  # prelu
            out = F.prelu(out, sd[wk])
        i += 1
    out = F.pixel_shuffle(out, scale)
    base = F.interpolate(x, scale_factor=scale, mode="nearest")
    return out + base


def compact_dump_order(sd):
    order = []
    idxs = sorted({int(k.split(".")[1]) for k in sd if k.startswith("body.")})
    for i in idxs:
        wk = f"body.{i}.weight"
        if sd[wk].dim() == 4:
            order += [wk, f"body.{i}.bias"]
        else:
            order += [wk]
    return order


def conv3xc_eval(sd, prefix, x):
    return F.conv2d(x, sd[f"{prefix}.eval_conv.weight"], sd[f"{prefix}.eval_conv.bias"], padding=1)


def conv3xc_branch(sd, prefix, x):
    xp = F.pad(x, (1, 1, 1, 1))
    y = F.conv2d(xp, sd[f"{prefix}.conv.0.weight"], sd[f"{prefix}.conv.0.bias"])
    y = F.conv2d(y, sd[f"{prefix}.conv.1.weight"], sd[f"{prefix}.conv.1.bias"])
    y = F.conv2d(y, sd[f"{prefix}.conv.2.weight"], sd[f"{prefix}.conv.2.bias"])
    return y + F.conv2d(x, sd[f"{prefix}.sk.weight"], sd[f"{prefix}.sk.bias"])


def merge_conv3xc(sd, prefix):
    """Merge sk(1x1) + [1x1 -> 3x3(pad0) -> 1x1] into one 3x3 (+bias); install
    into sd under eval_conv keys after verifying against the branch forward."""
    w1, b1 = sd[f"{prefix}.conv.0.weight"], sd[f"{prefix}.conv.0.bias"]
    w2, b2 = sd[f"{prefix}.conv.1.weight"], sd[f"{prefix}.conv.1.bias"]
    w3, b3 = sd[f"{prefix}.conv.2.weight"], sd[f"{prefix}.conv.2.bias"]
    sk_w, sk_b = sd[f"{prefix}.sk.weight"], sd[f"{prefix}.sk.bias"]
    # compose 1x1 (w1) into 3x3 (w2):  [m,i] into [o,m,ky,kx]
    w12 = torch.einsum("omyx,mi->oiyx", w2, w1[:, :, 0, 0])
    b12 = (w2 * b1.view(1, -1, 1, 1)).sum(dim=(1, 2, 3)) + b2
    # compose 3x3 (w12) into 1x1 (w3)
    w123 = torch.einsum("pm,miyx->piyx", w3[:, :, 0, 0], w12)
    b123 = w3[:, :, 0, 0] @ b12 + b3
    # add skip 1x1 into the center tap
    w = w123.clone()
    w[:, :, 1, 1] += sk_w[:, :, 0, 0]
    b = b123 + sk_b
    # verify merged == branch forward
    x = torch.randn(1, sk_w.shape[1], 8, 8)
    ref = conv3xc_branch(sd, prefix, x)
    got = F.conv2d(x, w, b, padding=1)
    d = (ref - got).abs().max().item()
    if d > 1e-4:
        raise SystemExit(f"merge verification FAILED for {prefix}: {d}")
    sd[f"{prefix}.eval_conv.weight"] = w
    sd[f"{prefix}.eval_conv.bias"] = b
    return d


SPAN_MEAN = (0.4488, 0.4371, 0.4040)  # DIV2K rgb_mean, official SPAN default
SPAN_IMG_RANGE = 255.0


def span_forward(sd, x, scale):
    # Input [0,1]; normalized HERE, matching official: conv zero-padding happens
    # AFTER norm (border == mean gray). Folding the norm into conv_1 is NOT
    # equivalent at image borders (zero-pad would mean black) — measured 0.32
    # border error; do not re-attempt. Official SPAN uses SiLU(inplace=True),
    # which mutates out1 in place -> the final concat sees SiLU(out1), not out1.
    mean = torch.tensor(SPAN_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) * SPAN_IMG_RANGE
    def spab(prefix, inp):
        o1 = conv3xc_eval(sd, f"{prefix}.c1_r", inp)
        o1a = F.silu(o1)
        o = F.silu(conv3xc_eval(sd, f"{prefix}.c2_r", o1a))
        o3 = conv3xc_eval(sd, f"{prefix}.c3_r", o)
        att = torch.sigmoid(o3) - 0.5
        return (o3 + inp) * att, o1a

    feat = conv3xc_eval(sd, "conv_1", x)
    b1, _ = spab("block_1", feat)
    b2, _ = spab("block_2", b1)
    b3, _ = spab("block_3", b2)
    b4, _ = spab("block_4", b3)
    b5, _ = spab("block_5", b4)
    b6, b6o1 = spab("block_6", b5)
    b6c = conv3xc_eval(sd, "conv_2", b6)
    cat = torch.cat([feat, b6c, b1, b6o1], 1)
    out = F.conv2d(cat, sd["conv_cat.weight"], sd["conv_cat.bias"])
    ups_w = sd["upsampler.0.weight"]
    out = F.conv2d(out, ups_w, sd["upsampler.0.bias"], padding=1)
    return F.pixel_shuffle(out, scale)


SPAN_PREFIXES = ["conv_1", "conv_2"] + [
    f"block_{b}.{c}" for b in range(1, 7) for c in ("c1_r", "c2_r", "c3_r")
]


def prepare_span_sd(path):
    """Canonical span sd prep: merge every Conv3XC branch (checkpoint eval_conv
    can be stale). span_forward normalizes input itself."""
    sd = load_sd(path)
    worst = max(merge_conv3xc(sd, p) for p in SPAN_PREFIXES)
    return sd, worst


def span_dump_order(sd):
    order = ["conv_1.eval_conv.weight", "conv_1.eval_conv.bias"]
    for b in range(1, 7):
        for c in ("c1_r", "c2_r", "c3_r"):
            order += [f"block_{b}.{c}.eval_conv.weight", f"block_{b}.{c}.eval_conv.bias"]
    order += ["conv_2.eval_conv.weight", "conv_2.eval_conv.bias",
              "conv_cat.weight", "conv_cat.bias",
              "upsampler.0.weight", "upsampler.0.bias"]
    return order


def ramp(c, h, w, seed=0):
    n = c * h * w
    return (np.arange(seed, n + seed, dtype=np.int64) % 251).astype("<f4") / 251.0


def main():
    os.makedirs(OUT, exist_ok=True)
    for fname, arch, scale, name in MODELS:
        path = os.path.join(W, fname)
        if not os.path.exists(path):
            print(f"MISSING {fname}")
            continue
        from spandrel import ModelLoader  # reference implementation — REQUIRED

        ref_net = ModelLoader().load_from_file(path).model.eval()
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        if arch == "span48":
            assert ref_net.is_norm, f"{name}: expected is_norm SPAN checkpoint"
            assert ref_net.img_range == SPAN_IMG_RANGE
            assert torch.allclose(ref_net.mean.flatten(), torch.tensor(SPAN_MEAN)), \
                f"{name}: nonstandard rgb_mean {ref_net.mean.flatten().tolist()}"
            sd, worst = prepare_span_sd(path)
            print(f"{name}: branches merged+verified (worst diff {worst:.2e})")
            order = span_dump_order(sd)
            fwd = lambda x: span_forward(sd, x, scale)
            nf = sd["conv_1.eval_conv.weight"].shape[0]
            nc = 6
        else:
            sd = load_sd(path)
            order = compact_dump_order(sd)
            fwd = lambda x: compact_forward(sd, x, scale)
            nf = sd["body.0.weight"].shape[0]
            nc = sum(1 for k in order if k.endswith(".weight") and sd[k].dim() == 4) - 2
        # HARD GATE: my functional forward vs the reference implementation on
        # random [0,1] input. Consistency goldens alone once hid a broken graph
        # (constant-gray SPAN output) — never trust self-agreement again.
        xr = torch.rand(1, 3, 33, 29, generator=torch.Generator().manual_seed(11))
        with torch.no_grad():
            ref_out = ref_net(xr)
            my_out = fwd(xr)
        rd = (ref_out - my_out).abs().max().item()
        if rd > 2e-3:
            raise SystemExit(f"{name}: reference cross-check FAILED (maxdiff {rd})")
        print(f"{name}: spandrel cross-check OK (maxdiff {rd:.2e})")
        blobs, shapes = [], []
        for k in order:
            t = sd[k].detach().numpy().astype("<f4")
            blobs.append(t.tobytes())
            shapes.append([k, list(t.shape)])
        with open(os.path.join(d, "weights.raw"), "wb") as f:
            f.write(b"".join(blobs))
        # f16 ship file always regenerated with the f32 dump so they can't desync
        np.frombuffer(b"".join(blobs), dtype="<f4").astype("<f2").tofile(
            os.path.join(d, "weights_f16.raw"))
        total = sum(int(np.prod(s[1])) for s in shapes)
        for (h, wd) in [(40, 36), (17, 13)]:
            gi = ramp(3, h, wd)
            x = torch.from_numpy(gi.reshape(1, 3, h, wd).copy())
            with torch.no_grad():
                y = fwd(x).numpy().astype("<f4")
            gi.tofile(os.path.join(d, f"gold_in_{h}x{wd}.raw"))
            y.tofile(os.path.join(d, f"gold_out_{h}x{wd}.raw"))
        meta = {"arch": arch, "scale": scale, "nf": int(nf), "nc": int(nc),
                "total_floats": int(total), "source": fname}
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
        print(f"{name}: {arch} nf={nf} nc={nc} s={scale} floats={total}")


if __name__ == "__main__":
    main()
