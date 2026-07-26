#!/usr/bin/env python3
"""Ground-truth people fine-tune (P2-mini). Warm-start + charbonnier vs HR.

Data: ZENSR_DATA (make_people_gt_data.py output: lr_u8 + hr_u8, val tail 512).
Init: ZENSR_INIT = path to a .pth whose sd (or {"sd": sd}) matches the
SRVGGNetCompact body.* layout at ZENSR_NF/ZENSR_NC (e.g. the rtc student
checkpoint, or 2xNomosUni_compact params). Exports models/adopted/<ZENSR_OUT>/
exactly like train_distill.py (weights.raw + goldens + meta.json).

Usage: train_people.py [steps=20000] [batch=96] [lr=1e-4]
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D = os.path.expanduser(os.environ.get("ZENSR_DATA", "~/tmp/zensr-people-gt"))
NF = int(os.environ.get("ZENSR_NF", "24"))
NC = int(os.environ.get("ZENSR_NC", "8"))
SCALE = int(os.environ.get("ZENSR_SCALE", "2"))
OUT_NAME = os.environ.get("ZENSR_OUT", "people_rtc_2x")
SPACE = os.environ.get("ZENSR_SPACE", "rgb")  # rgb | ycbcr (JFIF full-range)
INIT = os.environ.get("ZENSR_INIT", "")
OUTM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "adopted", OUT_NAME)


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        body = [nn.Conv2d(3, NF, 3, 1, 1), nn.PReLU(NF)]
        for _ in range(NC):
            body += [nn.Conv2d(NF, NF, 3, 1, 1), nn.PReLU(NF)]
        body += [nn.Conv2d(NF, 3 * SCALE * SCALE, 3, 1, 1)]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        out = F.pixel_shuffle(self.body(x), SCALE)
        return out + F.interpolate(x, scale_factor=SCALE, mode="nearest")


def charbonnier(a, b, eps=1e-6):
    return torch.sqrt((a - b) ** 2 + eps).mean()


# JFIF full-range RGB->YCbCr, [0,1] planes, Cb/Cr carry +0.5 (matches
# zensr_micro::consist so the runtime pipeline is bit-consistent).
_M = torch.tensor([[0.299, 0.587, 0.114],
                   [-0.1687359, -0.3312641, 0.5],
                   [0.5, -0.4186876, -0.0813124]])
_OFF = torch.tensor([0.0, 0.5, 0.5])


def to_space(x):
    if SPACE != "ycbcr":
        return x
    m = _M.to(x.device).to(x.dtype)
    off = _OFF.to(x.device).to(x.dtype)
    return torch.einsum("ij,bjhw->bihw", m, x) + off.view(1, 3, 1, 1)


def load_init(m):
    sd = torch.load(INIT, map_location="cpu", weights_only=True)
    for k in ("sd", "params", "params_ema", "state_dict"):
        if k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    sd = {k: v for k, v in sd.items() if k.startswith("body.")}
    own = m.state_dict()
    keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    skipped = sorted(set(sd) - set(keep)) + sorted(k for k in own if k not in sd)
    m.load_state_dict(keep, strict=False)
    print(f"warm-started from {INIT}: {len(keep)} tensors loaded"
          f"{', skipped (shape/missing): ' + ','.join(skipped) if skipped else ''}", flush=True)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 96
    lr0 = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
    dev = os.environ.get("ZENSR_DEV") or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    # M4-class Apple silicon runs bf16 natively; ZENSR_AMP=0 to disable.
    amp_on = dev != "cpu" and os.environ.get("ZENSR_AMP", "1") != "0"
    print(f"device={dev} amp={amp_on}", flush=True)
    torch.manual_seed(7)
    lr_all = np.load(os.path.join(D, "lr_u8.npy"), mmap_mode="r")
    hr_all = np.load(os.path.join(D, "hr_u8.npy"), mmap_mode="r")
    n = lr_all.shape[0] - 512
    val_lr = torch.from_numpy(lr_all[n:].copy()).float().permute(0, 3, 1, 2).div_(255).to(dev)
    val_hr = torch.from_numpy(hr_all[n:].copy()).float().permute(0, 3, 1, 2).div_(255).to(dev)
    # ZENSR_CPU_DATA=1: keep the full dataset host-side and ship per-step
    # batches (~2.3MB) — required on MPS (multi-GB single .to(mps) copies
    # hang in waitUntilCompleted) and on small-VRAM cards (ian's 1660 Ti).
    cpu_data = os.environ.get("ZENSR_CPU_DATA", "1" if dev == "mps" else "0") == "1"
    dloc = "cpu" if cpu_data else dev
    lr_gpu = torch.from_numpy(lr_all[:n].copy()).to(dloc)
    hr_gpu = torch.from_numpy(hr_all[:n].copy()).to(dloc)
    # ZENSR_QBOOST: oversample high-q + clean pairs (index duplication) using
    # <data>/../pairs.tsv (dejpeg-v2 layout). Closes the q90 identity gap
    # without runtime gating.
    sample_pool = None
    qboost = int(os.environ.get("ZENSR_QBOOST", "0"))
    ptsv = os.path.join(D, "pairs.tsv")
    if not os.path.exists(ptsv):
        ptsv = os.path.join(os.path.dirname(D), "pairs.tsv")
    if qboost > 0 and os.path.exists(ptsv):
        import csv as _csv
        boosted = list(range(n))
        with open(ptsv) as f:
            rd = _csv.reader(f, delimiter="\t")
            next(rd)
            for row in rd:
                i = int(row[0])
                if i < n and (int(row[4]) == 1 or int(row[3]) >= 85):
                    boosted.extend([i] * qboost)
        sample_pool = torch.tensor(boosted, device=dloc)
        print(f"qboost x{qboost}: pool {len(boosted)} (from {n})", flush=True)

    m = Student().to(dev)
    if INIT:
        load_init(m)
    if dev == "cuda":  # inductor; MPS compile is flaky, eager is fine there
        m = torch.compile(m)
    opt = torch.optim.AdamW(m.parameters(), lr=lr0, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr0 / 50)
    rng = np.random.default_rng(11)
    print(f"train n={n} steps={steps} batch={batch} lr={lr0} nf={NF} nc={NC} "
          f"params={sum(p.numel() for p in m.parameters())}", flush=True)
    for step in range(1, steps + 1):
        if sample_pool is not None:
            idx = sample_pool[torch.from_numpy(rng.integers(0, len(sample_pool), batch)).to(dloc)]
        else:
            idx = torch.from_numpy(rng.integers(0, n, batch)).to(dloc)
        x = to_space(lr_gpu[idx].to(dev).permute(0, 3, 1, 2).float().div_(255))
        y = to_space(hr_gpu[idx].to(dev).permute(0, 3, 1, 2).float().div_(255))
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=amp_on):
            out = m(x)
            loss = charbonnier(out, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps:
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=amp_on):
                vps = []
                for i in range(0, 512, 128):
                    vo = m(to_space(val_lr[i:i + 128])).float()
                    mse = ((vo - to_space(val_hr[i:i + 128])) ** 2).mean().item()
                    vps.append(-10 * np.log10(max(mse, 1e-10)))
                print(f"step {step} loss {loss.item():.5f} val_psnr_vs_GT {np.mean(vps):.2f}", flush=True)
        if step % 10000 == 0 or step == steps:
            torch.save({"sd": m._orig_mod.state_dict(), "step": step},
                       os.path.join(D, f"{OUT_NAME}_{step}.pth"))

    sd = {k: v.float().cpu() for k, v in m._orig_mod.state_dict().items()}
    os.makedirs(OUTM, exist_ok=True)
    idxs = sorted({int(k.split(".")[1]) for k in sd if k.startswith("body.")})
    blobs = []
    for i in idxs:
        wk = f"body.{i}.weight"
        if sd[wk].dim() == 4:
            blobs += [sd[wk].numpy().astype("<f4").tobytes(), sd[f"body.{i}.bias"].numpy().astype("<f4").tobytes()]
        else:
            blobs += [sd[wk].numpy().astype("<f4").tobytes()]
    open(os.path.join(OUTM, "weights.raw"), "wb").write(b"".join(blobs))
    m2 = Student().cpu().eval()
    m2.load_state_dict(sd)
    for (h, w) in [(40, 36), (17, 13)]:
        gi = (np.arange(3 * h * w, dtype=np.int64) % 251).astype("<f4") / 251.0
        x = torch.from_numpy(gi.reshape(1, 3, h, w).copy())
        with torch.no_grad():
            y = m2(x).numpy().astype("<f4")
        gi.tofile(os.path.join(OUTM, f"gold_in_{h}x{w}.raw"))
        y.tofile(os.path.join(OUTM, f"gold_out_{h}x{w}.raw"))
    total = sum(int(np.prod(v.shape)) for v in sd.values())
    json.dump({"arch": "compact", "scale": SCALE, "nf": NF, "nc": NC, "space": SPACE,
               "total_floats": int(total),
               "source": f"people GT fine-tune (init={os.path.basename(INIT) or 'scratch'})"},
              open(os.path.join(OUTM, "meta.json"), "w"), indent=1)
    print("EXPORTED", total, "floats", flush=True)


if __name__ == "__main__":
    main()
