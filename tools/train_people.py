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
    dev = "cuda"
    torch.manual_seed(7)
    lr_all = np.load(os.path.join(D, "lr_u8.npy"), mmap_mode="r")
    hr_all = np.load(os.path.join(D, "hr_u8.npy"), mmap_mode="r")
    n = lr_all.shape[0] - 512
    val_lr = torch.from_numpy(lr_all[n:].copy()).float().permute(0, 3, 1, 2).div_(255).to(dev)
    val_hr = torch.from_numpy(hr_all[n:].copy()).float().permute(0, 3, 1, 2).div_(255).to(dev)
    lr_gpu = torch.from_numpy(lr_all[:n].copy()).to(dev)
    hr_gpu = torch.from_numpy(hr_all[:n].copy()).to(dev)

    m = Student().to(dev)
    if INIT:
        load_init(m)
    m = torch.compile(m)
    opt = torch.optim.AdamW(m.parameters(), lr=lr0, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr0 / 50)
    rng = np.random.default_rng(11)
    print(f"train n={n} steps={steps} batch={batch} lr={lr0} nf={NF} nc={NC} "
          f"params={sum(p.numel() for p in m.parameters())}", flush=True)
    for step in range(1, steps + 1):
        idx = torch.from_numpy(rng.integers(0, n, batch)).to(dev)
        x = lr_gpu[idx].permute(0, 3, 1, 2).float().div_(255)
        y = hr_gpu[idx].permute(0, 3, 1, 2).float().div_(255)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(x)
            loss = charbonnier(out, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                vps = []
                for i in range(0, 512, 128):
                    vo = m(val_lr[i:i + 128]).clamp(0, 1).float()
                    mse = ((vo - val_hr[i:i + 128]) ** 2).mean().item()
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
    json.dump({"arch": "compact", "scale": SCALE, "nf": NF, "nc": NC,
               "total_floats": int(total),
               "source": f"people GT fine-tune (init={os.path.basename(INIT) or 'scratch'})"},
              open(os.path.join(OUTM, "meta.json"), "w"), indent=1)
    print("EXPORTED", total, "floats", flush=True)


if __name__ == "__main__":
    main()
