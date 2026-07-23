#!/usr/bin/env python3
"""Train the realtime-2x student (Compact nf=24 nc=8) by output-distillation
from 2xNomosUni_span_multijpg (S-E pilot; PLAN S8/S9).

Data: ~/tmp/zensr-distill/{lr_u8.npy, teacher_f16.npy} (val = last 512).
Out:  ~/tmp/zensr-distill/student_{step}.pth + models/adopted/rt_distill_2x/
      (weights.raw + goldens, same layout as dump_adopted compact).
Usage: train_distill.py [steps=60000] [batch=96]
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D = os.path.expanduser("~/tmp/zensr-distill")
OUTM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "adopted", "rt_distill_2x")
NF, NC, SCALE = 24, 8, 2


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


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 96
    dev = "cuda"
    torch.manual_seed(7)
    lr_all = np.load(os.path.join(D, "lr_u8.npy"), mmap_mode="r")
    tg_all = np.load(os.path.join(D, "teacher_f16.npy"), mmap_mode="r")
    n = lr_all.shape[0] - 512
    val_lr = torch.from_numpy(lr_all[n:].copy()).float().permute(0, 3, 1, 2).div_(255).to(dev)
    val_tg = torch.from_numpy(tg_all[n:].astype(np.float32)).to(dev)
    # GPU-resident train set (u8 0.37GB + f16 3.0GB): host-side gather/convert/H2D
    # was the bottleneck (3.3 steps/s, GPU 32%); sample + convert on device instead.
    lr_gpu = torch.from_numpy(lr_all[:n].copy()).to(dev)
    tg_gpu = torch.from_numpy(tg_all[:n].copy()).to(dev)

    m = Student().to(dev)
    m = torch.compile(m)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=2e-5)
    rng = np.random.default_rng(11)
    print(f"train n={n} steps={steps} batch={batch} params={sum(p.numel() for p in m.parameters())}", flush=True)
    for step in range(1, steps + 1):
        idx = torch.from_numpy(rng.integers(0, n, batch)).to(dev)
        x = lr_gpu[idx].permute(0, 3, 1, 2).float().div_(255)
        y = tg_gpu[idx].float()
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
                    mse = ((vo - val_tg[i:i + 128]) ** 2).mean().item()
                    vps.append(-10 * np.log10(max(mse, 1e-10)))
                print(f"step {step} loss {loss.item():.5f} val_psnr_vs_teacher {np.mean(vps):.2f}", flush=True)
        if step % 10000 == 0 or step == steps:
            torch.save({"sd": m._orig_mod.state_dict(), "step": step},
                       os.path.join(D, f"student_{step}.pth"))

    # export in compact dump order + goldens
    sd = {k: v.float().cpu() for k, v in m._orig_mod.state_dict().items()}
    os.makedirs(OUTM, exist_ok=True)
    order = []
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
               "total_floats": int(total), "source": "distilled from 2xNomosUni_span_multijpg"},
              open(os.path.join(OUTM, "meta.json"), "w"), indent=1)
    print("EXPORTED", total, "floats", flush=True)


if __name__ == "__main__":
    main()
