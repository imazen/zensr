#!/usr/bin/env python3
"""Conditioning ablation trainer (S2a/S7/S10 input-channel experiment).

ZENSR_COND = none | scalar | dmap  -> in_ch = 3 (+1 for scalar/dmap plane).
Data: ZENSR_DATA (make_dejpeg_data4 output). Warm start ZENSR_INIT loads all
body.* tensors whose shapes match (first conv skipped on conditioned arms).
Final report: val PSNR overall + stratified by q-band and encoder
(pairs.tsv). Exports <ZENSR_DATA>/<ZENSR_OUT>_final.pth + models/adopted
raw dump with in_ch recorded in meta (runtime loads in_ch==3 only for now).

Usage: train_cond.py [steps=14000] [batch=64] [lr=7e-5]
"""
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D = os.path.expanduser(os.environ.get("ZENSR_DATA", "~/tmp/zensr-dejpeg-v4"))
COND = os.environ.get("ZENSR_COND", "none")
NF = int(os.environ.get("ZENSR_NF", "64"))
NC = int(os.environ.get("ZENSR_NC", "16"))
OUT_NAME = os.environ.get("ZENSR_OUT", f"dejpeg5_{COND}")
INIT = os.environ.get("ZENSR_INIT", "")
IN_CH = 3 + (0 if COND == "none" else 1)
QBOOST = int(os.environ.get("ZENSR_QBOOST", "3"))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        body = [nn.Conv2d(IN_CH, NF, 3, 1, 1), nn.PReLU(NF)]
        for _ in range(NC):
            body += [nn.Conv2d(NF, NF, 3, 1, 1), nn.PReLU(NF)]
        body += [nn.Conv2d(NF, 3, 3, 1, 1)]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        # residual add over the RGB planes only
        return self.body(x) + x[:, :3]


def charbonnier(a, b, eps=1e-6):
    return torch.sqrt((a - b) ** 2 + eps).mean()


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 14000
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    lr0 = float(sys.argv[3]) if len(sys.argv) > 3 else 7e-5
    dev = "cuda"
    torch.manual_seed(7)
    lr_all = np.load(os.path.join(D, "lr_u8.npy"), mmap_mode="r")
    hr_all = np.load(os.path.join(D, "hr_u8.npy"), mmap_mode="r")
    sc_all = np.load(os.path.join(D, "cond_scalar_f32.npy"))
    dm_all = np.load(os.path.join(D, "dmap_u16.npy"), mmap_mode="r")
    n = lr_all.shape[0] - 512

    rows = list(csv.reader(open(os.path.join(D, "pairs.tsv")), delimiter="\t"))[1:]

    # ZENSR_CPU_DATA=1: keep train arrays host-side (small-VRAM boxes); the
    # per-batch H2D copy is ~5 MB/step. Same data/order/batch as GPU mode.
    cpu_data = os.environ.get("ZENSR_CPU_DATA", "0") == "1"
    data_dev = "cpu" if cpu_data else dev
    lr_gpu = torch.from_numpy(lr_all[:n].copy()).to(data_dev)
    hr_gpu = torch.from_numpy(hr_all[:n].copy()).to(data_dev)
    sc_gpu = torch.from_numpy(sc_all[:n]).to(data_dev)
    dm_gpu = torch.from_numpy(dm_all[:n].astype(np.float32) / 65535.0).to(data_dev)
    val_lr = torch.from_numpy(lr_all[n:].copy()).float().permute(0, 3, 1, 2).div_(255).to(dev)
    val_hr = torch.from_numpy(hr_all[n:].copy()).float().permute(0, 3, 1, 2).div_(255).to(dev)
    val_sc = torch.from_numpy(sc_all[n:]).to(dev)
    val_dm = torch.from_numpy(dm_all[n:].astype(np.float32) / 65535.0).to(dev)

    def with_cond(x, sc, dm):
        if COND == "none":
            return x
        if COND == "scalar":
            plane = sc.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        else:
            plane = dm.unsqueeze(1)
            plane = plane.repeat_interleave(8, dim=2).repeat_interleave(8, dim=3)
        return torch.cat([x, plane], dim=1)

    m = Net().to(dev)
    if INIT:
        sd = torch.load(INIT, map_location="cpu", weights_only=True)
        for k in ("sd", "params", "state_dict"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
        own = m.state_dict()
        keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        m.load_state_dict(keep, strict=False)
        print(f"warm: {len(keep)}/{len(own)} tensors from {os.path.basename(INIT)}", flush=True)
    m = torch.compile(m)
    opt = torch.optim.AdamW(m.parameters(), lr=lr0, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr0 / 50)
    rng = np.random.default_rng(11)

    pool = list(range(n))
    if QBOOST > 0:
        for r in rows:
            i = int(r[0])
            if i < n and (r[4] == "1" or int(r[3]) >= 85):
                pool.extend([i] * QBOOST)
    pool_t = torch.tensor(pool, device=dev)
    print(f"cond={COND} in_ch={IN_CH} train n={n} steps={steps} pool={len(pool)} "
          f"params={sum(p.numel() for p in m.parameters())}", flush=True)

    for step in range(1, steps + 1):
        idx = pool_t[torch.from_numpy(rng.integers(0, len(pool), batch)).to(dev)]
        di = idx.to(data_dev)
        x = lr_gpu[di].to(dev).permute(0, 3, 1, 2).float().div_(255)
        y = hr_gpu[di].to(dev).permute(0, 3, 1, 2).float().div_(255)
        x = with_cond(x, sc_gpu[di].to(dev), dm_gpu[di].to(dev))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = charbonnier(m(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == steps:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                vps = []
                for i in range(0, 512, 128):
                    vo = m(with_cond(val_lr[i:i + 128], val_sc[i:i + 128], val_dm[i:i + 128])).clamp(0, 1).float()
                    mse = ((vo - val_hr[i:i + 128]) ** 2).mean().item()
                    vps.append(-10 * np.log10(max(mse, 1e-10)))
                print(f"step {step} loss {loss.item():.5f} val_psnr {np.mean(vps):.2f}", flush=True)

    # stratified val report (per-sample PSNR joined to pairs.tsv)
    per = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, 512, 64):
            vo = m(with_cond(val_lr[i:i + 64], val_sc[i:i + 64], val_dm[i:i + 64])).clamp(0, 1).float()
            mse = ((vo - val_hr[i:i + 64]) ** 2).mean(dim=(1, 2, 3))
            per.extend((-10 * torch.log10(mse.clamp_min(1e-10))).tolist())
    strat = {}
    for j, p in enumerate(per):
        r = rows[n + j]
        enc, q, clean = r[1], int(r[3]), r[4] == "1"
        band = ("clean" if clean else "q<=9" if q <= 9 else "q10-35" if q <= 35
                else "q36-75" if q <= 75 else "q76+")
        strat.setdefault(band, []).append(p)
        strat.setdefault(enc, []).append(p)
    print("STRATA " + json.dumps({k: round(float(np.mean(v)), 2) for k, v in sorted(strat.items())}), flush=True)

    sd = {k: v.float().cpu() for k, v in m._orig_mod.state_dict().items()}
    torch.save({"sd": sd, "cond": COND, "in_ch": IN_CH}, os.path.join(D, f"{OUT_NAME}_final.pth"))
    print("SAVED", os.path.join(D, f"{OUT_NAME}_final.pth"), flush=True)


if __name__ == "__main__":
    main()
