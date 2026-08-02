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
# Checkpoint cadence. A run on a box we do not exclusively own can lose
# everything since its last checkpoint (2026-08-01: 7k steps lost to a 10k
# cadence when a dual-boot box flipped OS), so short runs on shared hardware
# should set this well below the default.
CKPT_EVERY = int(os.environ.get("ZENSR_CKPT_EVERY", "10000"))
# Feature/affinity KD (ROADMAP 1.4). ZENSR_FKD_TEACHER points at a teacher .pth
# and switches targets to that teacher computed ONLINE — intermediate features
# cannot be precomputed at corpus scale, and computing the output target the
# same way keeps the paired arms identical apart from the affinity term.
# ZENSR_FKD_W=0 is the output-KD control arm; >0 adds affinity supervision.
FKD_TEACHER = os.environ.get("ZENSR_FKD_TEACHER", "")
FKD_W = float(os.environ.get("ZENSR_FKD_W", "0"))
FKD_TNF = int(os.environ.get("ZENSR_FKD_TNF", "64"))
FKD_TNC = int(os.environ.get("ZENSR_FKD_TNC", "16"))
FKD_POOL = int(os.environ.get("ZENSR_FKD_POOL", "16"))
FKD_TAPS = int(os.environ.get("ZENSR_FKD_TAPS", "3"))
LOSS_SPACE = os.environ.get("ZENSR_LOSS_SPACE", "")  # "" | ycbcr
CHROMA_W = float(os.environ.get("ZENSR_CHROMA_W", "1"))
EDGE_W = float(os.environ.get("ZENSR_EDGE_W", "0"))
OUTM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "adopted", OUT_NAME)


class Student(nn.Module):
    def __init__(self, nf=None, nc=None):
        super().__init__()
        nf = NF if nf is None else nf
        nc = NC if nc is None else nc
        body = [nn.Conv2d(3, nf, 3, 1, 1), nn.PReLU(nf)]
        for _ in range(nc):
            body += [nn.Conv2d(nf, nf, 3, 1, 1), nn.PReLU(nf)]
        body += [nn.Conv2d(nf, 3 * SCALE * SCALE, 3, 1, 1)]
        self.body = nn.Sequential(*body)

    def forward(self, x, taps=()):
        if not taps:
            out = F.pixel_shuffle(self.body(x), SCALE)
            return out + F.interpolate(x, scale_factor=SCALE, mode="nearest")
        h, got = x, []
        for i, layer in enumerate(self.body):
            h = layer(h)
            if i in taps:
                got.append(h)
        out = F.pixel_shuffle(h, SCALE)
        return out + F.interpolate(x, scale_factor=SCALE, mode="nearest"), got


def prelu_taps(nc, k):
    """Body indices of the k evenly-spaced PReLU outputs in a body of nc blocks.

    Layout is [conv, prelu] + nc*[conv, prelu] + [conv], so PReLU outputs sit
    at the odd indices 1..2*nc+1. Spacing them by relative depth is what lets a
    24-wide/8-deep student be compared against a 64-wide/16-deep teacher.
    """
    odd = list(range(1, 2 * nc + 2, 2))
    if k >= len(odd):
        return tuple(odd)
    step = len(odd) / (k + 1)
    return tuple(odd[min(len(odd) - 1, int(round((i + 1) * step)) - 1)] for i in range(k))


def affinity(f, pool):
    """Spatial self-similarity of a feature map: B,C,H,W -> B,P,P (P=pool*pool).

    Normalising along channels before the Gram product makes this independent
    of channel count, which is the whole reason affinity KD works between
    architectures of different width — no learned projector to tune, so the
    arm differs from output-KD by exactly one loss term.
    """
    g = F.adaptive_avg_pool2d(f, pool).flatten(2)
    g = F.normalize(g, dim=1)
    return torch.bmm(g.transpose(1, 2), g)


def affinity_loss(s_taps, t_taps, pool):
    return sum(
        (affinity(s.float(), pool) - affinity(t.float(), pool)).abs().mean()
        for s, t in zip(s_taps, t_taps)
    ) / max(1, len(s_taps))


def _edge_weight(y):
    """Per-pixel weight from the TARGET's 8x8 block contrast (max-min luma).

    ZENSR_EDGE_W=k -> weight = 1 + k*clamp(contrast/64, 0, 1). Shifts capacity
    toward the high-contrast blocks that carry ~15 dB more error and 33% of
    pixels (contrast_eval 2026-07-30). Zero runtime/architecture cost.
    """
    lum = (0.299 * y[:, 0:1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3])
    hi = F.max_pool2d(lum, 8, 8)
    lo = -F.max_pool2d(-lum, 8, 8)
    c = ((hi - lo) * 255.0 / 64.0).clamp(0.0, 1.0)
    return 1.0 + EDGE_W * F.interpolate(c, scale_factor=8, mode="nearest")


def charbonnier(a, b, eps=1e-6):
    # ZENSR_LOSS_SPACE=ycbcr computes the loss in YCbCr regardless of model
    # I/O space, with Cb/Cr weighted by ZENSR_CHROMA_W (chroma-ceiling probe
    # 2026-07-27: the RGB loss lets the model ignore ~half the remaining
    # 420 error mass; this reweights without changing the deployed model).
    if LOSS_SPACE == "ycbcr":
        m = _M.to(a.device).to(a.dtype)
        aw = torch.einsum("ij,bjhw->bihw", m, a)
        bw = torch.einsum("ij,bjhw->bihw", m, b)
        wv = torch.tensor([1.0, CHROMA_W, CHROMA_W], device=a.device, dtype=a.dtype).view(1, 3, 1, 1)
        return (torch.sqrt((aw - bw) ** 2 + eps) * wv).mean()
    if EDGE_W > 0.0:
        w = _edge_weight(b)
        pw = torch.sqrt((a - b) ** 2 + eps)
        return (pw * w).sum() / (w.expand_as(pw).sum())
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
    # AMP dtype by hardware: Ampere+ (cc>=8) and Apple silicon run bf16
    # natively; Turing (cc 7.x) has fp16 tensor cores but only EMULATED bf16
    # (torch.cuda.is_bf16_supported() returns True there anyway — check
    # compute capability, not that flag). fp16 needs a GradScaler.
    amp_on = dev != "cpu" and os.environ.get("ZENSR_AMP", "1") != "0"
    amp_dt = torch.bfloat16
    if dev == "cuda" and torch.cuda.get_device_capability()[0] < 8:
        amp_dt = torch.float16
    scaler = torch.amp.GradScaler(dev, enabled=amp_on and amp_dt == torch.float16)
    print(f"device={dev} amp={amp_on} dtype={amp_dt if amp_on else 'fp32'}", flush=True)
    torch.manual_seed(7)
    lr_all = np.load(os.path.join(D, "lr_u8.npy"), mmap_mode="r")
    # Online-teacher mode needs no target array at all: the teacher supplies
    # both the output target and the intermediate features, from the LR crops.
    teacher = None
    if FKD_TEACHER:
        teacher = Student(FKD_TNF, FKD_TNC).to(dev).eval()
        tsd = torch.load(FKD_TEACHER, map_location="cpu", weights_only=True)
        tsd = tsd.get("sd", tsd)
        teacher.load_state_dict({k: v for k, v in tsd.items() if k.startswith("body.")})
        for p in teacher.parameters():
            p.requires_grad_(False)
    hr_f16 = (teacher is None) and os.path.exists(os.path.join(D, "hr_f16.npy"))
    if teacher is None:
        # f16 targets (teacher distillation) take precedence over u8 GT/targets
        hr_all = np.load(os.path.join(D, "hr_f16.npy" if hr_f16 else "hr_u8.npy"), mmap_mode="r")
        tgt_div = 1.0 if hr_f16 else 255.0
    else:
        hr_all, tgt_div = None, 1.0
    n = lr_all.shape[0] - 512
    val_lr = torch.from_numpy(lr_all[n:].copy()).float().permute(0, 3, 1, 2).div_(255).to(dev)
    if teacher is None:
        val_hr = torch.from_numpy(hr_all[n:].copy()).float().permute(0, 3, 1, 2).div_(tgt_div).to(dev)
    else:
        with torch.no_grad():
            val_hr = torch.cat([teacher(val_lr[i:i + 128]) for i in range(0, 512, 128)])
    # ZENSR_CPU_DATA=1: keep the full dataset host-side and ship per-step
    # batches (~2.3MB) — required on MPS (multi-GB single .to(mps) copies
    # hang in waitUntilCompleted) and on small-VRAM cards (ian's 1660 Ti).
    # Auto: keep data host-side when it would not comfortably fit in VRAM.
    # MPS always (multi-GB .to(mps) wedges); CUDA when the arrays exceed half
    # of free device memory (the 100k-crop set is 9.9 GB vs 8 GB cards).
    need_bytes = lr_all[:n].nbytes + (hr_all[:n].nbytes if hr_all is not None else 0)
    auto_cpu = dev == "mps"
    if dev == "cuda":
        free, _total = torch.cuda.mem_get_info()
        auto_cpu = need_bytes > free * 0.5
        print(f"dataset {need_bytes/1e9:.1f} GB, free VRAM {free/1e9:.1f} GB "
              f"-> {'host-side batches' if auto_cpu else 'device-resident'}", flush=True)
    cpu_data = os.environ.get("ZENSR_CPU_DATA", "1" if auto_cpu else "0") == "1"
    dloc = "cpu" if cpu_data else dev
    lr_gpu = torch.from_numpy(lr_all[:n].copy()).to(dloc)
    hr_gpu = None if hr_all is None else torch.from_numpy(hr_all[:n].copy()).to(dloc)
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
    # Tap indices, chosen by relative depth so the shallower student lines up
    # with proportional points in the deeper teacher.
    s_taps_s = prelu_taps(NC, FKD_TAPS) if FKD_W > 0 else ()
    s_taps_t = prelu_taps(FKD_TNC, FKD_TAPS) if teacher is not None else ()
    if teacher is not None:
        if SPACE == "ycbcr":
            raise SystemExit("ZENSR_FKD_TEACHER with ZENSR_LOSS_SPACE=ycbcr is "
                             "unsupported: the teacher expects RGB input")
        print(f"teacher online: {os.path.basename(FKD_TEACHER)} nf={FKD_TNF} nc={FKD_TNC} "
              f"fkd_w={FKD_W} pool={FKD_POOL} taps student={s_taps_s} teacher={s_taps_t}",
              flush=True)
    print(f"train n={n} steps={steps} batch={batch} lr={lr0} nf={NF} nc={NC} "
          f"params={sum(p.numel() for p in m.parameters())}", flush=True)
    for step in range(1, steps + 1):
        if sample_pool is not None:
            idx = sample_pool[torch.from_numpy(rng.integers(0, len(sample_pool), batch)).to(dloc)]
        else:
            idx = torch.from_numpy(rng.integers(0, n, batch)).to(dloc)
        x = to_space(lr_gpu[idx].to(dev).permute(0, 3, 1, 2).float().div_(255))
        if teacher is None:
            y = to_space(hr_gpu[idx].to(dev).permute(0, 3, 1, 2).float().div_(tgt_div))
        with torch.autocast(dev, dtype=amp_dt, enabled=amp_on):
            if teacher is not None:
                with torch.no_grad():
                    y, t_taps = teacher(x, s_taps_t)
            if FKD_W > 0:
                out, s_taps = m(x, s_taps_s)
                loss = charbonnier(out, y) + FKD_W * affinity_loss(s_taps, t_taps, FKD_POOL)
            else:
                out = m(x)
                loss = charbonnier(out, y)
        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        sched.step()
        if step % 500 == 0 or step == steps:
            with torch.no_grad(), torch.autocast(dev, dtype=amp_dt, enabled=amp_on):
                vps = []
                for i in range(0, 512, 128):
                    vo = m(to_space(val_lr[i:i + 128])).float()
                    mse = ((vo - to_space(val_hr[i:i + 128])) ** 2).mean().item()
                    vps.append(-10 * np.log10(max(mse, 1e-10)))
                # hr_f16.npy is written only by make_teacher_targets.py, so when
                # it is present the target — and therefore this metric — is the
                # TEACHER's output, not ground truth. Reporting it as "vs_GT"
                # overstated what was measured.
                print(f"step {step} loss {loss.item():.5f} "
                      f"val_psnr_vs_{'teacher' if hr_f16 or teacher is not None else 'GT'} "
                      f"{np.mean(vps):.2f}", flush=True)
        if step % CKPT_EVERY == 0 or step == steps:
            torch.save({"sd": getattr(m, "_orig_mod", m).state_dict(), "step": step},
                       os.path.join(D, f"{OUT_NAME}_{step}.pth"))

    sd = {k: v.float().cpu() for k, v in getattr(m, "_orig_mod", m).state_dict().items()}
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
    # f16 SHIP format (measured transparent: <=9e-4 output delta on dejpeg7,
    # ~1/4 of an 8-bit step; benchmarks 2026-07-27). Goldens below are
    # generated FROM the f16-roundtripped weights so the verify gate tests
    # exactly what ships.
    wf32 = np.frombuffer(b"".join(blobs), dtype="<f4")
    wf16 = wf32.astype("<f2")
    wf16.tofile(os.path.join(OUTM, "weights_f16.raw"))
    sd_ship = {}
    off = 0
    for k in [k for i in idxs for k in
              ([f"body.{i}.weight", f"body.{i}.bias"] if sd[f"body.{i}.weight"].dim() == 4
               else [f"body.{i}.weight"])]:
        n = int(np.prod(sd[k].shape))
        sd_ship[k] = torch.from_numpy(
            wf16[off:off + n].astype("<f4").reshape(tuple(sd[k].shape)).copy())
        off += n
    m2 = Student().cpu().eval()
    m2.load_state_dict(sd_ship)
    for (h, w) in [(40, 36), (17, 13)]:
        gi = (np.arange(3 * h * w, dtype=np.int64) % 251).astype("<f4") / 251.0
        x = torch.from_numpy(gi.reshape(1, 3, h, w).copy())
        with torch.no_grad():
            y = m2(x).numpy().astype("<f4")
        gi.tofile(os.path.join(OUTM, f"gold_in_{h}x{w}.raw"))
        y.tofile(os.path.join(OUTM, f"gold_out_{h}x{w}.raw"))
    total = sum(int(np.prod(v.shape)) for v in sd.values())
    # ---- reproducibility record (autogen; user directive 2026-07-27) ----
    import hashlib
    import platform
    import shlex
    import subprocess as sp

    def sha(p):
        try:
            return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
        except OSError:
            return None
    env = {k: v for k, v in os.environ.items() if k.startswith("ZENSR_")}
    ds_meta = None
    dm = os.path.join(D, "meta.json")
    if os.path.exists(dm):
        ds_meta = json.load(open(dm))
    commit = env.get("ZENSR_COMMIT") or (
        sp.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
               cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip() or None)
    repro = {
        "argv": sys.argv, "env": env, "seed_torch": 7, "seed_numpy": 11,
        "trainer": os.path.basename(__file__), "commit": commit,
        "host": platform.node(), "torch": torch.__version__,
        "device": dev, "amp_dtype": str(amp_dt) if amp_on else "fp32",
        "init_sha256_16": sha(INIT) if INIT else None,
        "dataset_dir": D, "dataset_meta": ds_meta,
        "weights_f16_sha256_16": sha(os.path.join(OUTM, "weights_f16.raw")),
    }
    json.dump({"arch": "compact", "scale": SCALE, "nf": NF, "nc": NC, "space": SPACE,
               "total_floats": int(total),
               "source": f"train_people (init={os.path.basename(INIT) or 'scratch'})",
               "repro": repro},
              open(os.path.join(OUTM, "meta.json"), "w"), indent=1)
    with open(os.path.join(OUTM, "repro.sh"), "w") as f:
        f.write("#!/bin/sh\n# autogenerated by train_people.py — exact retrain command\n")
        f.write(f"# commit {commit}; dataset {D} (its meta.json embedded in model meta)\n")
        envs = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items()))
        f.write(f"env {envs} python3 tools/train_people.py "
                + " ".join(shlex.quote(a) for a in sys.argv[1:]) + "\n")
    print("EXPORTED", total, "floats", flush=True)


if __name__ == "__main__":
    main()
