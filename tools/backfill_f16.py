#!/usr/bin/env python3
"""Backfill weights_f16.raw + REGENERATED goldens for existing adopted dirs.

Goldens must test the SHIP (f16) path: torch-forward the f16-roundtripped
weights and overwrite gold_out_*. Only for arch=compact dirs whose goldens
this trainer lineage produced; span48 dirs keep f32 (their dump pipeline
owns them). Usage: backfill_f16.py [models/adopted]
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

root = sys.argv[1] if len(sys.argv) > 1 else "models/adopted"
for name in sorted(os.listdir(root)):
    d = os.path.join(root, name)
    mj = os.path.join(d, "meta.json")
    if not os.path.isfile(mj):
        continue
    meta = json.load(open(mj))
    if meta.get("arch") != "compact":
        print(f"skip {name} (arch={meta.get('arch')})")
        continue
    w = np.fromfile(os.path.join(d, "weights.raw"), dtype="<f4")
    w16 = w.astype("<f2")
    w16.tofile(os.path.join(d, "weights_f16.raw"))
    # rebuild torch model from f16-roundtripped weights, regen goldens
    os.environ["ZENSR_NF"] = str(meta["nf"])
    os.environ["ZENSR_NC"] = str(meta["nc"])
    os.environ["ZENSR_SCALE"] = str(meta["scale"])
    import importlib
    import train_people
    importlib.reload(train_people)
    m = train_people.Student().cpu().eval()
    sd = m.state_dict()
    off = 0
    for k in sd:
        n = int(np.prod(sd[k].shape))
        sd[k] = torch.from_numpy(w16[off:off + n].astype("<f4").reshape(tuple(sd[k].shape)).copy())
        off += n
    assert off == len(w), f"{name}: layout mismatch {off} vs {len(w)}"
    m.load_state_dict(sd)
    for (h, wd) in [(40, 36), (17, 13)]:
        gi = (np.arange(3 * h * wd, dtype=np.int64) % 251).astype("<f4") / 251.0
        with torch.no_grad():
            y = m(torch.from_numpy(gi.reshape(1, 3, h, wd).copy())).numpy().astype("<f4")
        y.tofile(os.path.join(d, f"gold_out_{h}x{wd}.raw"))
    print(f"backfilled {name}")
