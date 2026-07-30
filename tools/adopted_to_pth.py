#!/usr/bin/env python3
"""adopted model dir -> torch state_dict .pth (inverse of train_people's export).

Lets any box use a shipped model as a distillation teacher without needing
the original training checkpoint. Usage: adopted_to_pth.py <dir> <out.pth>
"""
import json
import os
import sys

import numpy as np
import torch

d, out = sys.argv[1], sys.argv[2]
meta = json.load(open(os.path.join(d, "meta.json")))
os.environ.update(ZENSR_NF=str(meta["nf"]), ZENSR_NC=str(meta["nc"]),
                  ZENSR_SCALE=str(meta["scale"]))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_people as tp  # noqa: E402

w = np.fromfile(os.path.join(d, "weights.raw"), dtype="<f4")
m = tp.Student()
sd, off = m.state_dict(), 0
for k in sd:
    n = int(np.prod(sd[k].shape))
    sd[k] = torch.from_numpy(w[off:off + n].reshape(tuple(sd[k].shape)).copy())
    off += n
assert off == len(w), f"layout mismatch {off} vs {len(w)}"
m.load_state_dict(sd)
torch.save({"sd": sd, "step": -1}, out)
print(f"wrote {out} ({len(w)} params, nf={meta['nf']} nc={meta['nc']})")
