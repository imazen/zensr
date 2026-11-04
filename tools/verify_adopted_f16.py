#!/usr/bin/env python3
"""Verify an adopted compact model's f16 roundtrip and PyTorch goldens.

Usage: python3 tools/verify_adopted_f16.py models/adopted/<model>
Reads the local f32 source weights to check the export. It writes no files.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def main(path):
    d = Path(path)
    meta = json.loads((d / "meta.json").read_text())
    if meta["arch"] != "compact":
        raise ValueError("expected compact model")
    os.environ.update(ZENSR_NF=str(meta["nf"]), ZENSR_NC=str(meta["nc"]), ZENSR_SCALE=str(meta["scale"]))
    import train_people

    w32 = np.fromfile(d / "weights.raw", dtype="<f4")
    w16 = np.fromfile(d / "weights_f16.raw", dtype="<f2")
    if len(w32) != meta["total_floats"] or not np.array_equal(w32.astype("<f2"), w16):
        raise ValueError("f16 weights do not match the f32 roundtrip")
    model = train_people.Student().cpu().eval()
    state = model.state_dict()
    offset = 0
    for key, tensor in state.items():
        count = tensor.numel()
        part = w16[offset:offset + count].astype("<f4").reshape(tuple(tensor.shape)).copy()
        state[key] = torch.from_numpy(part)
        offset += count
    if offset != len(w16):
        raise ValueError(f"weight layout has {offset} values, file has {len(w16)}")
    model.load_state_dict(state)
    for h, w in ((40, 36), (17, 13)):
        inp = np.fromfile(d / f"gold_in_{h}x{w}.raw", dtype="<f4")
        expected_in = (np.arange(3 * h * w, dtype=np.int64) % 251).astype("<f4") / 251.0
        if not np.array_equal(inp, expected_in):
            raise ValueError(f"unexpected golden input {h}x{w}")
        gold = np.fromfile(d / f"gold_out_{h}x{w}.raw", dtype="<f4")
        with torch.no_grad():
            actual = model(torch.from_numpy(inp.reshape(1, 3, h, w).copy())).numpy().reshape(-1)
        if len(gold) != len(actual) or not np.isfinite(gold).all() or not np.isfinite(actual).all():
            raise ValueError(f"invalid golden output {h}x{w}")
        error = float(np.max(np.abs(actual - gold)))
        print(f"{h}x{w}: max_abs_error={error:.9g}")
        if error >= 5e-4:
            raise ValueError(f"golden output mismatch: {h}x{w}")
    print("f16 roundtrip and PyTorch goldens verified")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
