#!/usr/bin/env python3
"""Frozen quality-model PTQ/calibration and explicit integer ONNX exports."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import onnx
from onnx import helper as H, numpy_helper as N, TensorProto as T
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_people

BASELINE = "7ac1b1f9bf7952336fb27ad0132378fd02b5e77b6b6958ea4ced6612317f0b39"
ARMS = [
    "float",
    "weight-int8",
    "int8",
    "final-f32",
    "final2-f32",
    "integer-f32",
    "integer-final-f32",
    "integer-final2-f32",
]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        while b := f.read(1024 * 1024):
            h.update(b)
    return h.hexdigest()


def load(path):
    if sha(path) != BASELINE:
        raise ValueError("wrong quality baseline")
    train_people.SCALE = 1
    m = train_people.Student(64, 16).eval()
    raw = np.fromfile(path, dtype="<f2").astype("<f4")
    offset = 0
    with torch.no_grad():
        for p in m.parameters():
            n = p.numel()
            p.copy_(torch.from_numpy(raw[offset : offset + n].reshape(p.shape).copy()))
            offset += n
    assert offset == len(raw) == 595459
    return m, raw


def qparams(lo, hi):
    lo = min(float(lo), 0.0)
    hi = max(float(hi), 0.0)
    scale = np.float32(max(hi - lo, 1e-7) / 255)
    zero = np.uint8(np.clip(np.rint(-lo / scale), 0, 255))
    return scale, zero


def export(m, ranges, path, arm):
    nodes = []
    const = []
    current = "input"
    ci = 0

    def constant(name, value):
        const.append(N.from_array(np.asarray(value), name))
        return name

    def node(op, args, name, **attrs):
        nodes.append(H.make_node(op, args, [name], name=name, **attrs))
        return name

    convs = [v for v in m.body if isinstance(v, nn.Conv2d)]
    for layer in m.body:
        if isinstance(layer, nn.Conv2d):
            prefix = f"c{ci}"
            w = layer.weight.detach().cpu().numpy()
            b = layer.bias.detach().cpu().numpy()
            ws = np.maximum(
                np.max(np.abs(w), axis=(1, 2, 3)), np.finfo(np.float32).eps
            ).astype("f4") / np.float32(127)
            codes = np.clip(np.rint(w / ws[:, None, None, None]), -127, 127).astype(
                "i1"
            )
            integer = arm in [
                "int8",
                "final-f32",
                "final2-f32",
                "integer-f32",
                "integer-final-f32",
                "integer-final2-f32",
            ] and ci < len(convs) - (
                {
                    "final-f32": 1,
                    "final2-f32": 2,
                    "integer-final-f32": 1,
                    "integer-final2-f32": 2,
                }.get(arm, 0)
            )
            if integer:
                xs, xz = (
                    (np.float32(1 / 255), np.uint8(0))
                    if ci == 0
                    else qparams(*ranges[ci]["x"])
                )
                ys, yz = qparams(*ranges[ci]["y"])
                xsname = constant(prefix + "xs", xs)
                xzname = constant(prefix + "xz", xz)
                qx = node("QuantizeLinear", [current, xsname, xzname], prefix + "qx")
                if arm.startswith("integer-"):
                    output = node(
                        "ConvInteger",
                        [
                            qx,
                            constant(prefix + "wi8", codes),
                            xzname,
                            constant(prefix + "wz", np.int8(0)),
                        ],
                        prefix + "iconv",
                        pads=[1, 1, 1, 1],
                    )
                    current = node("Cast", [output], prefix + "accf32", to=T.FLOAT)
                    current = node(
                        "Mul",
                        [
                            current,
                            constant(prefix + "scale", (xs * ws).reshape(1, -1, 1, 1)),
                        ],
                        prefix + "scaled",
                    )
                    current = node(
                        "Add",
                        [current, constant(prefix + "bf32", b.reshape(1, -1, 1, 1))],
                        prefix + "biased",
                    )
                else:
                    qb = np.rint(b / (xs * ws)).astype("<i4")
                    output = node(
                        "QLinearConv",
                        [
                            qx,
                            xsname,
                            xzname,
                            constant(prefix + "wi8", codes),
                            constant(prefix + "ws", ws),
                            constant(prefix + "wz", np.zeros(len(ws), dtype="i1")),
                            constant(prefix + "ys", ys),
                            constant(prefix + "yz", yz),
                            constant(prefix + "bi32", qb),
                        ],
                        prefix + "qconv",
                        pads=[1, 1, 1, 1],
                    )
                    current = node(
                        "DequantizeLinear",
                        [output, prefix + "ys", prefix + "yz"],
                        prefix + "dq",
                    )
            else:
                if arm == "weight-int8":
                    w = codes.astype("f4") * ws[:, None, None, None]
                current = node(
                    "Conv",
                    [current, constant(prefix + "w", w), constant(prefix + "b", b)],
                    prefix + "conv",
                    pads=[1, 1, 1, 1],
                )
            ci += 1
        else:
            slope = layer.weight.detach().cpu().numpy().reshape(-1, 1, 1)
            current = node("PRelu", [current, constant(f"p{ci}a", slope)], f"p{ci}")
    node("Add", [current, "input"], "output")
    graph = H.make_graph(
        nodes,
        arm,
        [H.make_tensor_value_info("input", T.FLOAT, [1, 3, "H", "W"])],
        [H.make_tensor_value_info("output", T.FLOAT, [1, 3, "H", "W"])],
        const,
    )
    model = H.make_model(
        graph,
        opset_imports=[H.make_opsetid("", 13)],
        producer_name="zensr-quality-int8",
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return sum(n.op_type in ["QLinearConv", "ConvInteger"] for n in nodes)


def main(a):
    if not torch.cuda.is_available():
        raise ValueError("CUDA calibration explicitly required")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m, raw = load(a.weights)
    a.output.mkdir()
    raw.tofile(a.output / "weights.raw")
    meta = json.loads((a.dataset / "meta.json").read_text())
    rows = list(csv.DictReader((a.dataset / "pairs.tsv").open(), delimiter="\t"))
    train = len(rows) - int(meta["val_tail"])
    indices = np.linspace(0, train - 1, 256, dtype=np.int64)
    assert all(rows[i]["canonical_split"] == "train" for i in indices)
    splits = {
        r["id"]: r["split"] for r in csv.DictReader(a.split_map.open(), delimiter="\t")
    }
    assert all(splits[rows[i]["imazen_id"]] == "train" for i in indices)
    data = np.load(a.dataset / "lr_u8.npy", mmap_mode="r")
    ranges = [
        dict(x=[float("inf"), -float("inf")], y=[float("inf"), -float("inf")])
        for _ in range(18)
    ]
    m = m.cuda()
    with torch.no_grad():
        for start in range(0, len(indices), 16):
            x = (
                torch.tensor(
                    data[indices[start : start + 16]].copy(),
                    device="cuda",
                    dtype=torch.float32,
                )
                .permute(0, 3, 1, 2)
                .contiguous()
                / 255
            )
            ci = 0
            for layer in m.body:
                if isinstance(layer, nn.Conv2d):
                    for key, v in [("x", x), ("y", layer(x))]:
                        ranges[ci][key][0] = min(ranges[ci][key][0], v.min().item())
                        ranges[ci][key][1] = max(ranges[ci][key][1], v.max().item())
                        if key == "y":
                            y = v
                    x = y
                    ci += 1
                else:
                    x = layer(x)
            print(
                f"calibration {min(start + 16, len(indices))}/{len(indices)}",
                flush=True,
            )
    m = m.cpu()
    exports = {
        arm: dict(
            quantized_convolutions=export(m, ranges, a.output / (arm + ".onnx"), arm),
            sha256=sha(a.output / (arm + ".onnx")),
        )
        for arm in ARMS
    }
    gold = np.fromfile(a.gold_input, dtype="<f4").reshape(1, 3, 17, 13)
    with torch.no_grad():
        out = m(torch.tensor(gold.copy())).numpy()
    gold.tofile(a.output / "gold-input.raw")
    out.astype("<f4").tofile(a.output / "gold-float.raw")
    plan = dict(
        schema=1,
        baseline_sha256=BASELINE,
        parameters=595459,
        nf=64,
        nc=16,
        scale=1,
        calibration_indices=indices.tolist(),
        calibration_sources=sorted({rows[i]["imazen_id"] for i in indices}),
        train_rows=train,
        ranges=ranges,
        exports=exports,
        torch=torch.__version__,
        onnx=onnx.__version__,
        numpy=np.__version__,
        source_sha256=sha(__file__),
        dataset_hashes={
            n: sha(a.dataset / n) for n in ["meta.json", "pairs.tsv", "lr_u8.npy"]
        },
        split_map_sha256=sha(a.split_map),
        quantization="explicit QLinearConv, uint8 activations, per-channel signed int8 weights, int32 bias, float PReLU/residual",
        selection="fixed 256 evenly-spaced optimizer rows; min/max includes zero; no evaluation scores",
    )
    (a.output / "PLAN.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(exports, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ["weights", "dataset", "split-map", "output", "gold-input"]:
        p.add_argument("--" + n, type=Path, required=True)
    main(p.parse_args())
