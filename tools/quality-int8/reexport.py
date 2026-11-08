#!/usr/bin/env python3
"""Reuse exactly frozen calibration to compare integer graph representations."""

import argparse, json, shutil
from pathlib import Path
from export import load, export, sha, ARMS
import onnx
from onnx import helper as H, numpy_helper as N, TensorProto as T
import numpy as np


def audit(path, destination):
    model = onnx.load(path)
    constants = {v.name: N.to_array(v) for v in model.graph.initializer}
    nodes = []
    count = 0
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear":
            prefix = "audit" + str(count)
            scale = float(constants[node.input[1]])
            zero = float(constants[node.input[2]])
            for suffix, value in [
                ("lo", np.float32(-zero * scale)),
                ("hi", np.float32((255 - zero) * scale)),
                ("axes", np.array([0, 1, 2, 3], dtype="i8")),
            ]:
                model.graph.initializer.append(
                    N.from_array(np.asarray(value), prefix + suffix)
                )
            nodes.extend(
                [
                    H.make_node(
                        "Less", [node.input[0], prefix + "lo"], [prefix + "below"]
                    ),
                    H.make_node(
                        "Greater", [node.input[0], prefix + "hi"], [prefix + "above"]
                    ),
                    H.make_node(
                        "Or", [prefix + "below", prefix + "above"], [prefix + "outside"]
                    ),
                    H.make_node(
                        "Cast", [prefix + "outside"], [prefix + "countable"], to=T.INT32
                    ),
                    H.make_node(
                        "ReduceSum",
                        [prefix + "countable", prefix + "axes"],
                        [prefix + "count"],
                        keepdims=0,
                    ),
                ]
            )
            model.graph.output.append(
                H.make_tensor_value_info(prefix + "count", T.INT32, [])
            )
            count += 1
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save(model, destination)
    return dict(range_checks=count, sha256=sha(destination))


def main(a):
    original = json.loads((a.calibration / "PLAN.json").read_text())
    if (
        sha(a.calibration / "PLAN.json")
        != "1aa0b0855ef329df1066a454a4a4eb67696cd0e6e42a7ceb917d8a46aae50668"
    ):
        raise ValueError("wrong frozen calibration")
    m, raw = load(a.weights)
    a.output.mkdir()
    raw.tofile(a.output / "weights.raw")
    for n in ["gold-input.raw", "gold-float.raw"]:
        shutil.copyfile(a.calibration / n, a.output / n)
    exports = {
        arm: dict(
            quantized_convolutions=export(
                m, original["ranges"], a.output / (arm + ".onnx"), arm
            ),
            sha256=sha(a.output / (arm + ".onnx")),
        )
        for arm in ARMS
    }
    for arm in ["float", "weight-int8", "int8", "final-f32", "final2-f32"]:
        assert exports[arm] == original["exports"][arm], (
            "original comparison graph changed"
        )
    audited = {
        arm: audit(a.output / (arm + ".onnx"), a.output / (arm + "-audit.onnx"))
        for arm in ARMS
    }
    plan = {
        **original,
        "exports": exports,
        "audited_exports": audited,
        "original_calibration_plan_sha256": sha(a.calibration / "PLAN.json"),
        "export_source_sha256": sha(Path(__file__).with_name("export.py")),
        "reexport_source_sha256": sha(__file__),
        "amendment": "Add ConvInteger with float accumulator scaling/bias/PReLU; quantize once per block, avoiding QLinearConv output quantization before activation. Original five exports remain byte-identical. Added after initial timing, before inspecting pilot quality scores.",
    }
    (a.output / "PLAN.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(exports, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ["weights", "calibration", "output"]:
        p.add_argument("--" + n, type=Path, required=True)
    main(p.parse_args())
