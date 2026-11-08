#!/usr/bin/env python3
"""Share identical pure graph nodes for quality scoring; keep timed models intact."""

import argparse, copy, hashlib, json, shutil
from pathlib import Path
import onnx
from onnx import helper as H, numpy_helper as N, TensorProto as T


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    a.output.mkdir()
    old = json.loads((a.source / "PLAN.json").read_text())
    models = list(old["exports"])
    for n in ["weights.raw", "gold-input.raw", "gold-float.raw"]:
        shutil.copyfile(a.source / n, a.output / n)
    nodes = []
    initializers = []
    outputs = []
    const_cache = {}
    node_cache = {}
    output_indices = {}
    layout = {}
    audited = {}

    def add_output(name, info):
        if name not in output_indices:
            output_indices[name] = len(outputs)
            v = copy.deepcopy(info)
            v.name = name
            outputs.append(v)
        return output_indices[name]

    for arm in models:
        path = a.source / (arm + ".onnx")
        assert sha(path) == old["exports"][arm]["sha256"]
        shutil.copyfile(path, a.output / path.name)
        model = onnx.load(a.source / (arm + "-audit.onnx"))
        # 64 * 1024^2 <= 2^26; int32 sums are exact throughout this study.
        for node in model.graph.node:
            if (
                node.op_type == "Cast"
                and node.output[0].startswith("audit")
                and node.output[0].endswith("countable")
            ):
                next(v for v in node.attribute if v.name == "to").i = T.INT32
        for output in model.graph.output:
            if output.name.startswith("audit"):
                output.type.tensor_type.elem_type = T.INT32
        onnx.checker.check_model(model)
        onnx.save(model, a.output / (arm + "-audit.onnx"))
        audited[arm] = dict(
            sha256=sha(a.output / (arm + "-audit.onnx")),
            range_checks=len(model.graph.output) - 1,
        )
        names = {"input": "input"}
        for item in model.graph.initializer:
            array = N.to_array(item)
            key = (
                array.dtype.str,
                tuple(array.shape),
                hashlib.sha256(array.tobytes()).digest(),
            )
            if key not in const_cache:
                name = "constant_" + str(len(initializers))
                const_cache[key] = name
                v = copy.deepcopy(item)
                v.name = name
                initializers.append(v)
            names[item.name] = const_cache[key]
        for node in model.graph.node:
            args = tuple(names[x] if x else "" for x in node.input)
            key = (
                node.domain,
                node.op_type,
                args,
                tuple(v.SerializeToString() for v in node.attribute),
                len(node.output),
            )
            if key not in node_cache:
                names_out = tuple(
                    "value_" + str(len(nodes)) + "_" + str(i)
                    for i in range(len(node.output))
                )
                n = copy.deepcopy(node)
                n.name = "operation_" + str(len(nodes))
                del n.input[:]
                n.input.extend(args)
                del n.output[:]
                n.output.extend(names_out)
                nodes.append(n)
                node_cache[key] = names_out
            for before, after in zip(node.output, node_cache[key]):
                names[before] = after
        mapped = [add_output(names[v.name], v) for v in model.graph.output]
        layout[arm] = dict(pixels=mapped[0], counts=mapped[1:])
    graph = H.make_graph(
        nodes,
        "shared-quality-only",
        [H.make_tensor_value_info("input", T.FLOAT, [1, 3, "H", "W"])],
        outputs,
        initializers,
    )
    model = H.make_model(
        graph,
        opset_imports=[H.make_opsetid("", 13)],
        producer_name="zensr-quality-shared-prefix",
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, a.output / "joint.onnx")
    (a.output / "joint-map.json").write_text(json.dumps(layout, indent=2) + "\n")
    plan = {
        **old,
        "previous_plan_sha256": sha(a.source / "PLAN.json"),
        "audited_exports": audited,
        "counter_type": "int32; per-map pixel*channels <= i32::MAX enforced by evaluator",
        "joint_sha256": sha(a.output / "joint.onnx"),
        "joint_map_sha256": sha(a.output / "joint-map.json"),
        "combine_tool_sha256": sha(__file__),
        "quality_execution": "Identical pure graph nodes shared. Original timed ONNX files remain byte-identical. Independently executed output parity is mandatory before scoring.",
    }
    (a.output / "PLAN.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                models=len(models), joint_nodes=len(nodes), joint_outputs=len(outputs)
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
