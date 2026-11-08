#!/usr/bin/env python3
"""Compare frozen Tract golden outputs with ONNX's independent reference evaluator."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from onnx.version_converter import convert_version


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    plan = json.loads((args.exports / "PLAN.json").read_text())
    source = args.exports / "gold-input.raw"
    x = np.fromfile(source, dtype="<f4").reshape(1, 3, 17, 13)
    args.output.mkdir()
    rows = {}
    for name, metadata in plan["exports"].items():
        path = args.exports / (name + ".onnx")
        if sha(path) != metadata["sha256"]:
            raise ValueError("model graph differs from frozen plan")
        model = onnx.load(path)
        if len(model.graph.input) != 1:
            raise ValueError("expected one RGB input")
        # ONNX 1.23.2's reference quantizers start at opset 19. The official
        # converter adds inferred value metadata for these graphs. Require the
        # actual computation, constants and interfaces to remain unchanged.
        reference_model = convert_version(model, 19)
        for field in ["node", "initializer", "input", "output"]:
            if list(getattr(model.graph, field)) != list(
                getattr(reference_model.graph, field)
            ):
                raise ValueError("opset adaptation changed model computation")
        y = ReferenceEvaluator(reference_model).run(
            None, {model.graph.input[0].name: x}
        )[0]
        actual_path = args.tract / (name + "-gold.raw")
        actual = np.fromfile(actual_path, dtype="<f4").reshape(x.shape)
        if y.shape != x.shape or y.dtype != np.float32:
            raise ValueError("reference output geometry/type differs")
        if not np.isfinite(y).all() or not np.isfinite(actual).all():
            raise ValueError("nonfinite model output")
        delta = np.abs(y.astype(np.float64) - actual.astype(np.float64))
        maximum = float(delta.max())
        if name == "float" and maximum > 5e-4:
            raise ValueError("float reference exceeds existing golden ceiling")

        def rgb8(value):
            return np.floor(
                np.clip(value, 0, 1) * np.float32(255) + np.float32(.5)
            ).astype(np.uint8)

        codes = np.abs(rgb8(y).astype(np.int16) - rgb8(actual).astype(np.int16))
        output = args.output / (name + "-reference.raw")
        y.astype("<f4").tofile(output)
        rows[name] = dict(
            model_sha256=sha(path),
            reference_model_sha256=hashlib.sha256(
                reference_model.SerializeToString()
            ).hexdigest(),
            tract_output_sha256=sha(actual_path),
            reference_output_sha256=sha(output),
            values=y.size,
            differing_f32_bits=int(np.count_nonzero(y.view(np.uint32) != actual.view(np.uint32))),
            max_absolute_error=maximum,
            mean_absolute_error=float(delta.mean()),
            differing_rgb8_values=int(np.count_nonzero(codes)),
            max_rgb8_code_error=int(codes.max()),
        )
        print(name, "max normalized error", maximum,
              "max RGB8 error", int(codes.max()), flush=True)
    report = dict(
        evaluator="onnx.reference.ReferenceEvaluator",
        exported_opset=13,
        reference_opset=19,
        adaptation="Official version conversion; nodes, constants and input/output declarations must remain identical. Only inferred value metadata and the opset declaration may change.",
        onnx_version=onnx.__version__,
        numpy_version=np.__version__,
        python=sys.version,
        input_shape=list(x.shape),
        input_sha256=sha(source),
        plan_sha256=sha(args.exports / "PLAN.json"),
        tool_sha256=sha(Path(__file__)),
        arms=rows,
        interpretation="Independent-backend diagnostic on the fixed golden input. Integer discrepancies are reported without treating them as an accuracy pass. This does not replace the complete Tract cohort or certify unseen inputs.",
    )
    (args.output / "REFERENCE-COMPARISON.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["exports", "tract", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
