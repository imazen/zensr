#!/usr/bin/env python3
"""Package frozen quality-model exports and complete comparison evidence, without pixels."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_archive import make_archive
from qat.package import sha, verify_archive

BASELINE = "7ac1b1f9bf7952336fb27ad0132378fd02b5e77b6b6958ea4ced6612317f0b39"


def main(args):
    root = args.root
    evaluation = root / "evaluation-v6"
    if (evaluation / "COMPLETE").read_text() != sha(evaluation / "results.tsv"):
        raise ValueError("quality evaluation is incomplete or changed")
    identity = json.loads((evaluation / "IDENTITY.json").read_text())
    if identity["required_inputs"] != 2960:
        raise ValueError("complete all 2960 quality inputs before packaging")
    summary = json.loads(args.summary.read_text())
    if summary["evaluation_identity"] != identity or summary[
        "input_manifest_sha256"
    ] != sha(args.inputs):
        raise ValueError("quality summary or input identity differs")
    if summary["results_sha256"] != sha(evaluation / "results.tsv"):
        raise ValueError("quality summary does not match results")
    if sha(root / "model/weights_f16.raw") != BASELINE:
        raise ValueError("wrong retained baseline")
    if len(args.latency) != 5:
        raise ValueError(
            "include both hosts, both threaded runs and the additional integer run"
        )
    if len({p.name for p in args.latency}) != 5:
        raise ValueError("latency report names must be distinct")
    for path in args.latency:
        evidence = json.loads(path.read_text())
        if len(evidence["process_reliability"]) != 3:
            raise ValueError("latency report lacks three independent processes")
    args.output.mkdir()
    payload = args.output / "payload"
    payload.mkdir()

    def copy(source, relative):
        destination = payload / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    exports = root / "exports-v2"
    if (
        sha(exports / "PLAN.json")
        != "4494de692405aafca0940a9ee308b165c51a3a99124710d3bd449a06ecb5c648"
    ):
        raise ValueError("wrong frozen export plan")
    plan = json.loads((exports / "PLAN.json").read_text())
    shared = root / "exports-shared-v1"
    if sha(shared / "PLAN.json") != identity["plan_sha256"]:
        raise ValueError("shared evaluation graph plan differs")
    shared_plan = json.loads((shared / "PLAN.json").read_text())
    retained_auxiliary = []
    auxiliary = [
        (exports / (name + "-audit.onnx"), metadata["sha256"])
        for name, metadata in plan["audited_exports"].items()
    ] + [
        (shared / "PLAN.json", identity["plan_sha256"]),
        (shared / "joint.onnx", identity["joint_sha256"]),
        (shared / "joint-map.json", shared_plan["joint_map_sha256"]),
    ]
    for path, expected_hash in auxiliary:
        if sha(path) != expected_hash:
            raise ValueError("retained auxiliary graph identity differs")
        retained_auxiliary.append(
            dict(
                path=str(path.relative_to(root)),
                bytes=path.stat().st_size,
                sha256=expected_hash,
                uploaded=False,
            )
        )
    catalog = []
    for name, metadata in plan["exports"].items():
        source = exports / (name + ".onnx")
        if sha(source) != metadata["sha256"]:
            raise ValueError("model graph changed")
        copy(source, "models/" + source.name)
        catalog.append(dict(name=name, bytes=source.stat().st_size, **metadata))
    for name in ["weights.raw", "gold-input.raw", "gold-float.raw", "PLAN.json"]:
        copy(exports / name, "models/" + name)
    for name in ["PLAN.json", "gold-input.raw", "gold-float.raw"]:
        copy(root / "exports-v1" / name, "calibration/" + name)
    copy(root / "model/weights_f16.raw", "baseline/weights_f16.raw")
    copy(root / "model/weights.raw", "historical-reference/original-unrounded-f32.raw")
    copy(root / "model/meta.json", "historical-reference/meta.json")
    for name in ["results.tsv", "COMPLETE", "IDENTITY.json"]:
        copy(evaluation / name, "quality/" + name)
    copy(args.summary, "quality/SUMMARY.json")
    copy(root / "rgb-grid-i270.json", "quality/RGB8-INPUT-QUANTIZER.json")
    copy(args.inputs, "quality/INPUTS-MANIFEST.tsv")
    for path in args.latency:
        copy(path, "latency/" + path.name)
    with args.inputs.open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = [
            name for name in reader.fieldnames if name not in {"jpeg", "reference_rgb8"}
        ]
        with (payload / "EVALUATION-INPUT-CATALOG.tsv").open("w") as destination:
            writer = csv.DictWriter(
                destination,
                fieldnames=fields,
                delimiter="\t",
                lineterminator="\n",
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(reader)
    inventory = dict(
        baseline_sha256=BASELINE,
        models=catalog,
        layout="ONNX opset 13, RGB float NCHW [1,3,H,W], x1 output with input residual already added; guard/projection outside model",
        status="Frozen PTQ format comparison; historical model provenance limits remain; no QAT or serving-default adoption",
        historical_reference="Original unrounded f32 weights included for retention; not used by this experiment",
        retained_auxiliary_graphs=retained_auxiliary,
        not_uploaded=[
            "JPEG/reference pixels",
            "raw profiler captures and machine logs",
            "auxiliary shared-prefix and range-counter graphs",
            "latency raw-round objects retained in the experiment archive",
        ],
    )
    name = "QUALITY-INT8-20261108.json"
    (args.output / name).write_text(json.dumps(inventory, indent=2) + "\n")
    copy(args.output / name, "MODEL-CATALOG.json")
    copy(args.readme, "README.md")
    shutil.copyfile(args.readme, args.output / "QUALITY-INT8-README.md")
    for license_name in ["LICENSE-AGPL3", "LICENSE-COMMERCIAL"]:
        copy(Path(license_name), license_name)
    names = [
        str(p.relative_to(payload)) for p in sorted(payload.rglob("*")) if p.is_file()
    ]
    archive_name = "quality-int8-ptq-20261108.tar.zst"
    artifact = make_archive(payload, args.output / archive_name, names)
    verify_archive(args.output / archive_name, artifact["members"])
    (args.output / "archive-verification.json").write_text(
        json.dumps(artifact, indent=2) + "\n"
    )
    assets = [archive_name, name, "QUALITY-INT8-README.md"]
    (args.output / "QUALITY-INT8-SHA256SUMS").write_text(
        "".join(sha(args.output / n) + "  " + n + "\n" for n in assets)
    )
    print(
        json.dumps(
            dict(
                archive_bytes=artifact["bytes"],
                verified_members=len(names),
                onnx_models=len(catalog),
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "summary", "inputs", "output", "readme"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--latency", type=Path, action="append", required=True)
    main(parser.parse_args())
