#!/usr/bin/env python3
"""Retain process means and reliability flags from a completed zenbench experiment."""

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_result(path, root):
    data = path.read_bytes()
    if data.startswith(b"\x1f\x8b"):
        data = gzip.decompress(data)
    value = json.loads(data)
    if value.get("zenbench_evidence") == 1:
        checksum = value["sha256"]
        packed = (root / "objects" / (checksum + ".json.gz")).read_bytes()
        if digest(packed) != checksum:
            raise ValueError("benchmark object hash differs")
        return json.loads(gzip.decompress(packed))
    return value


def summarize(experiment, runs):
    attempts = experiment["attempts"]
    required = experiment["plan"]["processes"]
    if experiment["status"] not in {"complete", "unconfirmed"} or len(runs) != required:
        raise ValueError("incomplete benchmark experiment")
    if len(attempts) != required or any(a["status"] != "completed" for a in attempts):
        raise ValueError("incomplete benchmark attempts")
    if any(r.get("incomplete", False) for r in runs):
        raise ValueError("incomplete process result")
    identities = {r.get("git_hash") for r in runs}
    if len(identities) != 1:
        raise ValueError("benchmark source identities differ")
    groups = collections.defaultdict(list)
    expected = {c["group_name"] for c in runs[0]["comparisons"]}
    for run in runs:
        if (
            len(run["comparisons"]) != len(expected)
            or {c["group_name"] for c in run["comparisons"]} != expected
        ):
            raise ValueError("process group coverage differs")
        for comparison in run["comparisons"]:
            groups[comparison["group_name"]].append(comparison)
    report = dict(
        experiment_status=experiment["status"],
        binary_sha256=experiment["binary_sha256"],
        plan=experiment["plan"],
        process_reliability=[
            dict(
                unreliable=r["unreliable"],
                gate_waits=r["gate_waits"],
                cpu_placement=r["cpu_placement"],
            )
            for r in runs
        ],
        interpretation="Observed process means; retain every process and gate flag. No clean confirmation is asserted by this report.",
        groups={},
    )
    for name, comparisons in groups.items():
        names = [b["name"] for b in comparisons[0]["benchmarks"]]
        if any([b["name"] for b in c["benchmarks"]] != names for c in comparisons):
            raise ValueError("process arm coverage differs")
        table = {}
        for index, arm in enumerate(names):
            values = [c["benchmarks"][index]["summary"]["mean"] for c in comparisons]
            if any(not (0 < v < float("inf")) for v in values):
                raise ValueError("invalid process mean")
            table[arm] = dict(
                process_means_ms=[v / 1e6 for v in values],
                mean_of_process_means_ms=statistics.fmean(values) / 1e6,
                completed_rounds=[c["completed_rounds"] for c in comparisons],
                retained_sample_counts=[
                    c["benchmarks"][index]["summary"]["n"] for c in comparisons
                ],
            )
        baseline = table["native"]["mean_of_process_means_ms"]
        for entry in table.values():
            entry["ratio_to_native"] = entry["mean_of_process_means_ms"] / baseline
        report["groups"][name] = table
    return report


def main(args):
    experiment = json.loads((args.processes / "experiment.json").read_text())
    runs = [
        read_result(args.processes / attempt["result"], args.root)
        for attempt in experiment["attempts"]
    ]
    report = summarize(experiment, runs)
    report["experiment_sha256"] = digest(
        (args.processes / "experiment.json").read_bytes()
    )
    report["report_tool_sha256"] = digest(Path(__file__).read_bytes())
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    for group, arms in report["groups"].items():
        print(
            group,
            {name: round(v["mean_of_process_means_ms"], 3) for name, v in arms.items()},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("processes", type=Path)
    parser.add_argument("output", type=Path)
    main(parser.parse_args())
