#!/usr/bin/env python3
"""Summarize every frozen quality/precision arm without selecting by test output."""

import argparse
import collections
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qat.report import limit, balanced, interval

ARMS = [
    "native",
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
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(root):
    if (root / "COMPLETE").read_text() != sha(root / "results.tsv"):
        raise ValueError("incomplete/changed evaluation")
    rows = list(csv.DictReader((root / "results.tsv").open(), delimiter="\t"))
    for r in rows:
        for k in [
            "identity_ssim2",
            "raw_ssim2",
            "projected_ssim2",
            "raw_mae",
            "raw_max",
            "projected_mae",
            "projected_max",
            "raw_butter_max",
            "projected_butter_max",
            "raw_butter_pnorm3",
            "projected_butter_pnorm3",
            "input_rgb8_roundtrip_max",
        ]:
            if not Decimal(r[k]).is_finite():
                raise ValueError("nonfinite measurement")
    return rows


def summarize(rows, mode):
    arms = collections.defaultdict(list)
    for r in rows:
        arms[r["arm"]].append(r)
    if set(arms) != set(ARMS):
        raise ValueError("arm coverage differs")
    native = {r["id"]: r for r in arms["native"]}
    float_control = {r["id"]: r for r in arms["float"]}
    if len(native) != len(arms["native"]):
        raise ValueError("duplicate baseline")
    result = {}
    for name, values in arms.items():
        if len(values) != len(native) or {r["id"] for r in values} != set(native):
            raise ValueError("paired coverage differs")
        delta = []
        float_delta = []
        gain = []
        harm = []
        added = []
        removed = []
        losses = []
        changed = gt1 = gt2 = count = outside = 0
        mae = []
        butter_max = []
        butter_pnorm = []
        range_count = range_values = 0
        for r in values:
            b = native[r["id"]]
            f = float_control[r["id"]]
            for key in [
                "origin",
                "identity_ssim2",
                "encoder",
                "ss",
                "q",
                "width",
                "height",
                "values",
            ]:
                if r[key] != b[key]:
                    raise ValueError("paired metadata differs")
            if int(r["values"]) != 3 * int(r["width"]) * int(r["height"]):
                raise ValueError("RGB channel-value denominator differs from geometry")
            counts = [
                int(r[mode + s]) for s in ["_rgb8_gt2", "_rgb8_gt1", "_rgb8_changed"]
            ] + [int(r["values"])]
            if counts != sorted(counts) or counts[0] < 0:
                raise ValueError("invalid RGB error histogram")
            o = r["origin"]
            actual = Decimal(r[mode + "_ssim2"])
            baseline = Decimal(b[mode + "_ssim2"])
            identity = Decimal(r["identity_ssim2"])
            delta.append((o, float(actual - baseline)))
            float_delta.append((o, float(actual - Decimal(f[mode + "_ssim2"]))))
            gain.append((o, float(actual - identity)))
            h = actual - identity < -limit(identity)
            hb = baseline - identity < -limit(identity)
            harm.append((o, float(h)))
            added.append((o, float(h and not hb)))
            removed.append((o, float(hb and not h)))
            losses.append(
                dict(
                    delta=float(actual - baseline),
                    gain_vs_decode=float(actual - identity),
                    id=r["id"],
                    source_id=r["source_id"],
                    encoder=r["encoder"],
                    ss=r["ss"],
                    q=r["q"],
                )
            )
            changed += int(r[mode + "_rgb8_changed"])
            gt1 += int(r[mode + "_rgb8_gt1"])
            gt2 += int(r[mode + "_rgb8_gt2"])
            count += int(r["values"])
            mae.append((o, float(r[mode + "_mae"])))
            range_count += int(r["activation_range_exceedances"])
            range_values += int(r["activation_values"])
            outside += int(r[mode + "_outside01"])
            butter_max.append(
                (
                    o,
                    float(
                        Decimal(r[mode + "_butter_max"])
                        - Decimal(b[mode + "_butter_max"])
                    ),
                )
            )
            butter_pnorm.append(
                (
                    o,
                    float(
                        Decimal(r[mode + "_butter_pnorm3"])
                        - Decimal(b[mode + "_butter_pnorm3"])
                    ),
                )
            )
        result[name] = dict(
            outside01_values=outside,
            outside01_fraction=outside / count,
            mean_vs_native_butter_max=statistics.fmean(balanced(butter_max).values()),
            mean_vs_native_butter_pnorm3=statistics.fmean(
                balanced(butter_pnorm).values()
            ),
            worst_gain_vs_decode=min(v for _, v in gain),
            worst_delta_vs_native=min(v for _, v in delta),
            worst_cases_vs_decode=sorted(losses, key=lambda r: r["gain_vs_decode"])[
                :10
            ],
            cells=len(values),
            origins=len(balanced(delta)),
            mean_vs_native_ssim2=statistics.fmean(balanced(delta).values()),
            paired_origin_95_vs_native=interval(balanced(delta)),
            mean_vs_tract_float_ssim2=statistics.fmean(balanced(float_delta).values()),
            mean_gain_vs_decode=statistics.fmean(balanced(gain).values()),
            harm_cases=sum(int(v) for _, v in harm),
            harm_fraction=statistics.fmean(balanced(harm).values()),
            additional_harm_cases=sum(int(v) for _, v in added),
            removed_harm_cases=sum(int(v) for _, v in removed),
            rgb8_values=count,
            rgb8_code_error_histogram={
                "0": count - changed,
                "1": changed - gt1,
                "2": gt1 - gt2,
                "3_or_more": gt2,
            },
            rgb8_changed_fraction=changed / count,
            rgb8_gt1_fraction=gt1 / count,
            rgb8_max_code_error=max(int(r[mode + "_rgb8_max"]) for r in values),
            float_mean_absolute_error=statistics.fmean(balanced(mae).values()),
            float_max_absolute_error=max(float(r[mode + "_max"]) for r in values),
            activation_range_exceedances=range_count,
            activation_values=range_values,
            activation_range_exceedance_fraction=range_count / range_values
            if range_values
            else None,
            worst_cases=sorted(losses, key=lambda r: r["delta"])[:10],
        )
    return result


def validate_manifest(rows, manifest, expected_inputs):
    expected = {r["id"]: r for r in manifest}
    if len(manifest) != len(expected) or len(expected) != expected_inputs:
        raise ValueError("input manifest required coverage differs")
    for row in rows:
        if row["id"] not in expected:
            raise ValueError("unplanned evaluation input")
        entry = expected[row["id"]]
        for key in [
            "source_id",
            "origin",
            "content",
            "cohort",
            "size",
            "encoder",
            "ss",
            "q",
            "width",
            "height",
        ]:
            if row[key] != entry[key]:
                raise ValueError(
                    "evaluation metadata/geometry differs from input manifest"
                )
    if len(rows) != expected_inputs * len(ARMS):
        raise ValueError("required study coverage missing")


def main(a):
    rows = read(a.root)
    identity = json.loads((a.root / "IDENTITY.json").read_text())
    if sha(a.inputs) != identity["inputs_sha256"]:
        raise ValueError("input manifest differs from evaluation identity")
    validate_manifest(
        rows, list(csv.DictReader(a.inputs.open(), delimiter="\t")), a.expected_inputs
    )
    report = dict(
        evaluation_identity=identity,
        input_manifest_sha256=sha(a.inputs),
        shared_report_helper_sha256=sha(
            Path(__file__).resolve().parents[1] / "qat/report.py"
        ),
        results_sha256=sha(a.root / "results.tsv"),
        report_tool_sha256=sha(__file__),
        cohorts={},
        limitations="Historical quality model; reused evaluation inputs do not establish unseen-source generalization. Frozen PTQ, not QAT. No model/default selection. Only model input range exceedances are counted; QLinearConv internal output saturation is not instrumented. RGB8 error histograms are pixel-weighted; SSIM2 changes are origin-balanced. No new artifact audit.",
        input_rgb8_roundtrip_max=max(
            float(r["input_rgb8_roundtrip_max"]) for r in rows
        ),
    )
    for cohort in sorted({r["cohort"] for r in rows}):
        selected = [r for r in rows if r["cohort"] == cohort]
        groups = {"all": selected}
        if cohort == "secondary":
            groups.update(
                {
                    size: [r for r in selected if r["size"] == size]
                    for size in sorted({r["size"] for r in selected}, key=int)
                }
            )
        report["cohorts"][cohort] = {
            k: {mode: summarize(v, mode) for mode in ["raw", "projected"]}
            for k, v in groups.items()
        }
    with a.output.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    for name, r in report["cohorts"]["primary"]["all"]["projected"].items():
        print(
            name,
            "SSIM2 delta",
            r["mean_vs_native_ssim2"],
            "harm cases",
            r["harm_cases"],
            "RGB8 changed fraction",
            r["rgb8_changed_fraction"],
            "max RGB8 delta",
            r["rgb8_max_code_error"],
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--expected-inputs", type=int, default=2960)
    main(p.parse_args())
