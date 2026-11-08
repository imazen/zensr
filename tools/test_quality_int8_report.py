"""Boundary, pairing and denominator checks for the quality precision report."""

import importlib.util
from pathlib import Path
import unittest
import json
import tempfile
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "quality_precision_report", Path(__file__).parent / "quality-int8/report.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def fixture():
    rows = []
    for arm in report.ARMS:
        for identity, origin, values in [
            ("a", "one", 300),
            ("b", "one", 300),
            ("c", "two", 3000),
        ]:
            r = dict(
                content="fixture",
                cohort="primary",
                size="512",
                id=identity,
                arm=arm,
                origin=origin,
                identity_ssim2="90.0001",
                encoder="test-fixture",
                ss="444",
                q="50",
                width=str(values // 30),
                height="10",
                values=str(values),
                source_id=identity,
                activation_range_exceedances="0",
                activation_values="0",
            )
            for mode in ["raw", "projected"]:
                r.update(
                    {
                        mode + "_ssim2": "90.0001",
                        mode + "_butter_max": "1",
                        mode + "_butter_pnorm3": ".8",
                        mode + "_outside01": "0",
                        mode + "_mae": "0",
                        mode + "_max": "0",
                        mode + "_rgb8_changed": "0",
                        mode + "_rgb8_gt1": "0",
                        mode + "_rgb8_gt2": "0",
                        mode + "_rgb8_max": "0",
                    }
                )
            rows.append(r)
    return rows


class QualityReportTests(unittest.TestCase):
    def test_decimal_curve_and_origin_denominator(self):
        rows = fixture()
        r = next(r for r in rows if r["arm"] == "int8" and r["id"] == "a")
        r["projected_ssim2"] = "89.500104"
        self.assertEqual(report.summarize(rows, "projected")["int8"]["harm_cases"], 0)
        r["projected_ssim2"] = "89.5001039"
        result = report.summarize(rows, "projected")["int8"]
        self.assertEqual(result["harm_cases"], 1)
        self.assertEqual(result["harm_fraction"], 0.25)

    def test_rgb_histogram_uses_channel_values(self):
        rows = fixture()
        r = next(r for r in rows if r["arm"] == "int8" and r["id"] == "a")
        r.update(
            raw_rgb8_changed="10", raw_rgb8_gt1="3", raw_rgb8_gt2="1", raw_rgb8_max="7"
        )
        result = report.summarize(rows, "raw")["int8"]
        self.assertEqual(
            result["rgb8_code_error_histogram"],
            {"0": 3590, "1": 7, "2": 2, "3_or_more": 1},
        )
        self.assertEqual(result["rgb8_changed_fraction"], 10 / 3600)

    def test_missing_duplicate_and_mismatched_pairs_fail(self):
        for change in ["missing", "duplicate", "metadata"]:
            with self.subTest(change=change):
                rows = fixture()
                if change == "missing":
                    rows.pop()
                elif change == "duplicate":
                    rows.append(rows[-1].copy())
                else:
                    rows[-1]["encoder"] = "changed"
                with self.assertRaises(ValueError):
                    report.summarize(rows, "raw")


class ManifestAdmissionTests(unittest.TestCase):
    def test_geometry_and_unique_manifest_membership_are_required(self):
        rows = fixture()
        manifest = [r.copy() for r in rows if r["arm"] == "native"]
        report.validate_manifest(rows, manifest, 3)
        for change in ["geometry", "duplicate", "unplanned"]:
            with self.subTest(change=change):
                entries = [r.copy() for r in manifest]
                if change == "geometry":
                    entries[0]["width"] = "99"
                elif change == "duplicate":
                    entries.append(entries[0].copy())
                else:
                    entries[0]["id"] = "unknown"
                with self.assertRaises(ValueError):
                    report.validate_manifest(rows, entries, 3)


latency_spec = importlib.util.spec_from_file_location(
    "quality_latency_report", Path(__file__).parent / "quality-int8/latency_report.py"
)
latency = importlib.util.module_from_spec(latency_spec)
latency_spec.loader.exec_module(latency)


class LatencyEvidenceTests(unittest.TestCase):
    def test_all_processes_and_actual_rounds_are_retained(self):
        experiment = dict(
            status="unconfirmed",
            binary_sha256="fixture",
            plan=dict(processes=3),
            attempts=[dict(status="completed") for _ in range(3)],
        )
        runs = []
        for index in range(3):
            runs.append(
                dict(
                    git_hash="same",
                    incomplete=False,
                    unreliable=index == 1,
                    gate_waits=index,
                    cpu_placement={},
                    comparisons=[
                        dict(
                            group_name="fixture",
                            completed_rounds=index + 1,
                            benchmarks=[
                                dict(
                                    name="native",
                                    summary=dict(mean=(index + 1) * 1e6, n=index + 1),
                                ),
                                dict(name="int8", summary=dict(mean=4e6, n=index + 1)),
                            ],
                        )
                    ],
                )
            )
        result = latency.summarize(experiment, runs)
        self.assertTrue(result["process_reliability"][1]["unreliable"])
        self.assertEqual(
            result["groups"]["fixture"]["int8"]["completed_rounds"], [1, 2, 3]
        )
        self.assertEqual(result["groups"]["fixture"]["int8"]["ratio_to_native"], 2)
        runs[0]["incomplete"] = True
        with self.assertRaises(ValueError):
            latency.summarize(experiment, runs)


package_spec = importlib.util.spec_from_file_location(
    "quality_package", Path(__file__).parent / "quality-int8/package.py"
)
package = importlib.util.module_from_spec(package_spec)
package_spec.loader.exec_module(package)


class PackagingAdmissionTests(unittest.TestCase):
    def test_a_complete_pilot_is_not_a_complete_study(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "evaluation-v6"
            evaluation.mkdir()
            table = evaluation / "results.tsv"
            table.write_text("pilot fixture\n")
            (evaluation / "COMPLETE").write_text(package.sha(table))
            (evaluation / "IDENTITY.json").write_text(
                json.dumps(dict(required_inputs=8))
            )
            with self.assertRaisesRegex(ValueError, "2960"):
                package.main(SimpleNamespace(root=root))


if __name__ == "__main__":
    unittest.main()
