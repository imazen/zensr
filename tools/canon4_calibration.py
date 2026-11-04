#!/usr/bin/env python3
"""Summarize paired canon4 production-grid deltas, split by image.

Usage: python3 tools/canon4_calibration.py PROD_TSV
Outputs one row per encoder/subsampling/q/split with probe metadata, raw model
and projected gain medians, and counts of negative deltas.  The historical
hash_stem partition is shared with gate_recalibrate.py. The emitted gains are
not fitted curves and do not imply a routing decision.
"""
import collections
import csv
import statistics
import sys

from gate_recalibrate import hash_stem


def main(path):
    cells = collections.defaultdict(dict)
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"file", "encoder", "ss", "q", "arm", "ssim2", "probe_family", "probe_q"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        for row in reader:
            key = (row["file"], row["encoder"], row["ss"], float(row["q"]))
            if row["arm"] in cells[key]:
                raise ValueError(f"duplicate arm: {key} {row['arm']}")
            cells[key][row["arm"]] = row
    if not cells:
        raise ValueError("empty input")
    files = {key[0] for key in cells}
    groups = collections.defaultdict(list)
    for (filename, encoder, ss, q), arms in cells.items():
        needed = {"identity_off", "model_off", "model_proj"}
        if not needed <= arms.keys():
            raise ValueError(f"incomplete cell: {(filename, encoder, ss, q)}")
        base = arms["identity_off"]
        for arm in needed:
            if (arms[arm]["probe_family"], arms[arm]["probe_q"]) != (base["probe_family"], base["probe_q"]):
                raise ValueError(f"probe differs across arms: {(filename, encoder, ss, q)}")
        phase = "calibrate" if hash_stem(filename) % 2 == 0 else "validate"
        groups[(encoder, ss, q, phase)].append((
            base["probe_family"], float(base["probe_q"]),
            float(arms["model_off"]["ssim2"]) - float(base["ssim2"]),
            float(arms["model_proj"]["ssim2"]) - float(base["ssim2"]),
        ))
    print(f"# files={len(files)} calibrate={sum(hash_stem(f) % 2 == 0 for f in files)} validate={sum(hash_stem(f) % 2 != 0 for f in files)}")
    print("encoder\tss\tq\tsplit\tn\tprobe_families\tprobe_q_range\traw_median\traw_harmed\tproj_median\tproj_harmed")
    for (encoder, ss, q, phase), values in sorted(groups.items()):
        families = collections.Counter(v[0] for v in values)
        fs = ",".join(f"{name}:{count}" for name, count in sorted(families.items()))
        probes = [v[1] for v in values]
        raw = [v[2] for v in values]
        proj = [v[3] for v in values]
        print(f"{encoder}\t{ss}\t{q:g}\t{phase}\t{len(values)}\t{fs}\t"
              f"{min(probes):g}..{max(probes):g}\t{statistics.median(raw):+.4f}\t"
              f"{sum(d < 0 for d in raw)}\t{statistics.median(proj):+.4f}\t{sum(d < 0 for d in proj)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
