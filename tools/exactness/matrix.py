#!/usr/bin/env python3
"""Cross-path exactness matrix over the adopted models.

For each (model, input shape) every configuration — whole-image forward
(threads 0) and the tiled runner at threads 1/4/8 (tile 128), on dispatch tiers
auto / v3 / scalar — must produce the same raw f32 output, byte for byte. Uses
the tools/coz-profile worker (deterministic synthetic input, raw dump). Exits 1
on any disagreement and records where, including the shipping RGB8 quantizer
`(v * 255 + 0.5).clamp(0, 255) as u8` applied to the disagreeing values.

Optional --compare-bin runs every configuration of each case on a second
worker binary too (for example a baseline build) and reports, per
configuration, whether it gives the same bytes; that comparison is informative
and does not affect the exit status.

Usage: matrix.py --bin tools/coz-profile/target/release/zensr-coz-profile \
                 --out <new dir> [--models-dir models/adopted] [--models a,b]
                 [--tiers auto,v3,scalar] [--winograd] [--compare-bin BIN]
"""

import argparse
import array
import hashlib
import json
import os
import struct
import subprocess
import sys

# name -> (dir under models/adopted, nf, nc, scale); nf None means span48.
MODELS = {
    "rt24g": ("dejpeg_rt24g", 24, 8, 1),
    "q7": ("dejpeg7_graphics", 64, 16, 1),
    "rtc2": ("people_rtc_2x", 24, 8, 2),
    "nomo2": ("nomosuni_compact_2x", 64, 16, 2),
    "gx4": ("general_x4v3", 64, 32, 4),
    "ax4": ("animevideo_x4v3", 64, 16, 4),
    "span4": ("nomosuni_span_4x", None, None, 4),
}
SMALL = [
    (1, 1),
    (1, 9),
    (2, 7),
    (3, 17),
    (4, 4),
    (5, 5),
    (7, 33),
    (8, 19),
    (13, 9),
    (17, 13),
    (31, 47),
    (64, 64),
]
MID = [(128, 96), (997, 17), (17, 997), (256, 256), (300, 200), (48, 4096), (48, 4093)]


def shapes(model):
    if model == "rt24g":
        return SMALL + MID + [(1024, 1024)]
    if model == "q7":
        return SMALL + [(128, 96), (997, 17), (256, 256), (48, 4096)]
    if model == "span4":
        return [(17, 13), (32, 32), (64, 64), (130, 70)]
    return [
        (1, 1),
        (5, 5),
        (13, 9),
        (17, 13),
        (33, 19),
        (64, 64),
        (128, 96),
        (150, 140),
    ]


def scalar_ok(model, hw):
    """The scalar tier runs a one-rounding software FMA per multiply-add, so it
    is limited to inputs small enough to finish in seconds."""
    nf = MODELS[model][1] or 48
    return hw[0] * hw[1] <= (64 * 64 if nf <= 24 else 17 * 13)


def configs(model, hw, tiers):
    """(threads, tier) pairs: whole-image forward (0) and tiled 1/4/8, per tier;
    the scalar tier runs whole and tiled-4 on small inputs only."""
    out = []
    for tier in tiers:
        if tier == "scalar" and not scalar_ok(model, hw):
            continue
        for t in (0, 1, 4, 8):
            if tier == "scalar" and t not in (0, 4):
                continue
            out.append((t, tier))
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(bin_path, models_dir, model, hw, threads, tier, out, tag, winograd):
    name, nf, nc, s = MODELS[model]
    weights = os.path.join(models_dir, name, "weights_f16.raw")
    if not os.path.isfile(weights):
        raise SystemExit(f"missing model weights {weights} (requested model {model})")
    tmp = os.path.join(out, "dumps", f"tmp-{tag}.f32")
    if nf is None:
        spec = os.path.join(out, "specs", f"{model}-{hw[0]}x{hw[1]}-t{threads}.spec")
        if not os.path.exists(spec):
            with open(spec, "w") as f:
                f.write(f"{weights} span48 {s} {hw[0]} {hw[1]} 1 {threads}\n")
        cmd = [bin_path, "spec", spec, tmp]
    else:
        cmd = [
            bin_path,
            weights,
            str(nf),
            str(nc),
            str(s),
            str(hw[0]),
            str(hw[1]),
            "1",
            str(threads),
            tmp,
        ]
    env = dict(os.environ, ZENSR_PROFILE_TIER=tier)
    if winograd:
        env["ZENSR_WINOGRAD"] = "1"
    else:
        env.pop("ZENSR_WINOGRAD", None)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"worker failed ({p.returncode}): {cmd}\n{p.stderr[-2000:]}")
    digest = sha256(tmp)
    dst = os.path.join(out, "dumps", f"{digest}.f32")
    if os.path.exists(dst):
        os.remove(tmp)
    else:
        os.rename(tmp, dst)
    return digest


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def rgb8(v):
    # (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8, each step rounded to f32.
    x = f32(f32(v * 255.0) + 0.5)
    return int(min(max(x, 0.0), 255.0))


def diff(out, a, b):
    va, vb = array.array("f"), array.array("f")
    with open(os.path.join(out, "dumps", f"{a}.f32"), "rb") as f:
        va.frombytes(f.read())
    with open(os.path.join(out, "dumps", f"{b}.f32"), "rb") as f:
        vb.frombytes(f.read())
    ba = struct.unpack(f"<{len(va)}I", va.tobytes())
    bb = struct.unpack(f"<{len(vb)}I", vb.tobytes())
    idx = [i for i in range(min(len(ba), len(bb))) if ba[i] != bb[i]]
    px = [i for i in idx if rgb8(va[i]) != rgb8(vb[i])]
    first = idx[0] if idx else None
    return {
        "len": [len(va), len(vb)],
        "f32_differ": len(idx),
        "rgb8_differ": len(px),
        "first": None if first is None else [first, va[first], vb[first]],
        "first_rgb8": None
        if not px
        else [px[0], va[px[0]], vb[px[0]], rgb8(va[px[0]]), rgb8(vb[px[0]])],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bin", required=True)
    ap.add_argument(
        "--out", required=True, help="new directory; refuses an existing path"
    )
    ap.add_argument("--models-dir", default="models/adopted")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--tiers", default="auto,v3,scalar")
    ap.add_argument(
        "--winograd", action="store_true", help="ZENSR_WINOGRAD=1 (opt-in algorithm)"
    )
    ap.add_argument("--compare-bin")
    args = ap.parse_args()
    if os.path.lexists(args.out):
        raise SystemExit(f"refusing existing path {args.out}")
    os.makedirs(os.path.join(args.out, "dumps"))
    os.makedirs(os.path.join(args.out, "specs"))
    tiers = [t for t in args.tiers.split(",") if t]
    models = [m for m in args.models.split(",") if m]
    for m in models:
        if m not in MODELS:
            raise SystemExit(f"unknown model {m}; choose from {', '.join(MODELS)}")
    results, bad, n = [], 0, 0
    for model in models:
        for hw in shapes(model):
            runs = {}
            for threads, tier in configs(model, hw, tiers):
                tag = f"{n}"
                n += 1
                runs[f"t{threads}-{tier}"] = run(
                    args.bin,
                    args.models_dir,
                    model,
                    hw,
                    threads,
                    tier,
                    args.out,
                    tag,
                    args.winograd,
                )
            distinct = sorted(set(runs.values()))
            row = {
                "model": model,
                "h": hw[0],
                "w": hw[1],
                "runs": runs,
                "consistent": len(distinct) == 1,
            }
            if len(distinct) > 1:
                bad += 1
                ref_key = next(iter(runs))
                row["diffs"] = {
                    k: diff(args.out, runs[ref_key], v)
                    for k, v in runs.items()
                    if v != runs[ref_key]
                }
                row["reference"] = ref_key
            if args.compare_bin:
                other = {}
                for threads, tier in configs(model, hw, tiers):
                    k = f"t{threads}-{tier}"
                    other[k] = run(
                        args.compare_bin,
                        args.models_dir,
                        model,
                        hw,
                        threads,
                        tier,
                        args.out,
                        f"cmp{n}-{k}",
                        args.winograd,
                    )
                row["compare_bin_runs"] = other
                row["compare_bin_same"] = other == runs
                if other != runs:
                    row["compare_bin_diff"] = {
                        k: diff(args.out, runs[k], v)
                        for k, v in other.items()
                        if v != runs[k]
                    }
            results.append(row)
            status = "OK" if row["consistent"] else "MISMATCH"
            cmp = (
                ""
                if not args.compare_bin
                else (
                    " compare=same"
                    if row["compare_bin_same"]
                    else f" compare=DIFF({len(row['compare_bin_diff'])}/{len(runs)})"
                )
            )
            print(
                f"{model:6s} {hw[0]}x{hw[1]:<5} configs={len(runs):2d} {status}{cmp} {distinct[0][:16]}",
                flush=True,
            )
            if not row["consistent"]:
                for k, d in row["diffs"].items():
                    print(
                        f"    {k} vs {row['reference']}: f32 {d['f32_differ']} rgb8 {d['rgb8_differ']} first {d['first']}"
                    )
    summary = {
        "bin_sha256": sha256(args.bin),
        "compare_bin_sha256": sha256(args.compare_bin) if args.compare_bin else None,
        "winograd": args.winograd,
        "tiers": tiers,
        "cases": len(results),
        "runs": n,
        "inconsistent_cases": bad,
        "compare_bin_differs": sum(
            1 for r in results if args.compare_bin and not r["compare_bin_same"]
        ),
    }
    with open(os.path.join(args.out, "matrix.json"), "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=1)
    print(json.dumps(summary))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
