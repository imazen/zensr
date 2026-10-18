#!/usr/bin/env python3
"""Final-pixel exactness of the shipped x1 pipeline.

Restores each JPEG with `restore_jpeg` (guarded model, quantisation-consistency
projection, RGB8) at every requested thread count with the default automatic
tile, on every requested dispatch tier, and requires every RGB8 output of one
(JPEG, model) pair to be byte-identical. Optionally checks the common SHA-256
against a recorded expectation file. Exits 1 on any disagreement.

Usage: product.py --bin tools/exactness/product/target/release/zensr-exactness-product \
                  --out <new dir> --jpeg a.jpg [--jpeg b.jpg ...] \
                  --model NAME=weights_f16.raw:NF:NC [--model ...] \
                  [--tiers auto,v3,scalar] [--threads 1,2,4,8,12,16]
                  [--scalar-threads 16] [--expect expected.json]

The scalar tier computes every multiply-add with a one-rounding software FMA,
so on full-size inputs it is slow; --scalar-threads selects the thread counts
it runs (default: all of --threads).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bin", required=True)
    ap.add_argument(
        "--out", required=True, help="new directory; refuses an existing path"
    )
    ap.add_argument("--jpeg", action="append", required=True)
    ap.add_argument(
        "--model", action="append", required=True, help="NAME=weights_f16.raw:NF:NC"
    )
    ap.add_argument("--tiers", default="auto,v3,scalar")
    ap.add_argument("--threads", default="1,2,4,8,12,16")
    ap.add_argument("--scalar-threads")
    ap.add_argument("--expect", help='JSON {"<jpeg stem> <model>": sha256}')
    args = ap.parse_args()
    if os.path.lexists(args.out):
        raise SystemExit(f"refusing existing path {args.out}")
    os.makedirs(args.out)
    for j in args.jpeg:
        if not os.path.isfile(j):
            raise SystemExit(f"missing input {j}")
    models = []
    for m in args.model:
        name, rest = m.split("=", 1)
        weights, nf, nc = rest.rsplit(":", 2)
        if not os.path.isfile(weights):
            raise SystemExit(f"missing weights {weights}")
        models.append((name, weights, nf, nc))
    tiers = [t for t in args.tiers.split(",") if t]
    record = {
        "bin_sha256": sha256(args.bin),
        "inputs": {os.path.basename(j): sha256(j) for j in args.jpeg},
        "models": {
            n: {"weights": w, "sha256": sha256(w), "nf": int(f), "nc": int(c)}
            for n, w, f, c in models
        },
        "outputs": {},
    }
    bad = 0
    for name, weights, nf, nc in models:
        for tier in tiers:
            threads = (
                args.scalar_threads
                if (tier == "scalar" and args.scalar_threads)
                else args.threads
            )
            d = os.path.join(args.out, f"{name}-{tier}")
            os.makedirs(d)
            cmd = [args.bin, weights, nf, nc, threads, d] + args.jpeg
            p = subprocess.run(
                cmd,
                env=dict(os.environ, ZENSR_EXACT_TIER=tier),
                capture_output=True,
                text=True,
            )
            with open(os.path.join(d, "log.txt"), "w") as f:
                f.write(f"$ {json.dumps(cmd)}\n{p.stdout}{p.stderr}")
            if p.returncode != 0:
                raise SystemExit(
                    f"probe failed ({p.returncode}) {name} {tier}:\n{p.stderr[-2000:]}"
                )
            for fn in sorted(os.listdir(d)):
                if fn.endswith(".rgb8"):
                    stem, t = fn[: -len(".rgb8")].rsplit("-t", 1)
                    record["outputs"].setdefault(f"{stem} {name}", {})[
                        f"{tier} t{t}"
                    ] = sha256(os.path.join(d, fn))
    expect = json.load(open(args.expect)) if args.expect else {}
    for key, runs in sorted(record["outputs"].items()):
        distinct = sorted(set(runs.values()))
        ok = len(distinct) == 1
        exp = expect.get(key)
        if exp is not None and distinct != [exp]:
            ok = False
        bad += not ok
        print(
            f"{key}: {len(runs)} outputs, {len(distinct)} distinct {'OK' if ok else 'MISMATCH'} {distinct[0]}"
            + (
                ""
                if exp is None
                else f" expect={'match' if distinct == [exp] else exp}"
            )
        )
        if len(distinct) > 1:
            for cfg, digest in sorted(runs.items()):
                print(f"    {cfg}: {digest}")
    record["inconsistent"] = bad
    with open(os.path.join(args.out, "product.json"), "w") as f:
        json.dump(record, f, indent=1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
