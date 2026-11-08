#!/usr/bin/env python3
"""Pack adopted CPU weights and their runtime policy into one .zensr file.

Standard library only. Built-in calibration belongs exclusively to the exact
canon4 f16 bytes. Other weights use --policy manual and explicit runtime routing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "crates/zensr-zenjpeg/src"
CANON4 = "c3db9373791270bbe5493d3ee98cd4f62296510fe6931073e2966f433e9a5cf4"
CURVES = ("G420", "GRAPHIC420", "PHOTO420", "G444", "GRAPHIC444", "PHOTO444",
          "DIST420", "GRAPHIC_DIST420", "PHOTO_DIST420", "DIST444", "GRAPHIC_DIST444",
          "PHOTO_DIST444", "AQ_OFF_DELTA_420", "AQ_OFF_DELTA_444")
GATES = ("GATE_IJG_420", "GATE_IJG_444", "GATE_MOZJPEG_420", "GATE_MOZJPEG_444",
         "GATE_DISTANCE_420", "GATE_DISTANCE_444")
LIMIT = 64 * 1024 * 1024


def string(value):
    raw = value.encode("utf-8")
    if not raw or len(raw) > 256 or b"\0" in raw:
        raise ValueError("invalid container string")
    return struct.pack("<H", len(raw)) + raw


def constant(source, name):
    match = re.search(r"\bconst " + re.escape(name) + r"\s*:[^=]+=(.*?);", source, re.S)
    if not match:
        raise ValueError(f"missing frozen constant {name}")
    return match[1].strip()


def numbers(source, name):
    return [float(n) for n in re.findall(r"[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?",
                                       constant(source, name))]


def floats(values, word="f"):
    return struct.pack("<" + word * len(values), *values)


def policy_bytes(mode, metadata, weights):
    data = bytearray(struct.pack("<BII", mode, 1, 0))  # pipeline ABI, automatic tiling
    data += floats([9.5, 0.25])
    data += b"\1"  # texture gate
    data += floats([0.35, 0.10, 0.06, 0.15, 0.35, 0.45, 0.45, 0.5, 1.5, 0.25, 6.0])
    data += hashlib.sha256(metadata + weights).digest()  # binds interpretation and weights
    if mode == 0:
        return bytes(data)
    if hashlib.sha256(weights).hexdigest() != CANON4:
        raise ValueError("canon4 calibration requires the exact canon4 f16 weight hash")
    calibration = (SRC / "calibration.rs").read_text()
    data += floats(numbers(calibration, "GRAPHIC_ZERO_AC_THRESHOLD"))
    data += floats([numbers(calibration, name)[0] for name in GATES])
    for name in CURVES:
        values = numbers(calibration, name)
        if len(values) % 2:
            raise ValueError(f"invalid curve {name}")
        data += struct.pack("<H", len(values) // 2) + floats(values)
    if mode == 1:
        data += hashlib.sha256((SRC / "decoded_router_features.rs").read_bytes()).digest()
        heads = (SRC / "decoded_router_candidate.rs").read_text()
        for name, count in (("MEAN", 37), ("SCALE", 37), ("GAIN", 47), ("RISK", 47)):
            values = numbers(heads, name)
            if len(values) != count:
                raise ValueError(f"invalid router {name}")
            data += floats(values, "d")
        data += floats([numbers(heads, name)[0] for name in
                        ("GAIN_BIAS", "RISK_BIAS", "PENALTY", "THRESHOLD")], "d")
        groups = re.findall(r"&\[(.*?)\]", constant(heads, "CATEGORIES"), re.S)
        if len(groups) != 3:
            raise ValueError("invalid router category schema")
        for group in groups:
            values = re.findall(r'"([^"\n]+)"', group)
            data += struct.pack("<B", len(values))
            data += b"".join(string(value) for value in values)
    return bytes(data)


def assemble(metadata, policy, weights):
    prefix = struct.pack("<8sIIQQ", b"ZSRMODEL", 1, len(metadata), len(policy), len(weights))
    body = metadata + policy + weights
    if 64 + len(body) > LIMIT:
        raise ValueError("container exceeds runtime size limit")
    return prefix + hashlib.sha256(prefix + body).digest() + body


def pack(directory, policy, budget):
    meta = json.loads((directory / "meta.json").read_text())
    arch = {"compact": 1, "span48": 2}[meta["arch"]]
    encoding = 1 if meta.get("f16_goldens", False) else 2
    filename = "weights_f16.raw" if encoding == 1 else "weights_f32.raw"
    weight_path = directory / filename
    if weight_path.stat().st_size > LIMIT:
        raise ValueError("weights exceed runtime size limit")
    weights = weight_path.read_bytes()
    nf, nc, scale = int(meta["nf"]), int(meta["nc"]), int(meta["scale"])
    if scale not in (1, 2, 4) or not 4 <= nf <= 64 or not 0 <= nc <= 128:
        raise ValueError("unsupported model geometry")
    if arch == 1:
        if nf % 4:
            raise ValueError("unsupported Compact channel geometry")
        count = 29 * nf + nc * (9 * nf * nf + 2 * nf) + 27 * nf * scale * scale + 3 * scale * scale
        if len(weights) != count * (2 if encoding == 1 else 4):
            raise ValueError("weight size does not match Compact geometry")
    else:
        if (nf, nc) != (48, 6) or scale == 1:
            raise ValueError("Span48 requires nf=48, nc=6 and scale=2 or 4")
        count = 28*nf + 19*(9*nf*nf+nf) + 4*nf*nf+nf + 27*nf*scale*scale+3*scale*scale
        if len(weights) != count * (2 if encoding == 1 else 4):
            raise ValueError("weight size does not match Span48 geometry")
    # Reject nonfinite values at packaging as well as at runtime loading.
    import math
    if any(not math.isfinite(v[0]) for v in struct.iter_unpack("<e" if encoding == 1 else "<f", weights)):
        raise ValueError("non-finite model weights")
    mode = {"manual": 0, "canon4": 1, "canon4-coefficients": 2}[policy]
    if mode and (arch, nf, nc, scale, meta["space"], encoding) != (1, 24, 8, 1, "rgb", 1):
        raise ValueError("canon4 policy requires canon4 architecture and colour space")
    metadata = string(directory.name) + struct.pack("<BBBHHBB", arch,
        {"rgb": 0, "ycbcr": 1}[meta["space"]], scale, nf, nc,
        {"realtime": 0, "quality": 1}[budget], encoding)
    return assemble(metadata, policy_bytes(mode, metadata, weights), weights)


def inspect_bytes(data):
    if not 64 <= len(data) <= LIMIT:
        raise ValueError("invalid container size")
    magic, version, m, p, w = struct.unpack_from("<8sIIQQ", data)
    if magic != b"ZSRMODEL" or version != 1 or 64 + m + p + w != len(data) or m > 4096 or p > 65536:
        raise ValueError("invalid container header")
    if hashlib.sha256(data[:32] + data[64:]).digest() != data[32:64]:
        raise ValueError("container checksum mismatch")
    n, = struct.unpack_from("<H", data, 64)
    if not 1 <= n <= 256 or m != 2 + n + 9:
        raise ValueError("invalid model metadata")
    name = data[66:66+n].decode("utf-8")
    arch, space, scale, nf, nc, budget, encoding = struct.unpack_from("<BBBHHBB", data, 66+n)
    mode = data[64+m]
    return dict(name=name, format_version=version, bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(), weights_bytes=w,
                weights_sha256=hashlib.sha256(data[64+m+p:]).hexdigest(),
                policy_bytes=p, policy={0: "manual", 1: "decoded-router", 2: "coefficients"}[mode],
                architecture={1: "compact", 2: "span48"}[arch], nf=nf, nc=nc, scale=scale,
                space={0: "rgb", 1: "ycbcr"}[space],
                budget={0: "realtime", 1: "quality"}[budget], encoding={1: "f16", 2: "f32"}[encoding])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("pack")
    create.add_argument("model_dir", type=Path)
    create.add_argument("output", type=Path)
    create.add_argument("--policy", required=True, choices=("manual", "canon4", "canon4-coefficients"))
    create.add_argument("--budget", required=True, choices=("realtime", "quality"))
    info = sub.add_parser("inspect", help="inspect header/checksum; runtime performs full validation")
    info.add_argument("model", type=Path)
    args = parser.parse_args()
    if args.command == "pack":
        data = pack(args.model_dir, args.policy, args.budget)
        report = inspect_bytes(data)
        with args.output.open("xb") as stream:
            stream.write(data)
    else:
        if args.model.stat().st_size > LIMIT:
            raise ValueError("container exceeds runtime size limit")
        report = inspect_bytes(args.model.read_bytes())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
