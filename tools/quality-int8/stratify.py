#!/usr/bin/env python3
"""Describe frozen repair scores by content, JPEG quality and encoder/sampling."""

import argparse
import collections
import csv
from decimal import Decimal
import json
from pathlib import Path
import statistics

from report import ARMS, limit, read, sha, validate_manifest

BANDS = [(5, 20), (25, 40), (45, 60), (65, 80), (85, 100)]


def band(q):
    q = int(q)
    if q not in range(5, 101, 5):
        raise ValueError("quality outside frozen q5..100 step-five grid")
    return next(f"{lo}–{hi}" for lo, hi in BANDS if lo <= q <= hi)


def summary(rows, stage):
    if not rows:
        raise ValueError("empty stratum")
    origins = collections.defaultdict(list)
    gains = []
    harms = []
    for r in rows:
        identity = Decimal(r["identity_ssim2"])
        actual = Decimal(r[stage + "_ssim2"])
        if not identity.is_finite() or not actual.is_finite():
            raise ValueError("nonfinite score")
        gain = actual - identity
        harm = gain < -limit(identity)
        origins[r["origin"]].append((float(gain), int(harm)))
        gains.append(gain)
        harms.append(harm)
    return dict(
        cases=len(rows),
        origins=len(origins),
        mean=statistics.fmean(
            statistics.fmean(g for g, _ in v) for v in origins.values()
        ),
        median=float(statistics.median(gains)),
        harmed=sum(harms),
        harm_percent=100 * statistics.fmean(
            statistics.fmean(h for _, h in v) for v in origins.values()
        ),
        worst=float(min(gains)),
    )


def group(rows, fields, stage):
    groups = collections.defaultdict(list)
    for row in rows:
        key = tuple(band(row["q"]) if f == "band" else row[f] for f in fields)
        groups[key].append(row)
    return {k: summary(v, stage) for k, v in groups.items()}


def label(content):
    name = content.split("-", 1)[1]
    for prefix in ["lilith-", "unsplash-"]:
        name = name.removeprefix(prefix)
    return name.replace("-", " ")


def markdown(rows, stage, identity):
    bands = [f"{lo}–{hi}" for lo, hi in BANDS]
    by_content = group(rows, ["content", "band"], stage)
    categories = sorted({r["content"] for r in rows})
    lines = [
        "# dejpeg7_graphics by JPEG quality and image category — 2026-11-08",
        "",
        "Forced native float repair with deployed f16-expanded weights; "
        + ("guard/coefficient projection applied." if stage == "projected" else "before projection."),
        "Routing and the near-pristine gate are off. All deltas are "
        "SSIMULACRA2 against the decoded JPEG, not against canon4.",
        "",
        "**Coverage: one source image in each of 17 categories, long side 512.** "
        "Each image has four encoders × two sampling modes × 20 quality settings "
        "= 160 cases; each category/quality-band cell has 32 cases. "
        "Repeated encodings do not supply independent category-level evidence. "
        "Inputs are reused, and legacy training overlap is not established.",
        "",
        "Quality bands contain four settings each: 5/10/15/20, 25/30/35/40, "
        "45/50/55/60, 65/70/75/80 and 85/90/95/100. Encoder q values are "
        "nominal settings, not equal perceptual severity across encoders. "
        "Means and harm percentages balance origins within each cell. "
        "Harm is loss strictly above 0.5 SSIM2 through identity score 90, "
        "tapering linearly to 0.1 at 100, with decimal boundary arithmetic.",
        "",
        "## Mean gain / harm percentage",
        "",
        "Each cell is **mean gain / harm %**; positive gain is better. "
        "Negative means are retained.",
        "",
        "| Image category (source ID) | " + " | ".join(bands) + " |",
        "|---|" + "---:|" * len(bands),
    ]
    for content in categories:
        ids = sorted({r["source_id"] for r in rows if r["content"] == content})
        values = [by_content[(content, b)] for b in bands]
        lines.append(
            f"| {label(content)} ({', '.join(ids)}) | "
            + " | ".join(f"{v['mean']:+.2f} / {v['harm_percent']:.1f}%" for v in values)
            + " |"
        )
    lines += [
        "", "## Worst per-case gain", "",
        "Minimum gain in each of the same 32-case cells; negative values are losses.",
        "", "| Image category | " + " | ".join(bands) + " |",
        "|---|" + "---:|" * len(bands),
    ]
    for content in categories:
        lines.append(
            f"| {label(content)} | "
            + " | ".join(f"{by_content[(content, b)]['worst']:+.2f}" for b in bands)
            + " |"
        )
    lines += [
        "", "## Every nominal JPEG quality", "",
        "Each row covers all 17 source images and eight encoder/sampling variants "
        "per image (136 cases). Median is the case median; mean balances origins.",
        "", "| q | Mean gain | Median gain | Harm cases / 136 | Harm % | Worst gain |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for (q,), v in sorted(group(rows, ["q"], stage).items(), key=lambda x: int(x[0][0])):
        lines.append(
            f"| {q} | {v['mean']:+.3f} | {v['median']:+.3f} | "
            f"{v['harmed']}/136 | {v['harm_percent']:.2f}% | {v['worst']:+.3f} |"
        )
    lines += [
        "", "## Encoder and chroma sampling", "",
        "Each cell has 68 cases (17 images × four qualities); mean gain / harm %.",
        "", "| Encoder / sampling | " + " | ".join(bands) + " |",
        "|---|" + "---:|" * len(bands),
    ]
    encoder = group(rows, ["encoder", "ss", "band"], stage)
    for enc, ss in sorted({(r["encoder"], r["ss"]) for r in rows}):
        values = [encoder[(enc, ss, b)] for b in bands]
        lines.append(
            f"| {enc} / {ss} | "
            + " | ".join(f"{v['mean']:+.2f} / {v['harm_percent']:.1f}%" for v in values)
            + " |"
        )
    lines += [
        "", "## Reproduction and scope", "",
        "Computed from the completed precision study without rerunning inference "
        "or changing selection. The complete exact-q × category × encoder × "
        "sampling table is retained outside Git alongside the experiment; "
        "each row includes cases, origins, mean, median, harm count/rate and worst gain.",
        "",
        f"Results SHA-256: `{identity['results_sha256']}`.",
        f"Input manifest SHA-256: `{identity['inputs_sha256']}`.",
        "",
        "```sh",
        "just quality-int8-stratify EVALUATION OUTPUT.md --inputs INPUTS.tsv "
        "--detail-output DETAILS.tsv",
        "```",
        "",
        "[Complete study and limitations](quality_int8_2026-11-08.md). "
        "The automatic router is calibrated for canon4, not this model. "
        "These descriptive tables do not select a router or establish quality "
        "across arbitrary resolutions.",
    ]
    return "\n".join(lines) + "\n"


def main(a):
    rows = read(a.root)
    identity = json.loads((a.root / "IDENTITY.json").read_text())
    if sha(a.inputs) != identity["inputs_sha256"]:
        raise ValueError("input manifest differs from evaluation identity")
    with a.inputs.open() as f:
        validate_manifest(rows, list(csv.DictReader(f, delimiter="\t")), a.expected_inputs)
    selected = [r for r in rows if r["arm"] == "native" and r["cohort"] == "primary"]
    expected = {(c, q, e, ss) for c in {r['content'] for r in selected}
                for q in range(5, 101, 5)
                for e in ['turbo', 'mozjpeg', 'jpegli', 'zenjpeg'] for ss in ['420', '444']}
    actual = [(r['content'], int(r['q']), r['encoder'], r['ss']) for r in selected]
    if len(set(r['origin'] for r in selected)) != 17 or len(expected) != 2720:
        raise ValueError("primary category/origin coverage differs")
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("primary category/quality/encoder/sampling coverage differs")
    for content in {r['content'] for r in selected}:
        if len({r['origin'] for r in selected if r['content'] == content}) != 1:
            raise ValueError("expected one origin per content stratum")
    identity['results_sha256'] = sha(a.root / 'results.tsv')
    details = group(selected, ['content', 'q', 'encoder', 'ss'], a.stage)
    with a.detail_output.open('x', newline='') as f:
        fields = ['content', 'q', 'encoder', 'ss', 'cases', 'origins', 'mean',
                  'median', 'harmed', 'harm_percent', 'worst']
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        for key, value in sorted(details.items()):
            writer.writerow(dict(zip(fields[:4], key), **value))
    with a.output.open('x') as f:
        f.write(markdown(selected, a.stage, identity))
    print(json.dumps(summary(selected, a.stage), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--detail-output', type=Path, required=True)
    p.add_argument('--expected-inputs', type=int, default=2960)
    p.add_argument('--stage', choices=['raw', 'projected'], default='projected')
    main(p.parse_args())
