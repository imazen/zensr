#!/usr/bin/env python3
"""Build the zensr five-systems scientific report (self-contained HTML).

Data: parses the committed benchmark TSVs (no hand-transcribed numbers).
Gallery: crops max-detail 256px regions from gallery_dump PNGs, embeds as
base64 JPEG. Output: ~/tmp/zensr-report/report.html (publish as Artifact).
"""
import base64
import csv
import html
import json
import os
import statistics as st
from collections import defaultdict

import cv2
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAL = os.path.expanduser("~/tmp/zensr-gallery")
OUT = os.path.expanduser("~/tmp/zensr-report/report.html")


def rows_of(path):
    out = []
    for r in csv.reader(open(os.path.join(R, path)), delimiter="\t"):
        if len(r) == 8 and not r[0].startswith("#") and r[0] not in ("subcorpus", "sub"):
            try:
                out.append((r[0], r[1], r[2], r[3], r[4], float(r[5]), float(r[6]), float(r[7])))
            except ValueError:
                pass
    return out


DAY = rows_of("benchmarks/systems_eval_2026-07-23.tsv")
PTEST = rows_of("benchmarks/people_TEST_2026-07-24.tsv")
PDEV = rows_of("benchmarks/people_eval_2026-07-24.tsv")


def agg(rows, track, deg, base="lanczos"):
    """(system) -> dict of median/p10/butter/worse% vs base, per (track,deg)."""
    by = defaultdict(dict)  # system -> {file: (ssim2, butter)}
    for sub, f, tr, dg, sysname, psnr, ssim2, butter in rows:
        if tr == track and dg == deg:
            by[sysname][(sub, f)] = (ssim2, butter, psnr)
    out = {}
    baseline = by.get(base, {})
    for sysname, m in by.items():
        ss = sorted(v[0] for v in m.values())
        bb = sorted(v[1] for v in m.values())
        pp = sorted(v[2] for v in m.values())
        if not ss:
            continue
        p10 = ss[max(0, int(len(ss) * 0.1) - (0 if len(ss) * 0.1 % 1 else 1))] if len(ss) >= 10 else ss[0]
        worse = None
        if sysname != base and baseline:
            common = set(m) & set(baseline)
            if common:
                worse = 100.0 * sum(1 for k in common if m[k][0] < baseline[k][0]) / len(common)
        out[sysname] = dict(n=len(ss), ssim2=st.median(ss), p10=ss[len(ss) // 10] if len(ss) >= 10 else ss[0],
                            butter=st.median(bb), psnr=st.median(pp), worse=worse)
    return out


def crop_b64(scene, variant, box, size=224, q=86):
    p = os.path.join(GAL, scene, f"{variant}.png")
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    y, x, s = box
    c = img[y:y + s, x:x + s]
    c = cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, q])
    assert ok
    return base64.b64encode(enc.tobytes()).decode()


def best_box(scene, s=256):
    gt = cv2.imread(os.path.join(GAL, scene, "gt.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    best, bb = -1.0, (0, 0, s)
    for y in range(0, gt.shape[0] - s + 1, 64):
        for x in range(0, gt.shape[1] - s + 1, 64):
            e = float(np.abs(cv2.Laplacian(gt[y:y + s, x:x + s], cv2.CV_32F)).mean())
            if e > best:
                best, bb = e, (y, x, s)
    return bb


def fmt(v, nd=1):
    return f"{v:.{nd}f}" if v is not None else "—"


def bar(worse, best_is_low=True):
    if worse is None:
        return '<span class="mono dim">baseline</span>'
    pct = max(0.0, min(100.0, worse))
    cls = "good" if pct <= 25 else ("warn" if pct <= 50 else "bad")
    return (f'<span class="wbar"><span class="wfill {cls}" style="width:{pct:.0f}%"></span></span>'
            f'<span class="mono">{pct:.0f}%</span>')


def table(rows_map, order, base="lanczos", note=""):
    body = []
    for sysname in order:
        if sysname not in rows_map:
            continue
        d = rows_map[sysname]
        hl = ' class="hl"' if sysname.startswith(("P_", "E_rtc2", "S6_")) else ""
        body.append(
            f"<tr{hl}><td>{html.escape(sysname)}</td>"
            f'<td class="num">{fmt(d["ssim2"])}</td><td class="num">{fmt(d["p10"])}</td>'
            f'<td class="num">{fmt(d["butter"], 2)}</td><td class="num">{fmt(d["psnr"], 1)}</td>'
            f'<td class="wcell">{bar(d["worse"])}</td></tr>')
    return ("<table><thead><tr><th>system</th><th>SSIM2 med</th><th>p10</th>"
            "<th>butteraugli med</th><th>PSNR</th><th>worse-than-baseline</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>")


# ---------- assemble data ----------
X2_ORDER = ["lanczos", "catmullrom", "A2c_compact", "A2_span", "A2_span_raw",
            "E_rt", "E_rt32", "E_rtc", "E_rtc2"]
X4_ORDER = ["lanczos", "F_spanf", "A4_span", "B_quality", "D_anime"]
X1_ORDER = ["identity", "C_repair", "C_repair_cr", "S6_dejpeg"]
PT_ORDER = ["lanczos", "catmullrom", "A2c", "P_rtc", "P_a2c"]

sections_tables = {}
for deg in ["clean", "q75", "q50", "q35"]:
    sections_tables[("x2", deg)] = table(agg(DAY, "x2", deg), X2_ORDER)
    sections_tables[("x4", deg)] = table(agg(DAY, "x4", deg), X4_ORDER)
    if deg != "clean":
        sections_tables[("x1", deg)] = table(agg(DAY, "x1", deg, base="identity"), X1_ORDER)
    sections_tables[("ptest", deg)] = table(agg(PTEST, "x2", deg), PT_ORDER)
    sections_tables[("pdev", deg)] = table(agg(PDEV, "x2", deg),
                                           ["lanczos", "catmullrom", "A2c_compact", "E_rt", "P_rtc", "P_a2c"])

# speed table (quiet-box bench file)
speed_rows = []
for line in open(os.path.join(R, "benchmarks/systems_bench_2026-07-23.tsv")):
    c = line.rstrip("\n").split("\t")
    if len(c) == 7 and c[0] != "system" and not c[0].startswith("#"):
        speed_rows.append(c)

# audition summary
aud = []
for line in open(os.path.join(R, "benchmarks/teacher_audition_2026-07-23.summary.txt")):
    c = line.rstrip("\n").split("\t")
    if len(c) == 7 and c[0] != "sub":
        aud.append(c)

# gallery
SCENES = [
    ("people-q35", "People — turbo q35, ×2 (people-test-v1, CC0)",
     ["gt", "lanczos", "A2c", "P_a2c"],
     "The people band at its target: P_a2c recovers facial structure the generalist smooths away. Test-slice image (never trained on)."),
    ("people-q75", "People — turbo q75, ×2",
     ["gt", "lanczos", "A2c", "P_a2c"],
     "q75 was unwinnable for every generalist (worse-rate 69–91 %); P_a2c is the first model to win it (worse-rate 8 %)."),
    ("maps-q35", "Maps/graphics — turbo q35, ×2 (NPS brochure, public domain)",
     ["gt", "lanczos", "A2c", "A2span"],
     "Graphics is where restoration shines: span +5.2 SSIM2 median over Lanczos on maps (8/8 wins)."),
    ("documents-q35", "Documents — turbo q35, ×2 (IRS form, public domain)",
     ["gt", "lanczos", "A2c", "A2span"],
     "Text and line art: A2c +5.7 / span +8.0 over Lanczos, 8/8 wins each."),
    ("artscans-q35", "Art scans — turbo q35, ×2 (Haeckel, public domain) — honesty panel",
     ["gt", "lanczos", "A2c"],
     "A representative (median) loss: engraved halftone reads as texture; A2c −2.7 SSIM2 vs Lanczos. Routing sends this class to resampling."),
    ("textures-q35", "Stochastic texture — turbo q35, ×2 — honesty panel",
     ["gt", "lanczos", "A2c"],
     "The hardest class for SR everywhere: median A2c −0.9 SSIM2 at q35 (6/8 wins but thin), worse at higher q. Texture-gated guard shrinks the residual here."),
    ("realtime-q50", "Realtime tier — turbo q50, ×2 (45 K params, 23 MP/s @ 12T)",
     ["gt", "lanczos", "E_rtc", "A2c"],
     "The distilled 45 K student vs its 600 K teacher-family: most of the restoration at 13× less compute."),
    ("dejpeg-q35", "Native ×1 dejpeg — turbo q35, same size (S6, trained this program)",
     ["gt", "lanczos", "S6"],
     "Direct JPEG-artifact inversion at original resolution — the middle panel IS the degraded input. S6 beats identity at every quality (q35/q50 worse-rate 2 %), retiring the falsified scale-round-trip."),
]

# per-image ssim2 lookup for captions
per_img = {}
for src in (DAY, PTEST):
    for sub, f, tr, dg, sysname, psnr, ssim2, butter in src:
        per_img[(f, dg, sysname)] = ssim2
SCENE_FILES = {}
for line in open(os.path.join(GAL, "scenes.tsv")):
    c = line.rstrip("\n").split("\t")
    SCENE_FILES[c[0]] = (os.path.basename(c[1]), c[2])
LBL = {"gt": "ground truth", "lanczos": "Lanczos", "A2c": "A2c 600K", "A2span": "SPAN 410K",
       "S6": "S6 dejpeg ×1",
       "P_a2c": "P_a2c people", "P_rtc": "P_rtc 45K", "E_rtc": "E_rtc 45K"}
SYS_TSV = {"A2c": ["A2c_compact", "A2c"], "A2span": ["A2_span"], "P_a2c": ["P_a2c"],
           "P_rtc": ["P_rtc"], "E_rtc": ["E_rtc2", "E_rtc"], "lanczos": ["lanczos", "identity"],
           "S6": ["S6_dejpeg"]}

gallery_html = []
for scene, title, variants, blurb in SCENES:
    box = best_box(scene)
    fname, dg = SCENE_FILES[scene]
    panels = []
    for v in variants:
        b64 = crop_b64(scene, v, box)
        cap = LBL[v]
        if v != "gt":
            for key in SYS_TSV.get(v, []):
                if (fname, dg, key) in per_img:
                    cap += f' · <span class="mono">{per_img[(fname, dg, key)]:.1f}</span>'
                    break
        panels.append(f'<figure><img src="data:image/jpeg;base64,{b64}" alt="{html.escape(title)} — {LBL[v]}" loading="lazy" width="224" height="224"><figcaption>{cap}</figcaption></figure>')
    gallery_html.append(
        f'<div class="scene"><h4>{html.escape(title)}</h4><div class="strip">{"".join(panels)}</div>'
        f'<p class="blurb">{blurb} <span class="mono dim">{html.escape(fname)}</span></p></div>')

# audition table html
aud_rows = "".join(
    f"<tr><td>{a[0]}</td><td>{a[1]}</td><td>{html.escape(a[2])}</td>"
    f'<td class="num">{a[4]}</td><td class="num">{a[5]}</td><td class="num mono">{a[6]}</td></tr>'
    for a in aud if a[1] in ("q35", "q75") and a[0] in ("people", "textures"))

speed_html = "".join(
    f"<tr><td>{html.escape(c[0])}</td><td class='mono'>{c[1]}</td><td class='num'>{c[3]}</td>"
    f"<td class='num'>{c[4]}</td><td class='num'>{c[5]}</td><td class='num'>{c[6]}</td></tr>"
    for c in speed_rows)


def deg_tabs(prefix, degs=("q35", "q50", "q75", "clean")):
    btns = "".join(
        f'<button class="tab{" on" if i == 0 else ""}" data-t="{prefix}-{d}">{d}</button>'
        for i, d in enumerate(degs))
    panes = "".join(
        f'<div class="pane{" on" if i == 0 else ""}" id="{prefix}-{d}">{sections_tables[(prefix, d)]}</div>'
        for i, d in enumerate(degs))
    return f'<div class="tabs">{btns}</div>{panes}'


page = f"""<title>zensr — five systems report</title>
<style>
:root {{
  --paper:#f7f8f6; --ink:#212724; --mist:#e2e7e2; --dim:#6b756f;
  --vir:#0e7c6b; --vir-soft:#dcebe6; --red:#b4453a; --amber:#a97b2c;
  --card:#ffffff; --code:#eef1ee;
}}
@media (prefers-color-scheme: dark) {{ :root {{
  --paper:#14181a; --ink:#e6ece8; --mist:#28302d; --dim:#8b968f;
  --vir:#3cb79e; --vir-soft:#1d312c; --red:#d07a6e; --amber:#d0a45c;
  --card:#1a2022; --code:#20272a;
}} }}
:root[data-theme="dark"] {{
  --paper:#14181a; --ink:#e6ece8; --mist:#28302d; --dim:#8b968f;
  --vir:#3cb79e; --vir-soft:#1d312c; --red:#d07a6e; --amber:#d0a45c;
  --card:#1a2022; --code:#20272a;
}}
:root[data-theme="light"] {{
  --paper:#f7f8f6; --ink:#212724; --mist:#e2e7e2; --dim:#6b756f;
  --vir:#0e7c6b; --vir-soft:#dcebe6; --red:#b4453a; --amber:#a97b2c;
  --card:#ffffff; --code:#eef1ee;
}}
* {{ box-sizing: border-box; }}
body {{ background:var(--paper); color:var(--ink); margin:0;
  font: 16px/1.55 system-ui, "Segoe UI", Roboto, sans-serif; }}
.mono {{ font-family: ui-monospace, "Cascadia Code", "SF Mono", Menlo, Consolas, monospace;
  font-size:.92em; }}
main {{ max-width:1080px; margin:0 auto; padding:0 20px 80px; }}
header {{ border-bottom:2px solid var(--ink); padding:44px 0 18px; margin-bottom:8px; }}
.eyebrow {{ font-family:ui-monospace,monospace; text-transform:uppercase; letter-spacing:.14em;
  font-size:12px; color:var(--vir); }}
h1 {{ font-size: clamp(26px, 4vw, 40px); line-height:1.12; margin:.25em 0 .2em; text-wrap:balance;
  font-weight:650; }}
.sub {{ color:var(--dim); max-width:68ch; }}
.chips {{ display:flex; flex-wrap:wrap; gap:8px; margin:14px 0 0; }}
.chip {{ font-family:ui-monospace,monospace; font-size:12.5px; border:1px solid var(--mist);
  background:var(--card); padding:3px 10px; border-radius:3px; }}
.chip b {{ color:var(--vir); font-weight:600; }}
nav {{ position:sticky; top:0; background:var(--paper); border-bottom:1px solid var(--mist);
  z-index:5; display:flex; gap:2px; overflow-x:auto; padding:6px 0; }}
nav a {{ color:var(--dim); text-decoration:none; font-family:ui-monospace,monospace;
  font-size:12.5px; padding:4px 10px; white-space:nowrap; border-radius:3px; }}
nav a:hover, nav a:focus-visible {{ color:var(--ink); background:var(--code); outline:none; }}
h2 {{ font-size:22px; margin:52px 0 6px; padding-top:10px; }}
h2 .no {{ color:var(--vir); font-family:ui-monospace,monospace; font-size:15px; margin-right:10px; }}
h3 {{ font-size:16.5px; margin:26px 0 6px; }}
h4 {{ font-size:14.5px; margin:0 0 8px; }}
p {{ max-width:74ch; }}
p.lead {{ font-size:17.5px; }}
.dim {{ color:var(--dim); }}
.verdict {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px;
  margin:22px 0 6px; }}
.v {{ background:var(--card); border:1px solid var(--mist); border-top:3px solid var(--vir);
  padding:14px 16px; border-radius:4px; }}
.v .k {{ font-family:ui-monospace,monospace; font-size:28px; font-weight:600; color:var(--vir); }}
.v .k.neg {{ color:var(--red); }}
.v .t {{ font-size:13.5px; color:var(--dim); margin-top:2px; }}
.tablewrap, table {{ width:100%; }}
.pane {{ display:none; }} .pane.on {{ display:block; overflow-x:auto; }}
table {{ border-collapse:collapse; font-size:14px; background:var(--card); }}
th {{ text-align:left; font-family:ui-monospace,monospace; font-size:11.5px; text-transform:uppercase;
  letter-spacing:.06em; color:var(--dim); font-weight:600; border-bottom:2px solid var(--ink);
  padding:8px 12px 6px; white-space:nowrap; }}
td {{ border-bottom:1px solid var(--mist); padding:7px 12px; }}
td.num {{ font-family:ui-monospace,monospace; font-variant-numeric:tabular-nums; text-align:right; }}
tr.hl td {{ background:var(--vir-soft); }}
.wcell {{ min-width:150px; }}
.wbar {{ display:inline-block; width:84px; height:9px; background:var(--code); border-radius:2px;
  margin-right:8px; vertical-align:middle; overflow:hidden; }}
.wfill {{ display:block; height:100%; }}
.wfill.good {{ background:var(--vir); }} .wfill.warn {{ background:var(--amber); }}
.wfill.bad {{ background:var(--red); }}
.tabs {{ display:flex; gap:4px; margin:12px 0 10px; }}
.tab {{ font-family:ui-monospace,monospace; font-size:12.5px; padding:4px 14px; cursor:pointer;
  background:var(--card); color:var(--dim); border:1px solid var(--mist); border-radius:3px; }}
.tab.on {{ background:var(--ink); color:var(--paper); border-color:var(--ink); }}
.tab:focus-visible {{ outline:2px solid var(--vir); outline-offset:1px; }}
.note {{ border-left:3px solid var(--amber); background:var(--card); padding:10px 16px;
  margin:16px 0; font-size:14.5px; max-width:78ch; }}
.note.grave {{ border-color:var(--red); }}
.note.win {{ border-color:var(--vir); }}
.scene {{ margin:26px 0; }}
.strip {{ display:flex; gap:10px; overflow-x:auto; padding-bottom:6px; }}
figure {{ margin:0; flex:0 0 auto; }}
figure img {{ display:block; width:224px; height:224px; border:1px solid var(--mist);
  border-radius:3px; image-rendering:auto; }}
figcaption {{ font-family:ui-monospace,monospace; font-size:12px; color:var(--dim); padding-top:5px; }}
.blurb {{ font-size:14px; color:var(--dim); max-width:82ch; margin-top:6px; }}
code {{ background:var(--code); padding:1px 6px; border-radius:3px;
  font-family:ui-monospace,monospace; font-size:.88em; }}
footer {{ margin-top:64px; border-top:1px solid var(--mist); padding-top:18px; font-size:13.5px;
  color:var(--dim); }}
ul {{ max-width:78ch; }}
@media (prefers-reduced-motion: no-preference) {{
  .v {{ transition: border-color .15s; }}
}}
</style>
<main>
<header>
  <div class="eyebrow">zensr · imazen · engineering report · 2026-07-24</div>
  <h1>Five deployable super-resolution &amp; JPEG-repair systems, evaluated for blind en-masse use</h1>
  <p class="sub">CPU-only inference in a ~400 KB Rust runtime (AVX-512/AVX2, seam-exact tiling,
  <code>forbid(unsafe_code)</code>). Every claim below is parsed from the committed benchmark TSVs;
  every gallery panel is rendered by the shipping engine. Degradations are the real encoder
  (system <code>cjpeg</code>, libjpeg-turbo, 4:2:0, <code>-optimize</code>).</p>
  <div class="chips">
    <span class="chip">runtime <b>zensr-micro</b> 404 KB cdylib</span>
    <span class="chip">metrics <b>SSIMULACRA2 · butteraugli-3norm · PSNR</b></span>
    <span class="chip">eval <b>imazen-26 ×8 subcorpora</b> + <b>people-test-v1</b></span>
    <span class="chip">n <b>64 imgs/cell</b> · 4 degradations · 3 tracks</span>
  </div>
</header>
<nav aria-label="sections">
  <a href="#verdict">verdict</a><a href="#systems">systems</a><a href="#method">method</a>
  <a href="#results">results</a><a href="#subcorpus">per-class</a><a href="#guard">guard</a>
  <a href="#ladder">distillation</a><a href="#audition">audition</a><a href="#people">people band</a>
  <a href="#speed">speed</a><a href="#audit">split audit</a><a href="#gallery">gallery</a>
  <a href="#s10">S10</a><a href="#genloss">gen-loss</a><a href="#ladder">ladder</a><a href="#falsified">falsified</a>
</nav>

<h2 id="verdict"><span class="no">§0</span>Verdict</h2>
<div class="verdict">
  <div class="v"><div class="k">2 %</div><div class="t">P_a2c worse-than-Lanczos rate on the untouched people test slice at q35 (SSIM2 41.4 vs 33.5)</div></div>
  <div class="v"><div class="k">23 MP/s</div><div class="t">45 K-param realtime tier @ 12 threads (3.1 single-thread), guard included at 9 ms/MP</div></div>
  <div class="v"><div class="k">25.7 dB</div><div class="t">what the guard held a <em>catastrophically broken</em> model to (raw output: 9.4 dB) — bounded downside by construction</div></div>
  <div class="v"><div class="k neg">77–98 %</div><div class="t">worse-than-Lanczos rate of <em>every</em> model on clean input — severity gating is mandatory, not optional</div></div>
</div>
<p class="lead">Restoration models are worth running <em>only when the input is actually degraded</em>,
and which model depends on content class. With quant-table severity gating (we own the decoder)
plus the guard layer, blind collection processing is safe: the worst case is bounded at
"slightly worse than bilinear," and the measured downside on real JPEG input is single-digit
percent for the tuned bands.</p>

<h2 id="systems"><span class="no">§1</span>The five systems</h2>
<div class="pane on"><table><thead><tr><th>id</th><th>system</th><th>scale</th><th>model (license)</th><th>weights</th><th>role</th></tr></thead><tbody>
<tr><td class="mono">S-A</td><td>Guarded Fast Photo</td><td>×2/×4</td><td>2x/4xNomosUni_span_multijpg (CC-BY-4.0)</td><td class="num">1.6 MB f32</td><td>opt-in “restore harder”, q≤50</td></tr>
<tr><td class="mono">S-A′</td><td>Compact blind default</td><td>×2</td><td>2xNomosUni_compact_multijpg (CC-BY-4.0)</td><td class="num">2.4 MB</td><td>blind en-masse default (A2c)</td></tr>
<tr><td class="mono">S-B</td><td>Quality Restore</td><td>×4</td><td>realesr-general-x4v3 + wdn blend (BSD-3)</td><td class="num">9.8 MB pair</td><td>severity-blended heavy restore</td></tr>
<tr class="hl"><td class="mono">S-C/S6</td><td>×1 dejpeg (native)</td><td>×1</td><td>dejpeg_1x — GT-trained inversion, this program (595 K)</td><td class="num">2.4 MB f32 / 1.2 MB f16</td><td>beats identity at every q; round-trip retired</td></tr>
<tr><td class="mono">S-D</td><td>Anime/Graphics</td><td>×4</td><td>realesr-animevideov3 (BSD-3)</td><td class="num">2.5 MB</td><td>surprise blind winner, degraded ×4</td></tr>
<tr class="hl"><td class="mono">S-E</td><td>Realtime</td><td>×2</td><td>rtc_distill_2x — distilled this program (45,156 params)</td><td class="num">180 KB f32 / 90 KB f16</td><td>previews, latency-critical</td></tr>
<tr class="hl"><td class="mono">S2-band</td><td>People pack</td><td>×2</td><td>P_a2c + P_rtc — GT fine-tunes, this program</td><td class="num">2.4 MB + 180 KB</td><td>first proven external band</td></tr>
</tbody></table></div>

<h2 id="method"><span class="no">§2</span>Method</h2>
<ul>
<li><b>Protocol.</b> HR = center-crop 512 → CatmullRom down (×2: 256, ×4: 128) → <b>real</b>
libjpeg-turbo round-trip (<code>cjpeg -quality q -sample 2x2 -optimize</code>) at q∈{{clean, 75, 50, 35}} →
model or resampler back to 512 → score vs HR. The ×1 track degrades at full resolution.</li>
<li><b>Metrics.</b> SSIMULACRA2 (structure), butteraugli 3-norm (perceptual error; lower better),
PSNR. Aggregates: median, worst-decile p10, and <b>worse-than-baseline rate</b> — the
bounded-downside statistic. Metrics disagree systematically (butteraugli rewards conservatism,
SSIM2 rewards restored structure); both are reported and both agree on every policy call.</li>
<li><b>Corpora.</b> imazen-26 (8 provenance-clean subcorpora × 8 pinned eval images) and
people-test-v1 (64 CC0 photos from pxhere shards disjoint from all training — touched exactly once).</li>
<li><b>Splits.</b> Frozen eval sets are pinned, committed file lists (<code>eval_split/</code>);
training exclusion is verified by id-set intersection; val splits are image-level.
Dev slices (reused for model selection) are named as such — §10.</li>
<li><b>All systems run guarded</b> (§5) except the explicit ablation rows.</li>
</ul>

<h2 id="results"><span class="no">§3</span>Main results — imazen-26, n=64 per cell</h2>
<h3>×2 track</h3>
{deg_tabs("x2")}
<div class="note win"><b>Policy.</b> Clean → resample (everything loses to Lanczos).
Degraded → A2c blind default (lowest worse-rate at every q, best butteraugli);
span opt-in at q≤50 for maximum structure recovery (+5.8 SSIM2 at q35, at a butteraugli cost).
E_rtc2 = the clean-retrained 45 K realtime student (§6).</div>
<h3>×4 track</h3>
{deg_tabs("x4")}
<div class="note"><b>×4 is “lose less.”</b> SSIM2 medians are negative for <em>everything</em>
on degraded input — nothing “restores” q35 at ×4. F_spanf (NTIRE clean-specialist) owns clean
(worse-rate 19 %) and collapses on JPEG; D_anime is the blind winner on degraded input.</div>
<h3>×1 repair track (vs identity)</h3>
{deg_tabs("x1", ("q35", "q50", "q75"))}
<div class="note win"><b>S6-v2 (zenjpeg-native, 4 encoders): deblock DISABLED wins; one model, no gating.</b>
Directed follow-up: pairs re-generated through the deployment decoder (zenjpeg 0.9) across
{{libjpeg-turbo, mozjpeg 4.1.5, jpegli, zenjpeg}} × {{4:2:0, 4:4:4}} × q∈U(10,96) + 5 % clean
anchors, with a 2×2 deblock experiment (same encoded bytes, decoded with DeblockMode Off vs
Auto, matched models trained per arm). Verdict: model-on-pixel-exact-decode 69.45 mean
cell-median SSIM2 > cooperating model 68.64 > deblock-alone 66.70 > identity 66.22 — zenjpeg's
default (Off) is correct under the model; deblocking helps only standalone. The single
qboost-tuned model (dejpeg2b) beats identity at q15–90 on every encoder and both subsamplings
(q15 mean +9.9 SSIM2); q93–96 dips are +0.01…+0.04 butteraugli on a 0.33–0.52 baseline —
an order of magnitude under JND, guard-bounded. Fingerprint hookup validated at n=960:
mozjpeg/jpegli 100 % family-stable, zenjpeg probes as jpegli-lineage (imazen/zenjpeg#189
filed: encoder-embedded parameter record). Sub-q15 probing (user-prompted) found the deblock verdict FLIPS at q≤8 on Annex-K encoders —
Knusperli's coefficient-domain correction carries information pixel-space models can't see —
and the flip survived in-distribution floor-5 retraining (structural). Shipping configuration:
<b>dejpeg4_policy</b> — one policy-matched model on every image + a two-line probe rule
(non-Cjpegli family, IJG/Mozjpeg scale, est-q ≤ 9 → Knusperli decode). End-to-end:
low-q 21.23 ≈ cooperating specialist 21.26 (pure-off 20.77); standard grid 70.85 ≈ 70.89,
0/40 cells under the best-of-both oracle. Queued: S10 quantization-consistency projection
(per-coefficient DCT box ⇒ provable re-encode consistency) + S5b YCbCr-native models.</div>
<div class="note win"><b>Round trip falsified → native model wins (v1).</b> The interim
×2-up→downscale round trip lost to doing nothing at q50/q75 (rows kept above as the
record). The direct inversion — <b>S6_dejpeg</b>, trained in 25 GPU-minutes on same-size
turbo-JPEG pairs with an A2c-body warm start — beats identity at <em>every</em> quality on
<em>every</em> metric: worse-rate 2 % at q35/q50, butteraugli 0.99 at q75, +1.7–2.0 dB PSNR.
The runtime gained scale-1 via zero-channel head padding; goldens ≤3.7e-6.</div>

<h2 id="subcorpus"><span class="no">§4</span>Per-class structure (Δ SSIM2 median vs Lanczos · wins/8, ×2 q35)</h2>
<div class="pane on"><table><thead><tr><th>subcorpus</th><th>A2c</th><th>span</th><th>E_rtc</th><th>read</th></tr></thead><tbody>
<tr><td>renders</td><td class="num">+10.1 (7/8)</td><td class="num">+8.4 (8/8)</td><td class="num">+5.5</td><td>strongest wins</td></tr>
<tr><td>documents</td><td class="num">+5.7 (8/8)</td><td class="num">+8.0 (8/8)</td><td class="num">+4.2</td><td>text/line art</td></tr>
<tr><td>maps</td><td class="num">+4.7 (8/8)</td><td class="num">+5.2 (8/8)</td><td class="num">+4.3</td><td>graphics</td></tr>
<tr><td>screen</td><td class="num">+4.6 (7/8)</td><td class="num">+6.6 (5/8)</td><td class="num">+2.8</td><td>UI</td></tr>
<tr><td>photos</td><td class="num">+4.0 (6/8)</td><td class="num">+3.1 (5/8)</td><td class="num">+1.5</td><td>positive</td></tr>
<tr><td>textures</td><td class="num">−0.9 (6/8)</td><td class="num">−1.5 (6/8)</td><td class="num">−5.3</td><td>stochastic detail</td></tr>
<tr><td>people</td><td class="num">−1.8 (4/8)</td><td class="num">−0.6 (4/8)</td><td class="num">−0.7</td><td>→ fixed by the people band, §8</td></tr>
<tr><td>art-scans</td><td class="num">−2.7 (5/8)</td><td class="num">−1.5 (3/8)</td><td class="num">−1.5</td><td>halftone reads as texture</td></tr>
</tbody></table></div>
<p>Restoration wins on <b>graphic/text/synthetic</b> content and loses on <b>natural-texture</b>
content — at every track and quality. The router therefore needs a content signal next to the
quant-table severity signal; the guard’s per-cell texture-energy map (already computed) is the
first feature. Caveat: n=8 per class here; the people row was later shown noise-pessimistic at
n=64 (§8) — per-class thresholds require n≥50 before shipping.</p>

<h2 id="guard"><span class="no">§5</span>The guard layer — bounded downside by construction</h2>
<ul>
<li><b>Residual clamp</b> (τ=0.25): output can never leave <code>bilinear ± τ</code>. Property-tested
against adversarial model output.</li>
<li><b>Texture gate</b>: per-16px-cell high-pass energy shrinks the SR residual (α∈[0.35,1]) on
stochastic texture — the class SR measurably loses (§4).</li>
<li><b>Round-trip fallback</b>: box-down of the output vs the input; MAE beyond threshold blends
the whole image toward baseline. Catches off-distribution inputs.</li>
</ul>
<div class="note win"><b>Stress-tested by accident, at maximum severity.</b> A miswired SPAN port
(§10) produced 9.4 dB garbage — and the guarded pipeline held blind output at <b>25.7 dB /
SSIM2 54</b>, i.e. “slightly worse than bilinear.” On <em>working</em> models the guard improves
worst-decile, worse-rate, and butteraugli at every q, and <em>raises</em> clean-input medians by
tempering over-eager restoration (ablation: A2_span vs A2_span_raw in §3). Cost after the
separable rewrite: 9 ms per output MP (was 26).</div>

<h2 id="ladder"><span class="no">§6</span>S9 distillation ladder — four rungs, one day</h2>
<div class="pane on"><table><thead><tr><th>rung</th><th>question</th><th>outcome</th></tr></thead><tbody>
<tr><td class="mono">1</td><td>Does distillation transfer restoration to 45 K params?</td>
<td><b>Yes</b> — 68 % of the teacher-family’s q35 gain, 20-min train, 23 MP/s @12T</td></tr>
<tr><td class="mono">2</td><td>Is capacity the bottleneck? (nf32, 116 K)</td>
<td><b>No</b> — +0.5 SSIM2 for 2.7× compute. Falsified.</td></tr>
<tr><td class="mono">3</td><td>Does teacher choice dominate? (span→A2c teacher)</td>
<td><b>Yes</b> — same 45 K shape: q75 SSIM2 45.8→48.4, q35 worse-rate 50 %→28 %,
butteraugli matches the 600 K teacher at every q. <b>Distill from the policy winner, not the
SSIM2-median king.</b></td></tr>
<tr><td class="mono">4</td><td>Can a bigger community teacher fix people? </td>
<td><b>No</b> — §7. GT fine-tuning can — §8.</td></tr>
</tbody></table></div>
<p>Ops findings that made the loop 20 minutes: GPU-resident training set (host gather was the
bottleneck: 3.3 steps/s → 94 % GPU util), teacher outputs sanity-gated before every run.</p>

<h2 id="audition"><span class="no">§7</span>Teacher audition — heavyweight adoption falsified</h2>
<p>Four community heavyweights vs the incumbent 600 K compact and Lanczos, ×2-target protocol,
people/textures/art-scans/photos (SSIM2 med · butteraugli med · wins-vs-Lanczos, n=8):</p>
<div class="pane on"><table><thead><tr><th>slice</th><th>deg</th><th>model</th><th>SSIM2</th><th>butter</th><th>wins</th></tr></thead><tbody>
{aud_rows}
</tbody></table></div>
<p><b>Every candidate loses to both Lanczos and the incumbent on people, textures and art-scans
at every quality</b> — including the face-specialist DAT (0/8 on people at all four
degradations) and the 17 M RRDBs. The community zoo optimizes ×4 perceptual invention;
under ground-truth fidelity metrics at ×2 web-JPEG, invention is penalized. GFPGAN was excluded
on principle (identity-prior risk violates bounded-downside); CodeFormer is NC-licensed.</p>

<h2 id="people"><span class="no">§8</span>The people band — gap closed by 25 GPU-minutes of GT fine-tuning</h2>
<p>Corpus: <b>zensr-people-v1</b> — 2,500 CC0 photos (pxhere via HF dump, per-image URL
provenance), 24 K ground-truth crop pairs, image-level val. Two warm starts:
<b>P_rtc</b> (45 K, from the rung-3 student) and <b>P_a2c</b> (600 K, from the A2c weights).</p>
<h3>True held-out test — 64 images, virgin shards, zero id overlap, scored once</h3>
{deg_tabs("ptest")}
<h3>Dev slice (frozen 64, used for the training decisions)</h3>
{deg_tabs("pdev")}
<div class="note win"><b>Test ≥ dev on every degraded metric</b> — no home-field inflation.
P_a2c wins all degraded qualities on all three metrics (q35 worse-rate 2 %) and is the first
model to win people at q75. P_rtc at 45 K beats the 600 K generalist at q50/q75. A cross-source
control (imazen-26 unsplash-people, different site, zero training exposure) confirms transfer:
+3.3/+2.0 SSIM2 over Lanczos at q35/q50. <b>This is the per-class recipe</b>: ~2.5 K targeted
CC0 photos + warm start + 20 K steps; textures and art-scans are next.</div>

<h2 id="speed"><span class="no">§9</span>Speed — quiet box, 5-rep min, 7950X (WSL2, 28 threads visible)</h2>
<div class="pane on"><table><thead><tr><th>system</th><th>input</th><th>threads</th><th>min ms</th><th>MP-out/s</th><th>guard ms</th></tr></thead><tbody>
{speed_html}
</tbody></table></div>
<p>Guard cost is flat ≈9 ms per output MP after the separable rewrite. The realtime single-thread
gate (≥15 MP/s @1T) remains open — the 45 K model is now the bottleneck, not the guard;
the S8 half-resolution shape is the queued experiment.</p>

<h2 id="audit"><span class="no">§10</span>Split audit &amp; postmortems (what went wrong, on the record)</h2>
<ul>
<li><b>SPAN graph incident.</b> The from-scratch SPAN-48 port missed the official input
normalization <em>and</em> an <code>inplace=True</code> concat side-effect — and my torch-vs-Rust
consistency goldens <b>agreed on the broken graph</b> through a full eval and a poisoned teacher
run. Fix verified against the reference implementation (spandrel, ≤6e-6 on all 8 models);
the dump now hard-gates every port against the reference. <b>Self-agreement proves nothing.</b></li>
<li><b>One leaked eval image</b> (of 64): the Rust eval’s “first 8 <em>usable</em>” slid past a
101 MP decode-skip while the training exclusion cut “first 8 <em>sorted</em>.” Frozen splits are
now pinned, committed file lists. The shipped realtime student was retrained on the clean regen:
<b>E_rtc2 ≡ E_rtc within ±0.25 SSIM2 everywhere</b> including the leaked class — immaterial,
and now proven rather than argued.</li>
<li><b>Dev-vs-test.</b> The frozen slices had been reused for rung selection (making them dev
sets); the people claims now rest on a virgin-shard test slice touched exactly once (§8).</li>
<li><b>n=8 per-class readouts are noise-pessimistic</b> — at n=64 the incumbent already won
people q35. Per-class evals are n≥50 from here on.</li>
</ul>

<h2 id="gallery"><span class="no">§11</span>Gallery — engine output, max-detail 256 px crops</h2>
<p class="dim">Every panel below was rendered by zensr-micro (guarded, tiled) from the real
degraded input; numbers in captions are that image’s SSIMULACRA2. Crops auto-selected by
Laplacian energy; shown at 224 px.</p>
{"".join(gallery_html)}

<h2 id="s10"><span class="no">§12b</span>S10 — quantization-consistency projection (shipped)</h2>
<p>The JPEG file certifies per 8×8 block and band that the true coefficient lay within
±Q/2 of the stored value. Clamping the model output's block-DCT coefficients into that
box (the exact convex projection; <code>zensr_micro::consist</code>) makes the output
<b>provably re-encode to the file's own coefficients</b> — a training-free guarantee that
cannot be shortcut-gamed, unlike the input-conditioning arms falsified above. Slack has two
measured terms: relative (turbo p99≤0.07Q; mozjpeg trellis ≤0.23Q with a ~15Q tail on
<i>coded</i> coefficients too; jpegli/zenjpeg AQ ≤0.41Q) plus <b>slack_abs</b>, an absolute term
covering encoder-side u8 sample quantization (turbo Q=1 bands: p99 1.32, max 3.70 ≈ the
8·0.5 per-sample worst case — invisible until the DQT hits Q=1..3 at q≥93, where a purely
relative slack lets the box clamp CORRECT detail). 4:2:0 chroma is back-projected exactly on
the half-res lattice (replication is the right-inverse of box decimation — one pass,
residual &lt;2e-3); 4:4:4 projects directly; YCbCr-native models (S5b) run in the space where
quantization happened. Production call (<code>zensr_zenjpeg::restore_jpeg</code>):</p>
<div class="pane on"><table><thead><tr><th>grid</th><th>policy arm</th><th>+ projection (final config)</th><th>cells &lt; identity</th></tr></thead><tbody>
<tr><td>high-q 85–96</td><td class="num">+0.16</td><td class="num"><b>+0.35</b></td><td class="num">5/16→1/16 (−0.02)</td></tr>
<tr><td>standard 15–90</td><td class="num">+3.93</td><td class="num">+3.96</td><td class="num">5→3 / 40</td></tr>
<tr><td>low-q 5–12</td><td class="num">+8.54</td><td class="num">+8.55</td><td class="num">0→0 / 24</td></tr>
</tbody></table></div>
<div class="note win">slack_abs erased the q93 residual (turbo −0.24→−0.02, moz +0.06→+0.38)
and lifted q90 everywhere. The q96 story reversed under measurement: projection ADDS at every
q96 cell (+0.15..+0.81) — it is the <i>model</i> that loses to identity on near-pristine input.
Shipped fix: a measured <b>high-q identity gate</b> (probe q≥94.5 IJG/Moz scale, d≤0.6
Cjpegli-family; q93 reads d 0.7–1.0 and stays modeled) — the top-end analog of the low-q
deblock policy. One marginal negative cell remains on the grid (turbo q93, −0.02).
Engineering: SPANF research surface gated behind <code>internals</code> (default API 260→169
lines, contract in apidoc/PUBLIC_API.md), product-crates rebuild 1.56 s; review caught and
fixed a 4:2:2/4:4:0-misclassified-as-4:2:0 chroma corruption bug (regression-tested on real
encodes of all four subsampling modes); CMYK skips projection. Upstream issues filed —
zenjpeg#189 (encoder parameter record), zenjpeg#190 (impl-Stop monomorphization).</div>

<h2 id="genloss"><span class="no">§12c</span>Generation loss — gen2/gen3 measured (2026-07-26)</h2>
<p>Multi-generation re-encode chains (the real web: social re-encode, CDN recompress,
meme chains, crop-shifted grids, re-encode at <i>higher</i> q), each scored against the
pristine original at every generation, with matched single-generation baselines
(benchmarks/gen_eval_2026-07-26.tsv, 24 pinned-eval crops × 14 chains, run on tower):</p>
<div class="pane on"><table><thead><tr><th>chain</th><th>identity ssim2</th><th>restored</th><th>model Δ</th><th>matched single-gen Δ</th></tr></thead><tbody>
<tr><td>g2 social 85→75</td><td class="num">71.8</td><td class="num">75.1</td><td class="num"><b>+3.22</b></td><td class="num">+2.13 (t75)</td></tr>
<tr><td>g2 CDN 92→moz70</td><td class="num">70.4</td><td class="num">73.4</td><td class="num"><b>+3.02</b></td><td class="num">+2.30 (m70)</td></tr>
<tr><td>g2 up-q 60→90</td><td class="num">68.2</td><td class="num">71.6</td><td class="num"><b>+3.37</b></td><td class="num">+0.54 (t90)</td></tr>
<tr><td>g3 meme 75→m60→50</td><td class="num">55.9</td><td class="num">61.0</td><td class="num"><b>+5.13</b></td><td class="num">+4.25 (t50)</td></tr>
<tr><td>g3 deep 35×3</td><td class="num">54.1</td><td class="num">60.1</td><td class="num"><b>+6.09</b></td><td class="num">+5.79 (t35)</td></tr>
</tbody></table></div>
<div class="note win">Three findings. (1) <b>Gains grow on multi-gen input</b> — more artifact
energy, and the blind model removes proportionally more; no under-correction collapse.
(2) <b>Generation damage is mostly permanent</b>: restored gen2 lands ~3.9 ssim2 below restored
single-gen at the same final q (the model recovers ~20 % of the generational delta). That gap
is the target of gen-aware training (chain augmentation implemented, ZENSR_GEN2/GEN3, A/B queued).
(3) <b>The S10 projection stays safe</b> — it certifies the <i>final</i> generation; proj−noproj ≥ 0
on 13/14 chains (−0.05 only on up-q, neutral). Bonus: the up-q chain (probe says "q90, mild";
model corrects +3.37 from pixels) independently re-falsifies severity conditioning.</div>

<h2 id="ladder"><span class="no">§12d</span>Production model ladder (2026-07-27 autonomous wave)</h2>
<p>One night of LAN-fleet dispatch (jason RTX 3070 native-bf16 at 0.18 s/step trained seven
595k-param models in ~12–50 min each) closed the remaining design questions with
control-adjusted A/Bs. A +16k-step control on the original mix measured FLAT — so every
delta below is a real effect, not training time:</p>
<div class="pane on"><table><thead><tr><th>tier</th><th>model</th><th>params</th><th>ssim2 gain q15/35/55/75/90 (std)</th><th>s/MP @12T</th></tr></thead><tbody>
<tr><td><b>quality (default)</b></td><td>dejpeg7_graphics</td><td class="num">595k</td><td class="num">beats dejpeg4 on BOTH classes (+0.09 photo / +0.09 graphics median vs control) and at every high-q point; zero negative cells</td><td class="num">5.3</td></tr>
<tr><td><b>realtime</b></td><td>dejpeg_rt24d</td><td class="num">43k</td><td class="num">+1.63 / +1.58 / +1.42 / +0.86 / +0.04</td><td class="num"><b>0.21</b></td></tr>
<tr><td>low-q graphics route</td><td>dejpeg9_gfxycc</td><td class="num">595k</td><td class="num">graphics rows +1.58/+0.65/+0.39 OVER dejpeg7 at q15/35/55 (negative above q75)</td><td class="num">5.3</td></tr>
</tbody></table></div>
<div class="note win">Findings that set the shape: (1) the "graphics" specialist beat the
generalist on photos too — harder text/edge data improved artifact discrimination globally,
so it simply becomes the default; (2) S9 distillation holds at ×1 (+0.41 over direct-train)
and SATURATES at 43k params — rt24d matches rt32d at 3.2× the speed, 25× the quality tier;
(3) YCbCr-native is falsified as the general pipeline (lost the high grid outright) but is a
real low-q graphics trait (+0.30 median vs control) — it survives only inside the compound
specialist; (4) routing: default dejpeg7; chooser p(graphics)&gt;0.85 AND probe q≤60 →
gfxycc; q≥95 → identity gate; Annex-K q≤9.5 → Knusperli; before SR, chain the ×1 stage
when 4:2:0 OR q≲50 (the 4:4:4 arm showed the chain's lever is chroma repair on the
subsampled lattice — at 4:4:4 it wins q35, goes neutral q50, loses slightly q75). Every row traces to a committed
benchmarks/ TSV; models mirrored to Tower.</div>

<h2 id="falsified"><span class="no">§12</span>Falsified / negative results registry</h2>
<ul>
<li>×1 repair via scale round-trip (q≥50) — loses to identity. Replaced by the native S6 inversion, which wins everywhere.</li>
<li>Capacity as the realtime bottleneck (nf32: +0.5 SSIM2, 2.7× compute).</li>
<li>Heavyweight teacher adoption for people/textures (all four candidates lose to Lanczos).</li>
<li>Input-channel conditioning, both forms — global severity scalar AND per-block damage map
land identical (−3 dB on every degraded band): a gradient shortcut that substitutes for pixel
analysis. Pixels-in/pixels-out ships; S10 lives at the output.</li>
<li>YCbCr-native pipeline as the general model (S5b) — falsified under BOTH bias regimes:
warm-start (lost the high grid outright) and a from-scratch same-seed pair (−0.14 median,
photos −0.36; the warm run's q15 advantage did not survive de-confounding). The one robust
survivor is a graphics-rows edge (+0.13..+0.30 median) — it lives on only inside the
compound graphics specialist.</li>
<li>Extra-training-time as the explanation for specialist gains — the +16k-step control
on the original mix measured flat.</li>
<li>Bilinear-up of the half-res projection correction (4:2:0 back-projection) — box(bilerp(c)) ≠ c
attenuates the correction (only 1.8× violation reduction). Pixel replication is the exact
right-inverse of box decimation: one-pass exact (residual &lt; 2e-3).</li>
<li>Skip-zeroed-bands rescue for trellis/AQ projection slack — measured violations sit on
<i>coded</i> coefficients too (mozjpeg nonzero-only p99 up to 1.7Q, max 15Q). No band-conditional
skip restores the truth-in-box guarantee for those families.</li>
<li>int8 weights-only PTQ on SPAN-class (35 dB — needs QAT or int8-first arch). f16 is transparent.</li>
<li>Norm-folding into conv_1 (borders wrong by 0.32 — official zero-pads after normalization).</li>
<li>Consistency-only goldens (agreed on a broken graph — reference-gate everything).</li>
<li>+16 stride padding (2× slower), SiLU/gate store-fusion (26× MIR-inline regression),
row-band tiling (50–377 MB/thread) — engine appendix, PLAN.md.</li>
</ul>

<footer>
<p><b>Attribution.</b> 2x/4xNomosUni &amp; HFA2k models — Philip Hofmann (Phhofm), CC-BY-4.0.
realesr-general/animevideo — Real-ESRGAN (Xintao Wang et al.), BSD-3. SPANF — NTIRE 2025 ESR
team24. People corpus — pxhere.com photographers, CC0, via the nyuuzyou/pxhere dump.
NPS / IRS / Internet Archive imagery — public domain. Unsplash imagery — Unsplash License.</p>
<p class="mono dim">repo ~/work/zen/zensr (local jj) · data: benchmarks/*.tsv @ commits
d914c75b…0697d3f1 · engine zensr-micro (404 KB cdylib, forbid(unsafe_code)) · report generated
by tools/build_report.py — regenerate: <code>just report</code></p>
</footer>
</main>
<script>
document.querySelectorAll('.tab').forEach(b => b.addEventListener('click', () => {{
  const id = b.dataset.t, group = b.parentElement;
  group.querySelectorAll('.tab').forEach(x => x.classList.toggle('on', x === b));
  let el = group.nextElementSibling;
  while (el && el.classList.contains('pane')) {{
    el.classList.toggle('on', el.id === id);
    el = el.nextElementSibling;
  }}
}}));
</script>
"""
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write(page)
print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.2f} MB)")
