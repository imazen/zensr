"""Build an inspection gallery from exact retained images and scored endpoints."""
import argparse
import html
import json
from urllib.parse import urlencode
from pathlib import Path

import polars as pl


def main(a):
    data = json.loads((a.output / 'WITNESSES.json').read_text())
    scores = pl.read_parquet(a.scores).filter(pl.col('scale') == 4)
    sections = []
    fixed = data.get('projection_status') == 'fixed'
    previous = {c['id']: c for c in data.get('previous_cases', [])}
    checked = 0
    for case in data['cases']:
        case_scores = scores.filter(pl.col('id') == case['id'])
        reference = case['reference']
        figures = []

        def figure(label, image, row=None):
            nonlocal checked
            path = html.escape(image['path'], quote=True)
            if row is not None:
                if row['rgb8_sha256'] != image['pixels_sha256']:
                    raise ValueError('Gallery image differs from scored endpoint')
                caption = f"A {row['zensim_a']:.4f} · PSNR {row['psnr']:.4f} dB"
                checked += 1
            else:
                caption = 'Scoring reference'
            figures.append(f'<figure><a href="{path}"><img src="{path}" alt="{html.escape(label)}"></a>'
                           f'<figcaption>{html.escape(label)}<br><small>{caption}</small></figcaption></figure>')

        figure('HR reference', reference)
        baseline = next(r for r in case['filters'] if r['filter'] == 'catmull_rom')
        row = case_scores.filter((pl.col('repair') == 'direct') & (pl.col('model') == 'spanf') &
                                (pl.col('filter') == 'catmull_rom') & (pl.col('intensity_quarter') == 0)).to_dicts()
        if len(row) != 1:
            raise ValueError('Plain resize score missing or ambiguous')
        figure('CatmullRom — no repair', baseline, row[0])
        for model in ['spanf', 'dscf']:
            for repair in ['direct', 'repaired']:
                image = next(r for r in case['sr'] if r['model'] == model and r['repair'] == repair)
                row = case_scores.filter((pl.col('repair') == repair) & (pl.col('model') == model) &
                                        (pl.col('filter') == 'triangle') & (pl.col('intensity_quarter') == 4)).to_dicts()
                if len(row) != 1:
                    raise ValueError('SR score missing or ambiguous')
                figure(f"{model.upper()} — {'no repair' if repair == 'direct' else 'full repair first'}", image, row[0])
        # Verify all exported resize controls, even those not shown in the overview.
        for image in case['filters']:
            row = case_scores.filter((pl.col('repair') == 'direct') & (pl.col('model') == 'spanf') &
                                    (pl.col('filter') == image['filter']) & (pl.col('intensity_quarter') == 0)).to_dicts()
            if len(row) != 1 or row[0]['rgb8_sha256'] != image['pixels_sha256']:
                raise ValueError('Resize control differs from scored endpoint')
            checked += 1
        inputs = []
        input_images = [('Clean LR', case['clean_lr']), ('Decoded JPEG', case['decoded_lr'])]
        guard = next((r for r in case.get('repair_stages', []) if r['label'].endswith('/guard')), None)
        if guard is not None:
            input_images.append(('Guarded CNN — before constraints', guard))
        if case['id'] in previous:
            input_images.append(('Previous coefficient projection', previous[case['id']]['repaired_lr']))
        input_images.append(('Corrected coefficient projection' if fixed else 'After coefficient projection', case['repaired_lr']))
        for label, image in input_images:
            path = html.escape(image['path'], quote=True)
            inputs.append(f'<div><a href="{path}"><img src="{path}" alt="{label}"></a><p>{label} · '
                          f'{image["width"]} × {image["height"]}</p></div>')
        if guard is not None:
            query = urlencode(dict(reference='/gallery/' + guard['path'],
                                   candidate='/gallery/' + case['repaired_lr']['path'],
                                   beforeLabel='Guarded CNN — before constraints',
                                   afterLabel='After coefficient projection'))
            inputs.append(f'<p><a href="/?{html.escape(query, quote=True)}">Compare native LR: guarded CNN ↔ coefficient projection</a></p>')
        if case['id'] in previous:
            old = previous[case['id']]
            for model in ['spanf', 'dscf']:
                before = next(r for r in old['sr'] if r['model'] == model and r['repair'] == 'repaired')
                after = next(r for r in case['sr'] if r['model'] == model and r['repair'] == 'repaired')
                query = urlencode(dict(reference='/gallery/' + before['path'], candidate='/gallery/' + after['path'],
                                       beforeLabel=model.upper() + ' — previous repair', afterLabel=model.upper() + ' — corrected repair'))
                inputs.append(f'<p><a href="/?{html.escape(query, quote=True)}">Compare {model.upper()}: previous repair → corrected repair</a></p>')
        note = ('Previously failing size' if fixed else 'Worst-case size') if case['edge'] == 32 else 'Same source and quality at a larger size'
        sections.append(f'<section><h2>{html.escape(case["id"])} · {note}</h2><p>Output '
                        f'{reference["width"]} × {reference["height"]}; both SR models at 100% intensity. '
                        'Full repair is a forced candidate, not a runtime routing decision.</p>'
                        '<div class="grid">' + ''.join(figures) + '</div>'
                        '<details><summary>Inspect native LR inputs (magnified)</summary><div class="inputs">' +
                        ''.join(inputs) + '</div></details></section>')
    source = data['source']['source']
    original = html.escape(source['raw_url'], quote=True)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Repair/SR worst-case witnesses</title><style>
body{font:16px system-ui,sans-serif;line-height:1.5;background:#f4f5f1;color:#18251b;margin:24px auto;max-width:1500px;padding:0 20px}
h1,h2{line-height:1.2}section{margin:32px 0;padding:20px;background:white;border:1px solid #cbd5c9;border-radius:12px}
.grid,.inputs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}figure{margin:0}img{width:100%;height:auto;image-rendering:pixelated;border:1px solid #a4b2a2;background:#ddd}
figcaption{margin:8px 0}small{color:#48594b}a{color:#175a31}details{margin:20px 0}.controls{position:sticky;top:0;background:#f4f5f1;padding:12px;border-bottom:1px solid #bac9b9;z-index:2}
@media(max-width:800px){.grid,.inputs{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:500px){.grid,.inputs{grid-template-columns:1fr}}
</style><h1>Repair/SR worst-case witnesses</h1>
<p>One validation manuscript image, JPEG 420 at q40 and q50, at three input sizes. The 32-pixel cases contain the two reported worst losses. All images are lossless exports of retained pixels; no inference rerun or metric maps.</p>
<p>The gallery magnifies small images. Open a comparison and select <b>100%</b> to inspect actual output pixels. Drag to pan, wheel to zoom, use Flicker or hold Space to alternate. The viewer also has a smooth/pixelated toggle.</p>
<div class="controls"><label><input id="smooth" type="checkbox"> Smooth overview magnification</label> · Score: zensim-A (higher is better; can be negative) and PSNR RGB.</div>
'''
    page += f'<p><a href="{original}">Original source ({source["width"]} × {source["height"]})</a>. The HR comparison references below are downscaled, whole-frame references for the measured cases.</p>'
    oracle_path = a.output / 'JPEG_ORACLE.json'
    if oracle_path.exists():
        oracle = json.loads(oracle_path.read_text())
        if len(oracle['comparisons']) != len(data['cases']) or any(r['max_rgb_difference'] != 0 for r in oracle['comparisons']):
            raise ValueError('JPEG oracle comparison is incomplete or disagrees')
        page += '<p><b>JPEG cross-check:</b> libjpeg-turbo and zenjpeg decode all six retained JPEGs to identical RGB8. The tiny 32 × 8 JPEGs already contain a green cast; the larger versions here remain grayscale. Forced repair strengthens that existing color error. These inputs were generated by the study’s pinned JPEG encoder; this is not a claim about a newer encoder revision.</p>'
    if fixed:
        page = page.replace('Repair/SR worst-case witnesses', 'Corrected repair/SR witnesses')
        page = page.replace('The 32-pixel cases contain the two reported worst losses.', 'The 32-pixel cases previously showed the two reported worst losses; these outputs use the corrected shared projection.')
        page += '<p><b>Corrected runtime:</b> native and browser repair now reconstruct hidden DCT samples from the stored coefficients. JPEG inputs and models are unchanged here, isolating the projection fix. Expand the LR inputs for before/after constraints, or compare the previous and corrected repair-first upscales. The encoder padding fix is also in the native review PR.</p>'
    if not fixed and any(c.get('repair_stages') for c in data['cases']):
        page += '<p><b>Edge diagnosis:</b> the encoder padding bug is fixed in the draft PR; these frozen inputs are preserved. On the two tiny cases, coefficient projection increases maximum channel spread from 38/43 to 92/94. A controlled experiment using the encoded hidden block samples instead of replicated edge samples gives exactly the guarded CNN pixels. Expand the LR inputs below to compare before and after constraints. Runtime projection remains unchanged.</p>'
    page += ''.join(sections)
    page += '''<script>document.getElementById('smooth').onchange=e=>{for(const image of document.images)image.style.imageRendering=e.target.checked?'auto':'pixelated'}</script></html>'''
    (a.output / 'witnesses.html').write_text(page)
    result = dict(cases=len(data['cases']), checked_scored_endpoints=checked,
                  source_record_sha256=data['source_record_sha256'])
    (a.output / 'GALLERY_CHECKS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scores', type=Path, required=True)
    main(p.parse_args())
