#!/usr/bin/env python3
"""Select an outcome-blind, audited input for a descriptive cost pilot."""
import argparse
import hashlib
import json
from pathlib import Path

from publication_routing_analysis import read_cells
from publication_routing_cells import digest


def choose(rows,size,encoder,ss,quality):
    eligible=[r for r in rows if
        (r['size'],r['encoder'],r['ss'],r['nominal_q'])==
        (str(size),encoder,ss,str(quality))]
    if not eligible:raise ValueError('required cost-pilot cell missing')
    return min(eligible,key=lambda r:(r['file'],r['reference_sha256']))


def prepare(root,output,binary,weights,size,encoder,ss,quality):
    if output.exists():raise ValueError('refusing to overwrite cost inputs')
    rows,cells_sha=read_cells(root,'router_development')
    row=choose(rows,size,encoder,ss,quality)
    jpeg=root/row['jpeg_artifact']
    jpeg_sha=digest(jpeg)
    if jpeg_sha!=jpeg.name.split('.')[0]:raise ValueError('cost JPEG hash changed')
    label=f"{row['reference_sha256'][:12]}-{size}-{encoder}{ss}-q{quality}"
    manifest=f'label\tpath\n{label}\t{jpeg.resolve()}\n'
    record=dict(rule='First canonical filename, reference-hash tie break, in the requested audited development geometry/encoder/subsampling/quality cell. No quality outcomes used.',
        scope='Descriptive resource pilot on one input; not a population latency estimate or complete publication cost study.',
        cells_sha256=cells_sha,audit_sha256=digest(root/'AUDITED.json'),
        selector_sha256=digest(Path(__file__)),binary_sha256=digest(binary),weights_sha256=digest(weights),
        manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(),
        input={k:row[k] for k in ('file','reference_sha256','origin','width','height','size','encoder','ss','nominal_q','jpeg_artifact')},
        jpeg_sha256=jpeg_sha,jpeg_bytes=jpeg.stat().st_size)
    output.mkdir(parents=True)
    (output/'inputs.tsv').write_text(manifest)
    (output/'selection.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','output','binary','weights'):p.add_argument(name,type=Path)
    p.add_argument('--size',type=int,required=True)
    p.add_argument('--encoder',required=True)
    p.add_argument('--ss',choices=('420','444'),required=True)
    p.add_argument('--quality',type=int,required=True)
    a=p.parse_args()
    prepare(a.root,a.output,a.binary,a.weights,a.size,a.encoder,a.ss,a.quality)
