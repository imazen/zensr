#!/usr/bin/env python3
"""Owner-authorized score-table comparison without a completed pixel audit.

Uses the frozen evaluator, models and policy selections. Its table-loader hook
checks table completeness and coverage, without requiring AUDITED.json. Every
output explicitly records that artifact verification is incomplete.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import publication_routing_analysis as study
from publication_routing_cells import required_coverage
from publication_routing_tables import cell_bytes


def read_score_tables(root, expected_role):
    if not (root/'COMPLETE').is_file():raise ValueError('incomplete score grid')
    rows=[];digest=hashlib.sha256();keys=set()
    for name,data in cell_bytes(root):
        digest.update(name.encode());digest.update(data)
        cell=list(csv.DictReader(data.decode().splitlines(),delimiter='\t'))
        if len(cell)!=1:raise ValueError('incomplete score cell')
        r=cell[0]
        if r['role']!=expected_role:raise ValueError('unexpected dataset role')
        key=tuple(r[c] for c in ('reference_sha256','size','encoder','ss','nominal_q'))
        if key in keys:raise ValueError('duplicate score cell')
        keys.add(key);rows.append(r)
    if not rows:raise ValueError('empty score grid')
    expected=required_coverage(root)
    actual={tuple(r[c] for c in ('file','size','encoder','ss','nominal_q')) for r in rows}
    if actual!=expected:raise ValueError('incomplete coverage')
    return rows,digest.hexdigest()


def evaluate(root,selection,output,partial_audit):
    if output.exists():raise ValueError('refusing to replace report')
    temporary=output.with_suffix('.scores-only.partial')
    original_loader=study.read_cells
    try:
        # Explicit compatibility hook: frozen numerical evaluation stays intact;
        # only the owner's cancelled artifact-audit prerequisite is removed.
        study.read_cells=read_score_tables
        study.evaluate(root,selection,temporary)
    finally:
        study.read_cells=original_loader
    result=json.loads(temporary.read_text())
    result['artifact_verification']=dict(status='incomplete_owner_cancelled',
        interpretation='Comparison of completed score tables; not a completed pixel-artifact audit.',
        partial_audit=json.loads(partial_audit.read_text()),
        score_loader_sha256=study.sha(__file__),frozen_evaluator_sha256=study.sha(study.__file__))
    result['reference_exclusion']=json.loads((root/'REFERENCE_EXCLUSION.json').read_text())
    temporary.write_text(json.dumps(result,indent=2)+'\n')
    temporary.rename(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','selection','output','partial_audit'):p.add_argument(name,type=Path)
    p.add_argument('--allow-unaudited-artifacts',action='store_true',required=True)
    a=p.parse_args();evaluate(a.root,a.selection,a.output,a.partial_audit)
