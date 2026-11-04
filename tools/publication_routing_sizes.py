#!/usr/bin/env python3
"""Choose size-study representatives from decoded features, never outcomes."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from publication_routing_analysis import BASE, COEF, PIXELS, STUDY_SIZES, read_cells

SIZES=STUDY_SIZES


def representatives(rows,k):
    candidates={}
    for row in sorted(rows,key=lambda r:r['file']):
        if (row['encoder'],row['ss'],row['nominal_q'],row['size'])!=('turbo','420','50','512'):continue
        candidates.setdefault(row['origin'],row)
    if not candidates:raise ValueError('missing fixed clustering input cells')
    strata=collections.defaultdict(list)
    for row in candidates.values():strata[row['content']].append(row)
    selected=[];clusters=[]
    for content,members in sorted(strata.items()):
        features=BASE+COEF+PIXELS
        values=np.asarray([[float(r[c]) for c in features] for r in members])
        if not np.isfinite(values).all():raise ValueError('non-finite clustering input')
        x=StandardScaler().fit_transform(values)
        count=min(k,len(members))
        if count==len(members):
            labels=np.arange(count);centers=x
        else:
            model=KMeans(n_clusters=count,n_init=10,random_state=20261104,algorithm='lloyd').fit(x)
            labels=model.labels_;centers=model.cluster_centers_
        for label in sorted(set(labels)):
            indices=np.flatnonzero(labels==label)
            representative=min(indices,key=lambda i:(float(np.square(x[i]-centers[label]).sum()),members[i]['file']))
            row=members[representative];selected.append(row['reference_sha256'])
            clusters.append(dict(content=content,cluster=int(label),origins=len(indices),
                representative_sha256=row['reference_sha256'],
                members=[members[i]['origin'] for i in indices]))
    return selected,clusters



NATIVE_RULE=dict(cluster_floor=2048,all_origins_floor=4096)


def native_representatives(rows,references,k):
    """Supplement the feature sample using native geometry, never quality labels."""
    selected,clusters=representatives(rows,k)
    for cluster in clusters:cluster['native_floor']=0
    geometry={r['reference_sha256']:max(int(r['width']),int(r['height'])) for r in references}
    if any(r['reference_sha256'] not in geometry for r in rows):
        raise ValueError('missing native reference geometry')
    eligible=[r for r in rows if geometry[r['reference_sha256']]>=NATIVE_RULE['cluster_floor']]
    if eligible:
        extra,groups=representatives(eligible,k)
        selected.extend(extra)
        for group in groups:group['native_floor']=NATIVE_RULE['cluster_floor']
        clusters.extend(groups)
    # Choose an eligible canonical file, not the origin's possibly smaller file.
    large={}
    for r in sorted(rows,key=lambda r:r['file']):
        if (r['encoder'],r['ss'],r['nominal_q'],r['size'])!=('turbo','420','50','512'):continue
        if geometry[r['reference_sha256']]>=NATIVE_RULE['all_origins_floor']:
            large.setdefault(r['origin'],r)
    for origin,r in sorted(large.items()):
        selected.append(r['reference_sha256'])
        clusters.append(dict(content=r['content'],cluster=0,origins=1,
            native_floor=NATIVE_RULE['all_origins_floor'],
            representative_sha256=r['reference_sha256'],members=[origin]))
    return list(dict.fromkeys(selected)),clusters


def prepare(root,manifest,output,role,k):
    if output.exists():raise ValueError('refusing to overwrite representative selection')
    if k<1:raise ValueError('clusters per content must be positive')
    rows,digest=read_cells(root,role)
    with manifest.open() as f:
        reader=csv.DictReader(f,delimiter='\t');fields=reader.fieldnames;references=list(reader)
    selected,clusters=native_representatives(rows,references,k)
    subset=[r for r in references if r['reference_sha256'] in selected]
    if len(subset)!=len(selected) or any(r['role']!=role for r in subset):raise ValueError('manifest membership mismatch')
    output.mkdir(parents=True)
    target=output/'references.tsv'
    with target.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,delimiter='\t');writer.writeheader();writer.writerows(subset)
    record=dict(selector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_cells_sha256=digest,source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        selected_manifest_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        rule='Preserve per-content feature representatives; add per-content representatives among native-edge >=2048 files; include every native-edge >=4096 origin. Canonical eligible file per origin within each stratum; standardized turbo 420 q50 512 features; centroid-nearest including singleton clusters.',
        native_rule=NATIVE_RULE,clusters_per_content=k,seed=20261104,features=list(BASE+COEF+PIXELS),clusters=clusters,sizes=list(SIZES),
        native_origin_coverage={str(size):len({r['origin'] for r in subset if max(int(r['width']),int(r['height']))>=size}) for size in SIZES},
        native_size_coverage={str(size):sum(max(int(r['width']),int(r['height']))>=size for r in subset) for size in SIZES},
        limitation='Purposeful feature-cluster sample, not a random population sample; report size results separately from the complete primary cohort.')
    (output/'selection.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(references=len(subset),origins=len({r['origin'] for r in subset}),native_size_coverage=record['native_size_coverage']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('manifest',type=Path)
    parser.add_argument('output',type=Path);parser.add_argument('--role',choices=('router_development','locked_test'),required=True)
    parser.add_argument('--clusters-per-content',type=int,default=2)
    a=parser.parse_args();prepare(a.root,a.manifest,a.output,a.role,a.clusters_per_content)
