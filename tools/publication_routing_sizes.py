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
from publication_routing_analysis import BASE, COEF, PIXELS, read_cells

SIZES=(32,40,48,64,80,96,128,160,192,256,320,384,512,640,768,1024,1280,1536,2048,4096)


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


def prepare(root,manifest,output,role,k):
    if output.exists():raise ValueError('refusing to overwrite representative selection')
    if k<1:raise ValueError('clusters per content must be positive')
    rows,digest=read_cells(root,role)
    selected,clusters=representatives(rows,k)
    with manifest.open() as f:
        reader=csv.DictReader(f,delimiter='\t');fields=reader.fieldnames;references=list(reader)
    subset=[r for r in references if r['reference_sha256'] in selected]
    if len(subset)!=len(selected) or any(r['role']!=role for r in subset):raise ValueError('manifest membership mismatch')
    output.mkdir(parents=True)
    target=output/'references.tsv'
    with target.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,delimiter='\t');writer.writeheader();writer.writerows(subset)
    record=dict(input_cells_sha256=digest,source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        selected_manifest_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        rule='One canonical file per origin; per-content standardized decoded features at turbo 420 q50, 512 max edge; centroid-nearest representative of every cluster, including singleton clusters.',
        clusters_per_content=k,seed=20261104,features=list(BASE+COEF+PIXELS),clusters=clusters,sizes=list(SIZES),
        native_size_coverage={str(size):sum(max(int(r['width']),int(r['height']))>=size for r in subset) for size in SIZES},
        limitation='Purposeful feature-cluster sample, not a random population sample; report size results separately from the complete primary cohort.')
    (output/'selection.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(references=len(subset),origins=len({r['origin'] for r in subset}),native_size_coverage=record['native_size_coverage']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('manifest',type=Path)
    parser.add_argument('output',type=Path);parser.add_argument('--role',choices=('router_development','locked_test'),required=True)
    parser.add_argument('--clusters-per-content',type=int,default=2)
    a=parser.parse_args();prepare(a.root,a.manifest,a.output,a.role,a.clusters_per_content)
