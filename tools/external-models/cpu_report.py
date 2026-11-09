#!/usr/bin/env python3
"""Summarize complete codec-free CPU packets using whole-process observations."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics


def read_packet(path, matrix):
    packet=json.loads(gzip.decompress(path.read_bytes()))
    plan=packet['plan']
    if not packet['complete'] or plan['blocks']!=3:
        raise ValueError('requires a complete three-block packet')
    arms=[plan['baseline'],*plan['candidates']]
    expected=len(plan['cases'])*plan['blocks']*(len(arms)+int(plan['aa']))
    invocations=packet['invocations']
    if len(invocations)!=expected or any(i['status']!='completed' or i['exit_code']!=0 for i in invocations):
        raise ValueError('incomplete or failed benchmark invocations')
    placement=packet['placement']
    classes=placement.get('topology',{}).get('classes',[])
    selected=[c for c in classes if c['name']==placement.get('core_class')]
    cpus=placement.get('effective_cpus',[])
    if (placement.get('status')!='enforced_uniform' or placement.get('enforcement')!='cpuset'
            or len(selected)!=1 or not cpus or len(cpus)!=len(set(cpus))
            or not set(cpus)<=set(selected[0]['cpus'])
            or selected[0]['performance_rank']!=max(c['performance_rank'] for c in classes)):
        raise ValueError('uniform performance-core CPU placement not verified')
    grouped=defaultdict(list)
    for invocation in invocations:
        case=plan['cases'][invocation['case']]
        arm=invocation['arm']
        if not 0<=arm<=len(arms) or (arm==len(arms) and not plan['aa']):
            raise ValueError('unexpected arm')
        name=arms[arm]['id'] if arm<len(arms) else '__aa'
        values=invocation['values_ns']
        if len(values)!=plan['metric']['expected'] or len(values)!=3 or any(not 0<x<float('inf') for x in values):
            raise ValueError('invalid internal timing vector')
        output=Path(invocation['output'])
        for filename in ['stdout.txt','stderr.txt']:
            p=output/filename
            if hashlib.sha256(p.read_bytes()).hexdigest()!=invocation['artifacts'][str(p)]:
                raise ValueError('retained worker output bytes changed')
        lines=[line[len('IDENTITY '):] for line in (output/'stdout.txt').read_text().splitlines() if line.startswith('IDENTITY ')]
        if len(lines)!=1:raise ValueError('worker identity count differs')
        identity=json.loads(lines[0])
        n=int(case['fields']['n'])
        if identity['shape']!=[1,3,n*4,n*4] or invocation['correctness']!=[identity['input_sha256'],identity['shape']]:
            raise ValueError('input/output correctness contract differs')
        rss=re.findall(r'Maximum resident set size \(kbytes\): (\d+)',(output/'stderr.txt').read_text())
        if len(rss)!=1:raise ValueError('missing worker RSS measurement')
        grouped[(case['id'],name)].append(dict(block=invocation['block'],mean=statistics.mean(values),
            rss=int(rss[0]),digest=identity['output_sha256'],identity=identity))
    rows=[]
    for (case_id,name),observations in sorted(grouped.items()):
        if len(observations)!=3 or {o['block'] for o in observations}!={0,1,2}:
            raise ValueError('missing or repeated process block')
        if len({o['digest'] for o in observations})!=1:
            raise ValueError('fresh-process output hashes differ')
        case=next(c for c in plan['cases'] if c['id']==case_id)
        n=int(case['fields']['n']);threads=int(case['fields']['threads'])
        times=[o['mean']/1e6 for o in observations]
        median=statistics.median(times)
        rows.append(dict(matrix=matrix,case=case_id,model=name,input_side=n,output_side=n*4,
            requested_threads=threads,processes=3,samples_per_process=3,
            median_process_mean_ms=median,min_process_mean_ms=min(times),max_process_mean_ms=max(times),
            mad_process_mean_ms=statistics.median(abs(t-median) for t in times),
            output_mpix_per_second=(n*4)**2/1e6/(median/1000),
            max_worker_rss_kib=max(o['rss'] for o in observations),output_sha256=observations[0]['digest']))
    if len(rows)!=len(plan['cases'])*(len(arms)+int(plan['aa'])):
        raise ValueError('missing case/model coverage')
    return rows,packet


def main(a):
    rows=[]
    identities={}
    for matrix,path in [('primary',a.primary),('native-large',a.large)]:
        result,packet=read_packet(path,matrix);rows.extend(result)
        identities[matrix]=dict(packet_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            fingerprint=packet['fingerprint'],elapsed_seconds=packet['elapsed_seconds'],
            placement=packet['placement'],invocations=len(packet['invocations']))
    a.output.mkdir(exist_ok=True)
    with (a.output/'results.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (a.output/'IDENTITY.json').write_text(json.dumps(identities,sort_keys=True,indent=2)+'\n')
    for row in rows:
        if row['model']!='__aa':print(json.dumps(row))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['primary','large','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
