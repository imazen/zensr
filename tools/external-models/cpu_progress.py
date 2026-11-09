#!/usr/bin/env python3
"""Observe a live zenbench CPU matrix without altering its scientific inputs."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess


def main(a):
    plan=json.loads(a.plan.read_text())
    run=a.root/a.name
    path=run/'events.jsonl'
    latest={}
    if path.exists():
        text=path.read_text()
        lines=text.splitlines()
        for index,line in enumerate(lines):
            try:record=json.loads(line)
            except json.JSONDecodeError:
                if index==len(lines)-1 and not text.endswith('\n'):break
                raise
            latest[(record['case'],record['block'],record['arm'])]=record
    processes=[]
    for line in subprocess.check_output(['ps','-eo','pid=,ppid=,comm=,args='],text=True).splitlines():
        parts=line.strip().split(None,3)
        if len(parts)==4:processes.append(dict(pid=int(parts[0]),parent=int(parts[1]),command=parts[2],args=parts[3]))
    leaders=[p for p in processes if p['command']=='zenbench' and str(a.plan) in p['args']]
    owned={p['pid'] for p in leaders}
    while True:
        descendants={p['pid'] for p in processes if p['parent'] in owned}
        if descendants<=owned:break
        owned|=descendants
    arms=[plan['baseline']['id'],*[c['id'] for c in plan['candidates']],'__aa']
    completed=[r for r in latest.values() if r['status']=='completed']
    required=len(plan['cases'])*plan['blocks']*(1+len(plan['candidates'])+int(plan['aa']))
    record=dict(stage=a.name,run=str(run),plan=str(a.plan),required_invocations=required,
        completed_invocations=len(completed),checked=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        live_processes=[p for p in processes if p['pid'] in owned],
        failure=[dict(case=r['case'],block=r['block'],arm=arms[r['arm']],error=r.get('error')) for r in latest.values() if r['status']=='failed'],
        active=[dict(case=plan['cases'][r['case']]['id'],block=r['block'],arm=arms[r['arm']]) for r in latest.values() if r['status']=='running'],
        completed_cases=[c['id'] for index,c in enumerate(plan['cases']) if sum(r['case']==index for r in completed)==plan['blocks']*(1+len(plan['candidates'])+int(plan['aa']))],
        packet_exists=(run/'packet.json.gz').exists(),source_commit='6275f9c9f4b59cc6e755f4e537efabed0e45a59f')
    temporary=a.root/'PROGRESS.partial'
    temporary.write_text(json.dumps(record,sort_keys=True)+'\n');temporary.replace(a.root/'PROGRESS.json')
    print(json.dumps(record))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--name',required=True)
    main(p.parse_args())
