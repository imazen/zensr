#!/usr/bin/env python3
"""Bounded NVMe staging with storage-local transfers and fail-closed control.

Command arrays and host-specific paths live in a private execution config.
Thresholds are polling guards, not filesystem quotas. A missing observation,
tool mismatch or transfer failure freezes the writer and raises an error.
"""
import argparse
import datetime
import json
from pathlib import Path
import re
import subprocess
import time
import uuid


def event(**values):
    print(json.dumps(dict(checked=datetime.datetime.now(datetime.timezone.utc).isoformat(),**values)),flush=True)


def writer_group(config):
    pid=config['writer_pid'];stat=Path(f'/proc/{pid}/stat')
    if not stat.exists():return None
    if stat.read_text().split(') ',1)[1].split()[19]!=config['writer_start_ticks']:
        raise ValueError('writer PID reused')
    group=subprocess.check_output(['systemctl','--user','show',config['writer_scope'],'--value','-p','ControlGroup'],text=True).strip()
    if not group:raise ValueError('writer scope disappeared')
    root=Path('/sys/fs/cgroup')/group.lstrip('/')
    if not any(str(pid) in p.read_text().split() for p in root.rglob('cgroup.procs')):
        raise ValueError('writer is outside its recorded scope')
    return root


def control(config,action):
    if writer_group(config) is None:raise ValueError('writer ended during staging operation')
    subprocess.run(['systemctl','--user',action,config['writer_scope']],check=True)


def observation(config):
    result=subprocess.run(config['usage_command'],capture_output=True,text=True,
                          timeout=config.get('observation_timeout_seconds',15),check=True)
    value=json.loads(result.stdout)
    if value['tools']!=config['tools']:raise ValueError('storage monitor tools changed')
    return value


def frozen_request(config):
    group=writer_group(config)
    if group is None or 'frozen 1' not in (group/'cgroup.events').read_text().splitlines():
        raise ValueError('writer scope is not frozen')
    excluded=set()
    for process_list in group.rglob('cgroup.procs'):
        for pid in process_list.read_text().split():
            for fd in Path(f'/proc/{pid}/fd').iterdir():
                try:path=fd.readlink()
                except FileNotFoundError:continue
                if path.parent.parent.name!='artifacts':continue
                relative=path.parent.name+'/'+path.name
                if re.fullmatch(r'[0-9a-f]{2}/[0-9a-f]{64}\.[a-z0-9.-]+',relative):excluded.add(relative)
    return dict(version='zensr-frozen-native-stage-v1',nonce=uuid.uuid4().hex,
                writer_frozen=True,writer_pid=config['writer_pid'],writer_scope=config['writer_scope'],
                alias_root=config['alias_root'],excluded=sorted(excluded),tools=config['tools'],
                batch_sync=config.get('batch_sync',False),batch_size=config.get('batch_size',256))


def ended_request(config):
    # A reused PID is an error too: do not infer completion from a different
    # process occupying the old PID or from a vanished scope alone.
    if Path(f"/proc/{config['writer_pid']}").exists():
        raise ValueError('writer PID still exists; final drain refused')
    group=subprocess.check_output(['systemctl','--user','show',config['writer_scope'],
                                   '--value','-p','ControlGroup'],text=True).strip()
    if group:
        root=Path('/sys/fs/cgroup')/group.lstrip('/')
        if any(p.read_text().split() for p in root.rglob('cgroup.procs')):
            raise ValueError('writer scope still has processes')
    if not Path(config['complete_file']).is_file():
        raise ValueError('successful generation marker missing')
    return dict(version='zensr-ended-native-stage-v1',nonce=uuid.uuid4().hex,
                writer_ended=True,writer_pid=config['writer_pid'],writer_scope=config['writer_scope'],
                alias_root=config['alias_root'],excluded=[],tools=config['tools'],
                batch_sync=config.get('batch_sync',False),batch_size=config.get('batch_size',256))


def transfer_batch(config,log_root,final=False):
    request=ended_request(config) if final else frozen_request(config)
    request_file=log_root/(request['nonce']+'.request.json')
    request_file.write_text(json.dumps(request,indent=2)+'\n')
    log=log_root/(request['nonce']+'.transfer.log')
    event(phase='native-transfer',request=str(request_file),log=str(log))
    with log.open('w') as output:
        subprocess.run(config['drain_command'],input=json.dumps(request),text=True,
                       stdout=output,stderr=subprocess.STDOUT,check=True)
    records=[]
    for line in log.read_text().splitlines():
        if line.startswith('{'):records.append(json.loads(line))
    completed=[r for r in records if r.get('phase')=='transfer-complete' and r.get('nonce')==request['nonce']]
    if len(completed)!=1 or completed[0]['tools']!=config['tools']:
        raise ValueError('missing or changed native transfer acknowledgement')
    event(phase='native-transfer-complete',result=completed[0])
    return completed[0]


def watch(config,log_root):
    log_root.mkdir(parents=True,exist_ok=True)
    if config['trigger_bytes']>config['approved_limit_bytes']:
        raise ValueError('staging trigger exceeds approved limit')
    if config['free_floor_bytes']<config['approved_free_floor_bytes']:
        raise ValueError('free-space trigger is below approved floor')
    try:
        while writer_group(config) is not None:
            value=observation(config)
            event(phase='usage',**value)
            if value['allocated_bytes']>=config['trigger_bytes'] or value['free_bytes']<=config['free_floor_bytes']:
                control(config,'freeze')
                transfer_batch(config,log_root)
                value=observation(config)
                if value['free_bytes']<=config['free_floor_bytes']:
                    raise ValueError('cache remains below free-space trigger; writer left frozen')
                if value['allocated_bytes']>=config['trigger_bytes']:
                    raise ValueError('staging remains above trigger; writer left frozen')
                control(config,'thaw')
                event(phase='writer-resumed')
            time.sleep(config.get('poll_seconds',5))
        event(phase='writer-ended',final_drain_required=True)
    except BaseException:
        # Never resume after a failed observation or transfer. Check identity
        # before controlling the scope, so a reused PID cannot affect other work.
        if writer_group(config) is not None:control(config,'freeze')
        event(phase='failed-writer-frozen')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('config',type=Path)
    parser.add_argument('logs',type=Path)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--final',action='store_true')
    mode.add_argument('--once',action='store_true',help='drain an already frozen writer without resuming it')
    args=parser.parse_args();config=json.loads(args.config.read_text())
    if args.final or args.once:
        args.logs.mkdir(parents=True,exist_ok=True)
        transfer_batch(config,args.logs,final=args.final)
    else:watch(config,args.logs)
