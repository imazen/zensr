#!/usr/bin/env python3
"""Run named SR arms sequentially, retaining compressed scored images and logs."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess


def main(a):
    for name in a.models:
        output=a.root/('sr-'+name+'-'+a.suffix)
        command=[str(a.python),str(Path(__file__).with_name('evaluate_sr.py')),
            '--inputs',str(a.root/'sr-inputs-v1/inputs.jsonl'),'--scorer',str(a.scorer),
            '--native-weights',str(a.root/'native-span/weights_f16.raw'),
            '--model-root',str(a.root/('resshift' if name.startswith('resshift') else name)),
            '--name',name,'--output',str(output),'--jobs',str(a.jobs),
            '--artifacts',str(a.root/('sr-images-'+a.suffix))]
        with (a.root/('sr-'+name+'-'+a.suffix+'.log')).open('a') as log:
            log.write(json.dumps(dict(command=command,started=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n');log.flush()
            process=subprocess.Popen(['/usr/bin/time','-v',*command],stdout=log,stderr=subprocess.STDOUT)
            (a.root/('SR_SWEEP_PROGRESS-'+a.suffix+'.json')).write_text(json.dumps(dict(stage=name,supervisor_pid=os.getpid(),child_pid=process.pid,output=str(output),checked=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')
            code=process.wait()
            if code:raise RuntimeError(f'{name} exit {code}; inspect retained log')
    (a.root/('SR_SWEEP_COMPLETE-'+a.suffix+'.json')).write_text(json.dumps(dict(arms=len(a.models),inputs_per_arm=697))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','python','scorer']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--models',nargs='+',choices=['dscf','s3diff','resshift-v3','resshift-bicubic'],default=['dscf','s3diff','resshift-v3','resshift-bicubic'])
    p.add_argument('--suffix',default='retained-v2')
    main(p.parse_args())
