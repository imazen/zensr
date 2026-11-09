#!/usr/bin/env python3
"""Run the three frozen SR arms sequentially on one GPU, with retained logs."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess


def main(a):
    for name in ['s3diff','resshift-v3','resshift-bicubic']:
        output=a.root/('sr-'+name+'-v1')
        command=[str(a.python),str(Path(__file__).with_name('evaluate_sr.py')),
            '--inputs',str(a.root/'sr-inputs-v1/inputs.jsonl'),'--scorer',str(a.scorer),
            '--native-weights',str(a.root/'native-span/weights_f16.raw'),
            '--model-root',str(a.root/('s3diff' if name=='s3diff' else 'resshift')),
            '--name',name,'--output',str(output),'--jobs',str(a.jobs)]
        with (a.root/('sr-'+name+'-v1.log')).open('a') as log:
            log.write(json.dumps(dict(command=command,started=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n');log.flush()
            process=subprocess.Popen(['/usr/bin/time','-v',*command],stdout=log,stderr=subprocess.STDOUT)
            (a.root/'SR_SWEEP_PROGRESS.json').write_text(json.dumps(dict(stage=name,supervisor_pid=os.getpid(),child_pid=process.pid,output=str(output),checked=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')
            code=process.wait()
            if code:raise RuntimeError(f'{name} exit {code}; inspect retained log')
    (a.root/'SR_SWEEP_COMPLETE.json').write_text(json.dumps(dict(arms=3,inputs_per_arm=697))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','python','scorer']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--jobs',type=int,default=4)
    main(p.parse_args())
