"""Check packaged natural data and partitions against reported training hashes."""
from pathlib import Path
import json
from inputs import ROOT,bootstrap,load
s=bootstrap();checks=[]
for task in json.loads((ROOT/'data/available_seeds.json').read_text()):
    d,b,db=load(task,1000)
    cfg=json.loads((ROOT/'configs/runs'/task/'mlp/soft/1000/component0/CONFIG.json').read_text())
    for k,v in cfg['data_hashes'].items():assert s.c.r.array_hash(d[k])==v,(task,k)
    for k,v in cfg['split_hashes'].items():assert s.c.r.array_hash(d['splits'][k])==v,(task,k)
    for name,bank in b.items():
        assert bank['corners'].shape[1]==len(bank['indices'])
        assert set(bank['indices'])<=set(d['splits']['train'])
    checks.append(dict(task=task,rows=len(d['y']),relations=len(b),data_and_splits_match=True))
    print(task,'PASS',flush=True)
out=ROOT/'reproduced';out.mkdir(exist_ok=True);(out/'input_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
