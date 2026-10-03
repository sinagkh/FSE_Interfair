"""Query complete FT setting panels using the validation-frozen multiplier."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
from pathlib import Path
import concurrent.futures,json,subprocess,sys
import numpy as np
import pandas as pd
import ft_trainer as e
P=e.P;c=e.c
def model_path(task,seed):
    reg=json.loads((P/'FT_SELECTION.json').read_text());base=Path(reg['tasks'][task]['run_root'])
    return base/'runs/confirmation/main/ft'/task/str(seed)/'soft/selected.pt'
def query(task,seed):
    out=P/'evaluation/main/ft'/task/str(seed)
    if (out/'DONE.json').exists():return
    freeze=json.loads((P/'FREEZE_ft.json').read_text())
    for path,digest in freeze['files'].items():assert c.sha(path)==digest,path
    d,b,db=e.data(task,seed);out.mkdir(parents=True,exist_ok=True);rows=[];edits=[];features=[]
    paths=[('erm',e.reference_path(task,seed,'ft',False)),('soft',model_path(task,seed))]
    for arm,path in paths:
        record=path.parent/'DONE.json';assert record.exists(),record
        j=json.loads(record.read_text())
        if arm=='soft':assert j['selection_completed_before_audit'] and c.sha(path)==j['checkpoint_sha256']
        m=c.load(path,d,'ft','cpu');v,ff,ordinary=e.evaluate(m,d,list(b),db,'audit',False,'cpu',out/(arm+'.npz'))
        base=dict(task=task,architecture='ft',seed=seed,arm=arm,checkpoint=str(path),arrays=str(out/(arm+'.npz')))
        rows.append({**base,**v});edits.extend([{**base,**f} for f in ff]);features.extend([{**base,**f} for f in ordinary]);del m
    pd.DataFrame(rows).to_csv(out/'per_seed.csv',index=False);pd.DataFrame(edits).to_csv(out/'per_edit.csv',index=False);pd.DataFrame(features).to_csv(out/'per_feature.csv',index=False)
    c.write(out/'DONE.json',dict(status='complete',selection_uses_audit=False))
def all_queries():
    def one(job):
        log=P/'logs'/f'query_ft_{job[0]}_{job[1]}.log'
        with log.open('w') as f:r=subprocess.run([sys.executable,__file__,job[0],str(job[1])],stdout=f,stderr=subprocess.STDOUT)
        return dict(job=job,returncode=r.returncode,log=str(log))
    jobs=[(t,s) for t in ['hmda_oh','acs_income'] for s in range(1000,1010)];results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(one,jobs):
            results.append(r);c.write(P/'FT_QUERY_JOBS.json',results);print(r,flush=True)
    assert all(r['returncode']==0 for r in results)
if __name__=='__main__':
    if len(sys.argv)>1:query(sys.argv[1],int(sys.argv[2]))
    else:all_queries()
