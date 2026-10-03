"""Audit selected scratch models after the direction study is frozen."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
from pathlib import Path
import concurrent.futures, hashlib, json, subprocess, sys
P=_PACKAGE / 'training/mlp'

def one(task, seed):
    import mlp_trainer as s
    freeze=json.loads((P/'FREEZE.json').read_text())
    for path,digest in freeze['files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    d,b,db=s.q.data(task,seed); names=list(b)
    out=P/'evaluation'/task/str(seed);out.mkdir(parents=True,exist_ok=True)
    rows=[];features=[]
    for arm in ['erm','equality','equality_direction']:
        rec=json.loads((s.directory(task,seed,arm)/'DONE.json').read_text())
        assert rec['selection_completed_before_audit'] and not rec['config']['pretrained_weights_loaded']
        path=Path(rec['checkpoint']);assert s.c.sha(path)==rec['checkpoint_sha256']
        m=s.c.load(path,d,'mlp','cpu')
        v,ff=s.q.evaluate(m,d,names,db,'audit',out/(arm+'.npz'))
        rows.append(dict(task=task,seed=seed,architecture='mlp',arm=arm,checkpoint=str(path),**v))
        features.extend([dict(task=task,seed=seed,architecture='mlp',arm=arm,**r) for r in ff])
    s.pd.DataFrame(rows).to_csv(out/'per_seed.csv',index=False)
    s.pd.DataFrame(features).to_csv(out/'per_edit.csv',index=False)
    s.c.write(out/'DONE.json',dict(status='complete',rows=len(rows),selection_uses_audit=False))

if __name__=='__main__':
    if len(sys.argv)>1:
        one(sys.argv[1],int(sys.argv[2]))
    else:
        tasks=['hmda_oh','credit_broad','acs_income','acs_employment_sex','acs_employment_age']
        def job(pair):
            task,seed=pair
            out=P/'logs'/f'query_{task}_{seed}.log'
            with out.open('w') as f:
                r=subprocess.run([sys.executable,str(P/'query.py'),task,str(seed)],stdout=f,stderr=subprocess.STDOUT)
            assert r.returncode==0,(task,seed,str(out))
            return dict(task=task,seed=seed,status='PASS')
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            for result in ex.map(job,[(t,s) for t in tasks for s in range(1000,1010)]):
                print(result,flush=True)
