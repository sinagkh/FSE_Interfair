"""Directional audits of complete published pipelines and verified saved corners."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np
import pandas as pd
from scipy.special import expit
P=_PACKAGE / 'training/transformer';ROOT=_PACKAGE
sys.path.insert(0,str(ROOT/'training/direction'))
import direction as q
c=q.c
from model_interfaces import Pipeline,load_pipeline

def primary():return pd.read_csv(ROOT/'training/direction/exports/primary_complete_per_seed.csv')

def pipeline(row,d):
    arm=row.arm;task=row.task;seed=int(row.seed);arch=row.architecture
    if task=='hmda_oh' and arch=='mlp' and arm in ['ltdd','cot_phi','dralign','neufair','mirrorfair']:
        return load_pipeline(ROOT/'data/partitions/runs','named',task,seed,arm,d,'mlp','cpu')
    if task in ['credit_broad','acs_income'] and arch=='mlp' and arm in ['ltdd','cot_phi','dralign','mirrorfair']:
        sys.path.insert(0,str(ROOT/'data/tasks'))
        from task_pipelines import pipeline as load_completed
        return load_completed(task,seed,arm,d,arch,'cpu')
    path=Path(row.checkpoint) if isinstance(row.checkpoint,str) else c.source_path(task,seed,arch)
    assert isinstance(row.checkpoint,str) or arm=='erm'
    return Pipeline(c.load(path,d,arch,'cpu'),[path],'cpu',batch=512 if arch=='ft' else 4096)

def metrics(p,sign,z=None,hard=False):
    e=np.stack([p[1]-p[0],p[3]-p[2]]);de=np.stack([(p[1]>=.5).astype(int)-(p[0]>=.5).astype(int),(p[3]>=.5).astype(int)-(p[2]>=.5).astype(int)])
    signed=sign*e
    r=dict(n=p.shape[1],sign=sign,adverse_decision=float((sign*de<0).any(0).mean()))
    if not hard:
        for margin,label in [(0.,'0'),(.005,'005'),(.01,'01')]:
            f=signed < -margin;r['adverse_'+label]=float(f.any(0).mean());r['both_adverse_'+label]=float(f.all(0).mean())
        r.update(adverse_magnitude=float(np.maximum(0,-signed).mean()),common_response=float(abs(e.mean(0)).mean()))
        if z is not None:
            residual=(z[3]-z[2])-(z[1]-z[0]);r['joint_pass']=float(((abs(residual)<=.05)&~(signed<-.005).any(0)).mean())
    return r

def query(task,arch,seed,arm):
    table=primary();row=table[(table.task==task)&(table.architecture==arch)&(table.seed==seed)&(table.arm==arm)].iloc[0]
    out=P/'baseline_directions'/arch/task/str(seed)/arm
    if (out/'DONE.json').exists():return
    out.mkdir(parents=True,exist_ok=True)
    d,b,db=q.data(task,seed);hard=arm=='mirrorfair'
    candidate=[row.get('arrays'),row.get('gap_source_arrays')]
    arrays=next(Path(x) for x in candidate if isinstance(x,str) and Path(x).exists())
    expected=row.get('arrays_sha256') if isinstance(row.get('arrays'),str) and Path(row.arrays)==arrays else row.get('gap_source_arrays_sha256')
    if isinstance(expected,str):assert c.sha(arrays)==expected
    saved=np.load(arrays);f=None;records=[];raw={};replay=[]
    for name in q.MAP[task]:
        bank=db['audit'][name];keep=bank['supported'];ids=bank['indices'][keep]
        if name.startswith('linked_'):
            if f is None:f=pipeline(row,d)
            xx=bank['corners'][:,keep];p=f(xx.reshape(-1,xx.shape[-1]),'P').reshape(4,-1)
            z=None if hard else f(xx.reshape(-1,xx.shape[-1]),'L').reshape(4,-1)
        else:
            key=name+('_decision' if hard else '_P')
            source_ids=saved[name+'_indices'] if name+'_indices' in saved else bank['indices']
            lookup={int(v):i for i,v in enumerate(source_ids)};jj=np.asarray([lookup[int(i)] for i in ids])
            z=saved[name+'_L'][:,jj] if name+'_L' in saved and not hard else None
            p=saved[key][:,jj] if key in saved else expit(z)
            if task=='hmda_oh':
                if f is None:f=pipeline(row,d)
                xx=bank['corners'][:,keep][:,:7]
                check=f(xx.reshape(-1,xx.shape[-1]),'P').reshape(4,-1)
                err=float(abs(check-p[:,:7]).max());assert err<1e-5,(task,arch,seed,arm,name,err)
                replay.append(err)
        sign=q.MAP[task][name]
        records.append(dict(task=task,architecture=arch,seed=seed,arm=arm,edit=name,output_kind='hard_decision' if hard else 'probability',**metrics(p,sign,z,hard)))
        raw[name+'_P' if not hard else name+'_decision']=p;raw[name+'_indices']=ids
        if z is not None:raw[name+'_L']=z
    pd.DataFrame(records).to_csv(out/'per_edit.csv',index=False);np.savez_compressed(out/'corners.npz',**raw)
    values=pd.DataFrame(records).drop(columns=['task','architecture','seed','arm','edit','output_kind','n','sign']).mean().to_dict()
    values={('direction_'+k):v for k,v in values.items()}
    pd.DataFrame([dict(task=task,architecture=arch,seed=seed,arm=arm,output_kind='hard_decision' if hard else 'probability',**values)]).to_csv(out/'per_seed.csv',index=False)
    c.write(out/'DONE.json',dict(status='complete',source_arrays=str(arrays),source_arrays_sha256=c.sha(arrays),
       pipeline=f.provenance() if f else None,replay_max_error=max(replay,default=0.),cached_ordinary=True,
       linked_edit_queried=task=='hmda_oh',specifications_sha256=c.sha(q.P/'DIRECTION_SPECIFICATIONS.json'),
       model_selection_performed=False,training_performed=False))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('task');a.add_argument('architecture');a.add_argument('seed',type=int);a.add_argument('arm');v=a.parse_args()
    query(v.task,v.architecture,v.seed,v.arm)
