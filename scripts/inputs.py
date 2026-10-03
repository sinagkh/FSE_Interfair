"""Portable loading of the exact public-data partitions and lossless edit banks."""
from pathlib import Path
import sys,lzma,pickle,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'interfair'
sys.path.insert(0,str(CODE/'core'))

def expand(obj,x):
    if not isinstance(obj,dict):return obj
    out={k:expand(v,x) for k,v in obj.items() if k!='_encoded_corners'}
    if '_encoded_corners' in obj:
        rec=obj['_encoded_corners'];base=x[obj['indices']]
        cc=np.repeat(base[None,:,:],rec['shape'][0],axis=0)
        for i,(cols,values) in enumerate(rec['changes']):cc[i][:,cols]=values
        assert cc.shape==tuple(rec['shape'])
        out['corners']=cc
    return out

def read(path):
    # These are trusted, checksum-listed public-data caches supplied with this artifact.
    with lzma.open(path,'rb') as f:return pickle.load(f)

def load(task,seed=1000):
    dpath=ROOT/'data/processed'/task
    d=read(dpath/'dataset.pkl.xz');x=d['x'];d['banks']=expand(d['banks'],x)
    bank=dpath/f'training_{seed}.pkl.xz'
    if not bank.exists():
        if task.startswith('hmda_'):raise ValueError('HMDA prepared banks cover seeds 1000–1009; choose one of those seeds.')
        bank=dpath/'training_1000.pkl.xz'
    b=expand(read(bank),x);d['banks']['train']=b
    db=expand(read(dpath/'direction.pkl.xz'),x)
    for name in list(db['train']):
        if name in b:db['train'][name]=b[name]
    return d,b,db

def bootstrap():
    sys.path.insert(0,str(CODE/'training/mlp'))
    import mlp_trainer as study
    study.q.data=load
    study.c.data=lambda task,seed:load(task,seed)[:2]
    study.q.MAP.update({t:{'income':1,'debt_to_income_ratio':-1,'loan_to_value_ratio':-1,'linked_income_plus10k':1} for t in ['hmda_md','hmda_va','hmda_pa']})
    return study
