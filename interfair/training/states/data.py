"""State-specific HMDA data using the paper's frozen Ohio construction rules."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
from pathlib import Path
import sys, pickle, json, hashlib, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
P=_PACKAGE / 'training/states'
ROOT=_PACKAGE
sys.path.insert(0,str(ROOT/'core'))
import core
from feature_edits import inventory, EXTRA
from support import fitted_support
from splits import reserve

STATES=('md','va','pa')
SIGNS={'income':1,'debt_to_income_ratio':-1,'loan_to_value_ratio':-1,'linked_income_plus10k':1}
CONT=['loan_amount','loan_to_value_ratio','property_value','income','debt_to_income_ratio',
      'tract_minority_population_percent','tract_to_msa_income_percentage','ffiec_msa_md_median_family_income']
CATS=['derived_sex','loan_type','loan_purpose','lien_status','preapproval','occupancy_type','applicant_age']

def read_state(task,max_rows=60000):
    state=task.removeprefix('hmda_'); assert state in STATES+('oh',)
    path=core.RAW/f'hmda_2019_{state}.csv'
    f=pd.read_csv(path,usecols=CONT+CATS+['derived_race','action_taken'],low_memory=False)
    f['row_id']=np.arange(len(f))
    f=f[f.derived_race.isin(['White','Black or African American'])&f.action_taken.isin([1,2,3])].copy()
    f['label']=f.action_taken.isin([1,2]).astype(int)
    f['protected']=(f.derived_race=='Black or African American').astype(int)
    values=f.debt_to_income_ratio.astype(str).str.strip();dt=core.numeric(values)
    for a,b in {'<20%':19.,'20%-<30%':25.,'30%-<36%':33.,'50%-60%':55.,'>60%':65.}.items():
        dt=dt.mask(values==a,b)
    f['debt_to_income_ratio']=dt
    if max_rows and len(f)>max_rows:
        chosen,_=train_test_split(np.arange(len(f)),train_size=max_rows,random_state=core.SPLIT_SEED,
            stratify=2*f.label.to_numpy()+f.protected.to_numpy())
        f=f.iloc[np.sort(chosen)].copy()
    edits=[core.Edit(f'income_plus{n}k','income',n,training=n==10) for n in (10,20,30,50)]
    return f.reset_index(drop=True),CONT,CATS,edits,[path],[
        '2019 state-specific HMDA sample; training-only encoder and support.',
        'Primary ordinary contrasts use observed p20/p80 or complete categorical values.',
        'Signed linked-income edits preserve representative debt and use public DTI reporting codes.']

def contrast(d,spec,ids):
    raw=d['frame'].iloc[ids];enc=d['encoder'];lo=raw.copy();hi=raw.copy();valid=np.ones(len(ids),bool)
    for j,col in enumerate(spec['columns']):
        a=spec['low'][j] if len(spec['columns'])>1 else spec['low']
        b=spec['high'][j] if len(spec['columns'])>1 else spec['high'];lo[col]=a;hi[col]=b
        if col in enc.continuous:
            valid &= core.numeric(raw[col]).notna().to_numpy()
            st=enc.stats[col];valid &= st['q01']<=a<=st['q99'] and st['q01']<=b<=st['q99']
        else:
            valid &= ~core.categorical(raw[col]).isin(['__MISSING__','Exempt','Unknown','Not applicable']).to_numpy()
            valid &= a in enc.vocab[col] and b in enc.vocab[col]
    corners=[]
    for s in (0,1):
        for f in (lo,hi):
            f=f.copy();f['protected']=s;corners.append(enc.transform(f))
    return np.stack(corners),valid

def linked(d,base,support,split):
    enc=d['encoder'];out={};j=enc.columns.index('debt_to_income_ratio__scaled')
    ji=enc.columns.index('income__scaled');st=enc.stats['debt_to_income_ratio'];ist=enc.stats['income']
    for amount in ([10] if split!='audit' else [10,20,30,50]):
        b=base[f'income_plus{amount}k'];ids=b['indices'];raw=d['frame'].iloc[ids]
        inc=core.numeric(raw.income).to_numpy();dt=core.numeric(raw.debt_to_income_ratio).to_numpy()
        with np.errstate(divide='ignore',invalid='ignore'): target=dt*inc/(inc+amount)
        valid=b['valid']&np.isin(dt,np.arange(36,50))&(inc>0)&np.isfinite(target)
        valid &= (dt>=st['q01'])&(dt<=st['q99'])&(target>=st['q01'])&(target<=st['q99'])
        # Compile the raw fixed-debt edit before applying the reporting representation.
        xx=b['corners'].copy();continuous=((np.clip(target,st['clip_low'],st['clip_high'])-st['median'])/st['scale'])
        continuous=np.nan_to_num(continuous,nan=0.,posinf=0.,neginf=0.).astype(np.float32)
        xx[1,:,j]=xx[3,:,j]=continuous
        # Recover the same train-fitted representation used by the existing producer.
        dt=np.round(xx[0,:,j].astype(float)*st['scale']+st['median'],4)
        inc=np.round(xx[0,:,ji].astype(float)*ist['scale']+ist['median'],4)
        with np.errstate(divide='ignore',invalid='ignore'): target=dt*inc/(inc+amount)
        bins=np.select([target<20,target<30,target<36,target<50,target<60],[19.,25.,33.,target,55.],default=65.)
        coded=np.where((target>=36)&(target<50),np.clip(np.round(target),36,49),bins)
        assert np.isin(coded,[19,25,33,*range(36,50),55,65]).all()
        cc=xx.copy();cc[1,:,j]=cc[3,:,j]=((coded-st['median'])/st['scale']).astype(np.float32)
        keep=support.mask(cc)[0]&valid
        if split=='audit':
            # Same common-support audit cohort as the current Ohio income bank.
            bb=xx.copy();bb[1,:,j]=bb[3,:,j]=((bins-st['median'])/st['scale']).astype(np.float32)
            keep &= support.mask(xx)[0]&support.mask(bb)[0]&b['supported']
        assert keep.sum()>=32,(d['metadata']['task'],split,amount,int(keep.sum()))
        out[f'linked_income_plus{amount}k']=dict(corners=cc,valid=valid,supported=keep,indices=ids,
            training=amount==10,column='income and debt_to_income_ratio',delta=amount)
    return out

def prepare(state):
    task='hmda_'+state;path=P/'cache'/f'{task}.pkl';path.parent.mkdir(exist_ok=True)
    if path.exists():return path
    original=core.read_task
    try:
        core.read_task=read_state;d=core.prepare_task(task)
    finally:core.read_task=original
    EXTRA[task]=EXTRA['hmda_oh'].copy()
    specs,excluded=inventory(d);support=fitted_support(d);base=d['banks'];master={};banks={sp:{} for sp in base}
    indices={sp:next(iter(bs.values()))['indices'] for sp,bs in base.items()}
    for spec in specs:
        if spec['status']!='candidate':continue
        name=spec['name'];ids=d['splits']['train'];cc,valid=contrast(d,spec,ids)
        keep=np.zeros(len(ids),bool);keep[valid]=support.mask(cc[:,valid])[0]
        assert keep.sum()>=8192,(task,name,int(keep.sum()))
        # Store identities only; regenerate selected corners from the raw schema.
        master[name]=ids[keep]
        for sp,ii in indices.items():
            xx,vv=contrast(d,spec,ii);kk=np.zeros(len(ii),bool);kk[vv]=support.mask(xx[:,vv])[0]
            assert kk.sum()>=32,(task,sp,name,int(kk.sum()))
            banks[sp][name]=dict(corners=xx,indices=ii,valid=vv,supported=kk,training=True)
        print(task,name,'supported training',int(keep.sum()),flush=True)
    links={sp:linked(d,bs,support,sp) for sp,bs in base.items()}
    d['banks']=banks;d,reservation,_=reserve(d,task)
    d['metadata'].update(task=task,status='state-specific current-method data',ordinary_specs=specs,
        ordinary_features=len(master),direction_signs=SIGNS,anchor_budget_per_feature=8192,
        source_adapter_sha256=core.sha(__file__),reservation=reservation)
    counts={sp:{n:dict(anchors=len(b['indices']),valid=int(b['valid'].sum()),supported=int(b['supported'].sum())) for n,b in bs.items()} for sp,bs in banks.items()}
    report=dict(task=task,rows=len(d['y']),input_dim=d['x'].shape[1],split_counts={k:len(v) for k,v in d['splits'].items()},
        label_rate=float(d['y'].mean()),protected_rate=float(d['s'].mean()),specs=specs,ordinary_counts=counts,
        direction_counts={sp:{n:int(b['supported'].sum()) for n,b in bs.items()} for sp,bs in links.items()},
        training_support={n:len(ids) for n,ids in master.items()},sources=d['metadata']['source_files'],
        support=d['metadata']['support'],reservation=reservation,adapter_sha256=core.sha(__file__))
    with path.with_suffix('.tmp').open('wb') as f:pickle.dump(dict(data=d,master=master,links=links),f,pickle.HIGHEST_PROTOCOL)
    path.with_suffix('.tmp').replace(path)
    report['cache_sha256']=core.sha(path);core.save_json(path.with_suffix('.json'),report)
    return path

def data(task,seed):
    path=P/'cache'/f'{task}.pkl'
    with path.open('rb') as f:bundle=pickle.load(f)
    d=bundle['data'];rng=np.random.default_rng(seed+95103);banks={}
    specs={s['name']:s for s in d['metadata']['ordinary_specs'] if s['status']=='candidate'}
    for name,ids in sorted(bundle['master'].items()):
        ix=ids[rng.permutation(len(ids))[:8192]];cc,valid=contrast(d,specs[name],ix);assert valid.all()
        banks[name]=dict(corners=cc,indices=ix,valid=valid,supported=valid.copy(),training=True)
    d['banks']['train']=banks
    db={sp:{n:(banks if sp=='train' else d['banks'][sp])[n] for n in SIGNS if not n.startswith('linked_')} for sp in ('train','val','audit')}
    for sp in db:db[sp].update(bundle['links'][sp])
    return d,banks,db

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('state',choices=STATES+('oh',));args=ap.parse_args();prepare(args.state)
