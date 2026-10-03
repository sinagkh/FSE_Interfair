"""Credit-default task: data preparation and two-seed development runs."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[1]
import argparse, copy, hashlib, json, os, pickle, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from core import ROOT, RAW, Encoder, Support, Edit, compile_corners, save_json, sha

MONEY=['LIMIT_BAL']+[f'BILL_AMT{i}' for i in range(1,7)]+[f'PAY_AMT{i}' for i in range(1,7)]
CATS=['EDUCATION','MARRIAGE','PAY_0']+[f'PAY_{i}' for i in range(2,7)]
CACHE=_PACKAGE / 'core/cache_v2/default_credit_development.pkl'
OUT=_PACKAGE / 'core/runs/default_credit'

def prepare():
    if CACHE.exists():
        with CACHE.open('rb') as f:return pickle.load(f)
    f=pd.read_csv(ROOT/'cache_v2/default_credit_source.csv')
    assert len(f)==30000 and not f.isna().any().any() and f.ID.is_unique
    predictor=list(f.columns[1:-1]);profiles=pd.MultiIndex.from_frame(f[predictor]);codes,unique=pd.factorize(profiles,sort=False)
    f['label']=1-f['default payment next month'];f['protected']=(f.SEX==2).astype(int)
    f['row_id']=np.arange(len(f))
    g=f.assign(profile=codes).groupby('profile').agg(s=('protected','first'),y=('label','mean'),n=('label','size'),ny=('label','nunique'))
    strata=g.s*2+(g.y>=.5).astype(int)
    a,b=train_test_split(g.index.to_numpy(),test_size=.35,random_state=73003,stratify=strata)
    v,t=train_test_split(b,test_size=4/7,random_state=73004,stratify=strata.loc[b])
    splits={name:np.flatnonzero(np.isin(codes,ids)) for name,ids in [('train',a),('val',v),('audit',t)]}
    assert sum(map(len,splits.values()))==len(f)
    checks=[]
    for n1,n2 in [('train','val'),('train','audit'),('val','audit')]:
        ok=not np.intersect1d(codes[splits[n1]],codes[splits[n2]]).size;assert ok
        checks.append(dict(check='profile_disjoint',splits=[n1,n2],passed=ok))
    enc=Encoder(MONEY+['AGE'],CATS).fit(f.iloc[splits['train']]);x=enc.transform(f)
    support=Support(x[splits['train']]);rng=np.random.default_rng(93201)
    anchor_ix={k:np.sort(rng.choice(ix,min(len(ix),{'train':2048,'val':768,'audit':2048}[k]),replace=False)) for k,ix in splits.items()}
    specs=[]
    for c in MONEY:
        st=enc.stats[c];z=f.iloc[splits['train']][c];z=z[z.between(st['q01'],st['q99'])]
        low,high=z.quantile([.2,.8],interpolation='nearest').to_numpy();rule='p20/p80'
        if low==high:
            if (z==0).any() and (z>0).any():low=0;high=z[z>0].quantile(.5,interpolation='nearest');rule='zero/positive median'
            else:raise ValueError('Untestable contrast: '+c)
        assert low<high
        specs.append(dict(name=c,column=c,low=float(low),high=float(high),rule=rule,training=True))
    edits=[Edit('limit_plus10k','LIMIT_BAL',10000),Edit('payment1_plus1k','PAY_AMT1',1000),Edit('payment1_plus5k','PAY_AMT1',5000)]
    banks={};counts={};changed_columns=[]
    for split,ix in anchor_ix.items():
        raw=f.iloc[ix].copy();banks[split]={};counts[split]={}
        for spec in specs:
            frames=[]
            for s in (0,1):
                for endpoint in ('low','high'):
                    q=raw.copy();q[spec['column']]=spec[endpoint];q['protected']=s;frames.append(q)
            corners=np.stack([enc.transform(q) for q in frames]);valid=np.ones(len(ix),bool)
            keep,_=support.mask(corners)
            banks[split][spec['name']]=dict(corners=corners,indices=ix,valid=valid,supported=keep,training=True)
            counts[split][spec['name']]=dict(anchors=len(ix),valid=len(ix),supported=int(keep.sum()))
            cindex=enc.columns.index(spec['column']+'__scaled');changed=np.flatnonzero((corners[0]!=corners[1]).any(0)).tolist()
            assert changed==[cindex] and np.all(corners[0,:,0]==0) and np.all(corners[2,:,0]==1)
            assert np.array_equal(corners[0,:,1:],corners[2,:,1:])
            changed_columns.append(dict(split=split,feature=spec['name'],encoded_changed_columns=changed,passed=True))
            if split=='train':assert keep.sum()>=64,(spec['name'],keep.sum())
        for edit in edits:
            corners,valid=compile_corners(raw,edit,enc);keep=np.zeros(len(ix),bool)
            if valid.any():keep[valid],_=support.mask(corners[:,valid])
            banks[split][edit.name]=dict(corners=corners,indices=ix,valid=valid,supported=keep,training=False)
            counts[split][edit.name]=dict(anchors=len(ix),valid=int(valid.sum()),supported=int(keep.sum()))
    metadata=dict(task='default_credit',raw_sha256=sha(RAW/'default_credit/default_of_credit_card_clients.xls'),csv_sha256=sha(ROOT/'cache_v2/default_credit_source.csv'),
        label='non-default',protected='recorded sex: male0/female1',rows=len(f),profile_groups=len(g),excess_profiles=int(len(f)-len(g)),conflicting_profiles=int((g.ny>1).sum()),
        split_rows={k:len(v) for k,v in splits.items()},split_group_counts={k:len(np.unique(codes[v])) for k,v in splits.items()},
        split_s_label={k:pd.crosstab(f.iloc[ix].protected,f.iloc[ix].label).to_dict() for k,ix in splits.items()},encoder=enc.metadata(),specs=specs,bank_counts=counts,
        excluded_from_requirement={c:('additional protected/context variable' if c=='AGE' else 'nominal history/demographic input outside monetary specification') for c in ['AGE']+CATS},
        active_requirement='equal probability response to 13 monetary contrasts; signed ERM common-response preservation',
        support_thresholds=support.thresholds,support_counts=support.counts,checks=checks+changed_columns,
        model_outcomes_used_for_admission=False,final_confirmation=False)
    data=dict(frame=f,x=x,y=f.label.to_numpy(np.float32),s=f.protected.to_numpy(),splits=splits,encoder=enc,banks=banks,metadata=metadata,edits=edits,education_map={})
    with CACHE.open('wb') as stream:pickle.dump(data,stream,pickle.HIGHEST_PROTOCOL)
    save_json(ROOT/'reports/DEFAULT_DATA_GATE.json',metadata)
    np.savez_compressed(ROOT/'cache_v2/default_credit_splits.npz',**splits,profile=codes)
    print(json.dumps(dict(prepared=True,split_rows=metadata['split_rows'],supported={k:min(v['supported'] for v in counts[k].values()) for k in counts})),flush=True)
    return data

def train_data(data):
    d=copy.deepcopy(data);bb=[b for b in d['banks']['train'].values() if b['training']];nfields=len(bb)
    cs=[];ii=[];ww=[]
    for b in bb:
        k=b['supported'];n=int(k.sum());cs.append(b['corners'][:,k]);ii.append(b['indices'][k]);ww.append(np.full(n,1/(nfields*n),np.float64))
    ix=np.concatenate(ii);w=np.concatenate(ww);assert abs(w.sum()-1)<1e-12
    d['banks']['train']={'selected_mixture':dict(corners=np.concatenate(cs,axis=1),indices=ix,supported=np.ones(len(ix),bool),valid=np.ones(len(ix),bool),training=True,sampling_weights=w)}
    return d

def worker(seed,arm):
    assert seed in (2000,2001)
    torch.set_num_threads(1)
    from training_base import train
    data=train_data(prepare());base=OUT/str(seed)
    if arm in ('erm','removed'):
        train(data,'mlp',seed,arm,base/arm,score_scale='P');return
    erm=json.loads((base/'erm/DONE.json').read_text())
    train(data,'mlp',seed,arm,base/arm,source=base/'erm/task.pt',reference=erm['selections']['task']['validation'],weight=160,preserve_weight=160,selection_auc_budget=.005,selection_f1_ratio=.99,score_scale='P',interaction_enabled=arm=='interfair_preserve')

def queue():
    prepare();logs=ROOT/'job_logs/default_credit';logs.mkdir(exist_ok=True,parents=True)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','INTERFAIR_CONCURRENCY':'4','INTERFAIR_THREADS':'1'}
    def launch(job):
        seed,arm=job
        if (OUT/str(seed)/arm/'DONE.json').exists():return
        with (logs/f'{seed}_{arm}.log').open('w') as f:r=subprocess.run([sys.executable,__file__,'--job',str(seed),arm],stdout=f,stderr=subprocess.STDOUT,env=env)
        assert r.returncode==0,job
        print(json.dumps(dict(completed=job)),flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(launch,[(s,a) for s in (2000,2001) for a in ('erm','removed')]))
    with ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(launch,[(s,a) for s in (2000,2001) for a in ('control_preserve','interfair_preserve')]))
    save_json(ROOT/'reports/DEFAULT_EXECUTION.json',dict(completed=True,trajectories=8,seeds=[2000,2001],device='cpu',concurrency=4))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--job',nargs=2);a=p.parse_args()
    if a.prepare:prepare()
    elif a.job:worker(int(a.job[0]),a.job[1])
    else:queue()
