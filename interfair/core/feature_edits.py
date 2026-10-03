"""E10a: complete semantic input inventory and separate discovery/audit banks."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[1]
import argparse,json,pickle
import numpy as np
import pandas as pd
import torch
from core import ROOT,numeric,categorical,predict,summarize_effects,save_json
from training_base import prepare
from support import fitted_support
from analyze_direction import load_model

EXTRA={'adult':{'age':'additional protected age contrast belongs to E19','race':'additional protected race contrast belongs to E19'},'hmda_oh':{'derived_sex':'additional protected sex contrast belongs to E19','applicant_age':'additional protected age contrast belongs to E19'}}


def inventory(data):
    task=data['metadata']['task'];enc=data['encoder'];f=data['frame'];training=f.iloc[data['splits']['train']];specs=[]
    for col in enc.continuous+enc.categories:
        if col in EXTRA[task]:
            specs.append(dict(name=col,columns=[col],kind='additional_protected',status='separate_E19',reason=EXTRA[task][col]));continue
        if task=='adult' and col=='relationship':continue
        if task=='adult' and col=='marital_status':
            counts=training.groupby(['marital_status','relationship']).size().sort_values(ascending=False)
            vals=list(counts.index[:2]);columns=['marital_status','relationship'];kind='linked_nominal'
        elif col in enc.continuous:
            v=numeric(training[col]);st=enc.stats[col];v=v[(v>=st['q01'])&(v<=st['q99'])];vals=list(v.quantile([.2,.8],interpolation='nearest'));columns=[col];kind='ordinal' if col=='education_num' else 'numeric'
            if vals[0]==vals[1]:
                pos=v[v>0]
                vals=[0.,float(pos.quantile(.5,interpolation='nearest'))] if (v==0).any() and len(pos) else []
                kind='sparse_numeric'
        else:
            v=categorical(training[col]);v=v[~v.isin(['__MISSING__','Exempt','Unknown','Not applicable','Sex Not Available'])];vals=list(v.value_counts().index[:2]);columns=[col];kind='nominal'
        if len(vals)!=2 or vals[0]==vals[1]:
            specs.append(dict(name=col,columns=columns,kind=kind,status='excluded_constant_or_no_valid_contrast',reason='No distinct admissible training contrast'));continue
        specs.append(dict(name=col,columns=columns,kind=kind,status='candidate',low=list(vals[0]) if isinstance(vals[0],tuple) else vals[0],high=list(vals[1]) if isinstance(vals[1],tuple) else vals[1],target=0.,policy='developer-supplied direct effect invariance; screening evidence does not supply normative authority',direction=None,reason='Candidate ordinary-input block under expanded direct invariance; fixed raw probe, no causal/actionability claim'))
    used=set(enc.continuous+enc.categories)
    excluded=[]
    for col in f.columns:
        if col not in used:
            excluded.append(dict(column=col,status='protected_input' if col=='protected' else 'not_a_separate_prediction_input',reason='Explicit binary audit factor' if col=='protected' else 'Label, ID, raw protected identity or redundant/source metadata; see encoder columns'))
    assert set(c for s in specs for c in s['columns'])==used
    return specs,excluded


def banks(task):
    path=ROOT/'cache_v2'/f'{task}_feature_inventory.pkl'
    if path.exists():
        with path.open('rb') as f:return pickle.load(f)
    data=prepare(task);specs,excluded=inventory(data);support=fitted_support(data);enc=data['encoder'];f=data['frame'];used=next(iter(data['banks']['train'].values()))['indices'];available=np.setdiff1d(data['splits']['train'],used);rng=np.random.default_rng(93101);discovery=np.sort(rng.choice(available,1024,replace=False));assert not np.intersect1d(discovery,used).size
    indices={k:next(iter(data['banks'][k].values()))['indices'] for k in ('train','val','audit')};indices['discovery']=discovery;out={};counts={}
    for split,ix in indices.items():
        raw=f.iloc[ix].copy();out[split]={};counts[split]={}
        for spec in specs:
            if spec['status']!='candidate':continue
            low=raw.copy();high=raw.copy();valid=np.ones(len(ix),bool)
            for j,col in enumerate(spec['columns']):
                a=spec['low'][j] if len(spec['columns'])>1 else spec['low'];b=spec['high'][j] if len(spec['columns'])>1 else spec['high'];low[col]=a;high[col]=b
                valid &= numeric(raw[col]).notna().to_numpy() if col in enc.continuous else ~categorical(raw[col]).isin(['__MISSING__','Exempt','Unknown','Not applicable']).to_numpy()
                if col in enc.continuous:
                    st=enc.stats[col];valid &= st['q01']<=a<=st['q99'] and st['q01']<=b<=st['q99']
                else:valid &= a in enc.vocab[col] and b in enc.vocab[col]
                if col=='education_num':low['education']=data['education_map'][int(a)];high['education']=data['education_map'][int(b)]
            corners=[]
            for s in (0,1):
                for frame in (low,high):
                    frame=frame.copy();frame['protected']=s;corners.append(enc.transform(frame))
            corners=np.stack(corners);keep=np.zeros(len(ix),bool)
            if valid.any():keep[valid],_=support.mask(corners[:,valid])
            out[split][spec['name']]=dict(corners=corners,valid=valid,supported=keep,indices=ix,training=True)
            counts[split][spec['name']]=dict(anchors=len(ix),valid=int(valid.sum()),supported=int(keep.sum()))
    result=dict(banks=out,specs=specs,excluded=excluded,counts=counts)
    with path.open('wb') as f:pickle.dump(result,f,pickle.HIGHEST_PROTOCOL)
    save_json(ROOT/'reports'/f'{task}_inventory.json',dict(specs=specs,excluded=excluded,counts=counts,discovery_disjoint_from_steering=True,endpoint_fit='training only',protected= 'sex' if task=='adult' else 'race'))
    return result


def run(task):
    data=prepare(task);b=banks(task);rows=[]
    for seed in (2000,2001):
        base=ROOT/'runs/pilot_v3'/task/'mlp'/str(seed);head=ROOT/'runs/headroom'/task/'mlp'/str(seed)
        for arm in ('erm','removed','no_interaction','interfair_L','interfair_P'):
            selector='task' if arm=='erm' else ('L' if arm=='interfair_L' else 'P');directory=base/arm if arm in ('erm','removed') else head/arm;model=load_model(directory/f'{selector}.pt',data)
            for split in ('discovery','audit'):
                if split=='discovery' and arm!='erm':continue
                for name,bank in b['banks'][split].items():
                    n=len(bank['indices'])
                    for scale in ('L','P'):
                        v=predict(model,bank['corners'].reshape(4*n,-1),'cuda',scale).reshape(4,n)
                        for cohort in ('valid','supported'):
                            keep=bank[cohort];row=dict(task=task,seed=seed,arm=arm,selector=selector,split=split,feature=name,scale=scale,cohort=cohort,anchors=n,n=int(keep.sum()))
                            if keep.any():row.update(summarize_effects(v[:,keep],.01 if scale=='L' else .001,.05 if scale=='L' else .01))
                            rows.append(row)
            del model
    save_json(ROOT/'reports'/f'{task}_audit.json',rows)
    print(json.dumps(dict(task=task,rows=len(rows),ordinary_blocks=len(b['banks']['audit']))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['adult','hmda_oh'],required=True);args=p.parse_args();torch.set_num_threads(2);run(args.task)
