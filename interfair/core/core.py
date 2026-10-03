"""Schema-aware encoding, model definitions, edit banks, and fairness metrics."""
from __future__ import annotations
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[1]
import copy
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, f1_score, log_loss, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch import nn
from threadpoolctl import threadpool_limits

ROOT = _PACKAGE / 'core'
PROJECT = _PACKAGE
RAW = _PACKAGE / 'data/raw'
SPLIT_SEED = 73001
DEV_SEEDS = (2000, 2001, 2002)
ADULT_COLS = ['age','workclass','fnlwgt','education','education_num',
              'marital_status','occupation','relationship','race','sex',
              'capital_gain','capital_loss','hours_per_week','native_country','income']


def jsonable(x):
    if isinstance(x, dict): return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [jsonable(v) for v in x]
    if isinstance(x, Path): return str(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    return x


def save_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(jsonable(value),indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''): h.update(b)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def numeric(v):
    return pd.to_numeric(v.astype(str).str.replace('%','',regex=False),errors='coerce')


def categorical(v):
    return v.fillna('__MISSING__').astype(str).str.strip()


class Encoder:
    """Training-only robust scaling and categorical vocabulary, explicit unknown."""
    def __init__(self, continuous, categories):
        self.continuous=list(continuous); self.categories=list(categories)

    def fit(self, frame):
        self.stats={}; self.vocab={}; self.columns=['protected']
        for c in self.continuous:
            s=numeric(frame[c]); median=float(s.median())
            if not np.isfinite(median): median=0.
            quant=s.quantile([.005,.01,.25,.75,.99,.995]).to_numpy()
            quant=np.nan_to_num(quant,nan=median)
            iqr=float(quant[3]-quant[2])
            # Sparse features such as capital_gain have IQR=0 despite a large
            # nonzero tail. Fit the fallback scale on training data only.
            fallback=float(s.clip(quant[0],quant[5]).std(ddof=0))
            scale=max(iqr if iqr>0 else fallback,1.)
            self.stats[c]=dict(median=median,scale=scale,clip_low=float(quant[0]),
                               clip_high=float(quant[5]),q01=float(quant[1]),q99=float(quant[4]),
                               scale_rule='IQR' if iqr>0 else 'training winsorized std fallback')
            self.columns.extend([c+'__scaled',c+'__missing'])
        for c in self.categories:
            self.vocab[c]=sorted(set(categorical(frame[c])))
            self.columns.extend([c+'='+v for v in self.vocab[c]]+[c+'=__UNKNOWN__'])
        return self

    def transform(self, frame):
        chunks=[frame['protected'].to_numpy(np.float32)[:,None]]
        for c in self.continuous:
            t=self.stats[c]; raw=numeric(frame[c]); missing=raw.isna().to_numpy(np.float32)
            z=(raw.fillna(t['median']).clip(t['clip_low'],t['clip_high'])-t['median'])/t['scale']
            chunks.append(np.column_stack([z.to_numpy(np.float32),missing]))
        for c in self.categories:
            lookup={v:i for i,v in enumerate(self.vocab[c])}
            ids=np.array([lookup.get(v,len(lookup)) for v in categorical(frame[c])])
            one=np.zeros((len(frame),len(lookup)+1),np.float32)
            one[np.arange(len(frame)),ids]=1; chunks.append(one)
        out=np.concatenate(chunks,axis=1)
        assert np.isfinite(out).all()
        return out

    def metadata(self):
        return dict(continuous=self.continuous,categories=self.categories,
                    stats=self.stats,vocab=self.vocab,columns=self.columns,
                    fit_scope='training rows only',winsorization=[.005,.995])


@dataclass
class Edit:
    name: str
    column: str
    delta: float | None = None
    target: str | None = None
    training: bool = False

    def apply(self, frame, encoder, education_map=None):
        out=frame.copy(deep=True)
        if self.target is not None:
            out[self.column]=self.target
            valid=frame[self.column].notna().to_numpy() & (categorical(frame[self.column])!=self.target).to_numpy()
            valid &= self.target in encoder.vocab[self.column]
        else:
            values=numeric(frame[self.column]); after=values+self.delta
            out[self.column]=after
            st=encoder.stats[self.column]
            valid=(values.notna() & (values>=st['q01']) & (values<=st['q99']) &
                   (after>=st['q01']) & (after<=st['q99'])).to_numpy()
            if self.column=='education_num':
                out['education']=after.map(education_map or {})
                valid &= out['education'].notna().to_numpy()
        return out,valid


class Support:
    """Group-conditional 5-NN thresholds fit using only training rows."""
    def __init__(self,x,seed=9101,max_ref=6000):
        rng=np.random.default_rng(seed); self.models={}; self.thresholds={}; self.counts={}
        for s in (0,1):
            ids=np.flatnonzero(x[:,0]==s); rng.shuffle(ids)
            nc=min(1000,max(64,len(ids)//5)); cal=ids[:nc]; ref=ids[nc:nc+max_ref]
            if len(ref)<6: raise ValueError('Too few support-reference rows')
            nnm=NearestNeighbors(n_neighbors=5,algorithm='brute',n_jobs=2).fit(x[ref,1:])
            with threadpool_limits(limits=2):
                distances=nnm.kneighbors(x[cal,1:],return_distance=True)[0][:,-1]
            self.models[s]=nnm; self.thresholds[s]=float(np.quantile(distances,.95))
            self.counts[s]=dict(reference=len(ref),calibration=len(cal))

    def mask(self,corners):
        distances=[]
        for j,s in enumerate((0,0,1,1)):
            with threadpool_limits(limits=2):
                d=self.models[s].kneighbors(corners[j,:,1:],return_distance=True)[0][:,-1]
            distances.append(d)
        d=np.stack(distances)
        threshold=np.array([self.thresholds[s] for s in (0,0,1,1)])[:,None]
        return (d<=threshold).all(0),d


def compile_corners(frame,edit,encoder,education_map=None):
    high,valid=edit.apply(frame,encoder,education_map)
    out=[]
    for s in (0,1):
        for raw in (frame,high):
            changed=raw.copy(deep=True); changed['protected']=s
            out.append(encoder.transform(changed))
    return np.stack(out),valid


def read_task(name,max_rows=60000):
    if name=='adult':
        paths=[RAW/'adult/adult.data',RAW/'adult/adult.test']
        frames=[pd.read_csv(p,names=ADULT_COLS,na_values='?',skipinitialspace=True,comment='|') for p in paths]
        f=pd.concat(frames,ignore_index=True).dropna().reset_index(drop=True)
        f['row_id']=np.arange(len(f)); f['label']=(f.income.str.replace('.','',regex=False).str.strip()=='>50K').astype(int)
        f['protected']=(f.sex.str.strip()=='Female').astype(int)
        f['relationship']=f.relationship.str.strip().replace({'Husband':'Spouse','Wife':'Spouse'})
        continuous=['age','education_num','capital_gain','capital_loss','hours_per_week']
        cats=['workclass','marital_status','occupation','relationship','race','native_country']
        edits=[Edit('hours_plus5','hours_per_week',5,training=True),
               Edit('education_plus1','education_num',1,training=True),
               Edit('hours_plus10','hours_per_week',10),Edit('hours_plus15','hours_per_week',15),
               Edit('education_plus2','education_num',2)]
        notes=['Sex is present only as explicit protected input.',
               'Husband/Wife neutralized to Spouse to avoid sex-coded relationship contradictions.',
               'Education string is updated with education_num; only education_num enters the model.',
               'fnlwgt excluded as survey-weight metadata, not an audited personal attribute.',
               'No claim that education/income changes are realized causal interventions.']
    elif name=='hmda_oh':
        paths=[RAW/'hmda_2019_oh.csv']
        continuous=['loan_amount','loan_to_value_ratio','property_value','income','debt_to_income_ratio',
                    'tract_minority_population_percent','tract_to_msa_income_percentage','ffiec_msa_md_median_family_income']
        cats=['derived_sex','loan_type','loan_purpose','lien_status','preapproval','occupancy_type','applicant_age']
        f=pd.read_csv(paths[0],usecols=continuous+cats+['derived_race','action_taken'],low_memory=False)
        f['row_id']=np.arange(len(f))
        f=f[f.derived_race.isin(['White','Black or African American']) & f.action_taken.isin([1,2,3])].copy()
        f['label']=f.action_taken.isin([1,2]).astype(int)
        f['protected']=(f.derived_race=='Black or African American').astype(int)
        values=f.debt_to_income_ratio.astype(str).str.strip()
        dt=numeric(values)
        for a,b in {'<20%':19.,'20%-<30%':25.,'30%-<36%':33.,'50%-60%':55.,'>60%':65.}.items(): dt=dt.mask(values==a,b)
        f['debt_to_income_ratio']=dt
        edits=[Edit(f'income_plus{d}k','income',d,training=d==10) for d in (10,20,30,50)]
        notes=['HMDA income units are thousands of dollars; increments are input probes.',
               'Public DTI interval codes represented by declared midpoints for prediction.',
               'Income-only probes hold DTI fixed; linked income edits also update the debt-to-income ratio.',
               'Raw race input is used only to construct the explicit protected column.',
               'A fixed stratified sample is capped at 60000 Ohio records.']
    else: raise ValueError(name)
    if max_rows and len(f)>max_rows:
        selected,_=train_test_split(np.arange(len(f)),train_size=max_rows,random_state=SPLIT_SEED,
                                    stratify=2*f.label.to_numpy()+f.protected.to_numpy())
        f=f.iloc[np.sort(selected)].copy()
    return f.reset_index(drop=True),continuous,cats,edits,paths,notes


def prepare_task(name,max_rows=60000,bank_sizes=None):
    f,continuous,cats,edits,paths,notes=read_task(name,max_rows)
    labels=2*f.label.to_numpy()+f.protected.to_numpy(); ids=np.arange(len(f))
    trainval,audit=train_test_split(ids,test_size=.2,random_state=SPLIT_SEED,stratify=labels)
    train,val=train_test_split(trainval,test_size=.1875,random_state=SPLIT_SEED+1,stratify=labels[trainval])
    splits=dict(train=np.sort(train),val=np.sort(val),audit=np.sort(audit))
    encoder=Encoder(continuous,cats).fit(f.iloc[train]); x=encoder.transform(f)
    education_map={}
    if name=='adult':
        pairs=f.iloc[train][['education_num','education']].drop_duplicates()
        assert not pairs.education_num.duplicated().any()
        education_map=dict(zip(pairs.education_num.astype(int),pairs.education))
    support=Support(x[train]); banks={}
    sizes=bank_sizes or dict(train=2048,val=768,audit=2048)
    for k,split in splits.items():
        rng=np.random.default_rng(SPLIT_SEED+{'train':11,'val':12,'audit':13}[k])
        ix=np.sort(rng.choice(split,min(len(split),sizes[k]),replace=False)); bank={}
        for edit in edits:
            corners,valid=compile_corners(f.iloc[ix],edit,encoder,education_map)
            mask=np.zeros(len(ix),bool); distances=np.full((4,len(ix)),np.nan)
            if valid.any():
                okay,dist=support.mask(corners[:,valid]);mask[valid]=okay;distances[:,valid]=dist
            bank[edit.name]=dict(corners=corners,valid=valid,supported=mask,distances=distances,
                                 indices=ix,training=edit.training,column=edit.column,delta=edit.delta)
        banks[k]=bank
    meta=dict(task=name,rows=len(f),input_dim=x.shape[1],split_seed=SPLIT_SEED,
              source_files=[dict(path=str(p),sha256=sha(p)) for p in paths],notes=notes,
              splits={k:dict(n=len(v),row_sha256=hashlib.sha256(f.row_id.to_numpy()[v].tobytes()).hexdigest(),
                             label_rate=float(f.label.iloc[v].mean()),protected_rate=float(f.protected.iloc[v].mean()))
                      for k,v in splits.items()},encoder=encoder.metadata(),support=dict(thresholds=support.thresholds,counts=support.counts),
              bank_counts={k:{e:dict(anchors=len(b['indices']),valid=int(b['valid'].sum()),supported=int(b['supported'].sum()),training=b['training'])
                             for e,b in bank.items()} for k,bank in banks.items()},
              status='development data only; no untouched external confirmation used')
    return dict(frame=f,x=x,y=f.label.to_numpy(np.float32),s=f.protected.to_numpy(np.float32),
                splits=splits,encoder=encoder,banks=banks,metadata=meta,edits=edits,education_map=education_map)


class ResidualBlock(nn.Module):
    def __init__(self,width):
        super().__init__(); self.branch=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,width))
    def forward(self,x): return torch.relu(x+self.branch(x))


class Predictor(nn.Module):
    def __init__(self,input_dim,architecture='mlp',mode='erm',width=128):
        super().__init__(); self.mode=mode; self.architecture=architecture; self.input_dim=input_dim; self.width=width
        if architecture=='mlp':
            self.net=nn.Sequential(nn.Linear(input_dim,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1))
        elif architecture=='resnet':
            self.net=nn.Sequential(nn.Linear(input_dim,width),nn.ReLU(),ResidualBlock(width),ResidualBlock(width),nn.Linear(width,1))
        else: raise ValueError(architecture)
        self.offsets=nn.Parameter(torch.zeros(2))

    def base(self,x):
        if self.mode in ('removed','structural_L','structural_P'):
            x=x.clone();x[:,0]=0
        return self.net(x).squeeze(-1)

    def probability(self,x):
        g=self.base(x)
        if self.mode=='structural_P':
            # Evaluate only the bounded affine head in float64. The v2
            # float32 sum/subtraction could overshoot 1 by one ULP.
            c=self.offsets.double().sigmoid();low=c.amin();width=1-(c.amax()-low)
            p=width*g.double().sigmoid()+(c[x[:,0].long()]-low)
            # Mathematical bounds already hold; this clips roundoff only.
            return p.clamp(0.,1.).to(g.dtype)
        if self.mode=='structural_L':g=g+self.offsets[x[:,0].long()]
        return g.sigmoid()

    def forward(self,x):
        if self.mode=='structural_P':return torch.logit(self.probability(x).clamp(1e-7,1-1e-7))
        g=self.base(x)
        if self.mode=='structural_L':g=g+self.offsets[x[:,0].long()]
        return g

    def score(self,x,scale):
        return self.probability(x) if scale=='P' else self(x)


class Pipeline:
    def __init__(self,encoder,model,device='cpu'):
        self.encoder=encoder; self.model=model; self.device=device
    def predict_score(self,raw,scale='P'):
        x=torch.tensor(self.encoder.transform(raw),device=self.device)
        self.model.eval()
        with torch.no_grad():return self.model.score(x,scale).cpu().numpy()


def behavior(y,s,p):
    prediction=p>=.5
    result=dict(auc=float(roc_auc_score(y,p)),ap=float(average_precision_score(y,p)),
                f1=float(f1_score(y,prediction,zero_division=0)),accuracy=float(accuracy_score(y,prediction)),
                brier=float(brier_score_loss(y,p)),bce=float(log_loss(y,p,labels=[0,1])))
    rates={}
    for group in (0,1):
        keep=s==group;pos=keep&(y==1);neg=keep&(y==0)
        rates[group]=dict(tpr=float(prediction[pos].mean()),fpr=float(prediction[neg].mean()),
                          positive=float(prediction[keep].mean()),n=int(keep.sum()))
    dt=rates[1]['tpr']-rates[0]['tpr'];df=rates[1]['fpr']-rates[0]['fpr']
    result.update(eomax=max(abs(dt),abs(df)),aod=.5*(abs(dt)+abs(df)),
                  dp=abs(rates[1]['positive']-rates[0]['positive']),group_rates=rates)
    return result


def effects(scores):
    a=scores[1]-scores[0];b=scores[3]-scores[2]
    return a,b,b-a


def summarize_effects(scores,margin,tolerance):
    a,b,r=effects(scores)
    return dict(n=len(r),mean_abs=float(np.abs(r).mean()),q95_abs=float(np.quantile(abs(r),.95)),
                violation_rate=float((abs(r)>tolerance).mean()),opposed_rate=float((((a>margin)&(b<-margin))|((b>margin)&(a<-margin))).mean()),
                mean_common_abs=float(np.abs((a+b)/2).mean()),mean_effect_abs=float(((abs(a)+abs(b))/2).mean()),
                negative_effect_mean=float((np.maximum(-a,0)+np.maximum(-b,0)).mean()/2),
                near_flat_both_rate=float(((abs(a)<=margin)&(abs(b)<=margin)).mean()),
                negative_both_rate=float(((a<-margin)&(b<-margin)).mean()))


@torch.no_grad()
def predict(model,x,device,scale='P',batch=4096):
    model.eval(); out=[]
    for i in range(0,len(x),batch):out.append(model.score(torch.as_tensor(x[i:i+batch],device=device),scale).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def evaluate(model,data,split,device,primary_only=False,save_path=None):
    ids=data['splits'][split]
    result={'behavior':behavior(data['y'][ids],data['s'][ids],predict(model,data['x'][ids],device)),'rules':{}}
    arrays={}
    for name,b in data['banks'][split].items():
        if primary_only and not b['training']:continue
        valid=b['valid'];keep=b['supported'];n=len(keep)
        rule=dict(anchors=n,valid=int(valid.sum()),supported=int(keep.sum()),training=b['training'])
        scores={}
        for scale in ('L','P'):
            values=predict(model,b['corners'].reshape(4*n,-1),device,scale).reshape(4,n)
            scores[scale]=values; arrays[name+'_'+scale]=values
            for label,mask in [('valid',valid),('supported',keep)]:
                rule[scale+'_'+label]=summarize_effects(values[:,mask],.01 if scale=='L' else .001,.05 if scale=='L' else .01) if mask.any() else None
        dec=(scores['P']>=.5).astype(int)
        if keep.any():
            da,db,dr=effects(dec[:,keep]);rule['decision_transition_disagreement']=float((da!=db).mean())
        result['rules'][name]=rule
        arrays[name+'_supported']=keep;arrays[name+'_valid']=valid;arrays[name+'_indices']=b['indices']
    for scale in ('L','P'):
        vals=[r[scale+'_supported']['mean_abs'] for r in result['rules'].values() if r['training'] and r[scale+'_supported'] is not None]
        result['primary_'+scale]=float(np.mean(vals)) if vals else None
        directions=[r[scale+'_supported']['negative_effect_mean'] for r in result['rules'].values() if r['training'] and r[scale+'_supported'] is not None]
        result['primary_direction_'+scale]=float(np.mean(directions)) if directions else None
    if save_path:
        Path(save_path).parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(save_path,**arrays)
    return result


def guard_loss(p,y,s):
    values=[]
    for label in (0,1):
        a=(y==label)&(s==0);b=(y==label)&(s==1)
        if a.sum()>=4 and b.sum()>=4: values.append((p[a].mean()-p[b].mean()).square())
    return torch.stack(values).mean() if values else p.sum()*0
