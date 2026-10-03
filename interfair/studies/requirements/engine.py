"""Full-objective, fixed-budget supporting experiments (CPU)."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,copy,hashlib,json,math,pickle,sys,time
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from torch.nn import functional as F
P=_PACKAGE / 'studies/requirements';ROOT=_PACKAGE
sys.path.insert(0,str(ROOT/'training/mlp'))
import mlp_trainer as primary
q=primary.q;c=q.c;rb=q.rb
torch.set_num_threads(1)
TASKS=('hmda_oh','acs_income','credit_broad')
TARGET_ARMS=('full','features','contexts','population','task','augmentation')
POLICIES=('p20','valid_only','p10','p35','local')

def write(path,obj):c.write(Path(path),obj)
def read(path):return json.loads(Path(path).read_text())
def directory(task,seed,arm):return P/'runs'/c.phase(seed)/task/str(seed)/arm
def source(task,seed,arm='erm'):
    return ROOT/'training/mlp/runs'/c.phase(seed)/task/str(seed)/arm/'selected.pt'
def stable_seed(text):return int(hashlib.sha256(text.encode()).hexdigest()[:8],16)

def flags(z,sign=None):
    p=expit(z);e=np.stack([p[1]-p[0],p[3]-p[2]])
    r=z[3]-z[2]-z[1]+z[0];h=(z>=0).astype(int)
    dd=(h[1]-h[0])!=(h[3]-h[2]);opp=((e[0]>.005)&(e[1]<-.005))|((e[1]>.005)&(e[0]<-.005))
    za=z-r[None,:]*np.array([1.,-1.,-1.,1.])[:,None]/4
    ha=(za>=0).astype(int);da=(ha[1]-ha[0])!=(ha[3]-ha[2])
    stable=(abs(p-.5).min(0)>.005)&(abs(expit(za)-.5).min(0)>.005)
    equality=abs(r)>.05;eq_priority=equality&(opp|(dd&~da&stable))
    wrong=(sign*e<-.005).any(0) if sign is not None else np.zeros(z.shape[1],bool)
    return dict(equality=equality,decision=dd,opposite=opp,direction=wrong,
                priority=eq_priority|wrong,equality_priority=eq_priority,
                observed=opp|dd|wrong,joint=equality|wrong)

def discovery_ids(d,task):
    tr=d['splits']['train'];rng=np.random.default_rng(982801)
    if task=='acs_income':
        groups=d['frame'].SERIALNO.to_numpy();unique=np.unique(groups[tr]);rng.shuffle(unique);chosen=[]
        for g in unique:
            chosen.extend(tr[groups[tr]==g].tolist())
            if len(chosen)>=1024:break
        return np.sort(chosen)
    return np.sort(rng.choice(tr,min(1024,len(tr)),replace=False))

def nested(ids,d,key,limit):
    ids=np.unique(ids);rng=np.random.default_rng(stable_seed(key))
    groups=[rng.permutation(ids[(d['s'][ids]==a)&(d['y'][ids]==y)]) for a in (0,1) for y in (0,1)]
    order=np.array([v[i] for i in range(max(map(len,groups))) for v in groups if i<len(v)],dtype=np.int64)
    return order if not limit else order[:limit]

def pool(bank,d,excluded,limit=0,key='',mask='supported'):
    keep=bank[mask]&~np.isin(bank['indices'],excluded)
    ids=bank['indices'][keep];xx=bank['corners'][:,keep]
    if limit:
        take=nested(ids,d,key,limit);lookup={int(v):i for i,v in enumerate(ids)}
        jj=np.array([lookup[int(i)] for i in take]);ids=ids[jj];xx=xx[:,jj]
    assert len(ids)>0,key
    return dict(ids=ids,c=torch.as_tensor(xx),mask_hash=c.r.array_hash(keep))

def prepare_policy():
    dest=P/'banks/policies.pkl'
    if dest.exists():return
    d,b,db=q.data('hmda_oh',2000)
    from support import fitted_support
    support=fitted_support(d);enc=d['encoder'];tr=d['splits']['train']
    available=np.setdiff1d(tr,discovery_ids(d,'hmda_oh'))
    ids=np.random.default_rng(842103).choice(available,min(8192,len(available)),replace=False)
    banks={};counts=[]
    for policy in POLICIES:
        bb={}
        for name,orig in b.items():
            numeric=name in enc.continuous
            cols=[i for i,s in enumerate(enc.columns) if s in (name+'__scaled',name+'__missing') or s.startswith(name+'=')]
            assert cols,name
            xx=np.stack([d['x'][ids].copy() for _ in range(4)])
            xx[0:2,:,0]=0;xx[2:4,:,0]=1
            low=orig['corners'][0,0,cols];high=orig['corners'][1,0,cols]
            xx[0][:,cols]=xx[2][:,cols]=low;xx[1][:,cols]=xx[3][:,cols]=high
            valid=np.ones(len(ids),bool)
            if numeric:
                ix=enc.columns.index(name+'__scaled');missing=enc.columns.index(name+'__missing')
                observed=d['x'][tr[d['x'][tr,missing]==0],ix];grid=np.unique(observed)
                valid &= d['x'][ids,missing]==0
                if policy in ('p10','p35'):
                    quantiles=(.1,.9) if policy=='p10' else (.35,.65)
                    ends=[grid[np.argmin(abs(grid-np.quantile(observed,v)))] for v in quantiles]
                    xx[0,:,ix]=xx[2,:,ix]=ends[0];xx[1,:,ix]=xx[3,:,ix]=ends[1]
                    valid &= ends[1]>ends[0]
                elif policy=='local':
                    lo=d['x'][ids,ix];step=max(float(np.quantile(observed,.8)-np.quantile(observed,.2))/20,1e-6)
                    j=np.searchsorted(grid,lo+step);valid &= j<len(grid)
                    hi=grid[np.minimum(j,len(grid)-1)]
                    xx[0,:,ix]=xx[2,:,ix]=lo;xx[1,:,ix]=xx[3,:,ix]=hi
                assert np.isin(xx[1,:,ix],grid).all(),name
                valid &= xx[1,:,ix]>xx[0,:,ix]
            else:
                raw=d['frame'].iloc[ids][name].astype(str)
                valid &= ~raw.isin(['nan','__MISSING__','Exempt','Unknown','Not applicable']).to_numpy()
            supported=valid&support.mask(xx)[0]
            assert supported.sum()>=32,(policy,name,int(supported.sum()))
            bb[name]=dict(corners=xx,indices=ids.copy(),valid=valid,supported=supported,training=True)
            counts.append(dict(policy=policy,feature=name,valid=int(valid.sum()),supported=int(supported.sum())))
        banks[policy]=bb
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('wb') as f:pickle.dump(banks,f,pickle.HIGHEST_PROTOCOL)
    pd.DataFrame(counts).to_csv(P/'banks/POLICY_COUNTS.csv',index=False)
    write(P/'banks/POLICY_VERIFICATION.json',dict(status='PASS',source_train_only=True,
        observed_numeric_grid=True,shared_context_ids=True,sha256=c.sha(dest),data_hash=c.r.array_hash(d['x'])))

def discover(task,seed):
    out=P/'discovery'/task/str(seed)
    if (out/'DONE.json').exists():return
    d,b,db=q.data(task,seed);did=discovery_ids(d,task);src=source(task,seed)
    model=c.load(src,d,'mlp','cpu');allbanks={**b,**{n:v for n,v in db['train'].items() if n.startswith('linked_')}}
    arrays={};rows=[];selected=[]
    for name,bank in allbanks.items():
        count={};scores=[]
        for role in ('discovery','steering'):
            keep=bank['supported']&(np.isin(bank['indices'],did) if role=='discovery' else ~np.isin(bank['indices'],did))
            xx=bank['corners'][:,keep]
            if xx.shape[1]:z=c.logits(model,xx.reshape(-1,xx.shape[-1]),'cpu').reshape(4,-1)
            else:z=np.zeros((4,0))
            ff=flags(z,q.MAP[task].get(name));arrays[role+'__'+name+'__ids']=bank['indices'][keep]
            arrays[role+'__'+name+'__L']=z
            for k,v in ff.items():arrays[role+'__'+name+'__'+k]=v
            count[role]=dict(n=z.shape[1],**{k:int(v.sum()) for k,v in ff.items()})
            if role=='discovery':scores.append(float(abs(z[3]-z[2]-z[1]+z[0]).mean()) if z.shape[1] else 0.)
        if count['discovery']['priority']>0:selected.append(name)
        rows.append(dict(feature=name,mean_discovery_residual=scores[0],**count))
    out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'WITNESSES.npz',**arrays)
    write(out/'DONE.json',dict(task=task,seed=seed,source=str(src),source_sha256=c.sha(src),
        selected_features=selected,counts=rows,discovery_n=len(did),discovery_hash=c.r.array_hash(did),
        direction_counts_as_failure_independently_of_equality=True,selection_before_training=True,
        arrays_sha256=c.sha(out/'WITNESSES.npz')))

def settings(arm):
    return dict(task_only=arm in ('task','augmentation'),pop=arm not in ('task','augmentation'),
                direct=arm not in ('population','task','augmentation'),limit=int(arm[4:]) if arm.startswith('pool') else 0,
                policy=arm[7:] if arm.startswith('policy_') else None)

def train(task,seed,arm,epochs=40):
    out=directory(task,seed,arm)
    if (out/'DONE.json').exists():return
    if seed<2000:
        freeze=read(P/'FREEZE.json')
        for f,h in freeze['files'].items():assert c.sha(f)==h,f
    d,b,db=q.data(task,seed);names=list(b);did=discovery_ids(d,task)
    discovery=read(P/'discovery'/task/str(seed)/'DONE.json');saved=np.load(P/'discovery'/task/str(seed)/'WITNESSES.npz')
    cfg=copy.deepcopy(c.r.recipe(task,'mlp',1.));st=settings(arm)
    if st['policy']:
        b=pickle.load((P/'banks/policies.pkl').open('rb'))[st['policy']]
        sb={n:(b[n] if n in b else db['train'][n]) for n in q.MAP[task]}
    else:sb={n:db['train'][n] for n in q.MAP[task]}
    mask='valid' if st['policy']=='valid_only' else 'supported'
    ordinary={n:pool(v,d,did,st['limit'],task+':'+n,mask) for n,v in b.items()}
    signed={n:pool(v,d,did,st['limit'],task+':signed:'+n,mask) for n,v in sb.items()}
    chosen=set(names)|set(signed)
    if arm in ('features','contexts'):chosen=set(discovery['selected_features'])
    if arm=='domain_pair':chosen={'income','debt_to_income_ratio'}
    if arm=='ranked_pair':chosen=set(r['feature'] for r in sorted([x for x in discovery['counts'] if x['feature'] in names],key=lambda x:-x['mean_discovery_residual'])[:2])
    if arm=='random_pair':chosen=set(np.random.default_rng(seed+11709).choice(names,2,replace=False))
    def allocated(pools):
        result={}
        for n,v in pools.items():
            if n not in chosen:continue
            if arm=='contexts':
                ids=saved['steering__'+n+'__ids'];flags_=saved['steering__'+n+'__priority'];wanted=ids[flags_]
                keep=np.isin(v['ids'],wanted)
                if keep.any():result[n]=dict(ids=v['ids'][keep],c=v['c'][:,keep])
            else:result[n]=v
        return result
    direct=allocated(ordinary);sdir=allocated(signed)
    tr=d['splits']['train'];score_ids=nested(tr,d,task+':score',st['limit'])
    c.seed_all(seed);m=c.model(d,'mlp');initial=c.state_hash(m)
    reference=read(source(task,seed).parent/'DONE.json');ref=reference['selected']['validation']
    assert initial==reference['config']['initial_state_sha256']
    x=torch.as_tensor(d['x']);y=torch.as_tensor(d['y']);s=torch.as_tensor(d['s'])
    witness=[]
    for n in discovery['selected_features']:
        ids=saved['steering__'+n+'__ids'];witness.extend(ids[saved['steering__'+n+'__priority']].tolist())
    aug_ids=np.unique(witness).astype(int);augx=x[aug_ids];augy=y[aug_ids]
    config=dict(task=task,seed=seed,architecture='mlp',arm=arm,epochs=epochs,
        initialization='fresh seeded random model',initial_checkpoint=None,pretrained_weights_loaded=False,
        initial_state_sha256=initial,optimizer='AdamW',lr=.001,weight_decay=.0001,task_batch=512,
        population_batch=512,population_effect_contexts=256,direct_contexts=256,signed_contexts=256,
        validation_every=5,soft_recipe=cfg,direction_weight=1. if st['direct'] else 0.,
        decision_rate_weight=16. if st['pop'] else 0.,decision_response_weight=4. if st['direct'] else 0.,
        fairness_warmup_epochs=5,selection_uses_audit=False,selected_features=sorted(chosen),
        active_direct=list(direct),active_signed=list(sdir),pool_limit=st['limit'],policy=st['policy'],
        pool_counts={n:len(v['ids']) for n,v in ordinary.items()},signed_pool_counts={n:len(v['ids']) for n,v in signed.items()},
        source_erm=str(source(task,seed)),source_for_discovery_only=True,
        trainer_sha256=c.sha(__file__),protocol_sha256=None,
        discovery_sha256=c.sha(P/'discovery'/task/str(seed)/'DONE.json'),
        data_hashes={k:c.r.array_hash(d[k]) for k in ['x','y','s']},
        split_hashes={k:c.r.array_hash(v) for k,v in d['splits'].items()})
    out.mkdir(parents=True,exist_ok=True);write(out/'CONFIG.json',config);(out/'TRAINER.py').write_bytes(Path(__file__).read_bytes())
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    rngs=[np.random.default_rng(seed+i) for i in (201,11001,11002,11003,11004,11005)]
    hashes=[hashlib.sha256() for _ in range(5)];queries=0;steps=0;history=[];best=None;start=time.monotonic()
    basecfg={**cfg,'pair':0.}
    def sample(pools,rng,h):
        name=list(pools)[int(rng.integers(len(pools)))];p=pools[name]
        jj=c.r.balanced_positions(p['ids'],d,rng,256);ids=p['ids'][jj]
        h.update(name.encode());h.update(ids.tobytes());return name,ids,p['c'][:,jj]
    for epoch in range(1,epochs+1):
        order=rngs[0].permutation(tr);warm=min(epoch/5,1.)
        for off in range(0,len(order),512):
            ids=order[off:off+512];hashes[0].update(ids.tobytes());steps+=1
            m.train();opt.zero_grad(set_to_none=True);F.binary_cross_entropy_with_logits(m(x[ids]),y[ids]).backward()
            m.eval();ii=score_ids[c.r.balanced_positions(score_ids,d,rngs[1],512)];hashes[1].update(ii.tobytes())
            z=m(x[ii]);queries+=len(ii)
            loss=c.r.score_loss(z,s[ii],y[ii],cfg)+rb.rate_loss(z,y[ii],{'protected':s[ii]},16.,(.25,.5,1.))
            (warm*loss*(1. if st['pop'] else 0.)).backward()
            name,ii,cc=sample(ordinary,rngs[2],hashes[2]);z=m(cc.reshape(-1,cc.shape[-1])).reshape(4,-1);queries+=len(ii)*4
            with torch.no_grad():nz=m(x[ii]);queries+=len(ii)
            loss=c.r.pair_effect_loss(z,nz,s[ii],y[ii],basecfg)
            (warm*loss*(1. if st['pop'] else 0.)).backward()
            name,ii,cc=sample(direct or ordinary,rngs[3],hashes[3]);z=m(cc.reshape(-1,cc.shape[-1])).reshape(4,-1);queries+=len(ii)*4
            rv=z[3]-z[2]-z[1]+z[0]
            loss=cfg['pair']*(rv.square().mean()+.25*c.r.relations.topk_square(rv,.1))+4.*rb.transition_loss(z,False,(.15,.35,.7))
            (warm*loss*(1. if st['direct'] and direct else 0.)).backward()
            name,ii,cc=sample(sdir or signed,rngs[4],hashes[4]);z=m(cc.reshape(-1,cc.shape[-1])).reshape(4,-1);queries+=len(ii)*4
            e=torch.stack([z[1]-z[0],z[3]-z[2]]);rv=e[1]-e[0];ad=F.relu(-q.MAP[task][name]*e)
            loss=cfg['pair']*(rv.square().mean()+.25*c.r.relations.topk_square(rv,.1))+ad.square().mean()+.25*c.r.relations.topk_square(ad.flatten(),.1)
            (warm*loss*(1. if st['direct'] and sdir else 0.)).backward()
            # Equal query allocation for the augmentation slot, with zero loss in other arms.
            augpool=aug_ids if len(aug_ids) else tr;jj=augpool[rngs[5].integers(len(augpool),size=128)]
            ax=x[jj];alt=ax.clone();alt[:,0]=1-alt[:,0];az=m(torch.cat([ax,alt]));queries+=256
            al=F.binary_cross_entropy_with_logits(az,y[jj].repeat(2))
            (al*(1. if arm=='augmentation' and len(aug_ids) else 0.)).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step()
        if epoch%5==0 or epoch==epochs:
            v,_=q.evaluate(m,d,names,db,'val');key,admitted=primary.select_key(v,ref)
            item=dict(epoch=epoch,steps=steps,validation=v,key=key,admitted=admitted,seconds=time.monotonic()-start);history.append(item)
            if best is None or key<tuple(best['key']):
                best=copy.deepcopy(item)
                torch.save(dict(state_dict={k:v.detach().clone() for k,v in m.state_dict().items()},config=config,epoch=epoch),out/'selected.pt')
            write(out/'HISTORY.json',history)
            print(task,seed,arm,epoch,'R',round(v['L_R'],4),'D',round(v['decision_disagreement'],4),'W',round(v['direction_adverse_005'],4),'AUC',round(v['auc'],4),admitted,flush=True)
    assert steps==epochs*math.ceil(len(tr)/512)
    write(out/'DONE.json',dict(status='complete',config=config,selected=best,
      checkpoint=str(out/'selected.pt'),checkpoint_sha256=c.sha(out/'selected.pt'),
      initial_state_sha256=initial,optimizer_updates=steps,regularizer_model_rows=queries,
      task_batch_sha256=hashes[0].hexdigest(),population_batch_sha256=hashes[1].hexdigest(),
      population_effect_sha256=hashes[2].hexdigest(),direct_sha256=hashes[3].hexdigest(),signed_sha256=hashes[4].hexdigest(),
      selection_completed_before_audit=True,seconds=time.monotonic()-start))

def evaluate(task,seed,arm):
    out=directory(task,seed,arm);dest=P/'evaluation'/c.phase(seed)/task/str(seed)/arm
    if (dest/'DONE.json').exists():return
    j=read(out/'DONE.json');d,b,db=q.data(task,seed);m=c.load(out/'selected.pt',d,'mlp','cpu');dest.mkdir(parents=True,exist_ok=True)
    metrics,edits=q.evaluate(m,d,list(b),db,'audit',dest/'audit.npz')
    row=dict(task=task,architecture='mlp',seed=seed,arm=arm,checkpoint=j['checkpoint'],arrays=str(dest/'audit.npz'),admitted=j['selected']['admitted'],**metrics)
    pd.DataFrame([row]).to_csv(dest/'per_seed.csv',index=False)
    pd.DataFrame([{**{k:row[k] for k in ['task','architecture','seed','arm']},**r} for r in edits]).to_csv(dest/'per_edit.csv',index=False)
    write(dest/'DONE.json',dict(status='complete',checkpoint_sha256=j['checkpoint_sha256'],selection_before_audit=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','discover','train','evaluate']);p.add_argument('--task');p.add_argument('--seed',type=int);p.add_argument('--arm');p.add_argument('--epochs',type=int,default=40);a=p.parse_args()
    if a.command=='prepare':prepare_policy()
    elif a.command=='discover':discover(a.task,a.seed)
    elif a.command=='train':train(a.task,a.seed,a.arm,a.epochs)
    else:evaluate(a.task,a.seed,a.arm)
