"""Bounded continuation of soft IS; only train/validation used in fitting.

The baseline architecture and original soft requirement are retained. Additional
losses address decision responses and rates at the actual threshold.
"""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,copy,time,hashlib,itertools,argparse
HERE=_PACKAGE / 'training/behavior'
ROOT=_PACKAGE
sys.path.insert(0,str(ROOT/'training/shared'))
import common as c
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scipy.special import expit
torch.set_num_threads(1)
write=c.write
PAIRS=list(itertools.combinations(range(4),2))

def inputs(task,seed,arch,joint=False):
    d,banks=c.data(task,seed)
    if joint:
        bank=c.pickle.load((ROOT/'core/cache_v2/hmda_oh_protected_vector.pkl').open('rb'))
        d=copy.copy(d);d['banks']=bank['banks'];banks=d['banks']['train']
        ph=c.phase(seed)
        er=ROOT/'training/shared/runs'/ph/'joint'/arch/task/str(seed)/'erm/selected.pt'
        dr=ROOT/'training/equality/vector_selector_revision/runs'/ph/'vector/mlp/hmda_oh'/str(seed)/'vector'
    else:
        er=c.source_path(task,seed,arch)
        if task.startswith('acs_employment'):dr=c.directory(task,seed,arch,'soft')
        elif task=='hmda_oh':
            dr=ROOT/'training/equality/runs'/c.phase(seed)/'main'/arch/task/str(seed)/'interfair'
            if seed>=2000 and arch=='ft':dr=dr.parent/'interfair_strength0.3'
        else:
            select=ROOT/'data/tasks/runs'/c.phase(seed)/'primary'/arch/task/str(seed)/'METHOD_SELECTION.json'
            if select.exists():dr=Path(json.loads(select.read_text())['paths']['interfair'])
            else:dr=select.parent/'interfair'
    record=json.loads((dr/'DONE.json').read_text());soft=Path(record['checkpoint'])
    assert er.exists() and soft.exists(),(er,soft)
    return d,banks,er,soft

def natural_groups(d,ids,joint):
    s=d['s'][ids].astype(int)
    result={'protected':s}
    if joint:
        v=d['frame'].derived_sex.to_numpy()[ids]
        known=np.isin(v,['Male','Female']);sex=np.where(known,(v=='Female').astype(int),-1)
        result.update(sex=sex,joint=np.where(known,2*s+sex,-1))
    return result

def flat_metrics(y,s,z):
    b=c.behavior(y,s,expit(z))
    rates=b.pop('group_rates');a=rates.get('0',rates.get(0));q=rates.get('1',rates.get(1))
    b['tpr_gap']=abs(a['tpr']-q['tpr']);b['fpr_gap']=abs(a['fpr']-q['fpr'])
    b.update({f'group{k}_{name}':val for k,r in rates.items() for name,val in r.items()})
    return b

def corner_metrics(z,joint):
    v=z.reshape(4 if joint else 2,2,-1);e=v[:,1]-v[:,0];p=expit(v);ep=p[:,1]-p[:,0]
    dd=(v>=0).astype(int);de=dd[:,1]-dd[:,0]
    pairs=PAIRS if joint else [(0,1)]
    lr=np.stack([e[b]-e[a] for a,b in pairs]);pr=np.stack([ep[b]-ep[a] for a,b in pairs])
    r=dict(L_R=float(abs(lr).mean()),violation_005=float((abs(lr)>.05).mean()),P_R=float(abs(pr).mean()),
       decision_disagreement=float((np.ptp(de,axis=0)>0).mean()),
       opposite_005=float(((ep.max(0)>.005)&(ep.min(0)<-.005)).mean()),P_mean_common_abs=float(abs(ep.mean(0)).mean()))
    if joint:r['L_third']=float(abs(e[3]-e[2]-e[1]+e[0]).mean())
    return r

@torch.no_grad()
def evaluate(m,d,split,names,joint=False,device='cpu',save=None):
    ix=d['splits'][split];z=c.logits(m,d['x'][ix],device)
    r=flat_metrics(d['y'][ix],d['s'][ix],z)
    arr={'natural_L':z,'natural':expit(z),'natural_indices':ix}
    if joint:
        groups=natural_groups(d,ix,True);sex=groups['sex'];ok=sex>=0
        sx=flat_metrics(d['y'][ix][ok],sex[ok],z[ok]);r.update({'sex_'+k:v for k,v in sx.items() if k in ('aod','dp','eomax','tpr_gap','fpr_gap')})
        g=groups['joint'];rates=[]
        for k in range(4):
            mask=g==k;yy=d['y'][ix][mask];pred=z[mask]>=0
            rates.append([pred[yy==1].mean(),pred[yy==0].mean(),pred.mean()])
        rates=np.array(rates);diff=np.stack([abs(rates[b]-rates[a]) for a,b in PAIRS])
        r.update(subgroup_aod=float(diff[:,:2].mean(1).max()),subgroup_dp=float(diff[:,2].max()),subgroup_eomax=float(diff[:,:2].max()))
    features=[]
    for name in names:
        b=d['banks'][split][name];keep=b['supported'];ids=b['indices'][keep];cc=b['corners'][:,keep]
        zz=c.logits(m,cc.reshape(-1,cc.shape[-1]),device).reshape(cc.shape[0],-1)
        fm=corner_metrics(zz,joint)
        if not joint:
            pp=expit(zz);e0=pp[1]-pp[0];e1=pp[3]-pp[2]
            observed=np.where(d['s'][ids]>.5,e1,e0)
            fm['P_population_effect_gap']=float(c.r.relations.group_gap_np(observed,d['s'][ids],d['y'][ids])['conditional_mean'])
        features.append(dict(feature=name,n=len(ids),**fm))
        arr[name+'_L']=zz;arr[name+'_indices']=ids;arr[name+'_supported']=np.ones(len(ids),bool)
    ff=pd.DataFrame(features);r.update(ff.drop(columns=['feature','n']).mean().to_dict())
    if save:
        np.savez_compressed(save,**arr)
    return r,features

def fairness_names(joint):
    q=['aod','dp','eomax','tpr_gap','fpr_gap']
    if joint:q+=['sex_aod','sex_dp','sex_eomax','sex_tpr_gap','sex_fpr_gap','subgroup_aod','subgroup_dp','subgroup_eomax']
    return q

def select_key(v,ref,start,joint):
    fair=fairness_names(joint)
    guard=fair+['decision_disagreement']
    reg=[max(0.,v[k]-ref[k])/(ref[k]+.01) for k in guard]
    # One exploratory budget for all tasks. Export raw utility and the original
    # .01 loss policy as well; no test-dependent admission exception.
    utility=max(0.,ref['auc']-.02-v['auc'])/.02+max(0.,ref['accuracy']-.02-v['accuracy'])/.02+max(0.,ref['f1']-.02-v['f1'])/.02
    requirement=max(0.,v['L_R']-max(start['L_R']*1.25,.015))/max(start['L_R'],.015)
    accepted=max(reg)<=1e-12 and utility<=1e-12 and v['L_R']<ref['L_R']
    # Among feasible candidates pursue a safety margin on the declared rates,
    # rather than minimize a protected main effect absent from the requirement.
    ratios=[v[k]/(ref[k]+.01) for k in guard]
    key=(int(not accepted),max(reg)+utility+requirement,np.mean(ratios)+.1*v['L_R']/max(ref['L_R'],.01)+.05*v['bce'])
    return key,accepted

def rate_loss(z,y,groups,weight=16.,temperatures=(.25,.5,1.)):
    losses=[]
    for group in groups.values():
        vals=torch.unique(group[group>=0]);ss=[]
        for temp in temperatures:
            p=torch.sigmoid(z/temp)
            for label in (0,1,None):
                rates=[]
                for g in vals:
                    mask=group==g
                    if label is not None:mask=mask&(y==label)
                    if mask.sum()<8:continue
                    rates.append(p[mask].mean())
                if len(rates)>1:
                    rr=torch.stack(rates);pairs=[(rr[b]-rr[a]).square() for a,b in itertools.combinations(range(len(rr)),2)]
                    ss.append(torch.stack(pairs).mean()*(.5 if label is None else 1.))
        if ss:losses.append(torch.stack(ss).mean())
    return weight*torch.stack(losses).mean() if losses else z.sum()*0

def transition_loss(z,joint,temperatures=(.15,.35,.7)):
    v=z.reshape(4 if joint else 2,2,-1);pairs=PAIRS if joint else [(0,1)]
    terms=[]
    for temp in temperatures:
        p=(v/temp).sigmoid();e=p[:,1]-p[:,0]
        terms.append(torch.stack([(e[b]-e[a]).square().mean() for a,b in pairs]).mean())
    return torch.stack(terms).mean()

def normalized_reference(z,y,groups):
    out={}
    for name,group in groups.items():
        values=np.unique(group[group>=0])
        for temp in (.05,.15,.35):
            p=expit(z/temp)
            for label in (0,1,None):
                rates={int(g):p[(group==g)&((y==label) if label is not None else np.ones(len(y),bool))].mean() for g in values}
                for a,b in itertools.combinations(values,2):
                    gap=abs(rates[int(b)]-rates[int(a)])
                    out[(name,temp,label,int(a),int(b))]=(max(float(gap),.02),.25*float(gap))
    return out

def normalized_rate_loss(z,y,groups,reference):
    terms=[]
    for name,group in groups.items():
        vals=torch.unique(group[group>=0]).tolist()
        for temp in (.05,.15,.35):
            p=(z/temp).sigmoid()
            for label in (0,1,None):
                rates={}
                for g in vals:
                    mask=(group==g)
                    if label is not None:mask=mask&(y==label)
                    if mask.sum()>=8:rates[int(g)]=p[mask].mean()
                for a,b in itertools.combinations(sorted(rates),2):
                    scale,target=reference[(name,temp,label,a,b)];gap=(rates[b]-rates[a]).abs()
                    terms.append((F.relu(gap-target)/scale).square()+.1*(gap/scale).square())
    t=torch.stack(terms)
    return .5*(t.mean()+.25*t.max())

def train(task,seed,arch='mlp',joint=False,arm='threshold',steps=600,device='cpu'):
    dest=HERE/'runs'/c.phase(seed)/('joint' if joint else 'main')/arch/task/str(seed)/arm
    if (dest/'DONE.json').exists():return
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'TRAINER.py').write_bytes(Path(__file__).read_bytes())
    d,banks,er,soft=inputs(task,seed,arch,joint);names=list(banks)
    c.seed_all(seed);m=c.load(soft,d,arch,device);reference=c.load(er,d,arch,device)
    ref,_=evaluate(reference,d,'val',names,joint,device);initial,_=evaluate(m,d,'val',names,joint,device)
    x=torch.as_tensor(d['x'],device=device);y=torch.as_tensor(d['y'],device=device);s=torch.as_tensor(d['s'],device=device)
    tr=d['splits']['train'];groups={k:torch.as_tensor(v,device=device) for k,v in natural_groups(d,np.arange(len(d['x'])),joint).items()}
    start_logits=torch.zeros(len(d['x']),device=device,dtype=torch.float64)
    start_logits[tr]=torch.as_tensor(c.logits(m,d['x'][tr],device),device=device)
    source_config=torch.load(soft,map_location='cpu',weights_only=False)['config']
    cfg=copy.deepcopy(source_config.get('recipe',c.r.recipe(task,arch,1.)));cfg['anchors']=256
    pools={}
    for n,b in banks.items():
        keep=b['supported'];assert np.isin(b['indices'][keep],tr).all()
        pools[n]=dict(c=torch.as_tensor(b['corners'][:,keep],device=device),ids=b['indices'][keep])
    strong=arm in ('threshold_strong','normalized');extra=arm.startswith('threshold') or arm=='normalized'
    population_weight=64. if strong else 16.;transition_weight=32. if strong else 4.
    rate_temperatures=(.05,.15,.35) if strong else (.25,.5,1.)
    transition_temperatures=(.05,.15,.35) if strong else (.15,.35,.7)
    norm_ref=None
    if arm=='normalized':
        norm_ref=normalized_reference(c.logits(reference,d['x'][tr],device),d['y'][tr],natural_groups(d,tr,joint))
    configuration=dict(task=task,seed=seed,architecture=arch,joint=joint,arm=arm,steps=steps,lr=.00015 if arch=='mlp' else .00003,
      source_erm=str(er),source_soft=str(soft),source_erm_sha256=c.sha(er),source_soft_sha256=c.sha(soft),
      source_validation=ref,initial_validation=initial,soft_recipe=cfg,additional_population_weight=population_weight,additional_transition_weight=transition_weight,
      rate_temperatures=rate_temperatures,transition_temperatures=transition_temperatures,
      normalized_population=arm=='normalized',normalized_gap_floor=.02,normalized_target_fraction=.25,
      task_batch=512,population_batch=2048,corner_anchors=256,split_hashes={k:c.r.array_hash(v) for k,v in d['splits'].items()},
      selection_uses_audit=False,device=device,code_sha256=c.sha(__file__),protocol_sha256=None,
      evaluation_status='development validation only; audit queried separately after freeze',utility_budget=dict(auc_absolute=.02,accuracy_absolute=.02,f1_absolute=.02))
    write(dest/'CONFIG.json',configuration)
    opt=torch.optim.AdamW(m.parameters(),lr=configuration['lr'],weight_decay=.0001)
    rng=np.random.default_rng(seed+929010);history=[];best=None;start=time.monotonic();bh=hashlib.sha256()
    for step in range(steps+1):
        if step:
            opt.zero_grad(set_to_none=True);m.train();ii=tr[rng.integers(len(tr),size=512)];bh.update(ii.tobytes())
            z=m(x[ii]);taskloss=F.binary_cross_entropy_with_logits(z,y[ii])+.2*(z-start_logits[ii]).square().mean();taskloss.backward()
            m.eval();pp=tr[rng.integers(len(tr),size=2048)];z=m(x[pp]);sl=cfg['strength']*c.r.score_loss(z,s[pp],y[pp],cfg)
            if arm=='normalized':sl=sl+normalized_rate_loss(z,y[pp],{k:v[pp] for k,v in groups.items()},norm_ref)
            elif extra:sl=sl+rate_loss(z,y[pp],{k:v[pp] for k,v in groups.items()},population_weight,rate_temperatures)
            sl.backward()
            name=names[int(rng.integers(len(names)))];pool=pools[name]
            jj=c.r.balanced_positions(pool['ids'],d,rng,256);ids=pool['ids'][jj];cc=pool['c'][:,jj]
            z=m(cc.reshape(-1,cc.shape[-1])).reshape(cc.shape[0],-1)
            if joint:
                e=z.reshape(4,2,-1);e=e[:,1]-e[:,0]
                rr=torch.stack([e[b]-e[a] for a,b in PAIRS]);pl=cfg['pair']*(rr.square().mean()+cfg['tail']*c.r.relations.topk_square(rr.flatten(),cfg['tail_fraction']))
            else:
                with torch.no_grad():nz=m(x[ids])
                pl=c.r.pair_effect_loss(z,nz,s[ids],y[ids],cfg)
            pl=cfg['strength']*pl
            if extra:pl=pl+transition_weight*transition_loss(z,joint,transition_temperatures)
            pl.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step()
        if step%100==0:
            val,_=evaluate(m,d,'val',names,joint,device);key,admitted=select_key(val,ref,initial,joint)
            item=dict(step=step,validation=val,key=key,admitted=admitted,seconds=time.monotonic()-start)
            history.append(item)
            if best is None or tuple(key)<tuple(best['key']):
                best=copy.deepcopy(item)
                torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},config=configuration,step=step),dest/'selected.pt')
            write(dest/'HISTORY.json',history)
            print(task,arch,joint,seed,arm,step,admitted,'AUC',round(val['auc'],4),'AOD',round(val['aod'],4),'EO',round(val['eomax'],4),'D',round(val['decision_disagreement'],4),'L',round(val['L_R'],4),flush=True)
    write(dest/'DONE.json',dict(status='fit_complete',config=configuration,selected=best,admitted=best['admitted'],
      checkpoint=str(dest/'selected.pt'),checkpoint_sha256=c.sha(dest/'selected.pt'),seconds=time.monotonic()-start,batch_sha256=bh.hexdigest(),
      no_update_selected=best['step']==0,audit_not_yet_queried=True))

def query(path):
    path=Path(path);j=json.loads(path.read_text());cfg=j['config'];task=cfg['task'];seed=cfg['seed'];arch=cfg['architecture'];joint=cfg['joint']
    d,banks,er,soft=inputs(task,seed,arch,joint);rows=[];features=[]
    for arm,p in [('erm',er),('original_is',soft),('revised_is',Path(j['checkpoint']))]:
        m=c.load(p,d,arch,'cpu');v,f=evaluate(m,d,'audit',list(banks),joint,save=path.parent/(arm+'_audit.npz'))
        rows.append(dict(task=task,seed=seed,architecture=arch,joint=joint,arm=arm,**v));features.extend([dict(task=task,seed=seed,architecture=arch,joint=joint,arm=arm,**a) for a in f])
    pd.DataFrame(rows).to_csv(path.parent/'audit.csv',index=False);pd.DataFrame(features).to_csv(path.parent/'features.csv',index=False)
    write(path.parent/'AUDIT_COMPLETE.json',dict(status='complete',selection_sha256=c.sha(path),audit_used_for_selection=False))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['train','query']);a.add_argument('--task');a.add_argument('--seed',type=int);a.add_argument('--arch',default='mlp');a.add_argument('--joint',action='store_true');a.add_argument('--arm',default='threshold',choices=['threshold','threshold_strong','normalized','continuation']);a.add_argument('--steps',default=600,type=int);a.add_argument('--device',default='cpu');a.add_argument('--record');args=a.parse_args()
    if args.action=='train':train(args.task,args.seed,args.arch,args.joint,args.arm,args.steps,args.device)
    else:query(args.record)
