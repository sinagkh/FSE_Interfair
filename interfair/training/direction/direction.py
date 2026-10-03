"""Feature-specific direction requirements in addition to equal-response steering."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
from pathlib import Path
import sys,json,pickle,copy,time,hashlib,argparse
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scipy.special import expit
P=_PACKAGE / 'training/direction';ROOT=_PACKAGE
sys.path.insert(0,str(ROOT/'training/behavior'))
import behavior as rb
c=rb.c;torch.set_num_threads(1)
MAP={'hmda_oh':{'income':1,'debt_to_income_ratio':-1,'loan_to_value_ratio':-1,'linked_income_plus10k':1},
     'credit_broad':{f'PAY_AMT{i}':1 for i in range(1,7)},
     'acs_income':{'education':1,'hours':1},
     'acs_employment_sex':{'education':1},'acs_employment_age':{'education':1}}
write=c.write

def prepare():
    """Freeze schema-checked requirements and code-preserving linked rectangles."""
    (P/'banks').mkdir(exist_ok=True)
    manifest=[]
    for task,signs in MAP.items():
        d,banks=c.data(task,2000)
        for name,sign in signs.items():
            if name.startswith('linked_'):continue
            b=banks[name];changed=np.where(abs(b['corners'][1]-b['corners'][0]).max(0)>1e-6)[0]
            manifest.append(dict(task=task,edit=name,sign=sign,label=d['metadata'].get('label_definition',d['metadata'].get('label','approval')),
              changed_columns=[d['encoder'].columns[j] for j in changed],
              encoded_endpoints=[[float(b['corners'][i,0,j]) for j in changed] for i in (0,1)],
              support={sp:int((banks if sp=='train' else d['banks'][sp])[name]['supported'].sum()) for sp in ('train','val','audit')},
              basis='declared favorable-score monotonicity for this typed edit; not inferred from fitted model'))
    file=P/'banks/hmda_linked.pkl'
    if not file.exists():
        d,_=c.data('hmda_oh',2000)
        from support import fitted_support
        support=fitted_support(d)
        original=pickle.load((ROOT/'core/cache_v2/hmda_direction_joint.pkl').open('rb'))['banks']
        enc=d['encoder'];j=enc.columns.index('debt_to_income_ratio__scaled');ji=enc.columns.index('income__scaled')
        st=enc.stats['debt_to_income_ratio'];ist=enc.stats['income'];new={}
        for split in ('train','val','audit'):
            new[split]={}
            for amount in ([10] if split!='audit' else [10,20,30,50]):
                b=original[split][f'joint_income_plus{amount}k'];xx=b['corners'].copy()
                dt=np.round(xx[0,:,j].astype(float)*st['scale']+st['median'],4)
                income=np.round(xx[0,:,ji].astype(float)*ist['scale']+ist['median'],4)
                target=dt*income/(income+amount)
                raw=np.select([target<20,target<30,target<36,target<50,target<60],[19.,25.,33.,target,55.],default=65.)
                raw=np.where((target>=36)&(target<50),np.clip(np.round(target),36,49),raw)
                assert np.isin(raw,[19,25,33,*range(36,50),55,65]).all()
                xx[1,:,j]=xx[3,:,j]=((raw-st['median'])/st['scale']).astype(np.float32)
                mask=support.mask(xx)[0]&b['valid']
                if split=='audit':
                    with np.load(ROOT/'archive/INCOME_BANKS.npz') as old:
                        np.testing.assert_array_equal(xx,old[f'joint_income_plus{amount}k__disclosure_integer'])
                        mask &= old[f'joint_income_plus{amount}k__common_supported']
                assert np.isin(b['indices'],d['splits'][split]).all()
                new[split][f'linked_income_plus{amount}k']={**b,'corners':xx,'supported':mask,'training':amount==10}
                print('BANK',split,amount,int(mask.sum()),flush=True)
        with file.open('wb') as f:pickle.dump(new,f)
    linked=pickle.load(file.open('rb'))
    manifest.append(dict(task='hmda_oh',edit='linked_income_plus10k',sign=1,label='approval',
       changed_columns=['income','debt_to_income_ratio'],basis='income +10k with unchanged representative debt; disclosed DTI codes',
       support={sp:int(linked[sp]['linked_income_plus10k']['supported'].sum()) for sp in linked},banks_sha256=c.sha(file)))
    write(P/'DIRECTION_SPECIFICATIONS.json',manifest)
    # An exactly equal but adverse response must fail direction in both groups.
    z=np.array([[2.],[1.],[3.],[2.]])
    a=direction_metrics(z,1);assert a['adverse_005']==1 and a['L_R']==0
    a=direction_metrics(z,-1);assert a['adverse_005']==0 and a['L_R']==0
    write(P/'DIRECTION_PREFLIGHT.json',dict(status='PASS',ordered_features=sum(map(len,MAP.values())),opposite_signs_exercised=True,zero_interaction_adverse_response_detected=True))

def data(task,seed):
    d,b=c.data(task,seed);d['metadata']={**d['metadata'],'task':task}
    direction={sp:{n:(b if sp=='train' else d['banks'][sp])[n] for n in MAP[task] if not n.startswith('linked_')} for sp in ('train','val','audit')}
    if task=='hmda_oh':
        link=pickle.load((P/'banks/hmda_linked.pkl').open('rb'))
        for sp in direction:direction[sp].update(link[sp])
    return d,b,direction

def sources(task,seed):
    d,b,er,soft=rb.inputs(task,seed,'mlp')
    extra=None
    if task in ('credit_broad','acs_employment_sex','acs_employment_age'):
        base=ROOT/'training/behavior/runs'/c.phase(seed)/'main/mlp'/task/str(seed)
        choices=[json.loads((base/a/'DONE.json').read_text()) for a in ('threshold','threshold_strong')]
        j=min(choices,key=lambda v:tuple(v['selected']['key']));soft=Path(j['checkpoint']);extra=copy.deepcopy(j['config'])
        # The first moderate-temperature trainer stored its defaults in code.
        if 'rate_temperatures' not in extra:
            assert extra['additional_population_weight']==16 and extra['additional_transition_weight']==4
            legacy=ROOT/'training/behavior/repair_behavior_v1.py'
            assert extra['code_sha256']==c.sha(legacy)
            extra.update(rate_temperatures=[.25,.5,1.],transition_temperatures=[.15,.35,.7],temperature_source_sha256=c.sha(legacy))
    return er,soft,extra

def direction_metrics(z,sign):
    p=expit(z);e=np.stack((z[1]-z[0],z[3]-z[2]));ep=np.stack((p[1]-p[0],p[3]-p[2]));h=(z>=0).astype(int);ed=np.stack((h[1]-h[0],h[3]-h[2]))
    signed=sign*ep;adverse=np.maximum(0,-sign*e)
    a=rb.corner_metrics(z,False)
    a.update(n=z.shape[1],sign=sign,direction_logit_loss=float(np.square(adverse).mean()),
       adverse_magnitude=float(np.maximum(0,-signed).mean()),adverse_decision=float((sign*ed<0).any(0).mean()),
       common_response=float(abs(ep.mean(0)).mean()),signed_common_response=float(sign*ep.mean(0).mean()))
    for margin,label in [(0.,'0'),(.005,'005'),(.01,'01')]:
        flag=signed < -margin
        a['adverse_'+label]=float(flag.any(0).mean());a['both_adverse_'+label]=float(flag.all(0).mean())
    a['joint_pass']=float(((abs(e[1]-e[0])<=.05)&~(signed<-.005).any(0)).mean())
    return a

@torch.no_grad()
def evaluate(m,d,names,db,split,save=None):
    v,ordinary=rb.evaluate(m,d,split,names,device='cpu',save=save)
    rows=[];arr={}
    for name,b in db[split].items():
        keep=b['supported'];xx=b['corners'][:,keep];z=c.logits(m,xx.reshape(-1,xx.shape[-1]),'cpu').reshape(4,-1)
        sign=MAP[d['metadata']['task']].get(name,1)
        rows.append(dict(edit=name,training_edit=name in MAP[d['metadata']['task']],**direction_metrics(z,sign)))
        arr[name+'_L']=z;arr[name+'_indices']=b['indices'][keep]
    df=pd.DataFrame(rows);macro=df[df.training_edit].drop(columns=['edit','training_edit','n','sign']).mean()
    v.update({'direction_'+k:float(val) for k,val in macro.items()})
    if save:np.savez_compressed(Path(save).with_name('DIRECTION_'+Path(save).name),**arr)
    return v,rows

def key(v,erm,initial):
    k,accepted=rb.select_key(v,erm,initial,False)
    requirement=max(0,v['L_R']-max(1.25*initial['L_R'],.015))/max(initial['L_R'],.015)
    accepted=bool(accepted and requirement<=1e-12)
    return (int(not accepted),float(k[1]),v['direction_adverse_005'],v['direction_direction_logit_loss'],v['decision_disagreement'],v['bce']),accepted

def train(task,seed,weight):
    arm='equality' if weight==0 else 'equality_direction_'+str(weight).replace('.','p')
    dest=P/'direction_runs'/c.phase(seed)/task/str(seed)/arm
    if (dest/'DONE.json').exists():return
    dest.mkdir(parents=True,exist_ok=True)
    if seed<2000:
        freeze=json.loads((P/'DIRECTION_FREEZE.json').read_text());assert freeze['code_sha256']==c.sha(__file__)
        assert weight in [0,freeze['weight']]
    d,banks,db=data(task,seed);names=list(banks);er,soft,extra=sources(task,seed)
    c.seed_all(seed);m=c.load(soft,d,'mlp','cpu');erm=c.load(er,d,'mlp','cpu')
    ref,_=evaluate(erm,d,names,db,'val');initial,_=evaluate(m,d,names,db,'val')
    ck=torch.load(soft,map_location='cpu',weights_only=False)
    cfg=copy.deepcopy(ck['config'].get('soft_recipe',ck['config'].get('recipe',c.r.recipe(task,'mlp',1.))))
    x=torch.as_tensor(d['x']);y=torch.as_tensor(d['y']);s=torch.as_tensor(d['s']);tr=d['splits']['train']
    zstart=torch.zeros(len(x));zstart[tr]=torch.as_tensor(c.logits(m,d['x'][tr],'cpu'),dtype=torch.float32)
    def pools(bs):
        result={}
        for n,b in bs.items():
            keep=b['supported'];ids=b['indices'][keep];assert np.isin(ids,tr).all()
            result[n]={'ids':ids,'c':torch.as_tensor(b['corners'][:,keep])}
        return result
    ordinary=pools(banks);dp=pools(db['train']);assert set(dp)==set(MAP[task])
    config=dict(task=task,seed=seed,arm=arm,direction_weight=weight,source=str(soft),source_sha256=c.sha(soft),erm=str(er),
      steps=600,lr=.00015,soft_recipe=cfg,source_decision_config=extra,source_validation=initial,erm_validation=ref,
      specification_sha256=c.sha(P/'DIRECTION_SPECIFICATIONS.json'),code_sha256=c.sha(__file__),protocol_sha256=None,
      direction_signs=MAP[task],audit_used_for_selection=False,device='cpu')
    write(dest/'CONFIG.json',config);(dest/'TRAINER.py').write_bytes(Path(__file__).read_bytes())
    opt=torch.optim.AdamW(m.parameters(),lr=.00015,weight_decay=.0001);rng=np.random.default_rng(seed+929111)
    history=[];best=None;start=time.monotonic();draws=hashlib.sha256()
    for step in range(601):
        if step:
            m.train();opt.zero_grad(set_to_none=True);ids=tr[rng.integers(len(tr),size=512)];draws.update(ids.tobytes());z=m(x[ids])
            (F.binary_cross_entropy_with_logits(z,y[ids])+.2*(z-zstart[ids]).square().mean()).backward()
            m.eval();ids=tr[rng.integers(len(tr),size=2048)];draws.update(ids.tobytes());z=m(x[ids])
            pl=cfg['strength']*c.r.score_loss(z,s[ids],y[ids],cfg)
            if extra:pl=pl+rb.rate_loss(z,y[ids],{'protected':s[ids]},extra['additional_population_weight'],extra['rate_temperatures'])
            pl.backward()
            name=names[int(rng.integers(len(names)))];pool=ordinary[name];jj=c.r.balanced_positions(pool['ids'],d,rng,256)
            ids=pool['ids'][jj];cc=pool['c'][:,jj];draws.update(name.encode());draws.update(ids.tobytes());z=m(cc.reshape(-1,cc.shape[-1])).reshape(4,-1)
            with torch.no_grad():nz=m(x[ids])
            pl=cfg['strength']*c.r.pair_effect_loss(z,nz,s[ids],y[ids],cfg)
            if extra:pl=pl+extra['additional_transition_weight']*rb.transition_loss(z,False,extra['transition_temperatures'])
            pl.backward()
            name=list(dp)[int(rng.integers(len(dp)))];pool=dp[name];jj=c.r.balanced_positions(pool['ids'],d,rng,256)
            ids=pool['ids'][jj];cc=pool['c'][:,jj];draws.update(name.encode());draws.update(ids.tobytes());z=m(cc.reshape(-1,cc.shape[-1])).reshape(4,-1)
            e=torch.stack([z[1]-z[0],z[3]-z[2]]);res=e[1]-e[0]
            pl=cfg['strength']*cfg['pair']*(res.square().mean()+.25*c.r.relations.topk_square(res,.1))
            adverse=F.relu(-MAP[task][name]*e)
            pl=pl+weight*(adverse.square().mean()+.25*c.r.relations.topk_square(adverse.flatten(),.1))
            pl.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
        if step%100==0:
            val,fr=evaluate(m,d,names,db,'val');kk,admitted=key(val,ref,initial)
            item=dict(step=step,key=kk,admitted=admitted,validation=val,features=fr,seconds=time.monotonic()-start);history.append(item)
            if best is None or kk<tuple(best['key']):
                best=copy.deepcopy(item);torch.save(dict(state_dict={k:v.detach().clone() for k,v in m.state_dict().items()},config=config,step=step),dest/'selected.pt')
            write(dest/'HISTORY.json',history)
            print(task,seed,arm,step,'adverse',round(val['direction_adverse_005'],4),'L',round(val['L_R'],4),'admitted',admitted,flush=True)
    write(dest/'DONE.json',dict(status='complete',config=config,selected=best,checkpoint=str(dest/'selected.pt'),checkpoint_sha256=c.sha(dest/'selected.pt'),
      draw_sha256=draws.hexdigest(),seconds=time.monotonic()-start,selection_completed_before_audit=True))

def report(phase):
    rows=[]
    for p in (P/'direction_runs'/phase).glob('*/*/*/DONE.json'):
        j=json.loads(p.read_text());cfg=j['config'];v=j['selected']
        rows.append(dict(task=cfg['task'],seed=cfg['seed'],arm=cfg['arm'],weight=cfg['direction_weight'],step=v['step'],admitted=v['admitted'],**v['validation']))
    f=pd.DataFrame(rows);f.to_csv(P/(phase+'_direction_validation.csv'),index=False)
    cols=['direction_adverse_005','direction_joint_pass','L_R','decision_disagreement','auc','accuracy','aod','dp','eomax']
    print(f.groupby(['task','arm'])[cols+['admitted']].mean().to_string())

def query(task,seed):
    d,b,db=data(task,seed);names=list(b);er,soft,_=sources(task,seed);out=P/'direction_evaluation'/task/str(seed);out.mkdir(parents=True,exist_ok=True)
    paths=[('erm',er),('source_is',soft)]
    for p in (P/'direction_runs/confirmation'/task/str(seed)).glob('*/DONE.json'):
        j=json.loads(p.read_text());assert j['selection_completed_before_audit'];paths.append((p.parent.name,Path(j['checkpoint'])))
    rows=[];features=[]
    for arm,path in paths:
        m=c.load(path,d,'mlp','cpu');v,ff=evaluate(m,d,names,db,'audit',out/(arm+'.npz'))
        rows.append(dict(task=task,seed=seed,arm=arm,checkpoint=str(path),**v))
        features.extend([dict(task=task,seed=seed,arm=arm,**q) for q in ff])
    pd.DataFrame(rows).to_csv(out/'per_seed.csv',index=False);pd.DataFrame(features).to_csv(out/'per_edit.csv',index=False)
    write(out/'DONE.json',dict(status='complete',rows=len(rows),audit_used_for_selection=False))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','train','report','query']);a.add_argument('--task',choices=list(MAP));a.add_argument('--seed',type=int);a.add_argument('--weight',type=float,default=1.);a.add_argument('--phase',default='development');v=a.parse_args()
    if v.command=='prepare':prepare()
    elif v.command=='train':train(v.task,v.seed,v.weight)
    elif v.command=='query':query(v.task,v.seed)
    else:report(v.phase)
