"""Fresh equality-and-direction steering for FT and joint protected groups."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import os
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK']='1'
_cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES')
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,copy,hashlib,itertools,json,math,pickle,sys,time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scipy.special import expit
P=_PACKAGE / 'training/transformer';ROOT=_PACKAGE
sys.path.insert(0,str(ROOT/'training/direction'))
import direction as q
if _cuda_visible is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
else:os.environ['CUDA_VISIBLE_DEVICES']=_cuda_visible
c=q.c;rb=q.rb;torch.set_num_threads(1)
SCRATCH=_PACKAGE / 'training/mlp'

def directory(task,seed,arch,joint=False):
    return P/'runs'/c.phase(seed)/('joint' if joint else 'main')/arch/task/str(seed)/'soft'

def joint_linked():
    file=P/'banks/hmda_joint_linked.pkl'
    if file.exists():return pickle.load(file.open('rb'))
    from protected_vector import sex_columns
    from support import ordered_training_indices
    from core import Support
    d,_,db=q.data('hmda_oh',2000);cols,sexids=sex_columns(d)
    sex=d['frame'].derived_sex.to_numpy();tr=ordered_training_indices(d)
    supports=[Support(d['x'][tr[sex[tr]==label]]) for label in ('Male','Female')]
    result={}
    for sp,bank in db.items():
        result[sp]={}
        for name,b in bank.items():
            if not name.startswith('linked_'):continue
            n=len(b['indices']);v=b['corners'].reshape(2,2,n,-1)
            cube=np.empty((2,2,2,n,v.shape[-1]),np.float32)
            for race,sx in itertools.product((0,1),repeat=2):
                cc=v[race].copy();cc[:,:,cols]=0;cc[:,:,sexids[sx]]=1;cube[race,sx]=cc
            valid=b['valid']&np.isin(sex[b['indices']],['Male','Female']);keep=valid&b['supported']
            for sx in (0,1):keep &= supports[sx].mask(cube[:,sx].reshape(4,n,-1))[0]
            assert keep.sum()>=32,(sp,name,int(keep.sum()))
            result[sp][name]={**b,'corners':cube.reshape(8,n,-1),'valid':valid,'supported':keep}
    file.parent.mkdir(exist_ok=True)
    with file.open('wb') as f:pickle.dump(result,f)
    c.write(P/'banks/JOINT_LINKED_CHECK.json',dict(status='PASS',counts={sp:{n:int(b['supported'].sum()) for n,b in bs.items()} for sp,bs in result.items()},sha256=c.sha(file)))
    return result

def data(task,seed,joint=False):
    d,b,db=q.data(task,seed)
    if joint:
        bank=pickle.load((ROOT/'core/cache_v2/hmda_oh_protected_vector.pkl').open('rb'))
        d={**d,'banks':bank['banks']};b=d['banks']['train'];links=joint_linked()
        db={sp:{n:d['banks'][sp][n] for n in q.MAP[task] if not n.startswith('linked_')} for sp in ('train','val','audit')}
        for sp in db:db[sp].update(links[sp])
    return d,b,db

def direction_metrics(z,sign,joint):
    if not joint:return q.direction_metrics(z,sign)
    v=z.reshape(4,2,-1);e=v[:,1]-v[:,0];p=expit(v);ep=p[:,1]-p[:,0]
    de=(v[:,1]>=0).astype(int)-(v[:,0]>=0).astype(int);signed=sign*ep
    r=rb.corner_metrics(z,True);res=np.stack([e[b]-e[a] for a,b in rb.PAIRS])
    r.update(n=e.shape[1],sign=sign,direction_logit_loss=float(np.square(np.maximum(0,-sign*e)).mean()),
        adverse_magnitude=float(np.maximum(0,-signed).mean()),adverse_decision=float((sign*de<0).any(0).mean()),
        common_response=float(abs(ep.mean(0)).mean()),signed_common_response=float(sign*ep.mean(0).mean()))
    for margin,label in [(0.,'0'),(.005,'005'),(.01,'01')]:
        flag=signed < -margin;r['adverse_'+label]=float(flag.any(0).mean());r['both_adverse_'+label]=float(flag.all(0).mean())
    r['joint_pass']=float(((abs(res).max(0)<=.05)&~(signed<-.005).any(0)).mean())
    return r

@torch.no_grad()
def evaluate(m,d,names,db,split,joint=False,device='cpu',save=None):
    v,ordinary=rb.evaluate(m,d,split,names,joint=joint,device=device,save=save)
    rows=[];arr={}
    for name,b in db[split].items():
        keep=b['supported'];xx=b['corners'][:,keep]
        z=c.logits(m,xx.reshape(-1,xx.shape[-1]),device).reshape(xx.shape[0],-1)
        sign=q.MAP[d['metadata']['task']].get(name,1)
        rows.append(dict(edit=name,training_edit=name in q.MAP[d['metadata']['task']],**direction_metrics(z,sign,joint)))
        arr[name+'_L']=z;arr[name+'_indices']=b['indices'][keep]
    ff=pd.DataFrame(rows);macro=ff[ff.training_edit].drop(columns=['edit','training_edit','n','sign']).mean()
    v.update({'direction_'+k:float(x) for k,x in macro.items()})
    if save:np.savez_compressed(Path(save).with_name('DIRECTION_'+Path(save).name),**arr)
    return v,rows,ordinary

def select_key(v,ref,joint):
    fair=rb.fairness_names(joint)+['decision_disagreement']
    excess=[max(0.,v[k]-ref[k])/(ref[k]+.01) for k in fair]
    excess += [max(0.,ref[k]-.02-v[k])/.02 for k in ['auc','accuracy','f1']]
    excess += [max(0.,v['L_R']-ref['L_R'])/max(ref['L_R'],.01)]
    admitted=max(excess)<=1e-12
    score=v['violation_005']+v['direction_adverse_005']+v['decision_disagreement']+.1*(v['aod']+v['dp']+v['eomax'])
    return (int(not admitted),sum(excess),score,v['bce']),admitted

def group_loss(value,groups,y,cfg,weights=None):
    terms=[]
    for g in groups.values():
        vals=torch.unique(g[g>=0]).tolist();pairs=[]
        for a,b in itertools.combinations(vals,2):
            k=(g==a)|(g==b);s=(g[k]==b).to(value.dtype)
            if weights is None:loss=c.r.relations.group_gap_square_torch(value[k],s,y[k],cfg['min_cell'])
            else:loss=c.r.relations.weighted_group_gap_square_torch(value[k],s,y[k],cfg['min_cell'],weights[k])
            pairs.append(loss)
        if pairs:terms.append(torch.stack(pairs).mean())
    return torch.stack(terms).mean() if terms else value.sum()*0

def reference_path(task,seed,arch,joint):
    return SCRATCH/'runs'/c.phase(seed)/task/str(seed)/'erm/selected.pt' if joint else c.source_path(task,seed,arch)

def train(task,seed,arch,joint=False,device='cpu',smoke=None):
    out=directory(task,seed,arch,joint)
    if smoke:out=P/'smoke'/('joint' if joint else 'main')/arch/task/str(seed)
    if (out/'DONE.json').exists():return
    if seed<2000:
        freeze=json.loads((P/('FREEZE_joint.json' if joint else 'FREEZE_ft.json')).read_text())
        for path,digest in freeze['files'].items():assert c.sha(path)==digest,path
    out.mkdir(parents=True,exist_ok=True);d,banks,db=data(task,seed,joint);names=list(banks)
    refpath=reference_path(task,seed,arch,joint);rec=json.loads((refpath.parent/'DONE.json').read_text())
    refmodel=c.load(refpath,d,arch,device);ref,_,_=evaluate(refmodel,d,names,db,'val',joint,device);del refmodel
    c.seed_all(seed)
    if device=='cuda':torch.cuda.manual_seed_all(seed)
    m=c.model(d,arch).to(device);initial=c.state_hash(m)
    assert initial==rec['config']['initial_state_sha256'],(initial,rec['config']['initial_state_sha256'])
    cfg=c.r.recipe(task,arch,1.);epochs=smoke or (100 if arch=='ft' else 40)
    config=dict(task=task,seed=seed,architecture=arch,joint=joint,arm='soft',device=device,
        initialization='fresh seeded random model',initial_state_sha256=initial,pretrained_weights_loaded=False,
        initial_checkpoint=None,reference_checkpoint=str(refpath),reference_only_for_validation=True,
        epochs=epochs,patience=16 if arch=='ft' else None,validation_every=1 if arch=='ft' else 5,
        task_batch=512,population_batch=512,ordinary_anchors=256,signed_anchors=256,
        optimizer='official FT make_default_optimizer' if arch=='ft' else 'AdamW lr=.001 wd=.0001',
        direction_weight=1.,decision_rate_weight=16.,decision_response_weight=4.,fairness_warmup_epochs=5,
        soft_recipe=cfg,specification_sha256=c.sha(q.P/'DIRECTION_SPECIFICATIONS.json'),
        trainer_sha256=c.sha(__file__),protocol_sha256=None,selection_uses_audit=False,
        data_hashes={k:c.r.array_hash(d[k]) for k in ['x','y','s']},
        split_hashes={k:c.r.array_hash(v) for k,v in d['splits'].items()},
        joint_population_policy='mean over race, sex, and joint group partitions; exclude unknown sex only from sex/joint penalties' if joint else None)
    c.write(out/'CONFIG.json',config);(out/'TRAINER.py').write_bytes(Path(__file__).read_bytes())
    x=torch.as_tensor(d['x'],device=device);y=torch.as_tensor(d['y'],device=device);s=torch.as_tensor(d['s'],device=device);tr=d['splits']['train']
    groups={k:torch.as_tensor(v,device=device) for k,v in rb.natural_groups(d,np.arange(len(x)),joint).items()}
    def pools(bs):
        result={}
        for name,b in bs.items():
            keep=b['supported'];ids=b['indices'][keep];assert np.isin(ids,tr).all()
            result[name]=dict(ids=ids,c=torch.as_tensor(b['corners'][:,keep],device=device))
        return result
    ordinary=pools(banks);signed=pools({n:db['train'][n] for n in q.MAP[task]})
    rng=np.random.default_rng(seed+201);arng=np.random.default_rng(seed+929301)
    opt=m.optimizer() if arch=='ft' else torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    bh=hashlib.sha256();ah=hashlib.sha256();steps=0;queries=0;history=[];best=None;start=time.monotonic()
    for epoch in range(1,epochs+1):
        permutation=rng.permutation(tr);warm=min(epoch/5,1.);epstart=time.monotonic()
        for off in range(0,len(permutation),512):
            ids=permutation[off:off+512];bh.update(ids.tobytes());steps+=1;m.train();opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(m(x[ids]),y[ids]).backward();m.eval()
            ii=tr[c.r.balanced_positions(tr,d,arng,512)];ah.update(ii.tobytes());z=m(x[ii]);queries+=len(ii)
            gg={k:g[ii] for k,g in groups.items()}
            if joint:
                bw=c.r.relations.boundary_weights_from_logits(z,cfg['boundary_band'])
                loss=cfg['score']*group_loss(z,gg,y[ii],cfg)+cfg['boundary_score']*group_loss(z,gg,y[ii],cfg,bw)
            else:loss=c.r.score_loss(z,s[ii],y[ii],cfg)
            loss+=rb.rate_loss(z,y[ii],gg,16.,(.25,.5,1.));(warm*loss).backward()
            name=names[int(arng.integers(len(names)))];pool=ordinary[name]
            jj=c.r.balanced_positions(pool['ids'],d,arng,256);ii=pool['ids'][jj];ah.update(name.encode());ah.update(ii.tobytes());cc=pool['c'][:,jj]
            z=m(cc.reshape(-1,cc.shape[-1])).reshape(cc.shape[0],-1);queries+=z.numel()
            with torch.no_grad():nz=m(x[ii]);queries+=len(ii)
            if joint:
                e=z.reshape(4,2,-1);e=e[:,1]-e[:,0];rs=torch.stack([e[b]-e[a] for a,b in rb.PAIRS])
                loss=cfg['pair']*(rs.square().mean()+.25*c.r.relations.topk_square(rs.flatten(),.1))
                gg={k:g[ii] for k,g in groups.items()};valid=gg['joint']>=0
                observed=e[gg['joint'].clamp(min=0),torch.arange(len(ii),device=device)]
                gg={k:torch.where(valid,g,-torch.ones_like(g)) for k,g in gg.items()}
                bw=c.r.relations.boundary_weights_from_logits(nz,cfg['boundary_band'])
                loss+=cfg['effect']*group_loss(observed,gg,y[ii],cfg)+cfg['boundary_effect']*group_loss(observed,gg,y[ii],cfg,bw)
            else:loss=c.r.pair_effect_loss(z,nz,s[ii],y[ii],cfg)
            loss+=4.*rb.transition_loss(z,joint,(.15,.35,.7));(warm*loss).backward()
            name=list(signed)[int(arng.integers(len(signed)))];pool=signed[name]
            jj=c.r.balanced_positions(pool['ids'],d,arng,256);ii=pool['ids'][jj];ah.update(name.encode());ah.update(ii.tobytes());cc=pool['c'][:,jj]
            z=m(cc.reshape(-1,cc.shape[-1])).reshape(cc.shape[0],-1);queries+=z.numel()
            e=z.reshape(4 if joint else 2,2,-1);e=e[:,1]-e[:,0]
            rs=torch.stack([e[b]-e[a] for a,b in (rb.PAIRS if joint else [(0,1)])]);adverse=F.relu(-q.MAP[task][name]*e)
            loss=cfg['pair']*(rs.square().mean()+.25*c.r.relations.topk_square(rs.flatten(),.1))
            loss+=adverse.square().mean()+.25*c.r.relations.topk_square(adverse.flatten(),.1)
            (warm*loss).backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step()
        if epoch%config['validation_every']==0 or epoch==epochs:
            v,_,_=evaluate(m,d,names,db,'val',joint,device);key,admitted=select_key(v,ref,joint)
            item=dict(epoch=epoch,steps=steps,validation=v,key=key,admitted=admitted,seconds=time.monotonic()-start,epoch_seconds=time.monotonic()-epstart)
            history.append(item)
            if best is None or key<tuple(best['key']):
                best=copy.deepcopy(item);torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},config=config,epoch=epoch),out/'selected.pt')
            c.write(out/'HISTORY.json',history)
            print(task,arch,'joint' if joint else 'main',seed,epoch,'seconds',round(item['epoch_seconds'],1),'W',round(v['direction_adverse_005'],4),'R',round(v['L_R'],4),'D',round(v['decision_disagreement'],4),'AUC',round(v['auc'],4),'admitted',admitted,flush=True)
            if arch=='ft' and epoch-best['epoch']>=16:break
    assert steps==epoch*math.ceil(len(tr)/512)
    c.write(out/'DONE.json',dict(status='complete',config=config,selected=best,checkpoint=str(out/'selected.pt'),checkpoint_sha256=c.sha(out/'selected.pt'),
        initial_state_sha256=initial,task_batch_sha256=bh.hexdigest(),anchor_sha256=ah.hexdigest(),optimizer_updates=steps,
        regularizer_model_rows=queries,seconds=time.monotonic()-start,epochs_completed=epoch,selection_completed_before_audit=True))

def query(task,seed,arch,joint=False,device='cpu'):
    d,b,db=data(task,seed,joint);out=P/'evaluation'/('joint' if joint else 'main')/arch/task/str(seed);out.mkdir(parents=True,exist_ok=True)
    rows=[];edits=[];features=[]
    for arm,path in [('erm',reference_path(task,seed,arch,joint)),('soft',directory(task,seed,arch,joint)/'selected.pt')]:
        m=c.load(path,d,arch,device);v,ff,ordinary=evaluate(m,d,list(b),db,'audit',joint,device,out/(arm+'.npz'))
        base=dict(task=task,architecture=arch,seed=seed,arm=arm,joint=joint,checkpoint=str(path),arrays=str(out/(arm+'.npz')))
        rows.append({**base,**v});edits.extend([{**base,**f} for f in ff]);features.extend([{**base,**f} for f in ordinary]);del m
    pd.DataFrame(rows).to_csv(out/'per_seed.csv',index=False);pd.DataFrame(edits).to_csv(out/'per_edit.csv',index=False);pd.DataFrame(features).to_csv(out/'per_feature.csv',index=False)
    c.write(out/'DONE.json',dict(status='complete',selection_uses_audit=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','query','prepare']);p.add_argument('--task',default='hmda_oh');p.add_argument('--seed',type=int,default=2000);p.add_argument('--architecture',choices=['mlp','ft'],default='mlp');p.add_argument('--joint',action='store_true');p.add_argument('--device',default='cpu');p.add_argument('--smoke',type=int);a=p.parse_args()
    if a.command=='prepare':joint_linked()
    elif a.command=='train':train(a.task,a.seed,a.architecture,a.joint,a.device,a.smoke)
    else:query(a.task,a.seed,a.architecture,a.joint,a.device)
