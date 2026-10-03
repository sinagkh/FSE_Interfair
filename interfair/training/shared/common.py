"""Shared model construction, evaluation, and baseline training."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,pickle,hashlib,time,ast,itertools,copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import expit
P=_PACKAGE / 'training/shared';ROOT=_PACKAGE
CODE_BYTES=Path(__file__).read_bytes()
CODE_SHA256=hashlib.sha256(CODE_BYTES).hexdigest()
sys.path.insert(0,str(ROOT/'baselines/additive'))
import additive as old
r=old.repair
from core import Predictor,seed_all,state_hash,behavior,sha
from model_interfaces import Pipeline
from preprocessing_baselines import ltdd_fit,LTDDPipeline,cot_transform
from ft_architecture import FTPipeline
write=r.write
torch.set_num_threads(1)

def phase(seed):return 'development' if seed>=2000 else 'confirmation'
def directory(task,seed,arch,arm,joint=False):
    return P/'runs'/phase(seed)/('joint' if joint else 'main')/arch/task/str(seed)/arm

def data(task,seed):
    if task.startswith('acs_employment'):
        path=ROOT/'data/employment/cache'/f'{task}_2000.pkl'
        with path.open('rb') as f:d=pickle.load(f)
        mixture=d['banks']['train']['selected_mixture'];start=0;banks={}
        for name,n in d['metadata']['steering_contexts_per_block'].items():
            sl=slice(start,start+n);start+=n
            banks[name]={k:(v[:,sl] if k=='corners' else v[sl]) for k,v in mixture.items() if isinstance(v,np.ndarray)}
            banks[name]['training']=True
        assert start==len(mixture['indices'])
        d['banks']['train']=banks
        d['metadata']={**d['metadata'],'active_requirement':'equal logit responses on the same five typed ordinary blocks; no preservation term',
            'completion_source_cache':str(path),'completion_cache_sha256':sha(path)}
        return d,banks
    return old.data_and_banks(task,seed)

def model(d,arch):return Predictor(d['x'].shape[1],arch,'erm') if arch in ('mlp','resnet') else FTPipeline(d['encoder'],'erm')

@torch.no_grad()
def logits(m,x,device,batch=None):
    m.eval();batch=batch or (512 if hasattr(m,'typed') else 2048)
    return np.concatenate([m(torch.as_tensor(x[i:i+batch],device=device)).cpu().numpy().astype(float) for i in range(0,len(x),batch)])

def effect_metrics(z,ids,d):
    q=expit(z);e0=q[1]-q[0];e1=q[3]-q[2];rr=z[3]-z[2]-z[1]+z[0]
    dec=(z>=0).astype(int);ordinary=np.where(d['s'][ids]>.5,e1,e0)
    return dict(n=len(ids),L_R=float(abs(rr).mean()),violation_005=float((abs(rr)>.05).mean()),
        P_R=float(abs(e1-e0).mean()),decision_disagreement=float(((dec[1]-dec[0])!=(dec[3]-dec[2])).mean()),
        opposite_005=float((((e0>.005)&(e1<-.005))|((e1>.005)&(e0<-.005))).mean()),
        P_population_effect_gap=float(r.relations.group_gap_np(ordinary,d['s'][ids],d['y'][ids])['conditional_mean']),
        P_mean_common_abs=float(abs((e0+e1)/2).mean()))

def evaluate(m,d,device,out=None,joint=False):
    started=time.monotonic();ix=d['splits']['audit'];natural=logits(m,d['x'][ix],device)
    result=behavior(d['y'][ix],d['s'][ix],expit(natural));arrays=dict(natural=expit(natural),natural_L=natural,
        natural_indices=ix,natural_y=d['y'][ix],natural_s=d['s'][ix]);features=[];queries=len(ix)
    for name,b in d['banks']['audit'].items():
        z=logits(m,b['corners'].reshape(-1,d['x'].shape[1]),device).reshape(4,-1);queries+=z.size
        keep=b['supported'];ids=b['indices'][keep]
        row=dict(feature=name,training=bool(b['training']),**effect_metrics(z[:,keep],ids,d))
        features.append(row);arrays[name+'_L']=z;arrays[name+'_P']=expit(z)
        for k in ['indices','supported','valid']:arrays[name+'_'+k]=b[k]
    f=pd.DataFrame(features);result.update(f[f.training].drop(columns=['feature','training','n']).mean().to_dict())
    if joint:
        with (ROOT/'core/cache_v2/hmda_oh_protected_vector.pkl').open('rb') as stream:bank=pickle.load(stream)
        jf=[]
        for name,b in bank['banks']['audit'].items():
            z=logits(m,b['corners'].reshape(-1,d['x'].shape[1]),device).reshape(8,-1);queries+=z.size
            keep=b['supported'];ee=z[:,keep].reshape(4,2,-1);de=(ee>=0).astype(int);dd=de[:,1]-de[:,0]
            le=ee[:,1]-ee[:,0];pp=expit(ee);pe=pp[:,1]-pp[:,0]
            dif=np.stack([le[b]-le[a] for a,b in itertools.combinations(range(4),2)])
            pdif=np.stack([pe[b]-pe[a] for a,b in itertools.combinations(range(4),2)])
            jf.append(dict(feature=name,L_R=float(abs(dif).mean()),P_R=float(abs(pdif).mean()),
              violation_005=float((abs(dif)>.05).mean()),decision_disagreement=float((np.ptp(dd,axis=0)>0).mean()),
              opposite_005=float(((pe.max(0)>.005)&(pe.min(0)<-.005)).mean()),
              L_third=float(abs(le[3]-le[2]-le[1]+le[0]).mean())))
            arrays['joint__'+name+'_L']=z;arrays['joint__'+name+'_supported']=keep;arrays['joint__'+name+'_indices']=b['indices']
        result['joint_response']=pd.DataFrame(jf).drop(columns='feature').mean().to_dict()
        sex=d['frame'].derived_sex.to_numpy();valid=np.isin(sex[ix],['Male','Female']);sx=(sex[ix]=='Female').astype(int)
        result['sex_behavior']=behavior(d['y'][ix][valid],sx[valid],expit(natural[valid]))
        group=d['s'][ix].astype(int)*2+sx;rates={}
        for g in range(4):
            k=(group==g)&valid;y=d['y'][ix][k];pred=natural[k]>=0
            rates[g]=dict(n=int(k.sum()),tpr=float(pred[y==1].mean()),fpr=float(pred[y==0].mean()),positive=float(pred.mean()))
        result['joint_group_rates']=rates
        result['joint_aod']=max((abs(rates[a]['tpr']-rates[b]['tpr'])+abs(rates[a]['fpr']-rates[b]['fpr']))/2 for a,b in itertools.combinations(range(4),2))
        result['joint_dp']=max(abs(rates[a]['positive']-rates[b]['positive']) for a,b in itertools.combinations(range(4),2))
        result['joint_features']=jf
    result['audit_seconds']=time.monotonic()-started;result['audit_model_rows']=queries;result['features']=features
    if out:
        np.savez_compressed(out/'audit.npz',**arrays)
        write(out/'AUDIT.json',result)
    return result

# Compile only the official numerical functions, avoiding unused TensorFlow/AIF360 imports.
native_file=_PACKAGE / 'vendor/HIFI/bias_mitigation_methods/hifi.py'
tree=ast.parse(native_file.read_text());env={'torch':torch,'nn':nn,
 'get_all_subsets':lambda a:[list(c) for k in range(len(a)+1) for c in itertools.combinations(a,k)],
 'get_all_nonempty_subsets':lambda a:[list(c) for k in range(1,len(a)+1) for c in itertools.combinations(a,k)]}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['get_masked_inputs','custom_loss']],type_ignores=[]),str(native_file),'exec'),env)
native_hifi=env['custom_loss']

class Probability(nn.Module):
    def __init__(self,m):super().__init__();self.m=m
    def forward(self,x):return self.m(x).sigmoid().unsqueeze(1)

class JointView(nn.Module):
    """Expose race and binary sex as two sensitive coordinates to native HIFI."""
    def __init__(self,m,d):
        super().__init__();self.m=m
        from protected_vector import sex_columns
        self.sexcols,self.sexids=sex_columns(d)
        self.keep=[i for i in range(d['x'].shape[1]) if i not in self.sexcols and i!=0]
        self.width=d['x'].shape[1]
    def encode(self,x):return torch.cat([x[:,:1],x[:,self.sexids[1]:self.sexids[1]+1],x[:,self.keep]],1)
    def forward(self,u):
        x=u.new_zeros((len(u),self.width));x[:,0]=u[:,0];x[:,self.keep]=u[:,2:]
        x[:,self.sexids[0]]=1-u[:,1];x[:,self.sexids[1]]=u[:,1]
        return self.m(x).sigmoid().unsqueeze(1)

def source_path(task,seed,arch):
    if task.startswith('acs_employment'):
        return directory(task,seed,arch,'erm')/'selected.pt'
    return old.source_path(task,seed,arch)

def load(path,d,arch,device):
    ck=torch.load(path,map_location='cpu',weights_only=False);m=model(d,arch)
    state=ck.get('state_dict',ck.get('base_state'));m.load_state_dict(state)
    if ck.get('transform') is not None:
        m=LTDDPipeline(m,np.asarray(ck['transform']['slope']),np.asarray(ck['transform']['intercept']))
    return m.to(device).eval()

def standard(task,seed,arch,arm,device='cuda',joint=False):
    if arch=='mlp':device='cpu'
    out=directory(task,seed,arch,arm,joint)
    if (out/'DONE.json').exists():return
    if seed<2000:
        freeze=json.loads((P/'FREEZE.json').read_text())
        for path,digest in freeze['files'].items():assert sha(path)==digest,('frozen input changed',path)
    d,banks=data(task,seed);out.mkdir(parents=True,exist_ok=True)
    (out/'COMPLETION_ADAPTER.py').write_bytes(CODE_BYTES)
    seed_all(seed);base=model(d,arch).to(device);m=base;initial=state_hash(base)
    tr=d['splits']['train'];vi=d['splits']['val'];raw=d['x'][tr].copy();labels=d['y'][tr];weights=np.ones(len(tr),np.float32);extra={};transform=None
    hifi=arm in ('hifi','hifi_erm','hifi_shared');native=arm in ('hifi','hifi_erm');epochs=20 if native else (100 if arch=='ft' else 40)
    if arm=='ltdd':
        assert arch=='mlp' and not joint
        sl,it,records=ltdd_fit(raw,d['encoder'].columns,());m=LTDDPipeline(base,sl,it).to(device)
        transform=dict(slope=sl.tolist(),intercept=it.tolist());extra['transform_records']=records
    elif arm=='cot_phi':
        assert not joint
        rates=[labels[raw[:,0]==g].mean() for g in (0,1)];fav=int(rates[1]>rates[0]);oriented=raw.copy()
        if fav:oriented[:,0]=1-oriented[:,0]
        raw,extra=cot_transform(oriented,labels,seed+401)
        if fav:raw[:,0]=1-raw[:,0]
        extra['training_favored_group']=fav
    elif arm=='reweighing':
        group=d['s'][tr].astype(int)
        group_valid=np.ones(len(tr),bool)
        if joint:
            sx=d['frame'].derived_sex.to_numpy();group_valid=np.isin(sx[tr],['Male','Female'])
            group=2*group+(sx[tr]=='Female')
        for g in np.unique(group[group_valid]):
            for y in (0,1):
                k=(group==g)&(labels==y)&group_valid;assert k.any()
                weights[k]=((group[group_valid]==g).mean()*(labels[group_valid]==y).mean())/k[group_valid].mean()
        assert abs(weights.mean()-1)<1e-6
        extra['group_label_weights']={f'{g}/{y}':float(weights[(group==g)&(labels==y)&group_valid][0]) for g in np.unique(group[group_valid]) for y in (0,1)}
        extra['unclassified_sex_rows_unit_weight']=int((~group_valid).sum())
    elif arm not in ('erm','hifi','hifi_erm','hifi_shared'):raise ValueError(arm)
    x=torch.as_tensor(raw,device=device);y=torch.as_tensor(labels,device=device);w=torch.as_tensor(weights,device=device)
    opt=torch.optim.NAdam(base.parameters(),lr=.002) if native else (base.optimizer() if arch=='ft' else torch.optim.AdamW(base.parameters(),lr=.001,weight_decay=.0001))
    scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=10,gamma=.5) if native else None
    view=JointView(base,d) if joint and hifi else Probability(base)
    sensitive_valid=torch.as_tensor(np.isin(d['frame'].derived_sex.to_numpy()[tr],['Male','Female']),device=device) if joint else None
    cfg=dict(task=task,seed=seed,architecture=arch,arm=arm,joint=joint,device=device,initial_state_sha256=initial,
        epochs=epochs,batch=512,optimizer='NAdam lr=.002 StepLR(10,.5)' if native else ('native FT optimizer' if arch=='ft' else 'AdamW lr=.001 wd=.0001'),
        selector='native last epoch; training-loss convergence tolerance1e-4' if native else 'validation AUROC then BCE; FT patience16',
        eta=.1 if arm in ('hifi','hifi_shared') else 0.,hifi_source_sha256=sha(native_file),adapter_sha256=CODE_SHA256,
        data_hashes={k:r.array_hash(d[k]) for k in ['x','y','s']},split_hashes={k:r.array_hash(v) for k,v in d['splits'].items()},
        bank_hashes={n:r.array_hash(b['corners']) for n,b in banks.items()},training_only_preprocessing=True,
        selection_uses_audit=False,native_hifi_schedule=native,typed_hifi_adapter='binary sensitive view; ordinary fields kept in shared encoding',
        concurrent_runtime=True)
    cfg['joint_missing_sex_policy']='retain every row in classification; four-group penalty/reweighting only on recorded Male/Female rows' if joint else None
    cfg['batch_adapter']='shared paper batch512; HIFI publication uses dataset-specific batches for different benchmark datasets'
    write(out/'CONFIG.json',cfg);write(out/'TRANSFORM.json',extra)
    rng=np.random.default_rng(seed+201);bh=hashlib.sha256();history=[];best=None;last_loss=float('inf');start=time.monotonic()
    for ep in range(1,epochs+1):
        m.train();perm=rng.permutation(len(x));bh.update(perm.tobytes());losses=[];t=time.monotonic()
        for off in range(0,len(x),512):
            ii=perm[off:off+512];xx=x[ii];yy=y[ii];opt.zero_grad(set_to_none=True)
            if hifi:
                if joint:
                    loss=F.binary_cross_entropy(base(xx).sigmoid(),yy)
                    valid=sensitive_valid[ii]
                    if valid.any() and cfg['eta']:
                        inp=view.encode(xx[valid]);pr=view(inp)
                        loss=loss+native_hifi('dl',view,inp,pr,yy[valid,None],[0,1],cfg['eta'])-F.binary_cross_entropy(pr,yy[valid,None])
                else:
                    pr=view(xx);loss=native_hifi('dl',view,xx,pr,yy[:,None],[0],cfg['eta'])
            else:loss=(F.binary_cross_entropy_with_logits(m(xx),yy,reduction='none')*w[ii]).mean()
            assert torch.isfinite(loss),'nonfinite training loss'
            loss.backward();opt.step();losses.append(float(loss.detach()))
        if scheduler:scheduler.step()
        v=behavior(d['y'][vi],d['s'][vi],expit(logits(m,d['x'][vi],device)));key=(-v['auc'],v['bce'])
        item=dict(epoch=ep,validation=v,mean_loss=float(np.mean(losses)),seconds=time.monotonic()-t)
        history.append(item)
        if native or best is None or key<tuple(best['key']):
            best=dict(epoch=ep,key=key,validation=v)
            torch.save(dict(state_dict={k:q.detach().cpu().clone() for k,q in base.state_dict().items()},config=cfg,transform=transform,epoch=ep),out/'selected.pt')
        write(out/'HISTORY.json',history)
        print(task,arch,arm,seed,ep,round(v['auc'],5),round(item['seconds'],2),flush=True)
        if native and abs(last_loss-item['mean_loss'])<1e-4:break
        last_loss=item['mean_loss']
        if not native and arch=='ft' and ep-best['epoch']>=16:break
    fitted=load(out/'selected.pt',d,arch,device)
    audit=evaluate(fitted,d,device,out,joint)
    if arm=='erm' and arch=='mlp' and not (out/'audit_task.npz').exists():
        (out/'audit_task.npz').symlink_to('audit.npz')
    result=dict(status='complete',config=cfg,selected=best,validation={'behavior':best['validation']},
        selections={'task':{'validation':{'behavior':best['validation']}}},audit=audit,checkpoint=str(out/'selected.pt'),checkpoint_sha256=sha(out/'selected.pt'),
        arrays_sha256=sha(out/'audit.npz'),batch_sha256=bh.hexdigest(),seconds=time.monotonic()-start,epochs_completed=ep)
    write(out/'DONE.json',result)

def soft(task,seed,arch,device='cuda'):
    # Reuse the exact current objective, selector and sampler with a new task provider.
    if arch=='mlp':device='cpu'  # measured 8x faster under concurrent FT GPU load
    out=directory(task,seed,arch,'soft')
    if (out/'DONE.json').exists():return
    source=source_path(task,seed,arch)
    assert source.exists(),source
    if arch=='mlp' and not (source.parent/'audit_task.npz').exists():
        (source.parent/'audit_task.npz').symlink_to('audit.npz')
    r.HERE=P;r.source_path=source_path;r.get_data=lambda task,seed,study,arm:data(task,seed)
    r.reference_pipeline=lambda task,seed,arch,d,device='cpu':Pipeline(load(source_path(task,seed,arch),d,arch,device),[source_path(task,seed,arch)],device)
    r.make_model=model
    actual=out.parent/'interfair';actual.mkdir(parents=True,exist_ok=True)
    (actual/'COMPLETION_ADAPTER.py').write_bytes(CODE_BYTES)
    # Same canonical trainer; its generated arm directory is atomically named soft below.
    rr=r.train(task,seed,arch,'main','interfair',device,1.)
    actual=out.parent/'interfair'
    if actual!=out:
        # Keep paths in JSON/checkpoint consistent by retaining the canonical directory;
        # expose the paper arm as a relative link for the collection/export layer.
        out.symlink_to('interfair',target_is_directory=True)
    return rr

if __name__=='__main__':
    import argparse,fcntl
    a=argparse.ArgumentParser();a.add_argument('task');a.add_argument('seed',type=int);a.add_argument('arch');a.add_argument('arm');a.add_argument('--device',default='cuda');a.add_argument('--joint',action='store_true');args=a.parse_args()
    lock=P/'locks'/f'{args.task}_{args.seed}_{args.arch}_{args.arm}_{args.joint}.lock';lock.parent.mkdir(exist_ok=True)
    with lock.open('w') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        if args.arm=='soft':soft(args.task,args.seed,args.arch,args.device)
        else:standard(args.task,args.seed,args.arch,args.arm,args.device,args.joint)
