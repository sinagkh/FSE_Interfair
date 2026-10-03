"""E19b: explicit race/sex cube compiler and matched CPU repair."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[1]
import argparse,hashlib,itertools,json,os,pickle,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.nn import functional as F
from core import ROOT,Predictor,Support,behavior,guard_loss,predict,save_json,seed_all,sha,state_hash
from training_base import prepare
from feature_edits import banks
from support import ordered_training_indices

CACHE=_PACKAGE / 'core/cache_v2/hmda_oh_protected_vector.pkl'
BASE=_PACKAGE / 'core/runs/protected_vector'
PROTOCOL=_PACKAGE / 'core/reports/VECTOR_PROTOCOL.md'
ARMS=('control','race','vector','removed_vector')
PAIRS=list(itertools.combinations(range(4),2))

def sex_columns(data):
    enc=data['encoder'];cols=[i for i,c in enumerate(enc.columns) if c.startswith('derived_sex=')]
    ids=[enc.columns.index('derived_sex='+s) for s in ('Male','Female')]
    return cols,ids

def compile_banks():
    if CACHE.exists():
        with CACHE.open('rb') as f:return pickle.load(f)
    data=prepare('hmda_oh');iv=banks('hmda_oh');sexcols,sexids=sex_columns(data)
    train=ordered_training_indices(data);sex=data['frame'].derived_sex.to_numpy();support=[]
    for label in ('Male','Female'):
        selected=train[sex[train]==label];support.append(Support(data['x'][selected]))
    result={};counts={};checks=[]
    for split in ('train','val','audit'):
        result[split]={};counts[split]={}
        for feature,b in iv['banks'][split].items():
            n=len(b['indices']);v=b['corners'].reshape(2,2,n,-1);cube=np.empty((2,2,2,n,v.shape[-1]),dtype=np.float32)
            for race in (0,1):
                for sx in (0,1):
                    c=v[race].copy();c[:,:,sexcols]=0;c[:,:,sexids[sx]]=1;cube[race,sx]=c
            valid=b['valid']&np.isin(sex[b['indices']],['Male','Female']);keep=valid.copy()
            for sx in (0,1):
                ok,_=support[sx].mask(cube[:,sx].reshape(4,n,-1));keep &= ok
            # Verify full raw encoding for representative rows and every group/endpoint.
            spec=next(s for s in iv['specs'] if s['name']==feature)
            chosen=np.arange(min(n,7));raw=data['frame'].iloc[b['indices'][chosen]].copy()
            for race,sx,end in itertools.product((0,1),repeat=3):
                r=raw.copy();r['protected']=race;r['derived_sex']=('Male','Female')[sx]
                vals=spec['low'] if end==0 else spec['high']
                for j,col in enumerate(spec['columns']):r[col]=vals[j] if len(spec['columns'])>1 else vals
                err=float(abs(data['encoder'].transform(r)-cube[race,sx,end,chosen]).max());assert err==0,(split,feature,err)
            checks.append(dict(split=split,feature=feature,passed=True,check='all eight raw/encoded corners agree'))
            minimum=64 if split=='train' else 32
            counts[split][feature]=dict(anchors=n,binary_sex=int(np.isin(sex[b['indices']],['Male','Female']).sum()),valid=int(valid.sum()),supported=int(keep.sum()),minimum=minimum)
            assert keep.sum()>=minimum,(split,feature,counts[split][feature])
            result[split][feature]=dict(corners=cube.reshape(8,n,-1),valid=valid,supported=keep,indices=b['indices'])
    names=sorted(result['train']);chunks=[];weights=[]
    for name in names:
        b=result['train'][name];v=b['corners'][:,b['supported']];chunks.append(v);weights.append(np.full(v.shape[1],1/(len(names)*v.shape[1])))
    mixture=dict(corners=np.concatenate(chunks,axis=1),weights=np.concatenate(weights));assert abs(mixture['weights'].sum()-1)<1e-12
    d=dict(banks=result,mixture=mixture,features=names,sex_columns=sexcols,sex_ids=sexids)
    with CACHE.open('wb') as f:pickle.dump(d,f,pickle.HIGHEST_PROTOCOL)
    natural={}
    for split,ix in data['splits'].items():
        natural[split]={f'{race}/{label}':int(((data['s'][ix]==race)&(sex[ix]==label)).sum()) for race in (0,1) for label in ('Male','Female')}
        natural[split]['other_sex']=int((~np.isin(sex[ix],['Male','Female'])).sum())
    save_json(ROOT/'reports/VECTOR_BANKS.json',dict(status='PASS',counts=counts,raw_compiler_checks=checks,natural_counts=natural,support_counts={label:s.counts for label,s in zip(('Male','Female'),support)},support_thresholds={label:s.thresholds for label,s in zip(('Male','Female'),support)},cache_sha256=sha(CACHE),protocol_sha256=None))
    return d

class VectorModel(Predictor):
    def __init__(self,dim,mask_columns=()):
        super().__init__(dim,'mlp','erm');self.mask_columns=list(mask_columns)
    def base(self,x):
        if self.mask_columns:x=x.clone();x[:,self.mask_columns]=0
        return self.net(x).squeeze(-1)

def load(path,data,mask_columns=()):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    m=VectorModel(data['x'].shape[1],mask_columns);m.load_state_dict(ck['state_dict']);return m.eval()

def metric(v,t=None):
    v=v.reshape(4,2,-1);e=v[:,1]-v[:,0];rs=np.stack([e[b]-e[a] for a,b in PAIRS]);race=np.stack([e[2]-e[0],e[3]-e[1]]);sex=np.stack([e[1]-e[0],e[3]-e[2]])
    dec=(v>=.5).astype(float);de=dec[:,1]-dec[:,0];common=e.mean(0)
    out=dict(n=e.shape[1],all_pair_mean_abs=float(abs(rs).mean()),pair_violation_rate=float((abs(rs)>.01).mean()),any_pair_violation_rate=float((abs(rs).max(0)>.01).mean()),race_mean_abs=float(abs(race).mean()),sex_mean_abs=float(abs(sex).mean()),effect_range=float(np.ptp(e,axis=0).mean()),third_mean_abs=float(abs(e[3]-e[2]-e[1]+e[0]).mean()),decision_transition_disagreement=float((np.ptp(de,axis=0)>0).mean()),mean_common_abs=float(abs(common).mean()),pure_protected_offset_low=float(abs(v[3,0]-v[2,0]-v[1,0]+v[0,0]).mean()))
    if t is not None:
        te=t.reshape(4,2,-1);tc=(te[:,1]-te[:,0]).mean(0);active=abs(tc)>.001
        out.update(teacher_common_abs=float(abs(tc).mean()),common_mae=float(abs(common-tc).mean()),signed_retention=float((np.sign(tc[active])*common[active]).sum()/abs(tc[active]).sum()) if active.any() else None)
    return out

def evaluate(model,data,bank,split,teacher_values=None,mixture=False):
    def score(x):
        if not mixture:return predict(model,x,'cpu').astype(np.float64)
        cols,ids=sex_columns(data);out=[]
        for race,sx in itertools.product((0,1),repeat=2):
            xx=x.copy();xx[:,0]=race;xx[:,cols]=0;xx[:,ids[sx]]=1;out.append(predict(model,xx,'cpu').astype(np.float64))
        return np.mean(out,axis=0)
    ix=data['splits'][split];p=score(data['x'][ix]);sex=data['frame'].derived_sex.to_numpy()[ix];binary=np.isin(sex,['Male','Female'])
    b=behavior(data['y'][ix],data['s'][ix],p);bb=behavior(data['y'][ix][binary],data['s'][ix][binary],p[binary]);groups={}
    for race,sx in itertools.product((0,1),repeat=2):
        keep=(data['s'][ix]==race)&(sex==('Male','Female')[sx]);y=data['y'][ix][keep];pp=p[keep]
        groups[f'{race}/{sx}']=dict(n=int(keep.sum()),positive=int((y==1).sum()),negative=int((y==0).sum()),accuracy=float(((pp>=.5)==y).mean()),tpr=float((pp[y==1]>=.5).mean()),fpr=float((pp[y==0]>=.5).mean()))
    feature={};arrays={}
    for name,x in bank['banks'][split].items():
        cube=x['corners'][:,x['supported']];v=score(cube.reshape(-1,cube.shape[-1])).reshape(8,-1);arrays[name]=v;feature[name]=metric(v,teacher_values[name] if teacher_values else None)
    keys=[k for k,v in next(iter(feature.values())).items() if k!='n' and v is not None]
    macro={k:float(np.mean([v[k] for v in feature.values()])) for k in keys}
    return dict(behavior=b,binary_behavior=bb,binary_n=int(binary.sum()),groups=groups,features=feature,macro=macro),arrays

def worker(seed,arm):
    assert seed in (2000,2001) and arm in ARMS;torch.set_num_threads(1)
    assert not torch.cuda.is_available(),'CPU-only research queue'
    output=BASE/str(seed)/arm
    if (output/'DONE.json').exists():return
    output.mkdir(parents=True,exist_ok=True);data=prepare('hmda_oh');bank=compile_banks();src=ROOT/'runs/pilot_v3/hmda_oh/mlp'/str(seed)/'erm/task.pt'
    seed_all(seed);teacher=load(src,data);mask=[0]+bank['sex_columns'] if arm=='removed_vector' else []
    model=load(src,data,mask);ref,tval=evaluate(teacher,data,bank,'val');reference=ref['behavior']
    for p in teacher.parameters():p.requires_grad_(False)
    cubes=torch.tensor(bank['mixture']['corners']);w=bank['mixture']['weights'];ix=data['splits']['train'];x=torch.tensor(data['x'][ix]);y=torch.tensor(data['y'][ix])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(seed+2101);arng=np.random.default_rng(seed+2102);batch_hash=hashlib.sha256();anchor_hash=hashlib.sha256()
    config=dict(task='hmda_oh',seed=seed,arm=arm,mode='erm',architecture='mlp',mask_columns=mask,source=str(src),source_sha256=sha(src),initial_state_sha256=state_hash(model),bank_sha256=sha(CACHE),protocol_sha256=None,code_sha256=sha(__file__),device='cpu',epochs=20,lr=.001,batch=512,cubes_per_step=128,guard=.1,preservation_weight=160,interaction_weight=160 if arm in ('race','vector') else 0,selection='full-vector validation residual under original utility floor; epoch0 included',audit_used_for_selection=False)
    save_json(output/'CONFIG.json',config);history=[];selected=None
    def consider(epoch):
        nonlocal selected
        result,_=evaluate(model,data,bank,'val',tval);b=result['behavior'];ok=b['auc']>=reference['auc']-.005 and b['f1']>=.99*reference['f1'];key=(result['macro']['all_pair_mean_abs'],-b['auc'],b['bce'])
        history.append(dict(epoch=epoch,validation=result,feasible=ok))
        if ok and (selected is None or key<selected['key']):selected=dict(key=key,epoch=epoch,validation=result,state_dict={k:v.detach().clone() for k,v in model.state_dict().items()})
    start=time.time();consider(0)
    for epoch in range(1,21):
        model.train();perm=rng.permutation(len(x));batch_hash.update(perm.tobytes());losses=[]
        for i in range(0,len(x),512):
            ii=torch.as_tensor(perm[i:i+512]);logits=model(x[ii]);loss=F.binary_cross_entropy_with_logits(logits,y[ii])+.1*guard_loss(logits.sigmoid(),y[ii],x[ii,0])
            jj=arng.choice(cubes.shape[1],128,p=w);anchor_hash.update(jj.tobytes());probe=cubes[:,jj].reshape(-1,x.shape[1]);v=model.probability(probe).reshape(4,2,128);e=v[:,1]-v[:,0]
            with torch.no_grad():t=teacher.probability(probe).reshape(4,2,128);common=(t[:,1]-t[:,0]).mean(0)
            loss=loss+160*(e.mean(0)-common).square().mean()
            if arm=='race':penalty=torch.stack([e[2]-e[0],e[3]-e[1]]).square().mean()
            elif arm=='vector':penalty=torch.stack([e[b]-e[a] for a,b in PAIRS]).square().mean()
            else:penalty=e.sum()*0
            loss=loss+160*min(epoch/5,1)*penalty;optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        consider(epoch);history[-1]['mean_training_loss']=float(np.mean(losses));print(json.dumps(dict(seed=seed,arm=arm,epoch=epoch,val_R=history[-1]['validation']['macro']['all_pair_mean_abs'],feasible=history[-1]['feasible'])),flush=True)
    last=dict(epoch=20,validation=history[-1]['validation'],state_dict={k:v.detach().clone() for k,v in model.state_dict().items()});selections={'last':last}
    if selected is not None:selections['selected']=selected
    save_json(output/'SELECTION.json',dict(selection_finished_before_audit=True,no_feasible_candidate=selected is None,selections={k:{kk:vv for kk,vv in v.items() if kk not in ('key','state_dict')} for k,v in selections.items()}));save_json(output/'HISTORY.json',history)
    _,taudit=evaluate(teacher,data,bank,'audit');results={}
    for key,item in selections.items():
        torch.save(dict(state_dict=item['state_dict'],config=config,epoch=item['epoch']),output/(key+'.pt'));model.load_state_dict(item['state_dict']);audit,arrays=evaluate(model,data,bank,'audit',taudit);np.savez_compressed(output/(key+'_audit.npz'),**arrays)
        results[key]=dict(epoch=item['epoch'],validation=item['validation'],audit=audit,state_sha256=state_hash(model))
    save_json(output/'DONE.json',dict(config=config,selections=results,seconds=time.time()-start,batch_schedule_sha256=batch_hash.hexdigest(),anchor_schedule_sha256=anchor_hash.hexdigest(),concurrent_timing=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--job',nargs=2);p.add_argument('--queue',action='store_true');args=p.parse_args()
    if args.job:worker(int(args.job[0]),args.job[1]);return
    compile_banks()
    if not args.queue:return
    logs=ROOT/'job_logs/protected_vector';logs.mkdir(parents=True,exist_ok=True);env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    def launch(j):
        with (logs/f'{j[0]}_{j[1]}.log').open('w') as f:r=subprocess.run([sys.executable,__file__,'--job',*map(str,j)],stdout=f,stderr=subprocess.STDOUT,env=env)
        assert r.returncode==0,j
        print('completed',*j,flush=True)
    start=time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(launch,[(s,a) for s in (2000,2001) for a in ARMS]))
    save_json(ROOT/'reports/VECTOR_EXECUTION.json',dict(status='completed',trajectories=8,seconds=time.time()-start,device='cpu',concurrency=4,final_confirmation_started=False))

if __name__=='__main__':main()
