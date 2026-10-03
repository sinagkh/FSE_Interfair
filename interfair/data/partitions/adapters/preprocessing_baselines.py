"""Pinned LTDD / CoT-Phi transformations with explicit common-MLP adapters."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[3]
import ast,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import linregress
import torch
from torch import nn
from core import ROOT,Predictor,seed_all,state_hash,save_json,sha,predict,behavior,evaluate
from training_base import prepare

VENDOR=_PACKAGE / 'vendor'
def official_function(file,name):
    tree=ast.parse(Path(file).read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    env={'np':np,'pd':pd,'dataset_used':'adult'}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(file),'exec'),env)
    return env[name]

phi_native=official_function(VENDOR/'cot/CoT_Phi_Para.py','Parameter_phi0')
mutate_native=official_function(VENDOR/'cot/CoT_Phi_dl.py','get_decorelated_training_set')


def ltdd_fit(x,columns,exempt=()):
    x=np.asarray(x,dtype=np.float64);s=x[:,0];slope=np.zeros(x.shape[1]);intercept=np.zeros(x.shape[1]);records=[]
    for j,col in enumerate(columns):
        if j==0:continue
        excluded=any(col.startswith(c+'=') or col==c for c in exempt);constant=np.ptp(x[:,j])==0
        if constant:r=None;pv=None
        else:r=linregress(s,x[:,j]);pv=float(r.pvalue)
        selected=not excluded and r is not None and pv<.05
        if selected:slope[j]=r.slope;intercept[j]=r.intercept
        records.append(dict(column=col,p_value=pv,selected=selected,excluded_second_protected=excluded,constant=constant,slope=float(slope[j]),intercept=float(intercept[j])))
    return slope,intercept,records


def ltdd_numpy(x,slope,intercept):
    out=np.asarray(x,dtype=np.float64)-np.asarray(x)[:,0,None]*slope-intercept;out[:,0]=0
    return out


def cot_transform(x,y,seed):
    df=pd.DataFrame(np.asarray(x).copy(),columns=['protected']+[f'x{i}' for i in range(1,x.shape[1])]);df['protected']=1-df['protected'];df['label']=y
    table=pd.crosstab(df.protected,df.label).reindex(index=[0,1],columns=[0,1],fill_value=0);alpha=float(phi_native(table));assert 0<=alpha<=1,(table,alpha)
    np.random.seed(seed);transformed=mutate_native(df,0,alpha,'protected','label').sort_index();assert np.array_equal(transformed['label'],y)
    out=transformed.drop(columns='label').to_numpy(np.float32);out[:,0]=1-out[:,0]
    assert np.array_equal(out[:,1:],np.asarray(x,dtype=np.float32)[:,1:]) and len(out)==len(x)
    changed=out[:,0]!=x[:,0];assert np.all(np.asarray(y)[changed]==1) and np.all(x[changed,0]==0) and np.all(out[changed,0]==1)
    return out,dict(alpha=alpha,counts_privileged_coding=table.to_numpy().tolist(),changed=int(changed.sum()),training_rows=len(x),seed=seed,phi_before=float(np.corrcoef(x[:,0],y)[0,1]),phi_after=float(np.corrcoef(out[:,0],y)[0,1]),changed_index_sha256=hashlib.sha256(np.flatnonzero(changed).tobytes()).hexdigest())


class LTDDPipeline(nn.Module):
    def __init__(self,base,slope,intercept):
        super().__init__();self.base=base;self.register_buffer('slope',torch.tensor(slope,dtype=torch.float32));self.register_buffer('intercept',torch.tensor(intercept,dtype=torch.float32))
    def transform(self,x):
        out=x-x[:,0,None]*self.slope-self.intercept;return torch.cat([torch.zeros_like(out[:,:1]),out[:,1:]],dim=1)
    def forward(self,x):return self.base(self.transform(x))
    def probability(self,x):return self.forward(x).sigmoid()
    def score(self,x,scale='P'):return self.forward(x) if scale=='L' else self.probability(x)


def load_pipeline(directory,device='cuda'):
    ckpt=torch.load(Path(directory)/'selected.pt',map_location='cpu',weights_only=False);base=Predictor(ckpt['input_dim'],'mlp','erm');base.load_state_dict(ckpt['base_state'])
    model=LTDDPipeline(base,ckpt['slope'],ckpt['intercept']) if ckpt['method']=='ltdd' else base
    return model.to(device).eval()


def train_baseline(task,seed,method):
    torch.set_num_threads(1);data=partitions(task,seed);out=campaign_base/task/str(seed)/method
    if (out/'DONE.json').exists():return
    out.mkdir(parents=True,exist_ok=True);seed_all(seed);base=Predictor(data['x'].shape[1],'mlp','erm').cuda();initial=state_hash(base);ids=data['splits']['train'];raw=data['x'][ids].copy();y=data['y'][ids];extra={}
    if method=='ltdd':
        sl,it,records=ltdd_fit(raw,data['encoder'].columns,('race',) if task=='adult' else ('derived_sex',));model=LTDDPipeline(base,sl,it).cuda();extra=dict(selected_coordinates=sum(r['selected'] for r in records),coefficients=records)
        test=data['x'][data['splits']['val'][:256]];explicit=ltdd_numpy(test,sl,it).astype(np.float32);wrapped=model.transform(torch.tensor(test,device='cuda')).detach().cpu().numpy();error=float(abs(wrapped-explicit).max());assert error<3e-6,error;extra['transform_float32_max_error']=error
    elif method=='cot_phi':
        raw,extra=cot_transform(raw,y,seed+401);model=base;sl=it=None
    else:raise ValueError(method)
    save_json(out/'TRANSFORM.json',extra)
    config=dict(method=method,task=task,seed=seed,track='common MLP adapter; native preprocessing algorithm',epochs=40,batch=512,optimizer='AdamW',lr=.001,weight_decay=.0001,selector='validation AUROC then BCE; no specification access',initial_base_state_sha256=initial,train_indices_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),train_x_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),source_code_sha256=sha(__file__),source_manifests={m:json.loads((VENDOR/m/'MANIFEST.json').read_text()) for m in ('ltdd','cot')},audit_used_for_selection=False)
    save_json(out/'CONFIG.json',config);x=torch.tensor(raw,device='cuda');yy=torch.tensor(y,device='cuda');opt=torch.optim.AdamW(base.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(seed+201);batchhash=hashlib.sha256();history=[];best=None;bestkey=None;start=time.time()
    for epoch in range(1,41):
        model.train();perm=rng.permutation(len(x));batchhash.update(perm.tobytes())
        for i in range(0,len(x),512):
            j=torch.tensor(perm[i:i+512],device='cuda');loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x[j]),yy[j]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        vi=data['splits']['val'];v=behavior(data['y'][vi],data['s'][vi],predict(model,data['x'][vi],'cuda'));history.append(dict(epoch=epoch,behavior=v));key=(-v['auc'],v['bce'])
        if bestkey is None or key<bestkey:bestkey=key;best=dict(epoch=epoch,behavior=v,state={k:v.detach().cpu().clone() for k,v in base.state_dict().items()})
    save_json(out/'HISTORY.json',history);save_json(out/'SELECTION.json',dict(epoch=best['epoch'],validation=best['behavior'],audit_used=False));base.load_state_dict(best['state']);torch.save(dict(method=method,input_dim=data['x'].shape[1],base_state=best['state'],slope=sl,intercept=it,config=config),out/'selected.pt')
    audit=evaluate(model,data,'audit','cuda',save_path=out/'audit.npz');ref=json.loads((campaign_erm(task,seed).parent/'DONE.json').read_text())
    checks=dict(initial_weights_match_erm=initial==ref['config']['initial_state_sha256'],minibatches_match_erm=batchhash.hexdigest()==ref['batch_sha256']);assert all(checks.values()),checks
    save_json(out/'DONE.json',dict(status='completed',config=config,epoch=best['epoch'],validation=best['behavior'],audit=audit,checks=checks,batch_sha256=batchhash.hexdigest(),seconds=time.time()-start));print(json.dumps(dict(completed=[task,seed,method],audit_behavior=audit['behavior'],transform_summary={k:v for k,v in extra.items() if k!='coefficients'})),flush=True)
