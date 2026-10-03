"""Published preprocessing algorithms with explicit typed-data/common-MLP adapter."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import argparse
import hashlib
import importlib.util
import random
import time
import warnings
import pandas as pd
from sklearn.linear_model import LogisticRegression
from ablation import HERE,p,r,u,torch,np,read,write
from core import numeric,categorical
from pipeline_audit import audit
from torch.nn import functional as F


def fair_smote(d,seed,out):
    path=_PACKAGE / 'vendor/fair_smote/Generate_Samples.py'
    spec=importlib.util.spec_from_file_location('native_fair_smote',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ids=d['splits']['train'];raw=d['frame'].iloc[ids].copy();generation=pd.DataFrame(index=np.arange(len(ids)))
    generation['protected']=d['s'][ids].astype(int).astype(str)
    numeric_info={};cat_info={}
    for col in d['encoder'].continuous:
        vals=numeric(raw[col]).reset_index(drop=True);median=float(vals.median());filled=vals.fillna(median)
        low,high=float(filled.min()),float(filled.max());span=high-low if high>low else 1.
        generation[col]=(filled-low)/span
        generation[col+'__missing']=vals.isna().astype(int).astype(str)
        numeric_info[col]=(low,span)
    for col in d['encoder'].categories:
        vals=categorical(raw[col]).reset_index(drop=True);vocab=sorted(vals.unique());lookup={value:i for i,value in enumerate(vocab)}
        # String-coded observed categories keep native categorical sampling intact.
        generation[col]=(vals.map(lookup)/max(1,len(vocab)-1)).astype(str);cat_info[col]=vocab
    generation['label']=d['y'][ids].astype(int)
    random.seed(seed+401);np.random.seed(seed+401)
    counts=generation.groupby(['protected','label']).size();maximum=int(counts.max());pieces=[]
    start=time.monotonic()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='X does not have valid feature names.*')
        for s in (0,1):
            for y in (0,1):
                cell=generation[(generation.protected==str(s))&(generation.label==y)].copy()
                assert len(cell)>=5
                output=module.generate_samples(maximum-len(cell),cell,'typed_adapter')
                output.columns=generation.columns;pieces.append(output)
    balanced=pd.concat(pieces,ignore_index=True)
    assert (balanced.groupby(['protected','label']).size()==maximum).all()
    gx=balanced.drop(columns='label').astype(float).to_numpy();gy=balanced.label.to_numpy(np.float32)
    situation=LogisticRegression(C=1.,penalty='l2',solver='liblinear',max_iter=100,random_state=seed)
    situation.fit(gx,gy);flipped=gx.copy();flipped[:,0]=1-flipped[:,0]
    drop=situation.predict(gx)!=situation.predict(flipped);kept=balanced.loc[~drop].reset_index(drop=True)
    decoded=pd.DataFrame(index=np.arange(len(kept)));decoded['protected']=kept.protected.astype(float)
    for col,(low,span) in numeric_info.items():
        decoded[col]=kept[col].astype(float)*span+low
        decoded.loc[kept[col+'__missing'].astype(float)==1,col]=np.nan
    for col,vocab in cat_info.items():
        decoded[col]=[vocab[int(round(float(value)*max(1,len(vocab)-1)))] for value in kept[col]]
    x=d['encoder'].transform(decoded);y=kept.label.to_numpy(np.float32)
    # Verify native type decoding exactly preserves every original numeric/categorical value after encoding.
    original=pd.DataFrame(index=np.arange(len(ids)));original['protected']=d['s'][ids]
    for col in d['encoder'].continuous:original[col]=raw[col].to_numpy()
    for col in d['encoder'].categories:original[col]=raw[col].to_numpy()
    np.testing.assert_array_equal(d['encoder'].transform(original),d['x'][ids])
    assert np.isin(x[:,0],[0,1]).all() and np.isin(y,[0,1]).all()
    record=dict(before={str(k):int(v) for k,v in counts.items()},balanced_cell_size=maximum,balanced_rows=len(balanced),
                removed_by_situation=int(drop.sum()),retained=len(y),seconds=time.monotonic()-start,
                original_encoding_check=True,native_generator_sha256=p.sha(path),native_parameters=dict(f=.8,cr=.8,neighbors=3),
                categorical_encoding='native string branch; logical category codes',fit_partition='train only')
    write(out/'PREPROCESSING.json',record)
    return x,y,np.ones(len(y),np.float32)


def run(task,seed,method):
    torch.set_num_threads(1);assert seed in (2000,2001)
    out=HERE/'resampling'/f'runs/{task}/{seed}/{method}'
    if (out/'DONE.json').exists():return
    d,_=p.data_and_banks(task,seed);ids=d['splits']['train'];out.mkdir(parents=True,exist_ok=True)
    if method=='fairsmote':x,y,w=fair_smote(d,seed,out)
    else:
        assert method=='reweighing';x=d['x'][ids];y=d['y'][ids];s=d['s'][ids];w=np.ones(len(y),np.float32)
        cells=[]
        for a in (0,1):
            for b in (0,1):
                mask=(s==a)&(y==b);weight=(s==a).mean()*(y==b).mean()/mask.mean();w[mask]=weight
                cells.append(dict(group=a,label=b,weight=float(weight),weighted_count=float(w[mask].sum()),expected=float((s==a).sum()*(y==b).mean())))
        assert all(abs(v['weighted_count']-v['expected'])<.01 for v in cells)
        assert abs(float(w.sum())-len(y))<.01
        write(out/'PREPROCESSING.json',dict(cells=cells,normalization='mean weight one',fit_partition='train only'))
    np.savez_compressed(out/'TRAINING_DATA.npz',x=x,y=y,weights=w)
    p.seed_all(seed);m=u.model(d,'mlp','soft').cuda();initial=p.state_hash(m)
    native=read(p.source_path(task,seed,'mlp').parent/'DONE.json')
    assert initial==native['config']['initial_state_sha256']
    config=dict(task=task,seed=seed,method=method,architecture='mlp',mode='erm',device='cuda',epochs=40,batch=512,lr=.001,weight_decay=.0001,
                source_rows=len(ids),training_rows=len(y),initial_state_sha256=initial,train_x_sha256=r.array_hash(x),
                protocol_sha256=None,code_sha256=p.sha(__file__),audit_used_for_selection=False,
                comparison='common MLP, native task optimizer/selector; typed training-only preprocessing; GPU execution')
    write(out/'CONFIG.json',config)
    xx=torch.as_tensor(x,device='cuda');yy=torch.as_tensor(y,device='cuda');ww=torch.as_tensor(w,device='cuda')
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(seed+201)
    history=[];best=None;bestkey=None;bh=hashlib.sha256();start=time.monotonic()
    for epoch in range(1,41):
        perm=rng.permutation(len(y));bh.update(perm.tobytes());m.train()
        for start_idx in range(0,len(y),512):
            j=perm[start_idx:start_idx+512];opt.zero_grad(set_to_none=True)
            loss=(F.binary_cross_entropy_with_logits(m(xx[j]),yy[j],reduction='none')*ww[j]).mean()
            loss.backward();opt.step()
        vi=d['splits']['val'];pred=r.predict_logits(m,d['x'][vi],'cuda');v=r.behavior(d['y'][vi],d['s'][vi],1/(1+np.exp(-pred)))
        history.append(dict(epoch=epoch,validation=v));key=(-v['auc'],v['bce'])
        if bestkey is None or key<bestkey:
            bestkey=key;best=history[-1];torch.save(dict(state_dict=m.state_dict(),config=config,epoch=epoch),out/'selected.pt')
        write(out/'HISTORY.json',history)
    write(out/'SELECTION.json',dict(selected=best,selection_finished_before_audit=True))
    m.load_state_dict(torch.load(out/'selected.pt',map_location='cuda',weights_only=False)['state_dict'])
    pipeline=p.Pipeline(m,[out/'selected.pt'],'cuda');report,arrays=audit(pipeline,d)
    np.savez_compressed(out/'audit.npz',**arrays)
    write(out/'DONE.json',dict(status='completed',config=config,selection=best,audit=report,checkpoint=str(out/'selected.pt'),
                checkpoint_sha256=p.sha(out/'selected.pt'),seconds=time.monotonic()-start,batch_sha256=bh.hexdigest()))
    print('PUBLISHED COMPLETE',task,seed,method,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('task');ap.add_argument('seed',type=int);ap.add_argument('method',choices=['fairsmote','reweighing']);a=ap.parse_args()
    run(a.task,a.seed,a.method)
