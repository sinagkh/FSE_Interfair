"""Common mirror MLP training; freeze validation scenario before separate native gate/audit."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[3]
import argparse,hashlib,json,time
from types import SimpleNamespace
import numpy as np
import torch
from core import ROOT,Predictor,behavior,predict,save_json,seed_all,sha,state_hash
from maintenance import data_for,erm_path
from mirrorfair_source import literal

BASE=_PACKAGE / 'core/runs/mirrorfair_common'

def train(task,seed):
    torch.set_num_threads(1);assert seed in (1000,1001,1002,1003,1004,1005,1006,1007,1008,1009,2000,2001);assert not torch.cuda.is_available();d=data_for(task,seed);dest=BASE/task/str(seed);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'TRAIN_DONE.json').exists():return
    seed_all(seed);m=Predictor(d['x'].shape[1],'mlp','erm');initial=state_hash(m);ref=json.loads((erm_path(task,seed).parent/'DONE.json').read_text());assert initial==ref['config']['initial_state_sha256']
    ids=d['splits']['train'];x=d['x'][ids].copy();x[:,0]=1-x[:,0];assert np.array_equal(x[:,1:],d['x'][ids,1:]);yy=d['y'][ids].copy();x=torch.tensor(x);y=torch.tensor(yy);vi=d['splits']['val'];vx=d['x'][vi].copy();vx[:,0]=1-vx[:,0]
    rng=np.random.default_rng(seed+201);bh=hashlib.sha256();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);history=[];best=None;bestkey=None;start=time.monotonic()
    config=dict(task=task,seed=seed,epochs=40,batch=512,lr=.001,weight_decay=.0001,selector='AUROC then BCE on mirrored validation',initial_state_sha256=initial,source_erm_sha256=sha(erm_path(task,seed)),code_sha256=sha(__file__),protocol_sha256=None,protected_only_mutation=True,labels_unchanged=True,device='cpu',source_erm_device='cpu',audit_used_for_selection=False)
    save_json(dest/'CONFIG.json',config)
    for ep in range(1,41):
        m.train();perm=rng.permutation(len(x));bh.update(perm.tobytes())
        for i in range(0,len(x),512):
            j=perm[i:i+512];loss=torch.nn.functional.binary_cross_entropy_with_logits(m(x[j]),y[j]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        p=predict(m,vx,'cpu');b=behavior(d['y'][vi],d['s'][vi],p);key=(-b['auc'],b['bce']);history.append(dict(epoch=ep,validation=b))
        if best is None or key<bestkey:best=dict(epoch=ep,state_dict={k:v.detach().clone() for k,v in m.state_dict().items()},validation=b);bestkey=key
        save_json(dest/'progress.json',dict(epoch=ep,seconds=time.monotonic()-start))
    assert bh.hexdigest()==ref['batch_sha256'];m.load_state_dict(best['state_dict']);torch.save(dict(**best,config=config),dest/'selected.pt')
    orig=Predictor(d['x'].shape[1],'mlp','erm');orig.load_state_dict(torch.load(erm_path(task,seed),map_location='cpu',weights_only=False)['state_dict']);p=predict(orig,d['x'][vi],'cpu');q=predict(m,d['x'][vi],'cpu');g=native_groups(d,d['s'][vi]);select,combine=literal();scenario=int(select(np.c_[1-p,p],np.c_[1-q,q],SimpleNamespace(protected_attributes=g[:,None])));score,dec=combine(np.c_[1-p,p],np.c_[1-q,q],SimpleNamespace(protected_attributes=g[:,None]),scenario,False)
    np.savez_compressed(dest/'validation_rules.npz',p=p,q=q,g=g,scenario=np.array(scenario),combined_score=score,decisions=dec,indices=vi)
    save_json(dest/'PAIR_SELECTION.json',dict(scenario=scenario,mirror_epoch=best['epoch'],selection_finished_before_audit=True,selected_on='natural validation scores after mirror checkpoint fixed',native_group_code='training-favored-common-group maps to 1',bank_flag=False))
    save_json(dest/'TRAIN_DONE.json',dict(status='trained; native scenario gate and heldout audit pending',config=config,history=history,selected_epoch=best['epoch'],batch_sha256=bh.hexdigest(),matched_ERM_initial_weights=True,matched_ERM_minibatches=True,seconds=time.monotonic()-start));print(task,seed,'mirror trained; scenario',scenario,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task');p.add_argument('seed',type=int);a=p.parse_args();train(a.task,a.seed)
