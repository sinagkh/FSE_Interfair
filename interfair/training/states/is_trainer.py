"""State extension of the frozen fresh IS trainer; see canonical source hash in CONFIG."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import os
_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK']='1'
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import argparse, copy, hashlib, json, math, sys, time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

P = _PACKAGE / 'training/states'
ROOT = _PACKAGE
sys.path.insert(0, str(ROOT / 'training/direction'))
import direction as q
if _visible is None: os.environ.pop('CUDA_VISIBLE_DEVICES',None)
else: os.environ['CUDA_VISIBLE_DEVICES']=_visible
sys.path.insert(0,str(P))
from data import data as state_data, SIGNS, STATES
q.data=state_data
for state in STATES+('oh',): q.MAP['hmda_'+state]=SIGNS.copy()
sys.path.insert(0,str(ROOT/'training/transformer'))
import ft_trainer as ee
_DEVICE='cpu'
def evaluate_on_device(m,d,names,db,split,save=None):
    return ee.evaluate(m,d,names,db,split,device=_DEVICE,save=save)[:2]
q.evaluate=evaluate_on_device
c = q.c
rb = q.rb
torch.set_num_threads(1)

def directory(task, seed, arm):
    return P / 'runs' / c.phase(seed) / task / str(seed) / arm

def select_key(v, ref):
    fair = ['aod', 'dp', 'eomax', 'tpr_gap', 'fpr_gap', 'decision_disagreement']
    excess = [max(0., v[k] - ref[k]) / (ref[k] + .01) for k in fair]
    excess += [max(0., ref[k] - .02 - v[k]) / .02 for k in ['auc', 'accuracy', 'f1']]
    excess += [max(0., v['L_R'] - ref['L_R']) / max(ref['L_R'], .01)]
    admitted = max(excess) <= 1e-12
    score = v['violation_005'] + v['direction_adverse_005'] + v['decision_disagreement']
    score += .1 * (v['aod'] + v['dp'] + v['eomax'])
    return (int(not admitted), sum(excess), score, v['bce']), admitted

def train(task, seed, arm, weight=1., smoke_epochs=None, device='cuda'):
    global _DEVICE
    _DEVICE=device
    out = directory(task, seed, arm)
    if smoke_epochs is not None:
        out = P / 'smoke' / task / str(seed) / arm
    if (out / 'DONE.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    d, banks, db = q.data(task, seed)
    names = list(banks)
    ref = None
    if arm != 'erm':
        refpath = directory(task, seed, 'erm') / 'DONE.json'
        if smoke_epochs is not None:
            refpath = P / 'smoke' / task / str(seed) / 'erm/DONE.json'
        refrecord = json.loads(refpath.read_text())
        ref = refrecord['selected']['validation']
    c.seed_all(seed)
    model = c.model(d, 'mlp').to(device)
    initial = c.state_hash(model)
    if ref is not None:
        assert initial == refrecord['config']['initial_state_sha256']
    cfg = c.r.recipe('hmda_oh', 'mlp', 1.)
    epochs = 40 if smoke_epochs is None else smoke_epochs
    config = dict(task=task, seed=seed, architecture='mlp', arm=arm,
                  initialization='fresh seeded random model', initial_state_sha256=initial,
                  initial_checkpoint=None, pretrained_weights_loaded=False,
                  teacher_response_penalty=0., fidelity_penalty=0.,
                  epochs=epochs, task_batch=512, population_batch=512, ordinary_anchors=256,
                  signed_anchors=256, optimizer='AdamW', lr=.001, weight_decay=.0001,
                  early_stopping=False, validation_every=5, direction_weight=weight if arm=='equality_direction' else 0.,
                  decision_rate_weight=16., decision_response_weight=4., fairness_warmup_epochs=5,
                  soft_recipe=cfg, specification_sha256=c.sha(P/'cache'/f'{task}.json'),
                  trainer_sha256=c.sha(__file__), protocol_sha256=None,
                  data_hashes={k:c.r.array_hash(d[k]) for k in ['x','y','s']},
                  split_hashes={k:c.r.array_hash(v) for k,v in d['splits'].items()},
                  selection_uses_audit=False, device=device,
                  canonical_trainer_sha256=c.sha(ROOT/'training/mlp/mlp_trainer.py'),
                  data_adapter_sha256=c.sha(P/'data.py'))
    c.write(out/'CONFIG.json', config)
    (out/'TRAINER.py').write_bytes(Path(__file__).read_bytes())
    x=torch.as_tensor(d['x'],device=device); y=torch.as_tensor(d['y'],device=device); s=torch.as_tensor(d['s'],device=device)
    tr=d['splits']['train']
    def pools(bs):
        result={}
        for n,b in bs.items():
            keep=b['supported']; ids=b['indices'][keep]
            assert np.isin(ids,tr).all()
            result[n]={'ids':ids,'c':torch.as_tensor(b['corners'][:,keep],device=device)}
        return result
    ordinary=pools(banks); signed=pools({n:db['train'][n] for n in q.MAP[task]})
    rng=np.random.default_rng(seed+201); arng=np.random.default_rng(seed+929301)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    bh=hashlib.sha256(); ah=hashlib.sha256(); steps=0; queries=0
    best=None; history=[]; start=time.monotonic()
    for epoch in range(1,epochs+1):
        permutation=rng.permutation(tr); warm=min(epoch/5,1.)
        for off in range(0,len(permutation),512):
            ids=permutation[off:off+512]; bh.update(ids.tobytes()); steps+=1
            model.train(); opt.zero_grad(set_to_none=True)
            loss=F.binary_cross_entropy_with_logits(model(x[ids]),y[ids]); loss.backward()
            if arm!='erm':
                model.eval()
                ii=tr[c.r.balanced_positions(tr,d,arng,512)]; ah.update(ii.tobytes())
                z=model(x[ii]); queries+=len(ii)
                loss=c.r.score_loss(z,s[ii],y[ii],cfg)
                loss+=rb.rate_loss(z,y[ii],{'protected':s[ii]},16.,(.25,.5,1.))
                (warm*loss).backward()
                name=names[int(arng.integers(len(names)))]; pool=ordinary[name]
                jj=c.r.balanced_positions(pool['ids'],d,arng,256); ii=pool['ids'][jj]
                ah.update(name.encode()); ah.update(ii.tobytes()); cc=pool['c'][:,jj]
                z=model(cc.reshape(-1,cc.shape[-1])).reshape(4,-1); queries+=cc.shape[0]*cc.shape[1]
                with torch.no_grad(): nz=model(x[ii]); queries+=len(ii)
                loss=c.r.pair_effect_loss(z,nz,s[ii],y[ii],cfg)
                loss+=4.*rb.transition_loss(z,False,(.15,.35,.7))
                (warm*loss).backward()
                name=list(signed)[int(arng.integers(len(signed)))]; pool=signed[name]
                jj=c.r.balanced_positions(pool['ids'],d,arng,256); ii=pool['ids'][jj]
                ah.update(name.encode()); ah.update(ii.tobytes()); cc=pool['c'][:,jj]
                z=model(cc.reshape(-1,cc.shape[-1])).reshape(4,-1); queries+=cc.shape[0]*cc.shape[1]
                e=torch.stack([z[1]-z[0],z[3]-z[2]]); residual=e[1]-e[0]
                loss=cfg['pair']*(residual.square().mean()+.25*c.r.relations.topk_square(residual,.1))
                adverse=F.relu(-q.MAP[task][name]*e)
                loss+=config['direction_weight']*(adverse.square().mean()+.25*c.r.relations.topk_square(adverse.flatten(),.1))
                (warm*loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step()
        if epoch%5==0 or epoch==epochs:
            v,_=q.evaluate(model,d,names,db,'val')
            key,admitted=((-v['auc'],v['bce']),True) if arm=='erm' else select_key(v,ref)
            item=dict(epoch=epoch,steps=steps,validation=v,key=key,admitted=admitted,
                      seconds=time.monotonic()-start)
            history.append(item)
            if best is None or key<tuple(best['key']):
                best=copy.deepcopy(item)
                torch.save(dict(state_dict={k:v.detach().clone() for k,v in model.state_dict().items()},config=config,epoch=epoch),out/'selected.pt')
            c.write(out/'HISTORY.json',history)
            print(task,seed,arm,epoch,'W',round(v['direction_adverse_005'],4),'R',round(v['L_R'],4),
                  'D',round(v['decision_disagreement'],4),'AUC',round(v['auc'],4),'admitted',admitted,flush=True)
    assert steps==epochs*math.ceil(len(tr)/512)
    c.write(out/'DONE.json',dict(status='complete',config=config,selected=best,
       checkpoint=str(out/'selected.pt'),checkpoint_sha256=c.sha(out/'selected.pt'),
       initial_state_sha256=initial,task_batch_sha256=bh.hexdigest(),anchor_sha256=ah.hexdigest(),
       optimizer_updates=steps,regularizer_model_rows=queries,seconds=time.monotonic()-start,
       selection_completed_before_audit=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--arm',choices=['erm','equality_direction'],required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();train(a.task,a.seed,a.arm,device=a.device)
