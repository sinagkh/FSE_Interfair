"""Two-seed additive. Selection uses validation only; no final confirmation seeds."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[1]
import argparse
import copy
import hashlib
import json
import pickle
import time
import os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from core import (ROOT, DEV_SEEDS, Predictor, evaluate, guard_loss, prepare_task,
                  save_json, seed_all, sha, state_hash)

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'


def prepare(name):
    path=ROOT/'cache_v2'/f'{name}.pkl';meta=ROOT/'cache_v2'/f'{name}.json'
    if path.exists():
        with path.open('rb') as f:return pickle.load(f)
    start=time.time();data=prepare_task(name)
    path.parent.mkdir(exist_ok=True,parents=True)
    with path.with_suffix('.tmp').open('wb') as f:pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
    path.with_suffix('.tmp').replace(path)
    data['metadata']['preparation_seconds']=time.time()-start
    data['metadata']['compiler_sha256']=sha(ROOT/'core.py')
    save_json(meta,data['metadata'])
    print(json.dumps({'event':'prepared','task':name,'seconds':time.time()-start,'banks':data['metadata']['bank_counts']}),flush=True)
    return data


def feasible(result,reference,auc_budget=.01,f1_fraction=.98):
    b=result['behavior'];r=reference['behavior']
    return b['auc']>=r['auc']-auc_budget and b['f1']>=f1_fraction*r['f1']


def task_key(r):return (-r['behavior']['auc'],r['behavior']['bce'])


def train(data,architecture,seed,arm,output,source=None,reference=None,weight=None,guard_weight=.1,preserve_weight=0.,warmup=5,repair_epochs=20,lr=.001,selection_auc_budget=.01,selection_f1_ratio=.98,score_scale=None,interaction_enabled=None,direction_weight=0.,direction_sign=1.,selection_objective="invariance",preserve_target="signed"):
    output=Path(output)
    if (output/'DONE.json').exists():
        print(json.dumps({'event':'cached','path':str(output)}),flush=True)
        return json.loads((output/'DONE.json').read_text())
    output.mkdir(parents=True,exist_ok=True)
    is_repair=source is not None
    mode=arm if arm in ('removed','structural_L','structural_P') else 'erm'
    seed_all(seed);model=Predictor(data['x'].shape[1],architecture,mode).to(DEVICE)
    if source is not None:model.load_state_dict(torch.load(source,map_location=DEVICE,weights_only=False)['state_dict'])
    start_hash=state_hash(model)
    if DEVICE=='cuda':torch.cuda.reset_peak_memory_stats()
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
    ids=data['splits']['train'];x=torch.as_tensor(data['x'][ids],device=DEVICE)
    y=torch.as_tensor(data['y'][ids],device=DEVICE);s=x[:,0]
    rng=np.random.default_rng(seed+201);arng=np.random.default_rng(seed+301)
    banks=[];bank_weights=[]
    for name,b in data['banks']['train'].items():
        if not b['training']:continue
        bank=b['corners'][:,b['supported']]
        if bank.shape[1]<64:raise ValueError(f'Insufficient supported steering anchors: {name}: {bank.shape[1]}')
        banks.append(torch.as_tensor(bank,device=DEVICE))
        weights=b.get("sampling_weights")
        if weights is not None:
            weights=np.asarray(weights)[b["supported"]];assert np.isfinite(weights).all() and (weights>0).all()
            weights=weights/weights.sum()
        bank_weights.append(weights)
    scale=score_scale or ('P' if arm=='interfair_P' else 'L')
    interaction=arm in ('interfair_L','interfair_P') if interaction_enabled is None else interaction_enabled
    if weight is None:weight=16. if scale=='P' else 1.
    epochs=repair_epochs if is_repair else 40
    config=dict(task=data['metadata']['task'],architecture=architecture,prototype_architecture=architecture=='resnet',seed=seed,arm=arm,
                mode=mode,source=str(source) if source else None,source_sha256=sha(source) if source else None,
                initial_state_sha256=start_hash,epochs=epochs,lr=lr,weight_decay=.0001,batch=512,anchor_batch=128,
                interaction_weight=weight if interaction else 0.,interaction_scale=scale if interaction else None,
                guard_weight=guard_weight if is_repair else 0.,preserve_weight=preserve_weight,warmup=warmup,
                selection_auc_budget=selection_auc_budget,selection_f1_ratio=selection_f1_ratio,
                requirement_id=data['metadata'].get('active_requirement','additive fixed raw edits'),direction_weight=direction_weight,direction_sign=direction_sign,selection_objective=selection_objective,preserve_target=preserve_target,
                steering_bank_sha256=hashlib.sha256(b''.join(b.detach().cpu().numpy().tobytes() for b in banks)).hexdigest(),
                sampling_weight_sha256=hashlib.sha256(b''.join(w.tobytes() for w in bank_weights if w is not None)).hexdigest(),
                code_sha256={p.name:sha(p) for p in [ROOT/'core.py',ROOT/'training_base.py']},device=DEVICE,
                cuda_name=torch.cuda.get_device_name() if DEVICE=='cuda' else None,selection_split='val',audit_used_for_selection=False,
                campaign_concurrency=os.environ.get('INTERFAIR_CONCURRENCY','1'),runtime_is_isolated=os.environ.get('INTERFAIR_CONCURRENCY','1')=='1')
    save_json(output/'CONFIG.json',config)
    candidates={};history=[];batchhash=hashlib.sha256();anchorhash=hashlib.sha256()
    if preserve_weight:
        teacher=copy.deepcopy(model).eval()
        for p in teacher.parameters():p.requires_grad_(False)
    def consider(epoch):
        result=evaluate(model,data,'val',DEVICE,primary_only=True)
        ok=True if reference is None else feasible(result,reference,selection_auc_budget,selection_f1_ratio)
        info=dict(epoch=epoch,validation=result,feasible=ok)
        history.append(info)
        for selector in ('task','L','P'):
            if selector!='task' and not ok:continue
            if selector=='task':key=task_key(result)
            else:
                objective=result['primary_'+selector]
                if selection_objective=='direction':objective=result['primary_direction_'+selector]
                elif selection_objective=='combined':objective+=result['primary_direction_'+selector]
                key=(objective,)+task_key(result)
            if selector not in candidates or key<candidates[selector]['key']:
                candidates[selector]=dict(key=key,epoch=epoch,validation=result,feasible=ok,
                                          state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        return result
    start=time.time()
    if is_repair:consider(0)
    train_seconds=0.
    for epoch in range(1,epochs+1):
        model.train();perm=rng.permutation(len(x));batchhash.update(perm.tobytes())
        if DEVICE=='cuda':torch.cuda.synchronize()
        epstart=time.time();running=[]
        for i in range(0,len(x),512):
            b=torch.as_tensor(perm[i:i+512],device=DEVICE)
            if mode=='structural_P':
                p=model.probability(x[b]);loss=F.binary_cross_entropy(p,y[b])
            else:
                logits=model(x[b]);p=logits.sigmoid();loss=F.binary_cross_entropy_with_logits(logits,y[b])
            if is_repair:
                loss=loss+guard_weight*guard_loss(p,y[b],s[b])
                sampled=[];targets=[]
                for bank,weights in zip(banks,bank_weights):
                    j=arng.integers(0,bank.shape[1],size=128) if weights is None else arng.choice(bank.shape[1],size=128,p=weights)
                    anchorhash.update(j.tobytes())
                    sampled.append(bank[:,torch.as_tensor(j,device=DEVICE)])
                if interaction or preserve_weight or direction_weight:
                    joined=torch.cat([v.reshape(-1,x.shape[1]) for v in sampled],dim=0)
                    scores=model.score(joined,scale).reshape(len(sampled),4,128)
                    first=scores[:,1]-scores[:,0];second=scores[:,3]-scores[:,2]
                    if interaction:loss=loss+weight*min(epoch/max(warmup,1),1.)*(second-first).square().mean()
                    if direction_weight:loss=loss+direction_weight*min(epoch/max(warmup,1),1.)*(torch.relu(-direction_sign*first).square()+torch.relu(-direction_sign*second).square()).mean()/2
                    if preserve_weight:
                        with torch.no_grad():t=teacher.score(joined,scale).reshape(len(sampled),4,128);common=(t[:,1]-t[:,0]+t[:,3]-t[:,2])/2
                        if preserve_target=='positive_floor':
                            loss=loss+preserve_weight*torch.relu(common.clamp_min(0)-(first+second)/2).square().mean()
                        else:loss=loss+preserve_weight*((first+second)/2-common).square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            running.append(float(loss.detach()))
        if DEVICE=='cuda':torch.cuda.synchronize()
        train_seconds+=time.time()-epstart
        consider(epoch);history[-1]['mean_train_loss']=float(np.mean(running))
    # Finalize selection before any audit predictions.
    selections={k:{a:b for a,b in v.items() if a not in ('state_dict','key')} for k,v in candidates.items()}
    save_json(output/'SELECTION.json',dict(selectors=selections,no_feasible_selector=[k for k in ('L','P') if k not in candidates],selection_finished_before_audit=True))
    save_json(output/'HISTORY.json',history)
    results={}
    for selector,candidate in candidates.items():
        torch.save(dict(state_dict=candidate['state_dict'],config=config,epoch=candidate['epoch']),output/f'{selector}.pt')
        model.load_state_dict(candidate['state_dict'])
        audit=evaluate(model,data,'audit',DEVICE,save_path=output/f'audit_{selector}.npz')
        results[selector]=dict(epoch=candidate['epoch'],validation=candidate['validation'],validation_feasible=candidate['feasible'],audit=audit,state_sha256=state_hash(model))
    final=dict(config=config,selections=results,training_seconds=train_seconds,total_seconds=time.time()-start,
               batch_sha256=batchhash.hexdigest(),anchor_sha256=anchorhash.hexdigest(),parameter_count=sum(p.numel() for p in model.parameters()),
               peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if DEVICE=='cuda' else 0,
               status='completed',claim_scope='development experiment, not final confirmation')
    save_json(output/'DONE.json',final)
    print(json.dumps({'event':'completed','task':config['task'],'arch':architecture,'seed':seed,'arm':arm,'train_seconds':round(train_seconds,2),
                      'selected':{k:dict(epoch=v['epoch'],auc=v['audit']['behavior']['auc'],f1=v['audit']['behavior']['f1'],L=v['audit']['primary_L'],P=v['audit']['primary_P']) for k,v in results.items()}}),flush=True)
    return final


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--tasks',nargs='+',default=['adult','hmda_oh']);p.add_argument('--architectures',nargs='+',default=['mlp','resnet']);p.add_argument('--seeds',nargs='+',type=int,default=[2000,2001]);p.add_argument('--job',nargs=4,metavar=('TASK','ARCH','SEED','ARM'));args=p.parse_args()
    assert set(args.seeds)<=set(DEV_SEEDS),'Final seeds prohibited in development runner'
    torch.set_num_threads(int(os.environ.get("INTERFAIR_THREADS","2")))
    if args.job:
        task,architecture,raw_seed,arm=args.job;seed=int(raw_seed)
        assert seed in DEV_SEEDS
        out=ROOT/'runs/pilot_v3'/task/architecture/str(seed);data=prepare(task)
        reference=None;source=None
        if arm!='erm':
            erm=json.loads((out/'erm/DONE.json').read_text());reference=erm['selections']['task']['validation']
        if arm in ('no_interaction','interfair_L','interfair_P'):source=out/'erm/task.pt'
        train(data,architecture,seed,arm,out/arm,source=source,reference=reference)
        return
    for task in args.tasks:
        data=prepare(task)
        if args.prepare_only:continue
        for architecture in args.architectures:
            for seed in args.seeds:
                out=ROOT/'runs/pilot_v3'/task/architecture/str(seed)
                erm=train(data,architecture,seed,'erm',out/'erm')
                reference=erm['selections']['task']['validation'];source=out/'erm/task.pt'
                for arm in ('removed','structural_L','structural_P'):
                    train(data,architecture,seed,arm,out/arm,reference=reference)
                for arm in ('no_interaction','interfair_L','interfair_P'):
                    train(data,architecture,seed,arm,out/arm,source=source,reference=reference)


if __name__=='__main__':main()
