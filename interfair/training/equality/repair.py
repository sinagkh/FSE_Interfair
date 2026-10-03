"""Interaction losses and shared repair utilities."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.special import expit
from torch.nn import functional as F

HERE = _PACKAGE / 'training/equality'
ROOT = _PACKAGE
OLD = _PACKAGE / 'data/partitions'
DEV = _PACKAGE / 'core'
ART = _PACKAGE / 'core/relations.py'
sys.path[:0] = [str(OLD), str(DEV)]
from partitions import data_for, read_pickle, array_hash
from core import Predictor, behavior, seed_all, state_hash, sha
from model_interfaces import Pipeline, load_pipeline
from pipeline_audit import audit,feature_metrics

module_spec = importlib.util.spec_from_file_location('interaction_relations', ART)
relations = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = relations
module_spec.loader.exec_module(relations)
# Package integrity is checked by scripts/verify.py.


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist()) + '\n')
    tmp.replace(path)


def recipe(task, architecture='mlp', strength=None):
    if strength is None:
        registry=HERE/'ARCHITECTURE_STRENGTHS.json'
        strength=json.loads(registry.read_text()).get(architecture+'/'+task,1.) if registry.exists() else 1.
    hmda = task == 'hmda_oh'
    return dict(main=0., pair=1. if hmda else .35, score=.2, effect=.4,
                boundary_score=.2 if hmda else .15,
                boundary_effect=.3 if hmda else .25, tail=.25,
                tail_fraction=.1, boundary_band=.10 if hmda else .18,
                min_cell=8, strength=strength, batch=512, anchors=512,
                lr=.0007 if architecture == 'mlp' else .0001,
                weight_decay=.0001 if architecture == 'mlp' else .00001,
                epochs=100, patience=12, warmup=5, gradient_clip=5., selection_min_delta=.0001,
                auc_floor=.895 if task == 'adult' else None,
                credit_auc_budget=.01, f1_ratio=.98)


def old_root(seed):
    return OLD / ('admission_runs' if seed >= 2000 else 'runs')


def source_path(task, seed, arch):
    suffix=Path(arch)/task/str(seed)/'erm'/('task.pt' if arch == 'mlp' else 'selected.pt')
    path=old_root(seed)/'main'/suffix
    if not path.exists() and seed>=2000:path=DEV/'runs/splits'/suffix
    assert path.exists(),path
    return path


def reference_pipeline(task,seed,arch,d,device='cpu'):
    path=source_path(task,seed,arch)
    if arch=='mlp':
        from model_interfaces import plain
        return plain(path,d,device)
    from ft_architecture import load_ft
    return Pipeline(load_ft(path,d,device),[path],device,batch=512)


def get_data(task, seed, study='main', arm='interfair'):
    d = data_for(task, seed)
    names = None
    if study == 'selection':
        inv = read_pickle(DEV/'cache_v2'/f'{task}_feature_inventory.pkl')
        path=old_root(seed)/'selection'/task/str(seed)/'SELECTION_POLICY.json'
        if path.exists():policy=json.loads(path.read_text())
        else:
            from supplementary_data import selected_features
            policy=selected_features(task,seed,source_path(task,seed,'mlp'),HERE/'policies'/task/str(seed)/'SELECTION_POLICY.json')
        names = policy['selected'][arm]
        train = {n: inv['banks']['train'][n] for n in names}
    elif study.startswith('direction_'):
        from supplementary_data import auxiliary
        d = auxiliary(task, seed, 'direction', study.split('_')[1])
        train = {n:b for n,b in d['banks']['train'].items() if b['training']}
    elif study == 'gradients':
        from supplementary_data import auxiliary
        d = auxiliary(task, seed, 'gradients')
        train = {n:b for n,b in d['banks']['train'].items() if b['training']}
    elif task == 'default_credit':
        from credit import prepare
        train = {n:b for n,b in prepare()['banks']['train'].items() if b['training']}
    else:
        mixture = d['banks']['train']['selected_mixture']
        names = sorted(d['metadata']['anchor_indices'])
        train = {}
        for j,n in enumerate(names):
            sl = slice(j*8192,(j+1)*8192)
            train[n] = {k:v[:,sl] if k=='corners' else v[sl] for k,v in mixture.items() if isinstance(v,np.ndarray)}
            train[n]['training'] = True
    # The typed bank may restrict different features to different contexts.
    # Each feature retains equal loss mass, and each anchor draw balances S x Y.
    for name,b in train.items():
        assert np.isin(b['indices'], d['splits']['train']).all(), name
    return d, train


def make_model(d, arch):
    if arch == 'mlp': return Predictor(d['x'].shape[1], 'mlp', 'erm')
    from ft_architecture import FTPipeline
    return FTPipeline(d['encoder'], 'erm')


def balanced_positions(indices, d, rng, count):
    # Calls the canonical sampler, preserving its S x Y balancing and fallback.
    obj = SimpleNamespace(spec=SimpleNamespace(train_idx=np.arange(len(indices))),
                          y=d['y'][indices], s=d['s'][indices])
    return relations.sample_anchor_indices(obj, rng, count)


def pair_effect_loss(v, natural_logits, s, y, cfg, interaction=True, direction=False):
    """v order: S0 low/high, S1 low/high; natural S selects population effect."""
    first, second = v[1]-v[0], v[3]-v[2]
    residual = second-first
    zero = v.sum()*0
    direct = cfg['pair']*(residual.square().mean()+cfg['tail']*relations.topk_square(residual,cfg['tail_fraction'])) if interaction else zero
    effect = torch.where(s > .5, second, first)
    weights = relations.boundary_weights_from_logits(natural_logits, cfg['boundary_band'])
    population = cfg['effect']*relations.group_gap_square_torch(effect,s,y,cfg['min_cell']) if interaction else zero
    boundary = cfg['boundary_effect']*relations.weighted_group_gap_square_torch(effect,s,y,cfg['min_cell'],weights) if interaction else zero
    sign = cfg.get('direction_sign',1.)
    directional = cfg['pair']*(torch.relu(-sign*first).square()+torch.relu(-sign*second).square()).mean()/2 if direction else zero
    return direct+population+boundary+directional


def score_loss(z,s,y,cfg):
    w = relations.boundary_weights_from_logits(z,cfg['boundary_band'])
    return cfg['score']*relations.group_gap_square_torch(z,s,y,cfg['min_cell']) + cfg['boundary_score']*relations.weighted_group_gap_square_torch(z,s,y,cfg['min_cell'],w)


@torch.no_grad()
def predict_logits(model,x,device,batch=2048):
    model.eval()
    return np.concatenate([model(torch.as_tensor(x[i:i+batch],device=device)).cpu().numpy() for i in range(0,len(x),batch)])


def finite(x): return float(x) if np.isfinite(x) else 0.


@torch.no_grad()
def validation(model,d,names,device,cfg):
    ix=d['splits']['val'];z=predict_logits(model,d['x'][ix],device);p=expit(z)
    b=behavior(d['y'][ix],d['s'][ix],p)
    rng=np.random.default_rng(24680+len(ix));a=ix if len(ix)<=4096 else rng.choice(ix,4096,replace=False)
    xx=d['x'][a].copy();z0=xx.copy();z1=xx.copy();z0[:,0]=0;z1[:,0]=1
    main=float(abs(predict_logits(model,z1,device)-predict_logits(model,z0,device)).mean())
    natural=predict_logits(model,xx,device);s=d['s'][a];y=d['y'][a];mask=abs(expit(natural)-.5)<=.15
    score=finite(relations.group_gap_np(natural,s,y)['conditional_mean'])
    boundary_score=finite(relations.group_gap_np(natural[mask],s[mask],y[mask])['conditional_mean']) if mask.any() else 0.
    pairs=[];effects=[];boundaries=[]
    for name in names:
        bank=d['banks']['val'][name];keep=bank['supported'];ids=bank['indices'][keep];n=len(ids)
        assert np.isin(ids,ix).all()
        c=bank['corners'][:,keep];v=predict_logits(model,c.reshape(-1,c.shape[-1]),device).reshape(4,n)
        e0=v[1]-v[0];e1=v[3]-v[2];ss=d['s'][ids];yy=d['y'][ids]
        e=np.where(ss>.5,e1,e0);zz=predict_logits(model,d['x'][ids],device);kk=abs(expit(zz)-.5)<=.15
        pairs.append(float(abs(e1-e0).mean()));effects.append(finite(relations.group_gap_np(e,ss,yy)['conditional_mean']))
        boundaries.append(finite(relations.group_gap_np(e[kk],ss[kk],yy[kk])['conditional_mean']) if kk.any() else 0.)
    m=dict(main=main,pair=float(np.mean(pairs)),score=score,effect=float(np.mean(effects)),
           boundary_score=boundary_score,boundary_effect=float(np.mean(boundaries)))
    residual=m['main']+m['pair']+m['score']+m['effect']+.1*(m['boundary_score']+m['boundary_effect'])
    # Original utility_mechanism selector uses BCE as the utility tie-break.
    return dict(behavior=b,mechanism=m,score=residual+.001*b['bce'])


def eligible(b,ref,cfg):
    floor=cfg['auc_floor'] if cfg['auc_floor'] is not None else ref['auc']-cfg['credit_auc_budget']
    return b['auc']>=floor and b['f1']>=cfg['f1_ratio']*ref['f1'] and b['aod']<=ref['aod'] and b['dp']<=ref['dp']


def load_corrected(path,d,device='cpu'):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    m=make_model(d,ck['config']['architecture']);m.load_state_dict(ck['state_dict'])
    return Pipeline(m,[path],device,batch=512 if ck['config']['architecture']=='ft' else 4096)


def train(task,seed,arch='mlp',study='main',arm='interfair',device='cpu',strength=None):
    torch.set_num_threads(1)
    phase='development' if seed>=2000 else 'confirmation'
    out=HERE/'runs'/phase/study/arch/task/str(seed)/arm
    if strength is not None and strength!=1: out=out.parent/(arm+f'_strength{strength:g}')
    if (out/'DONE.json').exists(): return json.loads((out/'DONE.json').read_text())
    if phase=='confirmation':
        freeze=json.loads((HERE/'FREEZE.json').read_text())
        for path,digest in freeze['files'].items():assert sha(path)==digest,('frozen method input changed',path)
    d,banks=get_data(task,seed,study,arm);cfg=recipe(task,arch,strength);strength=cfg['strength']
    names=list(banks)
    interaction=arm not in ('control','direction','task_control','rerepair_control')
    direction=study.startswith('direction_') and arm in ('direction','both')
    source=source_path(task,seed,arch)
    reference=reference_pipeline(task,seed,arch,d,'cpu')
    refdone=json.loads((source.parent/'DONE.json').read_text())
    refval=refdone['validation'] if arch=='ft' else refdone['selections']['task']['validation']
    ref=refval['behavior']
    seed_all(seed);model=make_model(d,arch).to(device)
    init_source=None
    if study=='learning':
        assert arm=='continuation_interfair'
        init_source=source
    elif study=='maintenance':
        assert arm in ('rerepair_interfair','rerepair_control')
        init_source=HERE/'runs'/phase/'maintenance/mlp'/task/str(seed)/'update_repaired/selected.pt'
    if init_source is not None:model.load_state_dict(torch.load(init_source,map_location=device,weights_only=False)['state_dict'])
    initial=state_hash(model)
    if init_source is None:
        assert initial!=state_hash(reference.model),'Accidentally initialized from ERM'
        assert initial==refdone['config']['initial_state_sha256'],'Fresh initialization differs from the architecture/seed baseline initialization'
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']) if arch=='mlp' else model.optimizer()
    config=dict(task=task,seed=seed,architecture=arch,mode='erm',study=study,arm=arm,
        method='logit interaction steering',recipe=cfg,
        initialization='fresh seeded model' if init_source is None else 'explicit continuation/maintenance variant',
        initial_state_sha256=initial,erm_weights_loaded=study=='learning',initial_checkpoint=str(init_source) if init_source else None,
        initial_checkpoint_sha256=sha(init_source) if init_source else None,
        teacher_response_penalty=0,main_effect_penalty=0,score_scale='L',features=names,
        source_erm=str(source),source_erm_sha256=sha(source),reference_validation=ref,
        canonical_source_sha256=sha(ART),adapter_sha256=sha(__file__),device=device,
        baseline_comparison='same architecture/data/seed; original IS optimizer/schedule versus archived native ERM schedule',
        anchor_sampling='512 balanced S x Y contexts per feature, equal feature loss mass; typed supported banks',
        feature_sampling='all features per step' if arch=='mlp' and study!='selection' else 'one uniformly sampled feature per step; unbiased estimator of the same equal-feature mean, including its per-feature tail',
        validation_selection='original task AUROC floor; F1>=.98 ERM; AOD/DP<=ERM; minimum original mechanism score',
        test_used_for_selection=False,phase=phase,
        split_hashes={k:array_hash(v) for k,v in d['splits'].items()},
        data_hashes={k:array_hash(d[k]) for k in ('x','y','s')},
        bank_hashes={n:array_hash(b['corners']) for n,b in banks.items()})
    write(out/'CONFIG.json',config)
    x=torch.as_tensor(d['x'],device=device);y=torch.as_tensor(d['y'],device=device);s=torch.as_tensor(d['s'],device=device)
    rng=np.random.default_rng(seed+991);frng=np.random.default_rng(seed+992);indices=d['splits']['train']
    pools={n:dict(c=torch.as_tensor(b['corners'][:,b['supported']],device=device),ids=b['indices'][b['supported']]) for n,b in banks.items()}
    history=[];best=None;native=None;stale=0;start=time.monotonic();bh=hashlib.sha256();ah=hashlib.sha256()
    for epoch in range(1,cfg['epochs']+1):
        losses=[];epstart=time.monotonic();warm=min(epoch/cfg['warmup'],1)*strength
        for ids in relations.iter_minibatches(indices,cfg['batch'],rng):
            bh.update(ids.tobytes());optimizer.zero_grad(set_to_none=True);model.train()
            taskloss=F.binary_cross_entropy_with_logits(model(x[ids]),y[ids]);taskloss.backward();total=float(taskloss.detach())
            ai=indices[balanced_positions(indices,d,rng,cfg['anchors'])];ah.update(ai.tobytes())
            # Dropout is disabled for the deterministic specification function.
            # Task minibatches retain the architecture's native training dropout.
            model.eval();z=model(x[ai]);sl=warm*score_loss(z,s[ai],y[ai],cfg);sl.backward();total+=float(sl.detach())
            chosen=list(pools) if arch=='mlp' and study!='selection' else [list(pools)[int(frng.integers(len(pools)))]]
            for name in chosen:
                pool=pools[name]
                jj=balanced_positions(pool['ids'],d,rng,cfg['anchors']);ah.update(jj.tobytes())
                ah.update(name.encode())
                ii=pool['ids'][jj];c=pool['c'][:,jj]
                if interaction or direction:
                    v=model(c.reshape(-1,c.shape[-1])).reshape(4,len(jj))
                    with torch.no_grad(): zz=model(x[ii])
                    term=warm*pair_effect_loss(v,zz,s[ii],y[ii],cfg,interaction,direction)/len(chosen)
                    term.backward();total+=float(term.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);optimizer.step();losses.append(total)
        result=validation(model,d,names,device,cfg);b=result['behavior'];ok=eligible(b,ref,cfg)
        floor=cfg['auc_floor'] if cfg['auc_floor'] is not None else ref['auc']-cfg['credit_auc_budget']
        original_score=result['score'] if b['auc']>=floor else 1000+floor-b['auc']+result['score']-.001*b['bce']
        item=dict(epoch=epoch,validation=result,eligible=ok,original_selection_score=original_score,
                  mean_training_loss=float(np.mean(losses)),seconds=time.monotonic()-epstart,
                  cumulative_batch_sha256=bh.hexdigest(),cumulative_anchor_sha256=ah.hexdigest())
        history.append(item)
        state=None
        if native is None or original_score<native['original_selection_score']-cfg['selection_min_delta']:
            native=copy.deepcopy(item);stale=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(dict(state_dict=state,config=config,epoch=epoch),out/'native_selected.pt')
        else:stale+=1
        if ok and (best is None or result['score']<best['validation']['score']):
            best=copy.deepcopy(item)
            if state is None:state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(dict(state_dict=state,config=config,epoch=epoch),out/'selected.pt')
        write(out/'HISTORY.json',history)
        write(out/'PROGRESS.json',dict(epoch=epoch,best=best,native=native,elapsed=time.monotonic()-start))
        print(json.dumps(dict(task=task,seed=seed,arch=arch,study=study,arm=arm,epoch=epoch,eligible=ok,
              auc=b['auc'],f1=b['f1'],aod=b['aod'],dp=b['dp'],L=result['mechanism']['pair'],seconds=item['seconds'])),flush=True)
        if stale>=cfg['patience']:break
    selection=dict(selected=best,native=native,no_admitted_repair=best is None,
                   selection_finished_before_audit=True,selection_uses_audit=False)
    write(out/'SELECTION.json',selection)
    chosen=out/('selected.pt' if best is not None else 'native_selected.pt')
    pipeline=load_corrected(chosen,d,device)
    source_scores=source.parent/('audit.npz' if arch=='ft' else 'audit_task.npz')
    cached=np.load(source_scores)
    can_reuse=all(n+'_P' in cached and np.array_equal(cached[n+'_indices'],b['indices']) for n,b in d['banks']['audit'].items())
    report,arrays=audit(pipeline,d,None if can_reuse else reference)
    if can_reuse:
        for n,b in d['banks']['audit'].items():
            for sc in ('L','P'):
                for cohort in ('valid','supported'):
                    mask=b[cohort];report['features'][n][sc+'_'+cohort]=feature_metrics(arrays[n+'_'+sc][:,mask],cached[n+'_'+sc][:,mask],sc)
        retained=[v['P_supported']['signed_retention'] for v in report['features'].values() if v['training'] and v['supported']]
        report['macro']['signed_retention']=float(np.mean([v for v in retained if v is not None]))
        report['source_audit_reused']=dict(path=str(source_scores),sha256=sha(source_scores))
    report['macro_L']={k:float(np.mean([v['L_supported'][k] for v in report['features'].values() if v['training'] and v['supported']])) for k in ('R','q95_abs_R','violation_rate','signed_retention')}
    np.savez_compressed(out/'audit.npz',**arrays)
    if phase=='confirmation' and study=='main':
        from capture import capture
        capture(pipeline,d,out,task,seed,arch)
    result=dict(status='completed',admitted=best is not None,config=config,selection=selection,audit=report,
                checkpoint=str(chosen),checkpoint_sha256=sha(chosen),epochs_completed=epoch,
                seconds=time.monotonic()-start,batch_sha256=bh.hexdigest(),anchor_sha256=ah.hexdigest(),
                outcome='pending paired scientific assessment',no_admission_reporting='native checkpoint retained as failed-admission diagnostic; never substituted by ERM')
    write(out/'DONE.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task');p.add_argument('seed',type=int);p.add_argument('--arch',default='mlp');p.add_argument('--study',default='main');p.add_argument('--arm',default='interfair');p.add_argument('--device',default='cpu');p.add_argument('--strength',type=float,default=None)
    a=p.parse_args();train(a.task,a.seed,a.arch,a.study,a.arm,a.device,a.strength)
