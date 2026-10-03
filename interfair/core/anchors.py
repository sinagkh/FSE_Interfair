"""E14b: nested supported steering-bank sizes; fixed validation/audit banks."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[1]
import argparse,copy,hashlib,json,os,pickle,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from core import ROOT,numeric,categorical,save_json,sha
from training_base import train,evaluate
from maintenance import data_for,erm_path
from feature_edits import banks as inventory_banks
from support import fitted_support

PROTOCOL=_PACKAGE / 'core/reports/ANCHOR_BUDGET_PROTOCOL.md'
BASE=_PACKAGE / 'core/runs/anchor_budget'
BUDGETS=(32,128,512,2048,8192)


def prepare(task):
    path=ROOT/'cache_v2'/f'{task}_anchor_master.pkl'
    if path.exists():
        with path.open('rb') as f:return pickle.load(f)
    start=time.monotonic();d=data_for(task,2000);inv=inventory_banks(task);enc=d['encoder'];ix=d['splits']['train'];raw=d['frame'].iloc[ix].copy();support=fitted_support(d);result={};records=[];timing=[]
    for spec in inv['specs']:
        if spec['status']!='candidate':continue
        ts=time.monotonic();lo=raw.copy();hi=raw.copy();valid=np.ones(len(ix),bool)
        for j,col in enumerate(spec['columns']):
            a=spec['low'][j] if len(spec['columns'])>1 else spec['low'];b=spec['high'][j] if len(spec['columns'])>1 else spec['high'];lo[col]=a;hi[col]=b
            valid &= numeric(raw[col]).notna().to_numpy() if col in enc.continuous else ~categorical(raw[col]).isin(['__MISSING__','Exempt','Unknown','Not applicable']).to_numpy()
            if col in enc.continuous:
                st=enc.stats[col];valid &= st['q01']<=a<=st['q99'] and st['q01']<=b<=st['q99']
            else:valid &= a in enc.vocab[col] and b in enc.vocab[col]
            if col=='education_num':lo['education']=d['education_map'][int(a)];hi['education']=d['education_map'][int(b)]
        corners=[]
        for group in (0,1):
            for frame in (lo,hi):
                frame=frame.copy();frame['protected']=group;corners.append(enc.transform(frame))
        c=np.stack(corners);compile_seconds=time.monotonic()-ts;ts=time.monotonic();keep=np.zeros(len(ix),bool)
        if valid.any():keep[valid],_=support.mask(c[:,valid])
        support_seconds=time.monotonic()-ts
        old=inv['banks']['train'][spec['name']];pos=np.searchsorted(ix,old['indices']);assert np.array_equal(ix[pos],old['indices']) and np.array_equal(c[:,pos],old['corners']) and np.array_equal(keep[pos],old['supported'])
        count=int(keep.sum());assert count>=max(BUDGETS),(task,spec['name'],count,'insufficient unique supported training anchors')
        result[spec['name']]=dict(corners=c[:,keep].copy(),indices=ix[keep].copy())
        timing.append(dict(feature=spec['name'],candidate_contexts=len(ix),valid=int(valid.sum()),supported=count,compile_seconds=compile_seconds,support_seconds=support_seconds))
        records.append(dict(feature=spec['name'],old_training_bank_replayed_exactly=True,supported=count,passed=True));print(task,spec['name'],'supported',count,flush=True)
    obj=dict(banks=result,task=task,checks=records,preparation=timing,seconds=time.monotonic()-start,protocol_sha256=None,code_sha256=sha(__file__),support_thresholds=support.thresholds,inventory_sha256=sha(ROOT/'cache_v2'/f'{task}_feature_inventory.pkl'))
    with path.with_suffix('.tmp').open('wb') as f:pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)
    path.with_suffix('.tmp').replace(path)
    save_json(ROOT/'reports'/f'{task}_ANCHOR_MASTER.json',{k:v for k,v in obj.items() if k!='banks'})
    return obj


def budget_data(task,seed,n):
    d=data_for(task,seed);master=prepare(task);rng=np.random.default_rng(seed+95103);parts=[];indices=[];selected={}
    for name in sorted(master['banks']):
        b=master['banks'][name];order=rng.permutation(len(b['indices']))[:8192];chosen=order[:n];parts.append(b['corners'][:,chosen]);indices.append(b['indices'][chosen]);selected[name]=b['indices'][chosen].tolist()
    c=np.concatenate(parts,axis=1);ix=np.concatenate(indices);mask=np.ones(len(ix),bool)
    d['banks']['train']={'selected_mixture':dict(corners=c,indices=ix,valid=mask,supported=mask,training=True,sampling_weights=np.full(len(ix),1/len(ix),dtype=np.float64))}
    d['metadata']['active_requirement']=f'E14 supported steering budget {n} per feature; broad P invariance with signed-response preservation'
    d['metadata']['anchor_budget_per_feature']=n;d['metadata']['anchor_indices']=selected
    return d


def gate():
    torch.set_num_threads(1);checks=[]
    for task in ('adult','hmda_oh'):
        master=prepare(task);checks+=master['checks'];reference=data_for(task,2000)
        for seed in (2000,2001):
            rng=np.random.default_rng(seed+95103)
            for name,b in sorted(master['banks'].items()):
                order=rng.permutation(len(b['indices']))[:8192];assert len(np.unique(b['indices'][order]))==8192
                for n in BUDGETS:assert len(b['indices'][order[:n]])==n
            checks.append(dict(check='nested unique supported training contexts per field at all budgets',task=task,seed=seed,passed=True))
        # Fixed validation/audit banks and training-only source construction are explicit.
        for b in master['banks'].values():assert not np.intersect1d(b['indices'],np.concatenate([reference['splits']['val'],reference['splits']['audit']])).size
        checks.append(dict(check='steering contexts disjoint from validation and audit',task=task,passed=True))
    save_json(ROOT/'reports/ANCHOR_BUDGET_GATE.json',dict(status='PASS',checks=checks,code_sha256=sha(__file__),protocol_sha256=None,master_sha256={t:sha(ROOT/'cache_v2'/f'{t}_anchor_master.pkl') for t in ('adult','hmda_oh')}));print('Anchor budget gate PASS',len(checks),flush=True)


def worker(task,seed,n,arm):
    assert seed in (2000,2001) and n in BUDGETS and arm in ('control','interfair');assert not torch.cuda.is_available();torch.set_num_threads(1)
    g=json.loads((ROOT/'reports/ANCHOR_BUDGET_GATE.json').read_text());assert g['code_sha256']==sha(__file__) and g['protocol_sha256']==sha(PROTOCOL)
    d=budget_data(task,seed,n);out=BASE/task/str(seed)/str(n)/arm;out.mkdir(parents=True,exist_ok=True)
    if (out/'DONE.json').exists():return
    save_json(out/'BANK.json',dict(budget_per_feature=n,selected=d['metadata']['anchor_indices'],master_sha256=g['master_sha256'][task],protocol_sha256=None,runner_sha256=sha(__file__),validation_bank='unchanged original broad bank',audit_bank='unchanged original broad bank'))
    ref=json.loads((erm_path(task,seed).parent/'DONE.json').read_text())['selections']['task']['validation']
    train(d,'mlp',seed,arm,out,source=erm_path(task,seed),reference=ref,weight=160,preserve_weight=160,preserve_target='signed',guard_weight=.1,warmup=5,selection_auc_budget=.005,selection_f1_ratio=.99,score_scale='P',interaction_enabled=arm=='interfair')


def queue():
    logs=ROOT/'job_logs/anchor_budget';logs.mkdir(parents=True,exist_ok=True);env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','INTERFAIR_CONCURRENCY':'4'}
    def run(j):
        with (logs/('_'.join(map(str,j))+'.log')).open('w') as f:r=subprocess.run([sys.executable,__file__,'--job',*map(str,j)],stdout=f,stderr=subprocess.STDOUT,env=env)
        assert r.returncode==0,j
        print(json.dumps(dict(completed=j)),flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(run,[(t,s,n,a) for t in ('adult','hmda_oh') for s in (2000,2001) for n in BUDGETS for a in ('control','interfair')]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',choices=['adult','hmda_oh']);p.add_argument('--gate',action='store_true');p.add_argument('--job',nargs=4);a=p.parse_args()
    if a.prepare:prepare(a.prepare)
    elif a.gate:gate()
    elif a.job:worker(a.job[0],int(a.job[1]),int(a.job[2]),a.job[3])
    else:queue()
