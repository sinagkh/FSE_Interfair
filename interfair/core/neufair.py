"""Pinned native NeuFair SA with a verified common-MLP compute adapter."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[1]
import argparse
import ast
import copy
import datetime
import gzip
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import List,Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder,MinMaxScaler

from core import ROOT,Predictor,behavior,evaluate,predict,save_json,sha,seed_all
from maintenance import data_for,erm_path

VENDOR=_PACKAGE / 'vendor/neufair'
PROTOCOL=_PACKAGE / 'core/reports/NEUFAIR_PROTOCOL.md'
BASE=_PACKAGE / 'core/runs/neufair'
DEVICE='cuda'

# Execute literal definitions; unrelated preprocessing imports are unnecessary.
tree=ast.parse((VENDOR/'sa.py').read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SimulatedAnnealingRepair')
utree=ast.parse((VENDOR/'utils.py').read_text())
funcs=[n for n in utree.body if isinstance(n,ast.FunctionDef) and n.name in ('equal_opp_difference','load_adult_census','preprocess_adult_census')]
exec(compile(ast.Module(body=funcs+[cls],type_ignores=[]),str(VENDOR/'sa.py'),'exec'),globals())


class MaskedMLP(torch.nn.Module):
    def __init__(self,source):
        super().__init__()
        layers=[l for l in source.net if isinstance(l,torch.nn.Linear)] if hasattr(source,'net') else [source.layer1,source.layer2,source.layer3,source.layer4]
        self.layers=torch.nn.ModuleList(copy.deepcopy(layers))
        self.widths=[l.out_features for l in self.layers[:-1]]
        self.state=0

    def forward(self,x):
        bits=format(int(self.state),f'0{sum(self.widths)}b');offset=0
        for i,layer in enumerate(self.layers):
            if i<len(self.widths):
                keep=x.new_tensor([v=='0' for v in bits[offset:offset+self.widths[i]]]);offset+=self.widths[i]
                x=torch.relu(F.linear(x,layer.weight*keep[:,None],layer.bias))
            else:x=layer(x)
        return x.squeeze(-1)

    def score(self,x,scale):
        l=self(x)
        return l.sigmoid() if scale=='P' else l


class Adapter(SimulatedAnnealingRepair):
    def __init__(self,source,tensors,seed,dest,kmax=75,minutes=60,uphill=25000,native_f1=None):
        self.model=MaskedMLP(source).to(DEVICE).eval()
        self.tensors={name:tuple(t.to(DEVICE) for t in vals) for name,vals in tensors.items()}
        self.seed=seed;self.dest=dest;dest.mkdir(parents=True,exist_ok=True)
        self.layer_sizes=self.model.widths;self.num_layers=len(self.layer_sizes);self.state_size_bits=sum(self.layer_sizes)
        self.layer_size_prefix=[0,*np.cumsum(self.layer_sizes).tolist()]
        self.k_min=2;self.k_max=kmax;self.max_time=minutes;self.max_iter_temp_init=uphill;self.T=2.
        self.sens_classes=[0,1];self.sens_multi_dataset='none';self.log_file_name=str(dest/'search.log')
        self.cache={};self.calls=0;self.forward_evaluations=0;self.forward_rows=0
        self.best_state=None;self.best_cost=None
        self.phase='setup';self.last_progress=time.monotonic();self.started=time.monotonic();self.curve=[]
        self.baseline=self.compute_fairness(0)
        self.f1_threshold=.98*(self.baseline[1] if native_f1 is None else native_f1)
        self.f1_penalty=3*self.baseline[0]
        self.selection_frozen=False

    def freeze_selection(self):
        if not self.selection_frozen:
            assert self.best_state is not None
            save_json(self.dest/'SELECTION.json',dict(best_state=str(self.best_state),best_cost=self.best_cost,
                      selection_finished_before_audit=True,selected_by='native validation EOmax/soft-F1 cost',
                      dropped_weights=int(self.best_state).bit_count()))
            self.selection_frozen=True

    def compute_fairness(self,state,dataset='val'):
        if dataset!='val':self.freeze_selection()
        self.calls+=1;self.model.state=state;key=(state,dataset)
        if key not in self.cache:
            x,y,s=self.tensors[dataset]
            with torch.inference_mode():
                pred=self.model(x).sigmoid()>.5
                rates={(g,t):pred[(s==g)&(y==t)].float().mean() for g in (0,1) for t in (0,1)}
                assert all(torch.isfinite(v) for v in rates.values())
                eo=torch.maximum(abs(rates[0,1]-rates[1,1]),abs(rates[0,0]-rates[1,0]))
                tp=(pred&(y==1)).sum();fp=(pred&(y==0)).sum();fn=((~pred)&(y==1)).sum()
                f1=2*tp/(2*tp+fp+fn);acc=(pred==(y==1)).float().mean()
                self.cache[key]=tuple(float(v) for v in (eo,f1,acc))
            self.forward_evaluations+=1;self.forward_rows+=len(x)
        now=time.monotonic()
        if now-self.last_progress>=30:
            row=dict(phase=self.phase,elapsed_seconds=now-self.started,calls=self.calls,
                     forward_evaluations=self.forward_evaluations,forward_rows=self.forward_rows,
                     best_cost=self.best_cost,best_state=str(self.best_state),
                     dropped_weights=int(self.best_state).bit_count() if self.best_state is not None else None)
            self.curve.append(row);save_json(self.dest/'progress.json',row)
            save_json(self.dest/'curve.json',self.curve);self.last_progress=now
        return self.cache[key]

    def baseline_fairness(self):
        v=self.compute_fairness(0);t=self.compute_fairness(0,'test');tr=self.compute_fairness(0,'train')
        return v[1],v[0],v[2],t[1],t[0],t[2],tr[1],tr[0],tr[2]


def validate(source,obj):
    class OfficialShape(torch.nn.Module):
        def __init__(self,source):
            super().__init__()
            layers=[l for l in source.net if isinstance(l,torch.nn.Linear)] if hasattr(source,'net') else [source.layer1,source.layer2,source.layer3,source.layer4]
            for i,l in enumerate(layers):setattr(self,f'layer{i+1}',copy.deepcopy(l).cpu())
        def forward(self,x):
            return self.layer4(torch.relu(self.layer3(torch.relu(self.layer2(torch.relu(self.layer1(x)))))))
    native=SimulatedAnnealingRepair.__new__(SimulatedAnnealingRepair)
    native.model=OfficialShape(source).eval();native.layer_sizes=obj.layer_sizes;native.layer_size_prefix=obj.layer_size_prefix
    native.state_size_bits=obj.state_size_bits;native.sens_classes=[0,1];native.sens_multi_dataset='none'
    native.X_val,native.y_val,native.sens_val=[x.cpu() for x in obj.tensors['val']]
    original=copy.deepcopy(native.model.state_dict())
    rng=random.Random(97531);n=obj.state_size_bits
    states=[0,1,1<<(n-1),(1<<(n-1))|(1<<(obj.layer_sizes[0]-1)),sum(1<<i for i in rng.sample(range(n),obj.k_max))]
    states += [sum(1<<i for i in rng.sample(range(n),rng.randint(obj.k_min,obj.k_max))) for _ in range(7)]
    checks=[]
    for state in states:
        ref=native.compute_fairness(state);actual=obj.compute_fairness(state)
        metric_error=max(abs(float(a)-float(b)) for a,b in zip(ref,actual))
        literal=copy.deepcopy(native.model)
        bits=format(state,f'0{n}b');offset=0
        with torch.no_grad():
            for i,width in enumerate(obj.layer_sizes,1):
                for j,c in enumerate(bits[offset:offset+width]):
                    if c=='1':getattr(literal,f'layer{i}').weight[j].zero_()
                offset+=width
            expected=literal(native.X_val[:512]).reshape(-1).numpy()
            obj.model.state=state;observed=obj.model(obj.tensors['val'][0][:512]).cpu().numpy()
        error=float(abs(expected-observed).max())
        unchanged=all(torch.equal(v,native.model.state_dict()[k]) for k,v in original.items())
        assert error<2e-5 and metric_error<2e-6 and unchanged,(error,metric_error)
        checks.append(dict(state=str(state),logit_max_error=error,metric_max_error=metric_error,source_restored=unchanged,passed=True))
    obj.model.state=0
    return checks


def get_common(task,seed):
    data=data_for(task,seed);path=erm_path(task,seed)
    ck=torch.load(path,map_location='cpu',weights_only=False)
    source=Predictor(data['x'].shape[1],'mlp','erm');source.load_state_dict(ck['state_dict'])
    tensors={}
    for split,label in [('val','val'),('audit','test'),('train','train')]:
        ix=data['splits'][split];tensors[label]=tuple(torch.as_tensor(v[ix]) for v in (data['x'],data['y'],data['s']))
    return data,path,source,tensors


def get_native():
    mtree=ast.parse((VENDOR/'model.py').read_text());definition=next(n for n in mtree.body if isinstance(n,ast.ClassDef) and n.name=='AdultCensusModel')
    ns={'nn':torch.nn,'torch':torch};exec(compile(ast.Module(body=[definition],type_ignores=[]),str(VENDOR/'model.py'),'exec'),ns)
    source=ns['AdultCensusModel'](p=.1)
    path=VENDOR/'saved_models/adult/adult-0.684-42-ss-0.1.pt'
    source.load_state_dict(torch.load(path,map_location='cpu',weights_only=False));source.eval()
    old=Path.cwd()
    try:
        os.chdir(VENDOR/'sa_experiments')
        tr,te,va,yt,ye,yv,si=preprocess_adult_census(42)
    finally:os.chdir(old)
    tensors={name:(x,y,x[:,si]) for name,x,y in [('train',tr,yt),('test',te,ye),('val',va,yv)]}
    return path,source,tensors


def run(task,seed,native=False):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;seed_all(seed);random.seed(seed)
    if native:
        path,source,tensors=get_native();data=None;dest=BASE/'native_smoke_42';kmax=50;minutes=1;uphill=100
    else:
        assert seed in (2000,2001)
        data,path,source,tensors=get_common(task,seed);dest=BASE/task/str(seed);kmax=75;minutes=60;uphill=25000
    if (dest/'DONE.json').exists():return
    obj=Adapter(source,tensors,seed,dest,kmax,minutes,uphill,native_f1=.684 if native else None)
    checks=validate(source,obj)
    save_json(dest/'FIDELITY.json',dict(status='PASS',checks=checks))
    # Validation consumed no RNG used by SA; reset all search generators explicitly.
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    obj.cache.clear();obj.calls=0;obj.forward_evaluations=0;obj.forward_rows=0
    obj.best_cost=None;obj.best_state=None
    config=dict(task='native_adult' if native else task,seed=seed,source=str(path),source_sha256=sha(path),
                native_smoke=native,k_min=2,k_max=kmax,hidden_widths=obj.layer_sizes,search_minutes=minutes,
                uphill_initialization=uphill,f1_threshold=obj.f1_threshold,f1_penalty=obj.f1_penalty,
                protocol_sha256=None,adapter_sha256=sha(__file__),upstream_commit='953c80f82a85b916dc4ffbd76c9832d0d1dcb564',
                upstream_sha256={n:sha(VENDOR/n) for n in ('sa.py','utils.py','model.py')},
                isolated_GPU=True,torch_threads=1,pid=os.getpid(),audit_used_for_selection=False,
                baseline_validation=obj.baseline,bank_sha256=None if native else sha(ROOT/'cache_v2'/f'{task}_selection_{seed}_all.pkl'))
    save_json(dest/'CONFIG.json',config)
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True)
    (dest/'GPU_AT_START.txt').write_text(gpu)
    obj.started=time.monotonic();obj.last_progress=0;obj.phase='temperature_initialization'
    obj.T=obj.estimate_initial_temp(chi_0=.75,T0=5.,p=5,eps=.001)
    init_seconds=time.monotonic()-obj.started;initial_calls=obj.calls
    obj.phase='one_minute_smoke' if native else 'dedicated_60_minute_search'
    initial_temperature=obj.T;search_start=time.monotonic()
    obj.run_sa(decay='log');obj.log_file.close();search_seconds=time.monotonic()-search_start
    obj.freeze_selection();obj.model.state=obj.best_state
    best_val=obj.compute_fairness(obj.best_state)
    if native:
        audit={'native_metrics':obj.compute_fairness(obj.best_state,'test')}
    else:
        audit=evaluate(obj.model,data,'audit',DEVICE,save_path=dest/'audit.npz')
        torch.save(dict(state_dict=source.state_dict(),mask=str(obj.best_state),config=config),dest/'selected.pt')
    final=dict(status='completed',config=config,best_state=str(obj.best_state),dropped_weights=int(obj.best_state).bit_count(),
               best_validation=best_val,best_cost=obj.best_cost,initial_temperature=initial_temperature,
               initialization_seconds=init_seconds,initialization_calls=initial_calls,search_seconds=search_seconds,
               total_seconds=time.monotonic()-obj.started,calls=obj.calls,unique_forward_evaluations=obj.forward_evaluations,
               forwarded_rows=obj.forward_rows,audit=audit,selection_finished_before_audit=True)
    save_json(dest/'DONE.json',final);save_json(dest/'curve.json',obj.curve)
    with (dest/'search.log').open('rb') as f,gzip.open(dest/'search.log.gz','wb') as g:shutil.copyfileobj(f,g)
    (dest/'search.log').unlink()
    save_json(dest/'progress.json',dict(phase='completed',total_seconds=final['total_seconds'],best_cost=obj.best_cost))
    print(json.dumps(dict(task=config['task'],seed=seed,status='completed',seconds=final['total_seconds'],best_validation=best_val)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',nargs=2);p.add_argument('--native-smoke',action='store_true');p.add_argument('--queue',action='store_true');args=p.parse_args()
    if args.native_smoke:run('native_adult',42,True);return
    if args.job:run(args.job[0],int(args.job[1]));return
    if args.queue:
        logs=ROOT/'job_logs/neufair';logs.mkdir(parents=True,exist_ok=True)
        env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
        for task in ('adult','hmda_oh'):
            for seed in (2000,2001):
                with (logs/f'{task}_{seed}.log').open('w') as f:
                    p=subprocess.run([sys.executable,__file__,'--job',task,str(seed)],stdout=f,stderr=subprocess.STDOUT,env=env)
                assert p.returncode==0,(task,seed)
                print(json.dumps(dict(completed=[task,seed])),flush=True)


if __name__=='__main__':main()
