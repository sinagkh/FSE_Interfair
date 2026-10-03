"""Faithful retained named-method cells; no outer baseline tuning."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import argparse
import torch
from task_context import *
from task_pipelines import baseline_directory, evaluate, METHODS
from baseline_adapters import load

def run(task,seed,method):
    torch.set_num_threads(1)
    gate=json.loads((HERE/'reports/R1_ADMISSION.json').read_text());assert gate['status']=='PASS'
    if phase(seed)=='confirmation':
        freeze=json.loads((HERE/'FREEZE.json').read_text());assert freeze['status']=='frozen'
        for p,h in freeze['files'].items():assert sha(p)==h,('frozen input changed',p)
    directory=baseline_directory(task,seed,method)
    marker=directory/('TRAIN_DONE.json' if method=='mirrorfair' else 'DONE.json')
    if not marker.exists():
        assert not (task=='credit_broad' and seed in DEVELOPMENT),'Do not refit the compatible published development models'
        if method in ('ltdd','cot_phi'):load('preprocessing_baselines',seed).train_baseline(task,seed,method)
        elif method=='dralign':load('dralign',seed).run(task,seed,method)
        elif method=='mirrorfair':load('mirrorfair_common',seed).train(task,seed)
        else:raise ValueError(method)
    evaluate(task,seed,method)
    print('BASELINE DONE',task,seed,method,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=TASKS);p.add_argument('seed',type=int);p.add_argument('method',choices=METHODS);a=p.parse_args();run(a.task,a.seed,a.method)
