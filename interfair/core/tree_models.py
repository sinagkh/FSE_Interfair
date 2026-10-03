"""native-default histogram trees, whole-pipeline query audit only."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[1]
import argparse,json,os,pickle,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
import torch
from core import ROOT,save_json,sha,evaluate
from training_base import prepare
from feature_edits import banks
from credit import prepare as default_data

class TreeScore(torch.nn.Module):
    def __init__(self,tree,removed=False):
        super().__init__();self.tree=tree;self.removed=removed
    def score(self,x,scale):
        z=x.detach().cpu().numpy().copy()
        if self.removed:z[:,0]=0
        p=torch.from_numpy(self.tree.predict_proba(z)[:,1])
        return p if scale=='P' else torch.logit(p.clamp(1e-7,1-1e-7))

class UniformScore(torch.nn.Module):
    def __init__(self,base):super().__init__();self.base=base
    def score(self,x,scale):
        a=x.clone();b=x.clone();a[:,0]=0;b[:,0]=1
        p=(self.base.score(a,'P').double()+self.base.score(b,'P').double())/2
        return p if scale=='P' else torch.logit(p.clamp(1e-7,1-1e-7))

def data_for(task):
    if task=='default_credit':return default_data()
    d=prepare(task);d['banks']=banks(task)['banks'];return d

def job(task,seed):
    assert task in ('adult','hmda_oh','default_credit') and seed in (2000,2001)
    torch.set_num_threads(1);data=data_for(task);out=ROOT/'runs/tree_query'/task/str(seed);out.mkdir(parents=True,exist_ok=True)
    if (out/'DONE.json').exists():return
    ix=data['splits']['train'];results={};models={};config=dict(task=task,seed=seed,sklearn=sklearn.__version__,params=HistGradientBoostingClassifier(random_state=seed).get_params(),encoder=data['encoder'].metadata(),protocol_sha256=None,code_sha256=sha(__file__),training_rows=len(ix),outer_tuning=False,gradient_repair=False,selector='native internal early stopping on training only',audit_used_for_selection=False)
    save_json(out/'CONFIG.json',config)
    for arm in ('erm','removed'):
        x=data['x'][ix].copy()
        if arm=='removed':x[:,0]=0
        tree=HistGradientBoostingClassifier(random_state=seed).fit(x,data['y'][ix]);assert np.array_equal(tree.classes_,[0,1])
        with (out/f'{arm}.pkl').open('wb') as f:pickle.dump(tree,f,pickle.HIGHEST_PROTOCOL)
        save_json(out/f'{arm}_FROZEN.json',dict(model_sha256=sha(out/f'{arm}.pkl'),iterations=int(tree.n_iter_),early_stopping=bool(tree.do_early_stopping_),finished_before_audit=True))
        models[arm]=TreeScore(tree,arm=='removed')
    models['uniform_erm']=UniformScore(models['erm'])
    for arm,m in models.items():results[arm]=evaluate(m,data,'audit','cpu',save_path=out/f'audit_{arm}.npz')
    save_json(out/'DONE.json',dict(config=config,results=results,status='completed',claim_scope='development query audit; no learned tree repair'))
    print(json.dumps(dict(task=task,seed=seed,auc=results['erm']['behavior']['auc'],residual=results['erm']['primary_P'])),flush=True)

def queue():
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    logs=ROOT/'job_logs/tree_query';logs.mkdir(parents=True,exist_ok=True)
    def launch(j):
        with (logs/f'{j[0]}_{j[1]}.log').open('w') as f:r=subprocess.run([sys.executable,__file__,'--job',*map(str,j)],stdout=f,stderr=subprocess.STDOUT,env=env)
        assert r.returncode==0,j
        print(json.dumps(dict(completed=j)),flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(launch,[(t,s) for t in ('adult','hmda_oh','default_credit') for s in (2000,2001)]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',nargs=2);args=p.parse_args()
    if args.job:job(args.job[0],int(args.job[1]))
    else:queue()
