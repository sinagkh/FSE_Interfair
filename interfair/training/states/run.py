"""Resume-safe runner for the additional HMDA states (baseline adapters and InterFair)."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import os
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK']='1'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse, copy, hashlib, json, math, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
from scipy.special import expit
P=_PACKAGE / 'training/states'
sys.path.insert(0,str(P))
import is_trainer as t
from data import data, STATES, SIGNS
c=t.c;ee=t.ee
torch.set_num_threads(1)
BASELINES=('erm_source','ltdd','cot_phi','dralign','reweighing','hifi_shared')

def directory(task,seed,arm):return P/'runs'/c.phase(seed)/task/str(seed)/arm

def validate_freeze(seed):
    if seed>=2000:return
    j=json.loads((P/'FREEZE.json').read_text())
    for p,h in j['files'].items():assert c.sha(p)==h,('Frozen source changed',p)

def dralign(task,seed):
    import dralign as native
    from types import SimpleNamespace
    out=directory(task,seed,'dralign')
    if (out/'DONE.json').exists():return
    out.mkdir(parents=True,exist_ok=True)
    d,_,_=data(task,seed);source_path=directory(task,seed,'erm_source')/'selected.pt'
    source=c.load(source_path,d,'mlp','cpu');model=native.MLP(source)
    c.seed_all(seed);fn,_=native.native_function();writer=native.Writer();args=SimpleNamespace(epochs=20,pruning=False)
    opt=torch.optim.Adam(model.parameters(),lr=.001);tr=d['splits']['train']
    groups=[tr[d['s'][tr]==g] for g in (0,1)];batch=1000;steps=min(len(g)//batch for g in groups);assert steps>0
    rng=np.random.default_rng(seed+98710);bh=hashlib.sha256();history=[];best=None;start=time.monotonic()
    cfg=dict(task=task,seed=seed,architecture='mlp',arm='dralign',source=str(source_path),source_sha256=c.sha(source_path),
        adapter_sha256=c.sha(__file__),native_adapter_sha256=c.sha(native.__file__),native_sha256=c.sha(native.SOURCE),
        epochs=20,steps_per_epoch=steps,batch_per_group=1000,optimizer='Adam',lr=.001,
        endpoint_weight=.3,rationale_weight=.03,selector='minimum validation soft DP over epochs 3..19 (zero-based)',
        selection_uses_audit=False,device='cpu',initial_state_sha256=c.state_hash(source))
    c.write(out/'CONFIG.json',cfg)
    for epoch in range(20):
        items=[]
        for g in groups:
            use=rng.permutation(g)[:steps*batch];bh.update(use.tobytes())
            items.append([(torch.as_tensor(d['x'][ii]),torch.as_tensor(d['y'][ii])) for ii in use.reshape(steps,batch)])
        def validation(**kwargs):
            nonlocal best
            vi=d['splits']['val'];p=native.predict(model,d['x'][vi],'cpu')
            v=c.behavior(d['y'][vi],d['s'][vi],p);gap=abs(float(p[d['s'][vi]==0].mean()-p[d['s'][vi]==1].mean()))
            item=dict(epoch=epoch+1,soft_dp=gap,validation=v);history.append(item)
            if epoch>2 and (best is None or gap<best['soft_dp']):
                best=copy.deepcopy(item);source.net.load_state_dict(model.net.state_dict())
                torch.save(dict(state_dict=source.state_dict(),config=cfg,epoch=epoch+1),out/'selected.pt')
            return item.copy()
        fn(epoch,model,native.Loader(items[0]),native.Loader(items[0]),native.Loader(items[1]),mode='CAIGA',
           lam=.03,lam2=.3,args=args,criterion=torch.nn.BCELoss(),writer=writer,optimizer=opt,pretest_call=validation)
        c.write(out/'HISTORY.json',history)
    assert best is not None
    c.write(out/'DONE.json',dict(status='complete',config=cfg,selected=best,checkpoint=str(out/'selected.pt'),
        checkpoint_sha256=c.sha(out/'selected.pt'),optimizer_updates=20*steps,task_rows=40*steps*batch,
        batch_sha256=bh.hexdigest(),seconds=time.monotonic()-start,selection_completed_before_audit=True))

def baseline(task,seed,arm):
    if arm=='dralign':return dralign(task,seed)
    old_data=c.data;old_directory=c.directory;old_p=c.P
    try:
        c.data=lambda task,seed:data(task,seed)[:2]
        c.directory=lambda task,seed,arch,a,joint=False:directory(task,seed,'erm_source' if a=='erm' else a)
        c.P=P
        c.standard(task,seed,'mlp','erm' if arm=='erm_source' else arm,device='cpu')
    finally:c.data=old_data;c.directory=old_directory;c.P=old_p
    out=directory(task,seed,arm)
    c.write(out/'STATE_ADAPTER.json',dict(adapter_sha256=c.sha(__file__),data_adapter_sha256=c.sha(P/'data.py'),
        protocol_sha256=None,data_cache_sha256=c.sha(P/'cache'/f'{task}.pkl'),
        frozen_recipe=True,selection_uses_audit=False))

def query(task,seed,arm,device='cpu'):
    out=directory(task,seed,arm)
    if (out/'EVALUATION.json').exists():return
    d,b,db=data(task,seed);path=out/'selected.pt';m=c.load(path,d,'mlp',device)
    v,dr,ordinary=ee.evaluate(m,d,list(b),db,'audit',device=device,save=out/'audit_current.npz')
    c.write(out/'EVALUATION.json',dict(task=task,seed=seed,arm=arm,architecture='mlp',metrics=v,
        direction_features=dr,ordinary_features=ordinary,checkpoint=str(path),checkpoint_sha256=c.sha(path),
        arrays=str(out/'audit_current.npz'),arrays_sha256=c.sha(out/'audit_current.npz'),
        direction_arrays_sha256=c.sha(out/'DIRECTION_audit_current.npz'),
        data_cache_sha256=c.sha(P/'cache'/f'{task}.pkl'),selection_finished_before_audit=True))

def worker(task,seed,arm,device):
    validate_freeze(seed)
    if arm in ('erm','equality_direction'):t.train(task,seed,arm,device=device)
    else:baseline(task,seed,arm)
    query(task,seed,arm,device)

def launch(task,seed,arm,device):
    dest=P/'logs';dest.mkdir(exist_ok=True)
    log=dest/f'{task}_{seed}_{arm}.log'
    with log.open('a') as f:
        r=subprocess.run([sys.executable,__file__,'worker','--task',task,'--seed',str(seed),'--arm',arm,'--device',device],stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError(f'Worker failed: {log}')
    print('COMPLETE',task,seed,arm,flush=True)

def campaign(phase,gpu_workers=2,cpu_workers=8):
    seeds=(2000,2001) if phase=='development' else tuple(range(1000,1010))
    cells=[('hmda_'+state,seed) for state in STATES for seed in seeds]
    def gpu_cell(cell):
        task,seed=cell
        for arm in ('erm','equality_direction'):launch(task,seed,arm,'cuda')
    def cpu_cell(cell):
        task,seed=cell
        for arm in BASELINES:launch(task,seed,arm,'cpu')
    futures=[]
    with ThreadPoolExecutor(max_workers=gpu_workers) as gpu,ThreadPoolExecutor(max_workers=cpu_workers) as cpu:
        futures += [gpu.submit(gpu_cell,cell) for cell in cells]
        futures += [cpu.submit(cpu_cell,cell) for cell in cells]
        for f in as_completed(futures):f.result()
    print('CAMPAIGN COMPLETE',phase,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['worker','query','campaign'])
    ap.add_argument('--task');ap.add_argument('--seed',type=int);ap.add_argument('--arm')
    ap.add_argument('--device',default='cpu');ap.add_argument('--phase',default='development')
    ap.add_argument('--gpu-workers',type=int,default=2);ap.add_argument('--cpu-workers',type=int,default=8)
    a=ap.parse_args()
    if a.command=='worker':worker(a.task,a.seed,a.arm,a.device)
    elif a.command=='query':query(a.task,a.seed,a.arm,a.device)
    else:campaign(a.phase,a.gpu_workers,a.cpu_workers)
