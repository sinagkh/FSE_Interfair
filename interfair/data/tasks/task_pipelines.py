"""Full prediction pipelines, including preprocessing and native hard decisions."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
from types import SimpleNamespace
import torch
from task_context import *
from task_data import data_for
from run_core import source_path
from model_interfaces import Pipeline, Uniform, plain, load_pipeline

METHODS=('ltdd','cot_phi','dralign','mirrorfair')

def baseline_root(task,seed):
    if task=='credit_broad' and seed in DEVELOPMENT:
        return BREADTH/'runs','default_credit'
    return HERE/'runs'/phase(seed)/'baselines',task

def baseline_directory(task,seed,method):
    root,t=baseline_root(task,seed)
    return root/('preprocessing_baselines' if method in ('ltdd','cot_phi') else 'dralign' if method=='dralign' else 'mirrorfair_common')/t/str(seed)/(method if method!='mirrorfair' else '')

def favored_group(d):
    ix=d['splits']['train'];rates=[float(d['y'][ix][d['s'][ix]==g].mean()) for g in (0,1)]
    return int(rates[1]>rates[0]),rates

class NativeMirror:
    output_kind='hard_decision';queries_per_row=2
    def __init__(self,d,task,seed):
        from mirrorfair_source import literal
        directory=baseline_directory(task,seed,'mirrorfair')
        self.source=plain(source_path(task,seed),d);self.mirror=plain(directory/'selected.pt',d)
        self.path=directory/'PAIR_SELECTION.json';self.scenario=json.loads(self.path.read_text())['scenario']
        self.favored,_=favored_group(d);self.select,self.combine=literal()
        v=np.load(directory/'validation_rules.npz');assert np.array_equal(v['g'],(d['s'][v['indices']]==self.favored).astype(int))
        assert self.select(np.c_[1-v['p'],v['p']],np.c_[1-v['q'],v['q']],SimpleNamespace(protected_attributes=v['g'][:,None]))==self.scenario
        score,decision=self.combine(np.c_[1-v['p'],v['p']],np.c_[1-v['q'],v['q']],SimpleNamespace(protected_attributes=v['g'][:,None]),self.scenario,False)
        assert np.array_equal(score,v['combined_score']) and np.array_equal(decision,v['decisions'])
    def __call__(self,x,scale='P'):
        assert scale=='P','Native MirrorFair has no probability or logit output'
        p,q=self.source(x),self.mirror(x);g=(x[:,0]==self.favored).astype(int)
        _,decision=self.combine(np.c_[1-p,p],np.c_[1-q,q],SimpleNamespace(protected_attributes=g[:,None]),self.scenario,False)
        return decision.astype(np.float64)
    def provenance(self):
        return dict(output_kind=self.output_kind,queries_per_row=2,scenario=self.scenario,favored_common_group=self.favored,
                    source=self.source.provenance(),mirror=self.mirror.provenance(),selection_sha256=sha(self.path))

def pipeline(task,seed,arm,d=None,arch='mlp',device='cpu'):
    d=data_for(task,seed) if d is None else d
    if arm=='averaging':return Uniform(pipeline(task,seed,'erm',d,arch,device))
    if arm=='erm':
        path=source_path(task,seed,arch)
        if arch=='mlp':return plain(path,d,device)
        from ft_architecture import load_ft
        return Pipeline(load_ft(path,d,device),[path],device,batch=512)
    if arm in METHODS:
        if arm=='mirrorfair':return NativeMirror(d,task,seed)
        root,t=baseline_root(task,seed)
        return load_pipeline(root,'named',t,seed,arm,d,device=device)
    base=HERE/'runs'/phase(seed)/'primary'/arch/task/str(seed)
    selected=base/'METHOD_SELECTION.json'
    folder=Path(json.loads(selected.read_text())['paths'][arm]) if selected.exists() else base/arm
    j=json.loads((folder/'DONE.json').read_text())
    import repair
    return repair.load_corrected(Path(j['checkpoint']),d,device)

def evaluate(task,seed,arm,arch='mlp',tag='main'):
    torch.set_num_threads(1)
    d=data_for(task,seed);pipe=pipeline(task,seed,arm,d,arch)
    out=HERE/'evaluations'/phase(seed)/tag/arch/task/str(seed)/arm
    provenance=pipe.provenance();identity=dict(pipeline=provenance,data_sha256=sha(HERE/'cache'/f'{task}.pkl'))
    if (out/'DONE.json').exists():
        old=json.loads((out/'DONE.json').read_text());assert old['identity']==identity;return old
    from pipeline_audit import audit
    report,arrays=audit(pipe,d)
    out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'PREDICTIONS.npz',**arrays)
    result=dict(status='completed',task=task,seed=seed,arm=arm,architecture=arch,identity=identity,results=report,code_sha256=sha(__file__))
    write(out/'DONE.json',result);print('AUDITED',task,seed,arch,arm,flush=True)
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('task',choices=TASKS);p.add_argument('seed',type=int);p.add_argument('arm');p.add_argument('--arch',default='mlp');p.add_argument('--tag',default='main');a=p.parse_args();evaluate(a.task,a.seed,a.arm,a.arch,a.tag)
