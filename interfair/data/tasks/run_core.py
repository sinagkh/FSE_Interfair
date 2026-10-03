"""Data/path interface to the interaction-steering trainer and ERM."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import argparse
import torch
from task_context import *
from task_data import data_for
import repair as canonical

ORIGINAL_SOURCE = canonical.source_path

def source_path(task,seed,arch='mlp'):
    if task=='credit_broad':
        assert arch=='mlp'
        return ORIGINAL_SOURCE('default_credit',seed,arch)
    return HERE/'runs'/phase(seed)/'main'/arch/task/str(seed)/'erm'/('task.pt' if arch=='mlp' else 'selected.pt')

def native_training_data(d):
    # Native ERM does not consume rule gradients; retain the complete bank identity.
    bb=[b for b in d['banks']['train'].values() if b['training']]
    ix=np.concatenate([b['indices'][b['supported']] for b in bb])
    w=np.concatenate([np.full(int(b['supported'].sum()),1/(len(bb)*int(b['supported'].sum()))) for b in bb])
    mix=dict(corners=np.concatenate([b['corners'][:,b['supported']] for b in bb],axis=1),indices=ix,
             valid=np.ones(len(ix),bool),supported=np.ones(len(ix),bool),training=True,sampling_weights=w)
    return {**d,'banks':{**d['banks'],'train':{'selected_mixture':mix}}}

def install():
    canonical.HERE=HERE
    canonical.data_for=data_for
    canonical.source_path=source_path
    canonical.get_data=lambda task,seed,study='primary',arm='interfair': (lambda d:(d,{n:b for n,b in d['banks']['train'].items() if b['training']}))(data_for(task,seed))

def run(task,seed,arch,arm,device='cpu',strength=None):
    torch.set_num_threads(1)
    assert json.loads((HERE/'reports/R1_ADMISSION.json').read_text())['status']=='PASS'
    if phase(seed)=='confirmation':
        freeze=json.loads((HERE/'FREEZE.json').read_text())
        assert freeze['status']=='frozen'
        for path,h in freeze['files'].items():assert sha(path)==h,('frozen input changed',path)
    if arm=='erm':
        assert task=='acs_income','Compatible Credit ERM is reused, never refitted'
        d=data_for(task,seed);out=source_path(task,seed,arch).parent
        if (out/'DONE.json').exists():return
        if arch=='mlp':
            import training_base
            training_base.DEVICE=device
            training_base.train(d,'mlp',seed,'erm',out,interaction_enabled=False,score_scale='L')
        else:
            if device=='cpu':
                from baseline_adapters import load
                module=load('ft_architecture',seed)
            else:
                from method_adapters import load
                module=load('ft_architecture',HERE/'runs'/phase(seed),scope='ACSIncome native default FT ERM; no repair objective')
            module.train(task,seed,'erm',output_base=str(out.parents[2]),data_override=native_training_data(d))
        write(out/'COMPLETION_PROVENANCE.json',dict(adapter_sha256=sha(__file__),data_compiler_sha256=sha(HERE/'task_data.py'),data_sha256=sha(HERE/'cache'/f'{task}.pkl'),model_scope='new ACSIncome task; native ERM training'))
        return
    assert arm in ('interfair','control')
    install()
    result=canonical.train(task,seed,arch,'primary',arm,device,strength)
    out=HERE/'runs'/phase(seed)/'primary'/arch/task/str(seed)/(arm if strength in (None,1) else arm+f'_strength{strength:g}')
    write(out/'COMPLETION_PROVENANCE.json',dict(adapter_sha256=sha(__file__),canonical_trainer_sha256=sha(CORRECTED/'repair.py'),data_compiler_sha256=sha(HERE/'task_data.py'),data_sha256=sha(HERE/'cache'/f'{task}.pkl'),
                                              overrides=['data provider','output root','ERM reference path','study label primary; separate expanded-bank evaluator'],loss_optimizer_and_selection_changed=False))
    print('CORE DONE',task,seed,arch,arm,result['admitted'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=TASKS);p.add_argument('seed',type=int);p.add_argument('arch',choices=['mlp','ft']);p.add_argument('arm',choices=['erm','interfair','control']);p.add_argument('--device',default='cpu');p.add_argument('--strength',type=float);a=p.parse_args();run(a.task,a.seed,a.arch,a.arm,a.device,a.strength)
