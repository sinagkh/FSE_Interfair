"""Train the paper's ERM and full InterFair from fresh weights on packaged inputs."""
from pathlib import Path
import sys,os,argparse,json,shutil,importlib.util,hashlib
from inputs import ROOT,CODE,load,bootstrap

def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def relocate(m,out):
    original=m.P;m.P=out;out.mkdir(parents=True,exist_ok=True)
    for name in ['data.py']:
        if (original/name).is_file():shutil.copyfile(original/name,out/name)
    files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE.rglob('*.py')}
    for name in ['FREEZE.json','FREEZE_ft.json','FREEZE_joint.json']:
        (out/name).write_text(json.dumps({'files':files,'scope':'checksums for the packaged source modules; scientific settings are in CONFIG.json'},indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True,choices=['hmda_oh','hmda_md','hmda_va','hmda_pa','credit_broad','acs_income','acs_employment_sex','acs_employment_age'])
    p.add_argument('--architecture',choices=['mlp','ft'],default='mlp');p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--method',choices=['erm','is','both'],default='both');p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--smoke',action='store_true',help='One-epoch MLP execution check, written separately; not a reported experiment.')
    p.add_argument('--out',type=Path,default=ROOT/'retrained');a=p.parse_args()
    if a.architecture=='ft' and a.task not in ['hmda_oh','acs_income']:p.error('The paper uses FT for HMDA Ohio and ACSIncome only.')
    if a.smoke and a.architecture!='mlp':p.error('Use MLP for the short smoke test.')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES');s=bootstrap()
    if a.device=='cuda':
        if visible is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=visible
    if a.architecture=='mlp':
        if a.task in ['hmda_md','hmda_va','hmda_pa']:
            s=module(CODE/'training/states/is_trainer.py','paper_state_trainer');s.q.data=load
        elif a.device!='cpu':p.error('Use --device cpu for the original five MLP tasks, matching the paper.')
        relocate(s,a.out.resolve()/'mlp')
        if a.task.startswith('hmda_') and a.task!='hmda_oh':
            (s.P/'cache').mkdir(exist_ok=True)
            meta=json.loads((ROOT/'data/metadata.json').read_text())[a.task]
            (s.P/'cache'/f'{a.task}.json').write_text(json.dumps(meta,indent=2))
        arms=['erm','equality_direction'] if a.method=='both' else ['erm' if a.method=='erm' else 'equality_direction']
        for arm in arms:
            kw={'smoke_epochs':1 if a.smoke else None}
            if a.task in ['hmda_md','hmda_va','hmda_pa']:kw['device']=a.device
            s.train(a.task,a.seed,arm,**kw)
        print('Training complete:',s.P)
    else:
        c=s.c;relocate(c,a.out.resolve()/'ft_reference')
        if a.method in ['erm','both']:c.standard(a.task,a.seed,'ft','erm',a.device)
        ref=c.directory(a.task,a.seed,'ft','erm')/'selected.pt'
        if a.method in ['is','both']:
            path=CODE/'training/transformer'/('hmda/ft_trainer.py' if a.task=='hmda_oh' else 'ft_trainer.py')
            e=module(path,'paper_ft_trainer');relocate(e,a.out.resolve()/'ft_is')
            e.q.data=load;e.reference_path=lambda *args:ref
            e.train(a.task,a.seed,'ft',device=a.device)
        print('Training complete:',a.out)
if __name__=='__main__':main()
