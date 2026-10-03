"""Query adopted checkpoints and save coherent response/direction metrics."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
from engine import *

def current_path(task,seed,architecture,arm):
    if architecture=='mlp':return source(task,seed,'erm' if arm=='erm' else 'equality_direction')
    if arm=='soft':
        reg=read(ROOT/'training/transformer/FT_SELECTION.json')
        return Path(reg['tasks'][task]['run_root'])/'runs/confirmation/main/ft'/task/str(seed)/'soft/selected.pt'
    return c.source_path(task,seed,'ft')

def fit(task,seed,arm,kind):
    if kind=='engine':
        from engine import evaluate
        return evaluate(task,seed,arm)
    import ablation as a
    out=a.directory(task,seed,arm);dest=P/'ablation_evaluation'/c.phase(seed)/task/str(seed)/arm
    if (dest/'DONE.json').exists():return
    j=read(out/'DONE.json');d,b,db=q.data(task,seed);m=c.load(out/'selected.pt',d,'mlp','cpu');dest.mkdir(parents=True,exist_ok=True)
    metrics,edits=q.evaluate(m,d,list(b),db,'audit',dest/'audit.npz')
    row=dict(task=task,architecture='mlp',seed=seed,arm=arm,checkpoint=j['checkpoint'],arrays=str(dest/'audit.npz'),admitted=j['selected']['admitted'],**metrics)
    pd.DataFrame([row]).to_csv(dest/'per_seed.csv',index=False)
    pd.DataFrame([{**{k:row[k] for k in ['task','architecture','seed','arm']},**r} for r in edits]).to_csv(dest/'per_edit.csv',index=False)
    write(dest/'DONE.json',dict(status='complete',checkpoint_sha256=j['checkpoint_sha256'],selection_before_audit=True))

def alternate(task,seed,arch):
    out=P/'additional_edits'/arch/task/str(seed)
    if (out/'DONE.json').exists():return
    d,b,db=q.data(task,seed);out.mkdir(parents=True,exist_ok=True);rows=[];sources=[]
    for arm in ('erm','soft'):
        path=current_path(task,seed,arch,arm);assert (path.parent/'DONE.json').exists(),path
        m=c.load(path,d,arch,'cpu');arr={}
        for name,bank in d['banks']['audit'].items():
            keep=bank['supported'];ids=bank['indices'][keep];xx=bank['corners'][:,keep]
            z=c.logits(m,xx.reshape(-1,xx.shape[-1]),'cpu').reshape(4,-1)
            sign=q.MAP[task].get(name)
            # Additional edits share the prefix of their declared semantic block.
            if sign is None:
                candidates=[n for n in q.MAP[task] if name.startswith(n+'__') or name.startswith(n+'_')]
                if len(candidates)==1:sign=q.MAP[task][candidates[0]]
            metrics=c.effect_metrics(z,ids,d)
            if sign is not None:metrics.update({'direction_'+k:v for k,v in q.direction_metrics(z,sign).items() if k not in ['n','sign']})
            rows.append(dict(task=task,architecture=arch,seed=seed,arm=arm,feature=name,training=bool(bank['training']),signed=sign is not None,checkpoint=str(path),**metrics))
            arr[name+'_L']=z;arr[name+'_indices']=ids
        np.savez_compressed(out/(arm+'.npz'),**arr);sources.append(dict(arm=arm,checkpoint=str(path),sha256=c.sha(path)));del m
    df=pd.DataFrame(rows);df.to_csv(out/'per_feature.csv',index=False)
    keys=['task','architecture','seed','arm','training'];cols=[k for k in df.select_dtypes(include=np.number) if k not in ['seed','training','signed']]
    df.groupby(keys)[cols].mean().reset_index().to_csv(out/'per_seed.csv',index=False)
    write(out/'DONE.json',dict(status='complete',sources=sources,selection_before_audit=True))

def race_joint(seed):
    out=P/'race_trained_joint'/str(seed)
    if (out/'DONE.json').exists():return
    sys.path.insert(0,str(ROOT/'training/transformer'));import ft_trainer as e
    d,b,db=e.data('hmda_oh',seed,True);out.mkdir(parents=True,exist_ok=True);rows=[]
    for arm in ('erm','soft'):
        path=current_path('hmda_oh',seed,'mlp',arm);m=c.load(path,d,'mlp','cpu')
        v,edits,features=e.evaluate(m,d,list(b),db,'audit',True,'cpu',out/(arm+'.npz'))
        rows.append(dict(task='hmda_oh',architecture='mlp',seed=seed,arm=arm,checkpoint=str(path),**v))
    pd.DataFrame(rows).to_csv(out/'per_seed.csv',index=False);write(out/'DONE.json',dict(status='complete',joint_fitting=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['fit','alternate','race_joint']);p.add_argument('--task');p.add_argument('--seed',type=int);p.add_argument('--arm');p.add_argument('--kind',default='engine');p.add_argument('--architecture',default='mlp');a=p.parse_args()
    if a.command=='fit':fit(a.task,a.seed,a.arm,a.kind)
    elif a.command=='alternate':alternate(a.task,a.seed,a.architecture)
    else:race_joint(a.seed)
