"""Primary, external-population and held-out-edit exports for completed arms."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[3]
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,json,hashlib,subprocess,argparse,time
from concurrent.futures import ThreadPoolExecutor,as_completed
P=_PACKAGE / 'baselines/evaluation/rescore';STUDY=_PACKAGE / 'baselines/evaluation';ROOT=_PACKAGE
sys.path.insert(0,str(P));import score as S
import numpy as np,pandas as pd
sys.path.insert(0,str(STUDY/'ft'));import runner as r
c=r.c
METRICS=['auc','accuracy','f1','aod','dp','eomax','L_R','violation_005','decision_disagreement','direction_adverse_005']

def primary(run):
    run=Path(run);done=json.loads((run/'DONE.json').read_text());cfg=done['config'];task=cfg['task'];arch=cfg['architecture'];arm=cfg['arm'];seed=cfg['seed']
    track='ft' if arch=='ft' else 'employment';out=P/track/task/str(seed)/arm;out.mkdir(parents=True,exist_ok=True)
    if (out/'CHECKS.json').exists():return
    assert c.sha(run/'selected.pt')==done['checkpoint_sha256']
    d=S.ex.taskdata(task);a=np.load(run/'audit.npz')
    for name,b in d['banks']['audit'].items():
        np.testing.assert_array_equal(a[name+'_indices'],b['indices']);np.testing.assert_array_equal(a[name+'_supported'],b['supported'])
    np.testing.assert_array_equal(a['natural_indices'],d['splits']['audit'])
    m=c.load(run/'selected.pt',d,arch,'cpu');x=d['x'][d['splits']['audit'][:128]]
    replay=S.query(m,x,batch=512);err=float(np.max(abs(replay-a['natural_L'][:128])));assert err<2e-5,err
    row,ff,pe=S.score(task,run/'audit.npz',m)
    for k in ['auc','aod','L_R','violation_005','decision_disagreement']:assert abs(row[k]-done['audit'][k])<1e-7,(k,row[k],done['audit'][k])
    identity=dict(task=task,architecture=arch,arm=arm,seed=seed)
    row=dict(**identity,**row,checkpoint=str(run/'selected.pt'),checkpoint_sha256=done['checkpoint_sha256'],
       source_done=str(run/'DONE.json'),source_sha256=c.sha(run/'DONE.json'),arrays=str(run/'audit.npz'),arrays_sha256=c.sha(run/'audit.npz'),
       source_checkpoint=cfg['source_checkpoint'],source_checkpoint_sha256=cfg['source_checkpoint_sha256'],
       training_seconds=done['seconds'],protocol=cfg['optimizer']+'; '+cfg['selector'])
    assert all(np.isfinite(row[k]) for k in METRICS)
    pd.DataFrame([row]).to_csv(out/'per_seed.csv',index=False)
    for f,name in [(ff,'per_feature'),(pe,'per_edit')]:
        for k,v in identity.items():f[k]=v
        f.to_csv(out/(name+'.csv'),index=False)
    c.write(out/'CHECKS.json',dict(status='PASS',primary_banks_identical=True,selected_checkpoint_verified=True,replay_max_logit_error=err,selection_uses_audit=False))

def transfer(run,block):
    import auxiliary as A
    run=Path(run);cfg=json.loads((run/'CONFIG.json').read_text());seed=cfg['seed'];arm=cfg['arm'];arch=cfg['architecture']
    out=P/'ft_transfer'/block/arch/str(seed)/arm;out.mkdir(parents=True,exist_ok=True)
    if (out/'CHECKS.json').exists():return
    d,names,db=A.target_data(block);src,_=c.data('hmda_oh',seed);m=c.load(run/'selected.pt',src,arch,'cpu')
    metrics,edits,features=A.e.evaluate(m,d,names,db,'audit',False,'cpu',out/'audit.npz')
    row=dict(task=block,architecture=arch,seed=seed,arm=arm,block=block,checkpoint=str(run/'selected.pt'),
      checkpoint_sha256=c.sha(run/'selected.pt'),arrays=str(out/'audit.npz'),arrays_sha256=c.sha(out/'audit.npz'),**metrics)
    pd.DataFrame([row]).to_csv(out/'per_seed.csv',index=False)
    ident=dict(task=block,architecture=arch,seed=seed,arm=arm)
    pd.DataFrame([{**ident,**f} for f in features]).to_csv(out/'per_feature.csv',index=False)
    pd.DataFrame([{**ident,**f} for f in edits]).to_csv(out/'per_edit.csv',index=False)
    c.write(out/'CHECKS.json',dict(status='PASS',target_fitting=False,target_selection=False,source_checkpoint_verified=True))

def income(arm,seed):
    out=P/'income'/str(seed)/arm;out.mkdir(parents=True,exist_ok=True)
    if (out/'CHECKS.json').exists():return
    d,_,db=S.q.data('hmda_oh',seed);src=ROOT/'studies/learning/runs/hmda_oh'/arm/str(seed)/'selected.pt'
    m=c.load(src,d,'mlp','cpu');rows=[];arrays={}
    for name,b in db['audit'].items():
        if not name.startswith('linked_'):continue
        keep=b['supported'];ids=b['indices'][keep];xx=b['corners'][:,keep]
        z=S.query(m,xx.reshape(-1,xx.shape[-1])).reshape(4,-1);p=S.expit(z)
        vals={'direction_'+k:v for k,v in S.bd.metrics(p,1,z,False).items() if k not in ('n','sign')}
        vals.update(c.effect_metrics(z,ids,d));n=vals.pop('n')
        rows.append(dict(task='hmda_'+name.removeprefix('linked_income_'),source_task='hmda_oh',architecture='mlp',seed=seed,arm=arm,edit=name,n=n,output_kind='probability',**vals))
        arrays[name+'_L']=z;arrays[name+'_P']=p;arrays[name+'_indices']=ids
    assert len(rows)==4
    np.savez_compressed(out/'corners.npz',**arrays);pd.DataFrame(rows).to_csv(out/'per_seed.csv',index=False)
    c.write(out/'CHECKS.json',dict(status='PASS',checkpoint=str(src),checkpoint_sha256=c.sha(src),target_fitting=False,target_selection=False))

def campaign(kind,workers):
    if kind in ('ft','employment'):
        jobs=[['primary',str(f.parent)] for f in sorted((STUDY/kind/'runs/confirmation').glob('*/*/*/*/DONE.json'))]
    elif kind=='ft_transfer':
        jobs=[['transfer',str(f.parent),b] for f in sorted((STUDY/'ft/runs/confirmation/ft/hmda_oh').glob('*/*/DONE.json')) for b in ('hmda_2019_mi','hmda_2020_oh')]
    else:jobs=[['income',a,str(s)] for a in ('fairsmote','reweighing') for s in range(1000,1010)]
    logs=P/'logs';logs.mkdir(exist_ok=True)
    def launch(item):
        log=logs/(kind+'_'+hashlib.sha256('|'.join(item).encode()).hexdigest()[:12]+'.log')
        with log.open('a') as stream:ret=subprocess.run([sys.executable,__file__,'worker',*item],stdout=stream,stderr=subprocess.STDOUT)
        return dict(job=item,returncode=ret.returncode,log=str(log))
    results=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(launch,j) for j in jobs]):
            rec=f.result();results.append(rec)
            if rec['returncode'] or len(results)%10==0:print(kind,len(results),'/',len(jobs),rec if rec['returncode'] else '',flush=True)
    c.write(P/(kind+'_JOBS.json'),results);assert all(x['returncode']==0 for x in results)
    files=sorted((P/kind).glob('**/per_seed.csv'));out=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    if kind=='employment':out=out[out.arm!='fairsmote_reference']
    assert not out.duplicated(['task','architecture','arm','seed']).any()
    out.to_csv(P/kind/'per_seed_all.csv',index=False)
    print('EXPORTED',kind,len(out),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='worker':
        if sys.argv[2]=='primary':primary(sys.argv[3])
        elif sys.argv[2]=='transfer':transfer(sys.argv[3],sys.argv[4])
        else:income(sys.argv[3],int(sys.argv[4]))
    else:campaign(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 8)
