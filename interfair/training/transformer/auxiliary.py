"""Frozen-source transfer and directional audits of joint/transfer comparators."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import argparse,concurrent.futures,json,pickle,subprocess,sys
import numpy as np
import pandas as pd
import ft_trainer as e
import baseline_directions as bd
P=e.P;ROOT=e.ROOT;c=e.c
OLD=ROOT/'training/direction/exports'

def target_data(block):
    bank=ROOT/'data/partitions/external'/f'{block}.pkl'
    d=pickle.load(bank.open('rb'))['data'];d['splits']={'audit':np.arange(len(d['x']))}
    d['metadata']={**d['metadata'],'task':'hmda_oh'}
    db={'audit':{n:d['banks']['audit'][n] for n in e.q.MAP['hmda_oh'] if not n.startswith('linked_')}}
    file=P/'banks'/(block+'_linked.pkl')
    if not file.exists():
        src,_,_=e.q.data('hmda_oh',2000)
        from support import fitted_support
        support=fitted_support(src);ids=d['banks']['audit']['income']['indices'];raw=d['frame'].iloc[ids].copy();enc=d['encoder']
        inc=raw.income.to_numpy(float);dti=raw.debt_to_income_ratio.to_numpy(float);new=dti*inc/(inc+10)
        reported=np.select([new<20,new<30,new<36,new<50,new<60],[19.,25.,33.,new,55.],default=65.)
        reported=np.where((new>=36)&(new<50),np.clip(np.round(new),36,49),reported)
        valid=np.isfinite(inc)&np.isfinite(dti)&(inc>=enc.stats['income']['q01'])&(inc+10<=enc.stats['income']['q99'])
        # These reported DTI values are exact percentages, not interval representatives.
        valid &= np.isin(dti,np.arange(36,50))
        high=raw.copy();high['income']=inc+10;high['debt_to_income_ratio']=reported
        corners=[]
        for s in (0,1):
            for base in (raw,high):
                cc=base.copy();cc['protected']=s;corners.append(enc.transform(cc))
        corners=np.stack(corners);keep=valid&support.mask(corners)[0]
        assert keep.sum()>=32,(block,int(keep.sum()))
        b=dict(corners=corners,indices=ids,valid=valid,supported=keep,training=True)
        with file.open('wb') as f:pickle.dump(b,f)
        c.write(file.with_suffix('.json'),dict(block=block,edit='linked_income_plus10k',initial_DTI='exact reported integers 36 through 49',source_fitted_support=True,valid=int(valid.sum()),supported=int(keep.sum()),target_training=False))
    db['audit']['linked_income_plus10k']=pickle.load(file.open('rb'))
    return d,list(d['banks']['audit']),db

def query(panel,task,arch,seed,arm):
    out=P/'auxiliary'/panel/task/arch/str(seed)/arm
    if (out/'DONE.json').exists():return
    out.mkdir(parents=True,exist_ok=True);joint=panel=='joint'
    if joint:d,b,db=e.data('hmda_oh',seed,True);names=list(b)
    else:d,names,db=target_data(task)
    adopted=arm in ['erm','soft']
    if adopted:
        if arm=='erm':path=e.reference_path('hmda_oh',seed,arch,joint or arch=='mlp')
        elif arch=='mlp' and not joint:path=e.SCRATCH/'runs/confirmation/hmda_oh'/str(seed)/'equality_direction/selected.pt'
        elif arch=='ft':
            from query_ft import model_path
            path=model_path('hmda_oh',seed)
        else:path=e.directory('hmda_oh',seed,arch,joint)/'selected.pt'
        assert path.exists(),path
        src,_,_=e.q.data('hmda_oh',seed);m=c.load(path,src,arch,'cpu')
        metrics,edits,features=e.evaluate(m,d,names,db,'audit',joint,'cpu',out/'audit.npz')
        row=dict(task=task,architecture=arch,seed=seed,arm=arm,block=task if not joint else None,joint=joint,
            checkpoint=str(path),checkpoint_sha256=c.sha(path),arrays=str(out/'audit.npz'),arrays_sha256=c.sha(out/'audit.npz'),**metrics)
        pd.DataFrame([{**row,**f} for f in features]).to_csv(out/'per_feature.csv',index=False)
    else:
        if joint:
            table=pd.read_csv(OLD/'joint_complete_per_seed.csv');rr=table[(table.seed==seed)&(table.arm==arm)].iloc[0]
            path=Path(rr.checkpoint);src,_,_=e.q.data('hmda_oh',seed);f=bd.Pipeline(c.load(path,src,arch,'cpu'),[path],'cpu')
        else:
            rr=bd.primary();rr=rr[(rr.task=='hmda_oh')&(rr.architecture==arch)&(rr.seed==seed)&(rr.arm==arm)].iloc[0]
            src,_,_=e.q.data('hmda_oh',seed);f=bd.pipeline(rr,src)
        hard=arm=='mirrorfair';edits=[];arr={}
        for name,b in db['audit'].items():
            keep=b['supported'];xx=b['corners'][:,keep];ncorner=xx.shape[0]
            pp=f(xx.reshape(-1,xx.shape[-1]),'P').reshape(ncorner,-1);z=None if hard else f(xx.reshape(-1,xx.shape[-1]),'L').reshape(ncorner,-1)
            sign=e.q.MAP['hmda_oh'].get(name,1)
            values=e.direction_metrics(z,sign,True) if joint else bd.metrics(pp,sign,z,hard)
            edits.append(dict(edit=name,training_edit=name in e.q.MAP['hmda_oh'],**values));arr[name+('_decision' if hard else '_P')]=pp;arr[name+'_indices']=b['indices'][keep]
            if z is not None:arr[name+'_L']=z
        np.savez_compressed(out/'corners.npz',**arr)
        edf=pd.DataFrame(edits);vals=edf[edf.training_edit].drop(columns=['edit','training_edit','n','sign']).mean().to_dict()
        row=dict(task=task,architecture=arch,seed=seed,arm=arm,output_kind='hard_decision' if hard else 'probability',**{'direction_'+k:v for k,v in vals.items()})
    base=dict(task=task,architecture=arch,seed=seed,arm=arm)
    pd.DataFrame([{**base,**r} for r in edits]).to_csv(out/'per_edit.csv',index=False)
    pd.DataFrame([row]).to_csv(out/'per_seed.csv',index=False)
    c.write(out/'DONE.json',dict(status='complete',target_fitting=False,target_selection=False,source_panel=panel,specification_sha256=c.sha(e.q.P/'DIRECTION_SPECIFICATIONS.json'),script_sha256=c.sha(__file__)))

def run(which):
    jobs=[]
    for panel in ['joint','transfer']:
        table=pd.read_csv(OLD/(panel+'_complete_per_seed.csv'))
        for row in table.itertuples():
            if which=='baselines' and row.arm in ['erm','soft']:continue
            if which=='method' and row.arm not in ['erm','soft']:continue
            jobs.append((panel,row.task,row.architecture,row.seed,row.arm))
    def worker(job):
        log=P/'logs'/('aux_'+'_'.join(map(str,job))+'.log')
        with log.open('w') as f:r=subprocess.run([sys.executable,__file__,'query',*map(str,job)],stdout=f,stderr=subprocess.STDOUT)
        return dict(job=job,returncode=r.returncode,log=str(log))
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(worker,jobs):
            results.append(r);c.write(P/('AUXILIARY_'+which+'_JOBS.json'),results)
            if r['returncode'] or len(results)%20==0:print(r if r['returncode'] else f'Complete {len(results)}/{len(jobs)}',flush=True)
    assert all(r['returncode']==0 for r in results)

if __name__=='__main__':
    if sys.argv[1]=='query':query(sys.argv[2],sys.argv[3],sys.argv[4],int(sys.argv[5]),sys.argv[6])
    elif sys.argv[1]=='prepare':
        for block in ['hmda_2019_mi','hmda_2020_oh']:target_data(block)
    else:run(sys.argv[1])
