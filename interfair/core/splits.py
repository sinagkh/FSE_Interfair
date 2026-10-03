"""Bounded development admission of disjoint calibration and larger FT banks."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[1]
import argparse,copy,hashlib,json,os,pickle,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from scipy.special import expit
from core import ROOT,Predictor,predict,behavior,effects,save_json,sha
from anchors import budget_data
from credit import prepare as credit_data,train_data as credit_train
from calibration_audit import fit_temperature,ece,nll
import training_base

BASE=_PACKAGE / 'core/runs/splits'
REPORT=_PACKAGE / 'core/reports/splits'
PROTOCOL=_PACKAGE / 'core/reports/FINAL_PROTOCOL_PREFLIGHT_PROTOCOL.md'
TASKS=('adult','hmda_oh','default_credit')
ARMS=('erm','control_preserve','interfair_preserve')

def reserve(data,task):
    old=np.asarray(data['splits']['val']);probe=np.unique(np.concatenate([b['indices'] for b in data['banks']['val'].values()]));assert np.isin(probe,old).all()
    if task=='default_credit':
        groups=np.load(ROOT/'cache_v2/default_credit_splits.npz')['profile'];unique=np.unique(groups[old]);protected_groups=np.unique(groups[probe]);available=np.setdiff1d(unique,protected_groups);ncal=round(.2*len(unique))
        strata=[]
        for group in available:
            ix=old[groups[old]==group];assert len(set(data['s'][ix]))==1;strata.append(2*data['s'][ix[0]]+float(data['y'][ix].mean()>=.5))
        _,chosen=train_test_split(available,test_size=ncal,stratify=strata,random_state=73141);cal=old[np.isin(groups[old],chosen)]
    else:
        groups=np.arange(len(data['y']));available=np.setdiff1d(old,probe);ncal=round(.2*len(old));_,cal=train_test_split(available,test_size=ncal,stratify=2*data['s'][available]+data['y'][available],random_state=73141)
    cal=np.sort(cal);selection=np.setdiff1d(old,cal);parts=dict(train=data['splits']['train'],selection=selection,calibration=cal,audit=data['splits']['audit'])
    checks=[]
    import itertools
    for a,b in itertools.combinations(parts,2):
        assert not np.intersect1d(parts[a],parts[b]).size
        assert not np.intersect1d(groups[parts[a]],groups[parts[b]]).size
        checks.append(dict(check='partition/group disjoint',partitions=[a,b],passed=True))
    assert np.array_equal(np.sort(np.concatenate([selection,cal])),old) and np.isin(probe,selection).all()
    counts={k:{f'{s}/{y}':int(((data['s'][ix]==s)&(data['y'][ix]==y)).sum()) for s in (0,1) for y in (0,1)} for k,ix in parts.items()}
    assert min(counts['calibration'].values())>=10,counts
    checks.append(dict(check='complete original validation split; all probe contexts remain selection-only; calibration has each group/label cell',passed=True))
    data['splits']['val']=selection;data['splits']['calibration']=cal
    hashes={k:hashlib.sha256(ix.tobytes()).hexdigest() for k,ix in parts.items()}
    record=dict(task=task,split_rng=73141,original_validation_rows=len(old),partition_rows={k:len(ix) for k,ix in parts.items()},partition_group_counts={k:len(np.unique(groups[ix])) for k,ix in parts.items()},group_label_counts=counts,hashes=hashes,validation_probe_contexts=len(probe),checks=checks)
    data['metadata']['selection_calibration_roles']=record
    return data,record,parts

def data_for(task,seed):
    assert seed in (2000,2001)
    path=ROOT/'cache_v2'/f'{task}_final_preflight_{seed}.pkl'
    if path.exists():
        with path.open('rb') as f:return pickle.load(f)
    data=credit_train(credit_data()) if task=='default_credit' else budget_data(task,seed,8192)
    original_train=data['splits']['train'].copy();original_audit=data['splits']['audit'].copy();encoder_hash=hashlib.sha256(json.dumps(data['encoder'].metadata(),sort_keys=True).encode()).hexdigest();data,record,parts=reserve(data,task)
    assert np.array_equal(data['splits']['train'],original_train) and np.array_equal(data['splits']['audit'],original_audit)
    assert hashlib.sha256(json.dumps(data['encoder'].metadata(),sort_keys=True).encode()).hexdigest()==encoder_hash
    record.update(unchanged_train_encoder_audit=True,encoder_sha256=encoder_hash,steering_rectangles=data['banks']['train']['selected_mixture']['corners'].shape[1],seed=seed,recipe='old Default Credit candidate bank' if task=='default_credit' else '8192supported training anchors per ordinary feature')
    with path.open('wb') as f:pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    REPORT.mkdir(parents=True,exist_ok=True);save_json(REPORT/f'{task}_{seed}_DATA.json',record);np.savez_compressed(REPORT/f'{task}_{seed}_partitions.npz',**parts)
    return data

def gate():
    REPORT.mkdir(parents=True,exist_ok=True);records=[]
    for task in TASKS:
        for seed in (2000,2001):
            d=data_for(task,seed);r=json.loads((REPORT/f'{task}_{seed}_DATA.json').read_text());records.append(r)
            b=d['banks']['train']['selected_mixture'];assert b['corners'].shape[0]==4 and b['supported'].all() and abs(b['sampling_weights'].sum()-1)<1e-10
        a=json.loads((REPORT/f'{task}_2000_DATA.json').read_text());b=json.loads((REPORT/f'{task}_2001_DATA.json').read_text());assert a['hashes']==b['hashes']
    paths=['splits.py','reports/FINAL_PROTOCOL_PREFLIGHT_PROTOCOL.md','core.py','training_base.py','ft_architecture.py','anchors.py','calibration_audit.py','credit.py']
    save_json(REPORT/'GATE.json',dict(status='PASS',data=records,source_hashes={p:sha(ROOT/p) for p in paths},cache_hashes={t+'_'+str(s):sha(ROOT/'cache_v2'/f'{t}_final_preflight_{s}.pkl') for t in TASKS for s in (2000,2001)},final_confirmation_started=False));print('preflight data gate PASS',flush=True)

def verify():
    r=json.loads((REPORT/'GATE.json').read_text());assert r['status']=='PASS'
    for p,h in r['source_hashes'].items():assert sha(ROOT/p)==h,p

def worker(task,seed,arch,arm):
    verify();assert seed in (2000,2001) and arm in ARMS and arch in ('mlp','ft');assert arch=='mlp' or task!='default_credit';torch.set_num_threads(1);d=data_for(task,seed)
    out=BASE/arch/task/str(seed)/arm
    if (out/'DONE.json').exists():return
    if arch=='mlp':
        src=out.parent/'erm/task.pt';reference=json.loads((out.parent/'erm/DONE.json').read_text())['selections']['task']['validation'] if arm!='erm' else None
        training_base.train(d,'mlp',seed,arm,out,source=src if arm!='erm' else None,reference=reference,weight=160,preserve_weight=160 if arm!='erm' else 0,selection_auc_budget=.005,selection_f1_ratio=.99,score_scale='P',interaction_enabled=arm=='interfair_preserve')
    else:
        from ft_architecture import train
        native_arm='all_preserve' if arm=='interfair_preserve' else arm
        train(task,seed,native_arm,output_base='runs/splits/ft',data_override=d)
        # Keep native producer path/name intact; the interface ledger maps it.
    print('preflight fitted',task,seed,arch,arm,flush=True)

def analyze_cell(task,seed,arch):
    verify();torch.set_num_threads(1);d=data_for(task,seed);device='cuda' if torch.cuda.is_available() else 'cpu';paths={};results={};arrays={};teacher=None;ref_behavior=None
    for arm in ARMS:
        native='all_preserve' if arch=='ft' and arm=='interfair_preserve' else arm
        path=BASE/arch/task/str(seed)/native/('selected.pt' if arch=='ft' else 'task.pt' if arm=='erm' else 'P.pt');paths[arm]=sha(path)
        if arch=='ft':
            from ft_architecture import load_ft
            m=load_ft(path,d,device)
        else:
            m=Predictor(d['x'].shape[1]);m.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict']);m=m.to(device).eval()
        batch=512 if arch=='ft' else 4096
        cal=d['splits']['calibration'];z=predict(m,d['x'][cal],device,'L',batch).astype(float);fit=fit_temperature(z,d['y'][cal]);save_json(path.parent/'DISJOINT_CALIBRATION.json',dict(**fit,indices_sha256=hashlib.sha256(cal.tobytes()).hexdigest(),role='calibration only; excluded from source and repair checkpoint selection',source_sha256=paths[arm],fit_finished_before_calibrated_audit=True))
        ai=d['splits']['audit'];l=predict(m,d['x'][ai],device,'L',batch).astype(float);p=predict(m,d['x'][ai],device,'P',batch).astype(float);native_b=behavior(d['y'][ai],d['s'][ai],p)
        if arm=='erm':ref_behavior=native_b;teacher={}
        feature={};saved={}
        for name,b in d['banks']['audit'].items():
            if not b['training']:continue
            n=len(b['indices']);keep=b['supported'];cp=predict(m,b['corners'].reshape(4*n,-1),device,'P',batch).reshape(4,n)[:,keep].astype(float);cl=predict(m,b['corners'].reshape(4*n,-1),device,'L',batch).reshape(4,n)[:,keep].astype(float)
            if arm=='erm':teacher[name]=cp
            tv=teacher[name];ta,tb,_=effects(tv);tc=(ta+tb)/2;active=abs(tc)>.001
            for variant,pp in [('native',cp),('calibrated',expit(cl/fit['T']))]:
                a,c,r=effects(pp);common=(a+c)/2;dec=(pp>=.5).astype(int);da,db,_=effects(dec)
                feature.setdefault(variant,{})[name]=dict(n=int(keep.sum()),R=float(abs(r).mean()),violations=float((abs(r)>.01).mean()),signed_retention=float((np.sign(tc[active])*common[active]).sum()/abs(tc[active]).sum()) if active.any() else None,decision_disagreement=float((da!=db).mean()),new_calibration_violations=float(((abs(r)>.01)&(abs(effects(cp)[2])<=.01)).mean()))
            saved[name+'_P']=cp;saved[name+'_L']=cl;saved[name+'_indices']=b['indices'][keep]
        macro={v:{k:float(np.mean([r[k] for r in ff.values() if r[k] is not None])) for k in ('R','violations','signed_retention','decision_disagreement','new_calibration_violations')} for v,ff in feature.items()}
        calibrated=behavior(d['y'][ai],d['s'][ai],expit(l/fit['T']));model_done=json.loads((path.parent/'DONE.json').read_text());epoch=model_done['epoch'] if arch=='ft' else model_done['selections']['task' if arm=='erm' else 'P']['epoch']
        results[arm]=dict(epoch=epoch,source_fallback=arm!='erm' and epoch==0,fit=fit,behavior=native_b,calibrated_behavior=calibrated,strict_utility=native_b['auc']>=ref_behavior['auc']-.005 and native_b['f1']>=.99*ref_behavior['f1'],macro=macro,features=feature,calibrated_ece=ece(d['y'][ai],expit(l/fit['T'])),native_ece=ece(d['y'][ai],p))
        np.savez_compressed(REPORT/f'{task}_{seed}_{arch}_{arm}_audit.npz',**saved);del m
    save_json(REPORT/f'{task}_{seed}_{arch}_RESULTS.json',dict(status='completed',task=task,seed=seed,arch=arch,results=results,source_sha256=paths,protocol_sha256=None,scope='two-seed final-protocol development preflight, not confirmation'))

def queue():
    verify();logs=ROOT/'job_logs/splits';logs.mkdir(exist_ok=True);env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'};state=dict(status='running',new_trajectories=30,completed=[],pid=os.getpid(),started=time.time());save_json(REPORT/'EXECUTION.json',state)
    def launch(job):
        task,seed,arch,arm=job;log=logs/('_'.join(map(str,job))+'.log')
        with log.open('w') as f:r=subprocess.run([sys.executable,__file__,'--job',*map(str,job)],env=env,stdout=f,stderr=subprocess.STDOUT)
        assert r.returncode==0,(job,log);return job
    # Measured efficient regime: small GPU jobs in four workers; native FT exclusive.
    env['INTERFAIR_CONCURRENCY']='4'
    for arms in [('erm',),('control_preserve','interfair_preserve')]:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for job in pool.map(launch,[(t,s,'mlp',a) for t in TASKS for s in (2000,2001) for a in arms]):state['completed'].append(job);save_json(REPORT/'EXECUTION.json',state)
    env['INTERFAIR_CONCURRENCY']='1'
    for t in ('adult','hmda_oh'):
        for s in (2000,2001):
            for arm in ARMS:
                job=launch((t,s,'ft',arm));state['completed'].append(job);save_json(REPORT/'EXECUTION.json',state)
    for arch in ('mlp','ft'):
        for task in (TASKS if arch=='mlp' else TASKS[:2]):
            for seed in (2000,2001):
                with (logs/f'{task}_{seed}_{arch}_analysis.log').open('w') as f:r=subprocess.run([sys.executable,__file__,'--analyze',task,str(seed),arch],env=env,stdout=f,stderr=subprocess.STDOUT)
                assert r.returncode==0,(task,seed,arch,'analysis')
    state.update(status='completed',finished=time.time());save_json(REPORT/'EXECUTION.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gate',action='store_true');p.add_argument('--job',nargs=4);p.add_argument('--analyze',nargs=3);a=p.parse_args();torch.set_num_threads(1)
    if a.gate:gate()
    elif a.job:worker(a.job[0],int(a.job[1]),a.job[2],a.job[3])
    elif a.analyze:analyze_cell(a.analyze[0],int(a.analyze[1]),a.analyze[2])
    else:queue()
