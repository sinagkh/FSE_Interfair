"""E08a: frozen-model, CPU-only temperature calibration and scale audit."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[1]
import argparse,itertools,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
from core import ROOT,Predictor,predict,behavior,effects,save_json,sha
from training_base import prepare
from feature_edits import banks

OUT=_PACKAGE / 'core/reports/calibration';PROTOCOL=_PACKAGE / 'core/reports/CALIBRATION_PROTOCOL.md'

def entries(task,seed,arch):
    if arch=='ft':
        base=ROOT/'runs/ft_default'/task/str(seed)
        return [(a,base/b/'selected.pt') for a,b in [('erm','erm'),('control_preserve','control_preserve'),('interfair_preserve','all_preserve'),('removed','removed'),('uniform_erm','erm')]]
    base=ROOT/'runs/pilot_v3'/task/'mlp'/str(seed);head=ROOT/'runs/headroom'/task/'mlp'/str(seed)
    if task=='adult':b=ROOT/'runs/maintenance/adult'/str(seed);control=b/'initial_control/P.pt';repair=b/'initial_all/P.pt'
    else:b=ROOT/'runs/selection_preservation'/str(seed);control=b/'control/P.pt';repair=b/'all/P.pt'
    return [('erm',base/'erm/task.pt'),('control_preserve',control),('interfair_preserve',repair),('removed',base/'removed/task.pt'),('uniform_erm',base/'erm/task.pt'),('narrow_L',head/'interfair_L/L.pt'),('narrow_P',head/'interfair_P/P.pt'),('structural_L',base/'structural_L/L.pt'),('structural_P',base/'structural_P/P.pt')]

def nll(z,y):return float(np.mean(np.logaddexp(0,z)-y*z))

def fit_temperature(z,y):
    result=minimize_scalar(lambda u:nll(z/np.exp(u),y),bounds=(-3,3),method='bounded',options={'xatol':1e-8})
    assert result.success and np.isfinite(result.fun)
    T=float(np.exp(result.x)) if result.fun<=nll(z,y) else 1.
    return dict(T=T,logT=float(np.log(T)),native_logit_nll=nll(z,y),fitted_nll=nll(z/T,y),optimizer_success=bool(result.success),bound_hit=abs(np.log(T))>2.999,identity_fallback=T==1.)

def ece(y,p):
    bins=np.minimum((p*10).astype(int),9);total=0.
    for k in range(10):
        keep=bins==k
        if keep.any():total+=keep.mean()*abs(float(p[keep].mean()-y[keep].mean()))
    return total

def rule_metrics(L,P,L0,P0,T):
    a,b,r=effects(P);la,lb,lr=effects(L);oa,ob,orr=effects(P0);_,_,olr=effects(L0)
    common=(a+b)/2;teacher=(oa+ob)/2;active=abs(teacher)>.001;dec=(P>=.5).astype(int);da,db,_=effects(dec)
    origin=abs(orr)>.01;now=abs(r)>.01;adjusted=abs(lr)>.05/T
    return dict(n=P.shape[1],P_mean_abs=float(abs(r).mean()),P_violation=float(now.mean()),P_new_violation=float((now&~origin).mean()),P_lost_violation=float((~now&origin).mean()),L_mean_abs=float(abs(lr).mean()),L_violation_fixed=float((abs(lr)>.05).mean()),L_violation_rescaled=float(adjusted.mean()),L_rescaled_mask_mismatch=int((adjusted!=(abs(olr)>.05)).sum()),L_residual_scaling_error=float(abs(lr-olr/T).max()),mean_common_abs=float(abs(common).mean()),signed_retention=float((np.sign(teacher[active])*common[active]).sum()/abs(teacher[active]).sum()) if active.any() else None,opposed=float(((a*b)<0).mean()),sign_reversal_count=int(((a*oa<-1e-14)|(b*ob<-1e-14)).sum()),transition_disagreement=float((da!=db).mean()),P_protected_gap=float(abs(P[2:]-P[:2]).mean()),L_protected_gap=float(abs(L[2:]-L[:2]).mean()),P_saturation=float(((P<=1e-7)|(P>=1-1e-7)).mean()))

def analytic():
    values={};rows=[];checks=[]
    p=np.array([.2,.4,.35,.55],float);l=np.array([-1.,1.,-.2,1.8]);ind=np.array([.2,.4,.2,.4])
    for name,original in [('probability_additive',p),('logit_additive',expit(l)),('protected_independent',ind)]:
        L=logit(original)
        for T in (.5,1.,2.,5.):
            P=expit(L/T);r=float(P[3]-P[2]-P[1]+P[0]);lr=float((L[3]-L[2]-L[1]+L[0])/T)
            rows.append(dict(case=name,T=T,probabilities=P.tolist(),P_residual=r,L_residual=lr))
            if name=='protected_independent':assert abs(r)<1e-14 and abs(lr)<1e-14
            if name=='logit_additive':assert abs(lr)<1e-14
            if name=='probability_additive' and T==1:assert abs(r)<1e-14
    assert max(abs(r['P_residual']) for r in rows if r['case']=='probability_additive')>.01
    save_json(OUT/'ANALYTIC.json',dict(status='PASS',rows=rows,claim='Probability additivity with protected offset need not survive a shared nonlinear calibration; protected independence does.'))

def worker(task,seed,arch):
    torch.set_num_threads(1);assert not torch.cuda.is_available();data=prepare(task);iv=banks(task);dest=OUT/task/str(seed)/arch;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'DONE.json').exists():return
    rows=[];natural=[];checks=[];sources={};base_natural={};base_corners={};fitrecords=[]
    for arm,path in entries(task,seed,arch):
        key=dict(task=task,seed=seed,architecture=arch,arm=arm);sources[str(path)]=sha(path)
        if arch=='ft':
            from ft_architecture import load_ft,logits
            model=load_ft(path,data,'cpu');infer=lambda x:logits(model,x).astype(np.float64)
        else:
            ck=torch.load(path,map_location='cpu',weights_only=False);model=Predictor(data['x'].shape[1],'mlp',ck['config']['mode']);model.load_state_dict(ck['state_dict']);model.eval();infer=lambda x:predict(model,x,'cpu','L').astype(np.float64)
        def score(x):
            if arm=='uniform_erm':
                parts=[]
                for s in (0,1):
                    xx=x.copy();xx[:,0]=s
                    # Mean native probability, preserving the existing mixture definition.
                    parts.append(predict(model,xx,'cpu','P',batch=512 if arch=='ft' else 4096).astype(np.float64))
                P=np.mean(parts,axis=0);return logit(np.clip(P,1e-12,1-1e-12)),P
            L=infer(x)
            P=torch.tensor(L,dtype=torch.float32).sigmoid().numpy().astype(np.float64) if arch=='ft' else predict(model,x,'cpu','P').astype(np.float64)
            return L,P
        vi=data['splits']['val'];Lval,Pval=score(data['x'][vi]);fit=fit_temperature(Lval,data['y'][vi]);fit.update(key,source_sha256=sha(path),validation_rows=len(vi),calibration_uses_protected_groups=False)
        np.savez_compressed(dest/(arm+'_validation.npz'),L=Lval,P=Pval,y=data['y'][vi],indices=vi)
        save_json(dest/(arm+'_FIT.json'),fit);fitrecords.append(fit)
        # Fitting is frozen before these fresh audit evaluations.
        ai=data['splits']['audit'];La,Pa=score(data['x'][ai]);arrays=dict(natural_L=La,natural_P=Pa,natural_y=data['y'][ai],natural_s=data['s'][ai],natural_indices=ai)
        variant_T=[('native',1.),('identity',1.),('fitted',fit['T']),('T0.5',.5),('T2',2.),('T5',5.)]
        for variant,T in variant_T:
            L=La/T;P=Pa if variant=='native' else expit(L);b=behavior(data['y'][ai],data['s'][ai],P)
            natural.append(dict(**key,variant=variant,T=T,**{k:v for k,v in b.items() if k!='group_rates'},ece10=ece(data['y'][ai],P),logit_nll=nll(L,data['y'][ai]),rank_auc_logit=float(roc_auc_score(data['y'][ai],L)),decision_churn=float(((P>=.5)!=(Pa>=.5)).mean()),saturation=float(((P<=1e-7)|(P>=1-1e-7)).mean()),native_P_identity_max_error=float(abs(Pa-expit(La)).max())))
        ftarr=np.load(path.parent/'audit.npz') if arch=='ft' else None
        for name,bank in iv['banks']['audit'].items():
            keep=bank['supported'];N=len(keep)
            if arch=='ft':
                assert np.array_equal(ftarr[name+'_supported'],keep) and np.array_equal(ftarr[name+'_indices'],bank['indices'])
                L0=ftarr[name+'_L'][:,keep].astype(np.float64);P0=ftarr[name+'_P'][:,keep].astype(np.float64)
                if arm=='uniform_erm':
                    P0=np.tile((P0[:2]+P0[2:])/2,(2,1));L0=logit(np.clip(P0,1e-12,1-1e-12))
            else:
                L0,P0=score(bank['corners'][:,keep].reshape(-1,bank['corners'].shape[-1]));L0=L0.reshape(4,-1);P0=P0.reshape(4,-1)
            arrays[name+'_L']=L0;arrays[name+'_P']=P0;arrays[name+'_indices']=bank['indices'][keep]
            if arm=='erm':base_corners[name]=(L0,P0)
            for variant,T in variant_T:
                L=L0/T;P=P0 if variant=='native' else expit(L);met=rule_metrics(L,P,L0,P0,T)
                assert met['L_residual_scaling_error']<1e-10 and met['L_rescaled_mask_mismatch']==0 and met['sign_reversal_count']==0
                if arm in ('removed','uniform_erm'):assert met['P_mean_abs']<2e-6 and met['L_mean_abs']<2e-5
                rows.append(dict(**key,feature=name,variant=variant,T=T,**met))
            checks.append(dict(**key,feature=name,passed=True,scope='logit scaling/tolerance identity; no direction reversal; independent-score null where applicable',native_P_identity_max_error=float(abs(P0-expit(L0)).max())))
        np.savez_compressed(dest/(arm+'_audit.npz'),**arrays)
        print(json.dumps(dict(**key,T=fit['T'],status='audited')),flush=True)
    pd.DataFrame(rows).to_csv(dest/'rules.csv',index=False);pd.DataFrame(natural).to_csv(dest/'natural.csv',index=False)
    save_json(dest/'DONE.json',dict(status='completed',task=task,seed=seed,architecture=arch,checks=checks,fits=fitrecords,sources=sources,code_sha256=sha(__file__),protocol_sha256=None,score_variants=6,final_confirmation_started=False,external_outcomes_opened=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',nargs=3);p.add_argument('--queue',action='store_true');args=p.parse_args();OUT.mkdir(exist_ok=True)
    if args.job:worker(args.job[0],int(args.job[1]),args.job[2]);return
    analytic()
    if args.queue:
        log=ROOT/'job_logs/calibration';log.mkdir(exist_ok=True,parents=True);env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
        def launch(j):
            with (log/('_'.join(map(str,j))+'.log')).open('w') as f:r=subprocess.run([sys.executable,__file__,'--job',*map(str,j)],stdout=f,stderr=subprocess.STDOUT,env=env)
            assert r.returncode==0,j
            print('completed',*j,flush=True)
        start=time.time()
        jobs=[(t,s,a) for a in ('mlp','ft') for t in ('adult','hmda_oh') for s in (2000,2001)]
        with ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(launch,jobs))
        save_json(OUT/'EXECUTION.json',dict(status='completed',cells=8,seconds=time.time()-start,device='cpu',concurrency=4))

if __name__=='__main__':main()
