"""Common whole-pipeline audit; native probability and hard-decision roles differ."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import numpy as np
from core import behavior, effects

THRESHOLDS=tuple(np.arange(1,10)/10)


def feature_metrics(v,teacher=None,scale='P',output_kind='probability'):
    v=np.asarray(v,dtype=np.float64)
    n=v.shape[1]
    if not n:return dict(n=0,metrics=None)
    a,b,r=effects(v);common=(a+b)/2
    if output_kind=='hard_decision':
        return dict(n=n,decision_effect_disagreement_count=int((a!=b).sum()),
                    decision_effect_disagreement=float((a!=b).mean()),
                    decision_residual_mean_abs=float(abs(r).mean()))
    cutoff=.001 if scale=='P' else .01
    tol=.01 if scale=='P' else .05
    out=dict(n=n,R=float(abs(r).mean()),signed_R=float(r.mean()),
             median_abs_R=float(np.median(abs(r))),q95_abs_R=float(np.quantile(abs(r),.95)),
             max_abs_R=float(abs(r).max()),violations=int((abs(r)>tol).sum()),
             violation_rate=float((abs(r)>tol).mean()),tolerance=tol,
             opposite_direction_count=int(((a>cutoff)&(b< -cutoff)| (a< -cutoff)&(b>cutoff)).sum()),
             both_flat_count=int(((abs(a)<=cutoff)&(abs(b)<=cutoff)).sum()),
             mean_common_abs=float(abs(common).mean()))
    if scale=='P':
        out['thresholds']={}
        for t in THRESHOLDS:
            da,db,_=effects((v>=t).astype(int))
            out['thresholds'][f'{t:.1f}']=dict(disagreements=int((da!=db).sum()),n=n,
                                           rate=float((da!=db).mean()))
        out['decision_disagreement']=out['thresholds']['0.5']['rate']
    if teacher is not None:
        ta,tb,_=effects(np.asarray(teacher,dtype=np.float64));tc=(ta+tb)/2
        active=abs(tc)>cutoff;den=float(abs(tc[active]).sum())
        num=float((np.sign(tc[active])*common[active]).sum())
        out.update(source_active_count=int(active.sum()),retention_numerator=num,
                   retention_denominator=den,signed_retention=num/den if den>0 else None,
                   mean_common_error=float(abs(common-tc).mean()))
    return out


def decision_behavior(y,s,decision):
    # Binary decisions have no native probability/logit, proper score or AUROC.
    b=behavior(y,s,decision)
    return {k:v for k,v in b.items() if k in ('accuracy','f1','dp','aod','eomax','group_rates')}


def audit(pipeline,d,teacher=None,split='audit',banks=None):
    banks=d['banks'][split] if banks is None else banks
    ix=d['splits'][split];natural=pipeline(d['x'][ix]);kind=pipeline.output_kind
    natural_metrics=behavior(d['y'][ix],d['s'][ix],natural) if kind=='probability' else decision_behavior(d['y'][ix],d['s'][ix],natural)
    arrays=dict(natural=natural,natural_y=d['y'][ix],natural_s=d['s'][ix],natural_indices=ix)
    result=dict(output_kind=kind,behavior=natural_metrics,features={},provenance=pipeline.provenance())
    for name,b in banks.items():
        n=len(b['indices']);record=dict(anchors=n,valid=int(b['valid'].sum()),supported=int(b['supported'].sum()),
                                      training=bool(b['training']))
        x=b['corners'].reshape(4*n,-1)
        for scale in (('P','L') if kind=='probability' else ('decision',)):
            p=pipeline(x,'P' if scale=='decision' else scale).reshape(4,n)
            t=teacher(x,scale).reshape(4,n) if teacher is not None and kind=='probability' else None
            arrays[name+'_'+scale]=p
            for cohort in ('valid','supported'):
                keep=b[cohort]
                record[scale+'_'+cohort]=feature_metrics(p[:,keep],t[:,keep] if t is not None else None,scale,kind)
        result['features'][name]=record
        for k in ('indices','valid','supported'):arrays[name+'_'+k]=b[k]
    key='P_supported' if kind=='probability' else 'decision_supported'
    eligible=[r[key] for r in result['features'].values() if r['training'] and r[key]['n']>0]
    metric_keys=('R','violation_rate','signed_retention','decision_disagreement') if kind=='probability' else ('decision_effect_disagreement','decision_residual_mean_abs')
    result['macro']={k:float(np.mean([r[k] for r in eligible if r.get(k) is not None]))
                     if any(r.get(k) is not None for r in eligible) else None for k in metric_keys}
    result['macro']['features_with_support']=len(eligible)
    result['macro']['declared_features']=sum(r['training'] for r in result['features'].values())
    return result,arrays


def paired_context_interval(feature_arrays,groups=None,resamples=10000,seed=94102):
    """Cluster context resampling; every edit and both methods move together.

    Each field supplies matched indices, reference residual and method residual.
    Group IDs may map row indices to Default Credit predictor-profile clusters.
    These intervals condition on fixed models; they do not include model fitting.
    """
    universe=np.unique(np.concatenate([v['indices'] for v in feature_arrays.values()]))
    mapped=universe if groups is None else np.asarray(groups)[universe]
    unique=np.unique(mapped);lookup={g:i for i,g in enumerate(unique)}
    sums=np.zeros((len(unique),len(feature_arrays)));counts=np.zeros_like(sums)
    for j,v in enumerate(feature_arrays.values()):
        ix=np.asarray(v['indices']);g=ix if groups is None else np.asarray(groups)[ix]
        pos=np.array([lookup[x] for x in g]);diff=abs(np.asarray(v['reference']))-abs(np.asarray(v['method']))
        assert len(ix)==len(diff) and np.isfinite(diff).all()
        np.add.at(sums[:,j],pos,diff);np.add.at(counts[:,j],pos,1)
    rng=np.random.default_rng(seed);values=[]
    for start in range(0,resamples,100):
        draws=rng.integers(0,len(unique),(min(100,resamples-start),len(unique)))
        ss=sums[draws].sum(1);cc=counts[draws].sum(1)
        if (cc==0).any():raise ValueError('A bootstrap draw lost an entire feature; no silently dropped feature/replicate')
        values.extend((ss/cc).mean(1))
    return dict(point=float((sums.sum(0)/counts.sum(0)).mean()),interval=np.quantile(values,[.025,.975]).tolist(),
                groups=len(unique),resamples=resamples,rng=seed,
                scope='approximate paired fixed-model context/cluster sampling; no fitting uncertainty')
