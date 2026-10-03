"""Study-specific developed banks with final source/selection partitions."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import copy
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from partitions import HERE, DEV, data_for, read_pickle, array_hash, save_json, sha
from core import Predictor, predict, effects

DOMAIN={'adult':['hours_per_week','education_num'],'hmda_oh':['income','debt_to_income_ratio']}


def attach_partitions(d, task, seed):
    main=data_for(task,seed)
    for k in ('x','y','s'):assert array_hash(d[k])==array_hash(main[k]),(task,k)
    assert d['encoder'].metadata()==main['encoder'].metadata()
    d['splits']=copy.deepcopy(main['splits'])
    for name,b in d['banks'].get('val',{}).items():
        assert np.isin(b['indices'],d['splits']['val']).all(),(task,name,'validation probe outside selection')
    d['metadata']['selection_calibration_roles']=main['metadata']['selection_calibration_roles']
    return d


def auxiliary(task, seed, panel, policy=None):
    if panel in ('learning','selection'):
        d=read_pickle(DEV/'cache_v2'/f'{task}_selection_2000_all.pkl')
    elif panel=='gradients':
        d=read_pickle(DEV/'cache_v2'/f'{task}_input_gradient.pkl')
    elif panel=='direction':
        assert task=='hmda_oh' and policy in ('fixed','joint')
        d=read_pickle(DEV/'cache_v2'/f'hmda_direction_{policy}.pkl')
    else:raise ValueError(panel)
    return attach_partitions(d,task,seed)


def selected_features(task,seed,source,out):
    """Rank only the pre-existing training discovery bank; freeze before repair."""
    inv=read_pickle(DEV/'cache_v2'/f'{task}_feature_inventory.pkl')
    d=auxiliary(task,seed,'selection');allnames=sorted(inv['banks']['train'])
    ck=torch.load(source,map_location='cpu',weights_only=False);model=Predictor(d['x'].shape[1]);model.load_state_dict(ck['state_dict']);model.eval()
    records=[]
    for name,b in inv['banks']['discovery'].items():
        n=len(b['indices']);assert np.isin(b['indices'],d['splits']['train']).all()
        assert not np.intersect1d(b['indices'],inv['banks']['train'][name]['indices']).size
        p=predict(model,b['corners'].reshape(4*n,-1),'cpu','P').reshape(4,n)
        r=float(abs(effects(p[:,b['supported']])[2]).mean())
        records.append(dict(feature=name,residual=r,supported=int(b['supported'].sum()),
                            anchors=n,severity=r*b['supported'].mean()))
    ranking=[r['feature'] for r in sorted(records,key=lambda x:(-x['severity'],x['feature']))]
    random=list(np.random.default_rng(93103).choice(allnames,2,replace=False))
    result=dict(task=task,seed=seed,source_sha256=sha(source),ranking=ranking,rows=records,
                selected={'control':allnames,'domain':DOMAIN[task],'top2':ranking[:2],'random2':random,'all':allnames},
                split='training discovery; disjoint from this panel steering, not from source ERM fitting',
                selection_finished_before_repair=True,audit_used=False)
    out=Path(out)
    if out.exists():assert json.loads(out.read_text())==result
    else:save_json(out,result)
    return result


def selection_data(task,seed,names,policy):
    d=auxiliary(task,seed,'selection');inv=read_pickle(DEV/'cache_v2'/f'{task}_feature_inventory.pkl')
    chunks,indices,weights=[],[],[]
    for name in names:
        b=inv['banks']['train'][name];keep=b['supported'];n=int(keep.sum());assert n>=64
        chunks.append(b['corners'][:,keep]);indices.append(b['indices'][keep]);weights.append(np.full(n,1/(len(names)*n),np.float64))
    ix=np.concatenate(indices);mask=np.ones(len(ix),bool)
    d['banks']['train']={'selected_mixture':dict(corners=np.concatenate(chunks,axis=1),indices=ix,
                      valid=mask,supported=mask,training=True,sampling_weights=np.concatenate(weights))}
    d['metadata']['selected_features']=list(names);d['metadata']['selection_policy']=policy
    return d


def gate():
    checks=[]
    from training_base import prepare
    for task in ('adult','hmda_oh'):
        for seed in (2000,2001):
            for panel in ('learning','gradients'):
                d=auxiliary(task,seed,panel)
                for name,b in prepare(task)['banks']['val'].items():assert np.isin(b['indices'],d['splits']['val']).all()
                checks.append(dict(task=task,seed=seed,panel=panel,check='all validation probes exclude calibration; arrays and encoder unchanged',passed=True))
            inv=read_pickle(DEV/'cache_v2'/f'{task}_feature_inventory.pkl');names=sorted(inv['banks']['train'])
            for policy,ns in [('all',names),('domain',DOMAIN[task]),('random2',list(np.random.default_rng(93103).choice(names,2,replace=False)))]:
                d=selection_data(task,seed,ns,policy)
                old=read_pickle(DEV/'cache_v2'/f'{task}_selection_{seed}_{policy}.pkl')
                a=d['banks']['train']['selected_mixture'];b=old['banks']['train']['selected_mixture']
                assert all(np.array_equal(a[k],b[k]) for k in a if isinstance(a[k],np.ndarray))
                checks.append(dict(task=task,seed=seed,policy=policy,check='selection training bank equals developed policy',passed=True))
    from protected_vector import compile_banks
    cube=compile_banks()
    for seed in (2000,2001):
        d=data_for('hmda_oh',seed)
        for b in cube['banks']['val'].values():assert np.isin(b['indices'],d['splits']['val']).all()
        for policy in ('fixed','joint'):auxiliary('hmda_oh',seed,'direction',policy)
        checks.append(dict(seed=seed,check='direction and protected-vector validation exclude calibration',passed=True))
    save_json(HERE/'reports/SUPPLEMENTARY_DATA_ADMISSION.json',dict(status='PASS',checks=checks,source_sha256=sha(__file__)))
    print('supplementary data admission PASS',len(checks),flush=True)


if __name__=='__main__':gate()
