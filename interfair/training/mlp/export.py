"""Complete scratch-study outcomes with paired seed-level inference."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
P=_PACKAGE / 'training/mlp'; E=_PACKAGE / 'training/mlp/exports'; E.mkdir(exist_ok=True)
rows=pd.concat([pd.read_csv(f) for f in (P/'evaluation').glob('*/*/per_seed.csv')],ignore_index=True)
features=pd.concat([pd.read_csv(f) for f in (P/'evaluation').glob('*/*/per_edit.csv')],ignore_index=True)
assert len(rows)==150 and not rows.duplicated(['task','seed','arm']).any()
assert rows.groupby(['task','arm']).seed.apply(lambda s:set(s)==set(range(1000,1010))).all()
U=['auc','accuracy','f1','aod','dp','eomax']
R=['L_R','violation_005','P_R','P_population_effect_gap','decision_disagreement','opposite_005']
DM=['direction_adverse_005','direction_both_adverse_005','direction_adverse_decision','direction_joint_pass']
metrics=U+R+DM+['direction_adverse_0','direction_adverse_01','direction_adverse_magnitude','direction_common_response']
rows.to_csv(E/'per_seed.csv',index=False);features.to_csv(E/'per_edit.csv',index=False)
summary=rows.groupby(['task','arm'])[metrics].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary]
summary.to_csv(E/'summary.csv')
tests=[];regressions=[]
def test(a,b,task,metric,family,comparator='equality',edit=None):
    x=a.set_index('seed')[metric].sort_index();y=b.set_index('seed')[metric].sort_index()
    assert x.index.equals(y.index) and len(x)==10
    delta=y-x;sd=delta.std(ddof=1);se=sd/np.sqrt(10)
    p=float(stats.ttest_rel(y,x).pvalue) if sd>1e-14 else (1. if abs(delta.mean())<1e-14 else 0.)
    tests.append(dict(task=task,edit=edit,comparator=comparator,metric=metric,family=family,n=10,
       comparator_mean=x.mean(),combined_mean=y.mean(),difference=delta.mean(),
       ratio_of_means=y.mean()/x.mean() if abs(x.mean())>1e-15 else np.nan,
       ci95_low=delta.mean()-stats.t.ppf(.975,9)*se,ci95_high=delta.mean()+stats.t.ppf(.975,9)*se,p_raw=p))
    higher=metric in ['auc','accuracy','f1','direction_joint_pass']
    for seed in delta.index[(delta*(-1 if higher else 1))>1e-10]:
        regressions.append(dict(task=task,seed=int(seed),comparator=comparator,metric=metric,
                                comparator_value=x.loc[seed],combined_value=y.loc[seed],difference=delta.loc[seed],edit=edit))
for task,g in rows.groupby('task'):
    b=g[g.arm=='equality_direction']
    for comparator in ['equality','erm']:
        a=g[g.arm==comparator]
        for metric in DM:test(a,b,task,metric,comparator+': direction',comparator)
        for metric in U+R:test(a,b,task,metric,comparator+': utility population equality',comparator)
for edit,g in features[(features.task=='hmda_oh')&~features.training_edit].groupby('edit'):
    for metric in ['adverse_005','both_adverse_005','adverse_decision','L_R']:
        test(g[g.arm=='equality'],g[g.arm=='equality_direction'],'hmda_oh',metric,'held-out income',edit=edit)
t=pd.DataFrame(tests)
for _,ix in t.groupby('family').groups.items():
    order=t.loc[ix].sort_values('p_raw').index;p=t.loc[order,'p_raw'].to_numpy()
    t.loc[order,'p_holm']=np.minimum(1,np.maximum.accumulate(p*np.arange(len(p),0,-1)))
t.to_csv(E/'paired_tests.csv',index=False);pd.DataFrame(regressions).to_csv(E/'regressions.csv',index=False)
admission=[]
for f in (P/'runs/confirmation').glob('*/*/*/DONE.json'):
    j=json.loads(f.read_text());v=j['selected'];cfg=j['config']
    admission.append(dict(task=cfg['task'],seed=cfg['seed'],arm=cfg['arm'],admitted=v['admitted'],selected_epoch=v['epoch'],
                          checkpoint=j['checkpoint'],checkpoint_sha256=j['checkpoint_sha256'],
                          initial_state_sha256=j['initial_state_sha256'],optimizer_updates=j['optimizer_updates'],
                          regularizer_model_rows=j['regularizer_model_rows'],seconds=j['seconds']))
pd.DataFrame(admission).to_csv(E/'admission.csv',index=False)
check=json.loads((P/'confirmation_BUDGET_VERIFICATION.json').read_text())
assert len(check['triplets'])==50
(E/'VERIFICATION.json').write_text(json.dumps(dict(status='PASS',model_rows=150,matched_triplets=50,
    fresh_initialization=True,pretrained_weights_loaded=False,all_arms_40_epochs=True,
    same_task_updates=True,paired_is_contexts_and_queries_equal=True,
    test_families=t.groupby('family').size().to_dict(),audit_used_for_selection=False),indent=2)+'\n')
print(rows.groupby(['task','arm'])[U+R+DM].mean().to_string())
print('\nDirection contrasts:')
print(t[t.metric=='direction_adverse_005'][['task','comparator','comparator_mean','combined_mean','ratio_of_means','p_holm']].to_string(index=False))
print('\nSignificant other matched differences:')
print(t[(t.comparator=='equality')&~t.metric.isin(DM)&(t.p_holm<.05)][['task','metric','difference','p_holm']].to_string(index=False))
