"""Common evidence export; uses frozen outputs, not refitting or audit selection."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
from common import *
from scipy import stats
from functools import lru_cache
import argparse,subprocess
U=['auc','accuracy','f1','aod','dp','eomax']
R=['L_R','violation_005','P_R','P_population_effect_gap','decision_disagreement','opposite_005']
KEY=['task','architecture','arm','seed']
HIGH={'auc','accuracy','f1'}
SOURCES={}
def register(path):
    path=Path(path).resolve();SOURCES[str(path)]=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
    return path
def read(path):return json.loads(register(path).read_text())
@lru_cache(None)
def taskdata(task):return data(task,2000)[0]
def saved_metrics(task,path,target=False,d_override=None):
    d=d_override if d_override is not None else taskdata(task);a=np.load(register(path));ix=a['natural_indices']
    natural=expit(a['natural_L']) if 'natural_L' in a else a['natural']
    row=behavior(d['y'][ix],d['s'][ix],natural);ff=[]
    for name,b in d['banks']['audit'].items():
        if not b['training']:continue
        z=a[name+'_L'];ids=a[name+'_indices']
        keep=np.ones(len(ids),bool) if target else a[name+'_supported']
        # Targeting stores supported-only arrays; other producers retain all rows.
        ff.append(dict(feature=name,**effect_metrics(z[:,keep],ids[keep],d)))
    row.update(pd.DataFrame(ff).drop(columns=['feature','n']).mean().to_dict())
    for g, rates in row.pop('group_rates').items():
        row.update({f'group{g}_{k}':v for k,v in rates.items()})
    row['conformance_005']=1-row['violation_005']
    return row,ff

def load_new(phase):
    rows=[];joint=[];features=[]
    for path in sorted((P/'runs'/phase).glob('*/*/*/*/*/DONE.json')):
        scope=path.relative_to(P/'runs'/phase).parts[0]
        if scope not in ('main','joint','ft_deterministic'):continue
        if scope=='main' and '/ft/' in str(path) and path.parent.name in ('hifi','hifi_shared'):continue
        if path.parent.name=='soft':continue # relative alias of canonical interfair
        j=read(path);c=j['config'];task=c['task'];seed=c['seed'];arch=c['architecture'];arm=c['arm']
        arm='soft' if arm=='interfair' else arm
        ident=dict(task=task,seed=seed,architecture=arch,arm=arm)
        a,ff=saved_metrics(task,path.parent/'audit.npz')
        checkpoint=register(j['checkpoint'])
        assert sha(checkpoint)==j['checkpoint_sha256']
        row=dict(**ident,**a,admitted=j.get('admitted',np.nan),uses_admission_guard=arm=='soft',checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
            source_done=str(path),source_sha256=sha(path),arrays=str(path.parent/'audit.npz'),arrays_sha256=sha(path.parent/'audit.npz'),
            epochs_completed=j['epochs_completed'],training_seconds=j['seconds'],training_device=c['device'],
            phase=phase,protocol='current soft procedure' if arm=='soft' else ('HIFI native objective; shared architecture fitting and validation selector' if arm=='hifi_shared' else ('HIFI native objective/schedule; shared architecture and batch' if arm.startswith('hifi') else 'shared architecture/task training with method preprocessing')),
            feature_count=len(ff))
        if c.get('joint'):
            jr=j['audit']['joint_response'];r0={**row,**jr,**{f'sex_{k}':j['audit']['sex_behavior'][k] for k in ['aod','dp','eomax']},
                'subgroup_aod':j['audit']['joint_aod'],'subgroup_dp':j['audit']['joint_dp'],
                'training_protected_axes':'race and sex; unclassified sex classification-only'}
            for metric in ['P_population_effect_gap','P_mean_common_abs']:
                r0['race_'+metric]=r0.pop(metric)
            joint.append(r0)
            features.extend([dict(panel='joint',**ident,**v) for v in j['audit']['joint_features']])
        else:
            rows.append(row);features.extend([dict(panel='main',**ident,**v) for v in ff])
    return pd.DataFrame(rows),pd.DataFrame(joint),pd.DataFrame(features)

def load_transfer():
    rows=[];features=[];banks={}
    for path in sorted((P/'runs/confirmation/transfer').glob('*/*/*/*/DONE.json')):
        j=read(path);c=j['config'];block=j['block']
        if block not in banks:banks[block]=pickle.load(register(j['bank']).open('rb'))['data']
        a,ff=saved_metrics('hmda_oh',path.parent/'audit.npz',d_override=banks[block])
        ident=dict(task=block,source_task='hmda_oh',block=block,architecture=c['architecture'],arm=c['arm'],seed=c['seed'])
        assert sha(register(j['checkpoint']))==j['checkpoint_sha256']
        rows.append(dict(**ident,**a,checkpoint=j['checkpoint'],source_done=str(path),arrays=str(path.parent/'audit.npz'),
            audit_seconds=j['audit']['audit_seconds'],audit_model_rows=j['audit']['audit_model_rows'],target_used_for_fitting=False))
        features.extend([dict(panel='transfer',**ident,**f) for f in ff])
    return pd.DataFrame(rows),pd.DataFrame(features)

def load_target(phase):
    rows=[];features=[];seeds=[2000,2001] if phase=='development' else list(range(1000,1010))
    olddir=ROOT/'archive'
    for task in ['hmda_oh','acs_income','credit_broad']:
      for seed in seeds:
        reference=None;matched=[]
        discovery=P/'targeting_strict/discovery'/task/str(seed)
        if (discovery/'DONE.json').exists():
            policy=read(discovery/'DONE.json');register(discovery/'SOURCE.npz')
            assert sha(discovery/'SOURCE.npz')==policy['source_arrays_sha256']
        for arm in ['population_control','broad','behavior_features','behavior_contexts','pure_continuation','witness_augmentation']:
            root=P/'targeting_strict'
            path=root/'runs'/task/str(seed)/arm/'DONE.json'
            if not path.exists():continue
            j=read(path);a,ff=saved_metrics(task,path.parent/'AUDIT.npz',True);ident=dict(task=task,seed=seed,architecture='mlp',arm=arm)
            np.testing.assert_allclose([a['L_R'],a['decision_disagreement'],a['opposite_005']],
                [j['audit']['macro'][k] for k in ['L_R','decision','opposite']],rtol=1e-7,atol=1e-8)
            assert sha(register(j['checkpoint']))==j['checkpoint_sha256']
            extra={k:v for k,v in j['audit']['macro'].items() if isinstance(v,(int,float)) and k not in a}
            rows.append(dict(**ident,**a,**extra,source_done=str(path),arrays=str(path.parent/'AUDIT.npz'),
                checkpoint=j['checkpoint'],training_seconds=j['seconds'],selected_step=j['selected']['step'],
                no_repair_selected=j['no_repair_selected'],phase=phase,protocol='matched 600-step continuation; discovery/steering separated'))
            features.extend([dict(panel='target',**ident,**f) for f in ff]);matched.append(j)
            if reference is None:
                register(j['config']['initial_checkpoint'])
                reference,fr=saved_metrics(task,path.parent/'SOURCE_AUDIT.npz',True)
                rows.append(dict(task=task,seed=seed,architecture='mlp',arm='erm',**reference,
                    source_done=str(path),arrays=str(path.parent/'SOURCE_AUDIT.npz'),checkpoint=j['config']['initial_checkpoint'],
                    phase=phase,protocol='source for matched continuation'))
                features.extend([dict(panel='target',task=task,seed=seed,architecture='mlp',arm='erm',**f) for f in fr])
        if len(matched)==6:
            for key in ['task_draw_sha256','population_draw_sha256']:assert len({j[key] for j in matched})==1,(task,seed,key)
            assert len({j['config']['initial_checkpoint_sha256'] for j in matched})==1
    return pd.DataFrame(rows),pd.DataFrame(features)

def contrasts(df,pairs,metrics,prefix):
    rows=[]
    for task,arch in df[['task','architecture']].drop_duplicates().itertuples(index=False,name=None):
      group=df[(df.task==task)&(df.architecture==arch)]
      for reference,method in pairs(group):
        a=group[group.arm==reference].set_index('seed');b=group[group.arm==method].set_index('seed')
        ids=sorted(set(a.index)&set(b.index));assert ids==list(range(1000,1010)),(task,arch,reference,method,ids)
        for metric in metrics:
            if metric not in group or a.loc[ids,metric].isna().any() or b.loc[ids,metric].isna().any():continue
            x=a.loc[ids,metric].to_numpy(float);y=b.loc[ids,metric].to_numpy(float);delta=y-x
            exact=reference=='tied' and metric in ['L_R','violation_005','P_R','decision_disagreement','opposite_005']
            pval=np.nan if exact else (1. if np.all(delta==0) else (0. if np.std(delta,ddof=1)==0 else float(stats.ttest_rel(y,x).pvalue)))
            half=stats.t.ppf(.975,9)*stats.sem(delta)
            family=prefix+('_response' if metric not in U and metric not in ['sex_aod','sex_dp','sex_eomax','subgroup_aod','subgroup_dp'] else '_utility_population')
            rows.append(dict(task=task,architecture=arch,reference=reference,method=method,metric=metric,n=10,family=family,
                reference_mean=x.mean(),method_mean=y.mean(),mean_difference=delta.mean(),ratio_of_means=y.mean()/x.mean() if x.mean()!=0 else np.nan,
                ci95_low=delta.mean()-half,ci95_high=delta.mean()+half,p_t=pval,status='algebraic comparator' if exact else 'tested'))
    f=pd.DataFrame(rows);f['p_holm']=np.nan
    for fam,q in f.groupby('family'):
        q=q[q.status=='tested'].sort_values('p_t');f.loc[q.index,'p_holm']=np.minimum(1,np.maximum.accumulate(q.p_t.to_numpy()*np.arange(len(q),0,-1)))
    return f

def primary_pairs(g):
    arms=set(g.arm);pairs=[]
    if 'soft' in arms:pairs.extend((a,'soft') for a in sorted(arms-{'soft'}))
    if {'hifi','hifi_erm'}<=arms:pairs.append(('hifi_erm','hifi'))
    if {'hifi_shared','erm'}<=arms:pairs.append(('erm','hifi_shared'))
    for arm in ['cot_phi','reweighing']:
        if g.architecture.iloc[0]=='ft' and arm in arms:pairs.append(('erm',arm))
    return pairs
def target_pairs(g):
    return [(a,b) for a in ['erm','pure_continuation','witness_augmentation','population_control','broad'] for b in ['behavior_features','behavior_contexts'] if a!=b]
def joint_pairs(g):return [(a,'soft') for a in sorted(set(g.arm)-{'soft'})]+[('hifi_erm','hifi'),('erm','reweighing')]

def ranks(df,out,prefix,metrics):
    module=ast.parse((ROOT/'archive/tools/export_full_tables.py').read_text())
    code=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='R_CODE' for t in n.targets))
    z=df.copy()
    long=z.melt(id_vars=KEY,value_vars=metrics,var_name='metric',value_name='value').dropna(subset=['value'])
    long['algebraic']=(long.arm=='tied')&long.metric.isin(['L_R','violation_005','P_R','decision_disagreement','opposite_005'])
    long['direction']=np.where(long.metric.isin(HIGH),'higher','lower')
    long['family']=prefix+' / '+long.task+' / '+long.architecture+' / '+long.metric
    long.to_csv(out/(prefix+'_sk_input.csv'),index=False);script=out/(prefix+'_sk.R');script.write_text(code)
    q=subprocess.run(['Rscript',str(script),str(ROOT.parent),str(out/(prefix+'_sk_input.csv')),str(out/(prefix+'_sk_ranks.csv')),str(out/(prefix+'_sk_errors.csv')),str(out/(prefix+'_R_session.txt'))],capture_output=True,text=True)
    (out/(prefix+'_sk.log')).write_text(q.stdout+q.stderr);assert q.returncode==0,q.stderr

def write_summary(df,out,prefix):
    ms=[m for m in U+R+['sex_aod','sex_dp','subgroup_aod','subgroup_dp','L_third','ap','bce','brier',
        'group0_tpr','group1_tpr','group0_fpr','group1_fpr','group0_positive','group1_positive','P_mean_common_abs'] if m in df]
    df.to_csv(out/(prefix+'_per_seed.csv'),index=False)
    a=df.groupby(['task','architecture','arm'])[ms].agg(['mean','std','count']);a.columns=['_'.join(c) for c in a.columns]
    a.reset_index().to_csv(out/(prefix+'_summary.csv'),index=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['development','confirmation']);ap.add_argument('--allow-partial',action='store_true');ap.add_argument('--skip-ranks',action='store_true');args=ap.parse_args()
    out=P/'exports'/args.phase;out.mkdir(parents=True,exist_ok=True)
    new,joint,features=load_new(args.phase);target,tf=load_target(args.phase)
    write_summary(new,out,'new');write_summary(target,out,'target');write_summary(joint,out,'joint_new')
    pd.concat([features,tf],ignore_index=True).to_csv(out/'per_feature.csv',index=False)
    if args.phase=='confirmation' and not args.allow_partial:
        assert len(new)==290,(len(new),'expected 29 new arms x10')
        assert len(joint)==50 and len(target)==210
        primary=pd.read_csv(register(ROOT/'archive/results/01_primary/full_main_per_seed.csv'))
        primary=pd.concat([primary,new],ignore_index=True);assert not primary.duplicated(KEY).any()
        assert primary.groupby(KEY[:-1]).seed.apply(lambda x:set(x)==set(range(1000,1010))).all()
        write_summary(primary,out,'primary_complete')
        oldj=pd.read_csv(register(ROOT/'archive/generated/recovered_intersectional_per_seed.csv'))
        joint=pd.concat([oldj[oldj.arm=='soft'],joint],ignore_index=True)
        # Primary race-only effect gaps are not a defined four-group estimand.
        joint=joint.drop(columns=['P_population_effect_gap','P_mean_common_abs'],errors='ignore')
        write_summary(joint,out,'joint_complete')
        transfer,tff=load_transfer();assert len(transfer)==160,len(transfer)
        oldtransfer=pd.read_csv(register(ROOT/'archive/generated/recovered_transfer_per_seed.csv'))
        oldtransfer['source_task']=oldtransfer.task;oldtransfer['task']=oldtransfer.block
        transfer=pd.concat([oldtransfer,transfer],ignore_index=True)
        assert not transfer.duplicated(KEY).any()
        write_summary(transfer,out,'transfer_complete');tff.to_csv(out/'transfer_new_per_feature.csv',index=False)
        for df,pairs,metrics,prefix in [(primary,primary_pairs,U+R,'primary_complete'),(target,target_pairs,U+R,'target'),
                (joint,joint_pairs,U+['sex_aod','sex_dp','subgroup_aod','subgroup_dp','L_R','violation_005','P_R','decision_disagreement','opposite_005','L_third'],'joint_complete'),
                (transfer,primary_pairs,U+R,'transfer_complete')]:
            contrasts(df,pairs,metrics,prefix).to_csv(out/(prefix+'_paired_tests.csv'),index=False)
            if not args.skip_ranks:ranks(df,out,prefix,metrics)
    write(out/'MANIFEST.json',dict(status='PARTIAL' if args.allow_partial else 'COMPLETE',phase=args.phase,sources=list(SOURCES.values()),script_sha256=sha(__file__)))
    print(new.groupby(['task','architecture','arm'])[['auc','aod','L_R','decision_disagreement']].mean().to_string())
if __name__=='__main__':main()
