"""Exact current-model failure accounting and finite-bank audit precision."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
from engine import *

def primary_files(task,arch,seed):
    if arch=='mlp':
        base=ROOT/'training/mlp/evaluation'/task/str(seed)
        return base/'erm.npz',base/'equality_direction.npz'
    base=ROOT/'training/transformer/evaluation/main/ft'/task/str(seed)
    return base/'erm.npz',base/'soft.npz'

def corner_pair(a,b,name):
    ia=a[name+'_indices'];ib=b[name+'_indices'];np.testing.assert_array_equal(ia,ib)
    x=a[name+'_L'].astype(float);y=b[name+'_L'].astype(float);assert x.shape==y.shape==(4,len(ia))
    return x,y

def accounting(task,arch,seed,a,b,panel,arm):
    rows=[]
    for name in [k[:-2] for k in a.files if k.endswith('_L') and k!='natural_L']:
        if name+'_L' not in b:continue
        if panel.startswith('signed_') and name not in q.MAP[task]:continue
        x,y=corner_pair(a,b,name);sign=q.MAP[task].get(name)
        fx=flags(x,sign);fy=flags(y,sign);row=dict(panel=panel,task=task,architecture=arch,seed=seed,arm=arm,feature=name,n=x.shape[1],signed=sign is not None)
        for event in ['equality','decision','opposite','direction','priority','observed','joint']:
            old=fx[event];new=fy[event]
            if event=='priority':
                # A witness is resolved only if its observed event vanishes.
                new=fy['observed']
            row[event+'_source_n']=int(old.sum());row[event+'_resolved_n']=int((old&~new).sum());row[event+'_remaining_n']=int((old&new).sum())
            row[event+'_source_pass_n']=int((~old).sum());row[event+'_introduced_n']=int((~old&new).sum());row[event+'_final_n']=int(new.sum())
            assert row[event+'_remaining_n']+row[event+'_introduced_n']==row[event+'_final_n']
        px=expit(x);py=expit(y)
        row['response_drift']=float(abs((py[1]-py[0]+py[3]-py[2]-px[1]+px[0]-px[3]+px[2])/2).mean())
        rows.append(row)
    return rows

def precision(task,arch,seed,a,b,ad,bd):
    points={};ids={};metrics=['L_R','violation_005','decision_disagreement','opposite_005','direction_adverse_005']
    for arm,arr,dirs in [('erm',a,ad),('soft',b,bd)]:
        points[arm]={m:{} for m in metrics}
        for name in [k[:-2] for k in arr.files if k.endswith('_L') and k!='natural_L']:
            z=arr[name+'_L'].astype(float);r=z[3]-z[2]-z[1]+z[0];f=flags(z)
            for metric,value in zip(metrics[:4],[abs(r),f['equality'],f['decision'],f['opposite']]):points[arm][metric][name]=value.astype(float)
            if name in ids:np.testing.assert_array_equal(ids[name],arr[name+'_indices'])
            else:ids[name]=arr[name+'_indices']
        for name,sign in q.MAP[task].items():
            z=dirs[name+'_L'].astype(float);points[arm][metrics[-1]][name]=flags(z,sign)['direction'].astype(float)
            if name in ids:np.testing.assert_array_equal(ids[name],dirs[name+'_indices'])
            else:ids[name]=dirs[name+'_indices']
    full={arm:{m:np.mean([v.mean() for v in values.values()]) for m,values in mm.items()} for arm,mm in points.items()};rows=[]
    for rep in range(30):
        rng=np.random.default_rng(stable_seed(f'precision-full:{task}:{arch}:{seed}:{rep}'));perms={n:rng.permutation(len(v)) for n,v in ids.items()}
        for budget in (64,128,256,512,1024,-1):
            selected={n:(ix if budget==-1 else ix[:budget]) for n,ix in perms.items()}
            for metric in metrics:
                values={arm:np.mean([v[selected[n]].mean() for n,v in points[arm][metric].items()]) for arm in points}
                delta=values['soft']-values['erm'];truth=full['soft'][metric]-full['erm'][metric]
                rows.append(dict(task=task,architecture=arch,seed=seed,replicate=rep,budget='full' if budget==-1 else str(budget),metric=metric,
                  erm=values['erm'],soft=values['soft'],erm_full=full['erm'][metric],soft_full=full['soft'][metric],
                  erm_abs_error=abs(values['erm']-full['erm'][metric]),soft_abs_error=abs(values['soft']-full['soft'][metric]),
                  paired_abs_error=abs(delta-truth),direction_matches_full_bank=bool(np.sign(delta)==np.sign(truth)),
                  query_rows=4*sum(len(selected[n]) for n in points['erm'][metric])))
    return rows

def main():
    out=P/'replay';out.mkdir(exist_ok=True);rows=[];budget_rows=[];sources=[]
    settings=[(t,'mlp') for t in q.MAP]+[('hmda_oh','ft'),('acs_income','ft')]
    for task,arch in settings:
        for seed in range(1000,1010):
            x,y=primary_files(task,arch,seed);a=np.load(x);b=np.load(y)
            ad=np.load(x.with_name('DIRECTION_'+x.name));bd=np.load(y.with_name('DIRECTION_'+y.name))
            rows+=accounting(task,arch,seed,a,b,'primary','soft')
            # Keep linked-direction witnesses in a distinct replay panel, without
            # adding them to ordinary feature-macro equality measurements.
            rows+=accounting(task,arch,seed,ad,bd,'signed_primary','soft')
            budget_rows+=precision(task,arch,seed,a,b,ad,bd)
            sources.extend([dict(path=str(p),sha256=c.sha(p)) for p in [x,y]])
    for task in TASKS:
        for seed in range(1000,1010):
            a=np.load(primary_files(task,'mlp',seed)[0])
            for arm in TARGET_ARMS:
                file=P/'evaluation/confirmation'/task/str(seed)/arm/'audit.npz'
                if not file.exists():continue
                rows+=accounting(task,'mlp',seed,a,np.load(file),'targeting',arm)
                da=primary_files(task,'mlp',seed)[0]
                da=np.load(da.with_name('DIRECTION_'+da.name))
                rows+=accounting(task,'mlp',seed,da,np.load(file.with_name('DIRECTION_'+file.name)),'signed_targeting',arm)
    f=pd.DataFrame(rows);f.to_csv(out/'PER_FEATURE.csv',index=False)
    count_cols=[n for n in f if n.endswith('_n') or n=='n'];key=['panel','task','architecture','seed','arm']
    totals=f.groupby(key)[count_cols].sum().reset_index()
    for event in ['equality','decision','opposite','direction','priority','observed','joint']:
        totals[event+'_resolved_rate']=totals[event+'_resolved_n']/totals[event+'_source_n'].replace(0,np.nan)
        totals[event+'_introduced_rate']=totals[event+'_introduced_n']/totals[event+'_source_pass_n'].replace(0,np.nan)
    totals.to_csv(out/'PER_SEED.csv',index=False)
    b=pd.DataFrame(budget_rows);b.to_csv(out/'AUDIT_BUDGET_PER_REPLICATE.csv',index=False)
    summary=b.groupby(['task','architecture','budget','metric']).agg(erm_mean_abs_error=('erm_abs_error','mean'),soft_mean_abs_error=('soft_abs_error','mean'),paired_mean_abs_error=('paired_abs_error','mean'),paired_p95=('paired_abs_error',lambda x:x.quantile(.95)),direction_agreement=('direction_matches_full_bank','mean'),query_rows=('query_rows','mean')).reset_index()
    summary.to_csv(out/'AUDIT_BUDGET_SUMMARY.csv',index=False)
    assert b[b.budget=='full'].paired_abs_error.max()<1e-12
    write(out/'VERIFICATION.json',dict(status='PASS',primary_pairs=70,exact_failure_accounting=True,
        source_and_target_anchor_identity=True,full_bank_sampling_error_zero=True,inputs=sources))
    print('Replayed',len(f),'feature records and',len(b),'audit budget measurements.')

if __name__=='__main__':main()
