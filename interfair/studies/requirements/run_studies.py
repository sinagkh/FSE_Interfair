"""Bounded CPU scheduling; no confirmation before a reviewed development freeze."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
from pathlib import Path
import argparse,concurrent.futures,json,os,subprocess,sys,time
P=_PACKAGE / 'studies/requirements'

def jobs(seeds):
    result=[]
    base=['full','features','contexts','population','task','augmentation']
    for task in ['hmda_oh','acs_income','credit_broad']:
        arms=base.copy()
        if task in ('hmda_oh','acs_income'):arms+=['pool512','pool2048']
        if task=='hmda_oh':arms+=['domain_pair','ranked_pair','random_pair','policy_p20','policy_valid_only','policy_p10','policy_p35','policy_local']
        for seed in seeds:
            for arm in arms:result.append(dict(kind='engine',task=task,seed=seed,arm=arm))
    for task in ['hmda_oh','acs_income','credit_broad','acs_employment_sex','acs_employment_age']:
        for seed in seeds:
            for arm in (['no_decision','no_equal','no_population'] if task=='hmda_oh' else ['no_decision']):
                result.append(dict(kind='ablation',task=task,seed=seed,arm=arm))
    for seed in seeds:result.append(dict(kind='maintenance',task='hmda_oh',seed=seed,arm='sequence'))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['development','confirmation']);p.add_argument('--workers',type=int,default=16);p.add_argument('--kind',choices=['all','engine','ablation','maintenance'],default='all');a=p.parse_args()
    if a.phase=='confirmation':assert (P/'FREEZE.json').exists(),'Review development before confirmation.'
    seeds=[2000,2001] if a.phase=='development' else list(range(1000,1010));todo=jobs(seeds)
    if a.kind!='all':todo=[j for j in todo if j['kind']==a.kind]
    (P/'logs').mkdir(exist_ok=True)
    # Discovery is immutable and shared by all allocation arms.
    discovery=[(t,s) for t in ['hmda_oh','acs_income','credit_broad'] for s in seeds] if a.kind in ('all','engine') else []
    def prep(job):
        t,s=job;log=P/'logs'/f'discovery_{t}_{s}.log'
        with log.open('w') as f:r=subprocess.run([sys.executable,str(P/'engine.py'),'discover','--task',t,'--seed',str(s)],stdout=f,stderr=subprocess.STDOUT)
        return r.returncode
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(6,a.workers)) as pool:
        codes=list(pool.map(prep,discovery))
    assert not any(codes),'Discovery failed; inspect logs.'
    results=[];start=time.monotonic();failed=False
    def one(j):
        log=P/'logs'/f"{j['kind']}_{j['task']}_{j['seed']}_{j['arm']}.log"
        args=[sys.executable,str(P/(j['kind']+'.py'))]
        if j['kind']=='maintenance':args+=['--seed',str(j['seed'])]
        else:args+=['train','--task',j['task'],'--seed',str(j['seed']),'--arm',j['arm']]
        with log.open('w') as f:
            r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
            if r.returncode==0 and j['kind']!='maintenance':
                r=subprocess.run([sys.executable,str(P/'queries.py'),'fit','--kind',j['kind'],'--task',j['task'],'--seed',str(j['seed']),'--arm',j['arm']],stdout=f,stderr=subprocess.STDOUT)
        return {**j,'returncode':r.returncode,'log':str(log)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(one,j):j for j in todo}
        for future in concurrent.futures.as_completed(futures):
            r=future.result();results.append(r);failed|=bool(r['returncode'])
            status=dict(phase=a.phase,kind=a.kind,expected=len(todo),completed=len(results),failed=[x for x in results if x['returncode']],seconds=time.monotonic()-start,results=results)
            tmp=P/(f'STATUS_{a.phase}_{a.kind}.tmp');tmp.write_text(json.dumps(status,indent=2)+'\n');os.replace(tmp,P/f'STATUS_{a.phase}_{a.kind}.json')
            print({k:r[k] for k in ['kind','task','seed','arm','returncode']},f'{len(results)}/{len(todo)}',flush=True)
    if failed:raise RuntimeError('Some jobs failed; no complete-panel status is asserted.')

if __name__=='__main__':main()
