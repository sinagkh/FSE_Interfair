"""Common-bank linked income audits of complete source MLP pipelines."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
from engine import *
from queries import current_path
sys.path.insert(0,str(ROOT/'training/transformer'))
import baseline_directions as bd

def one(seed,arm):
    out=P/'income'/str(seed)/arm
    if (out/'DONE.json').exists():return
    d,b,db=q.data('hmda_oh',seed);hard=arm=='mirrorfair';out.mkdir(parents=True,exist_ok=True)
    if arm in ['erm','soft']:
        path=current_path('hmda_oh',seed,'mlp',arm);f=bd.Pipeline(c.load(path,d,'mlp','cpu'),[path],'cpu')
    else:
        table=bd.primary();row=table[(table.task=='hmda_oh')&(table.architecture=='mlp')&(table.seed==seed)&(table.arm==arm)].iloc[0]
        f=bd.pipeline(row,d)
    rows=[];raw={}
    for name,bank in db['audit'].items():
        if not name.startswith('linked_'):continue
        keep=bank['supported'];ids=bank['indices'][keep];xx=bank['corners'][:,keep];pp=f(xx.reshape(-1,xx.shape[-1]),'P').reshape(4,-1)
        z=None if hard else f(xx.reshape(-1,xx.shape[-1]),'L').reshape(4,-1)
        v={'direction_'+k:x for k,x in bd.metrics(pp,1,z,hard).items() if k not in ['n','sign']}
        decisions=(pp>=.5).astype(int);v['decision_disagreement']=float(((decisions[3]-decisions[2])!=(decisions[1]-decisions[0])).mean())
        if z is not None:v.update(c.effect_metrics(z,ids,d))
        v.pop('n',None)
        task='hmda_'+name.removeprefix('linked_income_')
        rows.append(dict(task=task,source_task='hmda_oh',architecture='mlp',seed=seed,arm=arm,edit=name,n=int(keep.sum()),output_kind='hard_decision' if hard else 'probability',**v))
        raw[name+'_indices']=ids;raw[name+('_decision' if hard else '_P')]=pp
        if z is not None:raw[name+'_L']=z
    np.savez_compressed(out/'corners.npz',**raw);pd.DataFrame(rows).to_csv(out/'per_seed.csv',index=False)
    write(out/'DONE.json',dict(status='complete',pipeline=f.provenance(),training=False,selection=False,queries_sha256=c.sha(out/'corners.npz')))

if __name__=='__main__':
    import concurrent.futures,subprocess
    if len(sys.argv)==3:one(int(sys.argv[1]),sys.argv[2])
    else:
        arms=['erm','soft','ltdd','cot_phi','dralign','neufair','mirrorfair','hifi_shared']
        jobs=[(s,a) for s in range(1000,1010) for a in arms]
        def run(j):
            seed,arm=j;log=P/'logs'/f'income_{seed}_{arm}.log'
            with log.open('w') as f:r=subprocess.run([sys.executable,__file__,str(seed),arm],stdout=f,stderr=subprocess.STDOUT)
            return dict(seed=seed,arm=arm,returncode=r.returncode,log=str(log))
        results=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for r in pool.map(run,jobs):
                results.append(r);write(P/'INCOME_JOBS.json',results)
                if r['returncode'] or len(results)%10==0:print(r,len(results),flush=True)
        assert all(r['returncode']==0 for r in results)
