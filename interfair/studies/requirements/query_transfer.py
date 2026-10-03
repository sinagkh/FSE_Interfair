"""Replay the fixed source no-decision ablation under the two HMDA shifts."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
from engine import *
sys.path.insert(0,str(ROOT/'training/transformer'))
import auxiliary as aux

def one(block,seed):
    out=P/'decision_transfer'/block/str(seed)
    if (out/'DONE.json').exists():return
    path=P/'ablations/confirmation/hmda_oh'/str(seed)/'no_decision/selected.pt'
    assert (path.parent/'DONE.json').exists(),path
    src,_,_=q.data('hmda_oh',seed);d,names,db=aux.target_data(block);m=c.load(path,src,'mlp','cpu');out.mkdir(parents=True,exist_ok=True)
    v,edits,features=aux.e.evaluate(m,d,names,db,'audit',False,'cpu',out/'audit.npz')
    row=dict(task=block,architecture='mlp',seed=seed,arm='no_decision',checkpoint=str(path),checkpoint_sha256=c.sha(path),arrays=str(out/'audit.npz'),**v)
    pd.DataFrame([row]).to_csv(out/'per_seed.csv',index=False)
    write(out/'DONE.json',dict(status='complete',target_fitting=False,target_selection=False,checkpoint_sha256=c.sha(path)))

if __name__=='__main__':
    import concurrent.futures,subprocess
    if len(sys.argv)==3:one(sys.argv[1],int(sys.argv[2]))
    else:
        jobs=[(b,s) for b in ['hmda_2019_mi','hmda_2020_oh'] for s in range(1000,1010)]
        def run(j):
            b,s=j;log=P/'logs'/f'ablation_transfer_{b}_{s}.log'
            with log.open('w') as f:r=subprocess.run([sys.executable,__file__,b,str(s)],stdout=f,stderr=subprocess.STDOUT)
            return dict(block=b,seed=s,returncode=r.returncode,log=str(log))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,jobs))
        write(P/'DECISION_TRANSFER_JOBS.json',results);assert all(r['returncode']==0 for r in results)
