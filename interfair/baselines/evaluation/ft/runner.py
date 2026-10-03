"""Published baseline adapters for FT-Transformer and employment tasks."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[3]
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys, ast, json, time, copy, hashlib, importlib.util, random, warnings, argparse
from types import SimpleNamespace
P=_PACKAGE / 'baselines/evaluation/ft'; STUDY=_PACKAGE / 'baselines/evaluation'; ROOT=_PACKAGE
sys.path.insert(0,str(ROOT/'training/shared'))
import common as c
from common import np,pd,torch,nn,F
from sklearn.linear_model import LogisticRegression
from core import numeric,categorical
torch.set_num_threads(1)
NATIVE=_PACKAGE / 'vendor/fairer/celeba/utils_train_alexnet.py'
PREPROCESS=_PACKAGE / 'baselines/resampling/resampling.py'
CODE_FILES=[Path(__file__),Path(c.__file__),NATIVE,PREPROCESS,
 ROOT/'core/preprocessing_baselines.py',
 ROOT/'core/ft_architecture.py',
 ROOT/'vendor/fair_smote/Generate_Samples.py']

def directory(task,seed,arch,arm):
    return STUDY/('ft' if arch=='ft' else 'employment')/'runs'/c.phase(seed)/arch/task/str(seed)/arm

def fairsmote(d,seed,out):
    # Execute the existing preprocessing function verbatim, without importing its
    # unrelated experiment orchestration (which has another module named common).
    node=next(n for n in ast.parse(PREPROCESS.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='fair_smote')
    env=dict(_PACKAGE=ROOT,HERE=ROOT/'baselines/resampling',importlib=__import__('importlib'),pd=pd,np=np,
             time=time,warnings=warnings,random=random,LogisticRegression=LogisticRegression,
             numeric=numeric,categorical=categorical,p=c,write=c.write)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(PREPROCESS),'exec'),env)
    return env['fair_smote'](d,seed,out)

def source(task,seed,arch):
    if seed<2000:
        f=pd.read_csv(ROOT/'training/transformer/exports/primary_complete_per_seed.csv')
        row=f[(f.task==task)&(f.architecture==arch)&(f.arm=='erm')&(f.seed==seed)].iloc[0]
        path=Path(row.checkpoint) if isinstance(row.get('checkpoint'),str) else Path(row.source_done).parent/'selected.pt'
        assert path.is_file(),path
        assert c.sha(path)==row.checkpoint_sha256,(path,'main-table source mismatch')
        return path
    candidates=[ROOT/'training/mlp/runs/development'/task/str(seed)/'erm/selected.pt',c.source_path(task,seed,arch)]
    # Direction-scratch checkpoints are MLP; FT uses its native source.
    if arch=='ft':candidates=candidates[1:]
    return next(q for q in candidates if q.is_file())

def load(run_dir,device='cpu'):
    run_dir=Path(run_dir);ck=torch.load(run_dir/'selected.pt',map_location='cpu',weights_only=False)
    cfg=ck['config'];d,_=c.data(cfg['task'],cfg['seed'])
    return c.load(run_dir/'selected.pt',d,cfg['architecture'],device)

class Loader:
    def __init__(self,items):self.items=items
    def __len__(self):return len(self.items)
    def __iter__(self):
        class It:
            def __init__(self,items):self.it=iter(items)
            def __iter__(self):return self
            def __next__(self):return next(self.it)
            def next(self):return next(self.it)
        return It(self.items)
class Writer:
    def __init__(self):self.last={}
    def add_scalar(self,name,v,step):self.last[name]=float(v.detach()) if torch.is_tensor(v) else float(v)

class NativeMLP(nn.Module):
    """FAIRER's existing MLP adapter exposes only active network parameters."""
    def __init__(self,base):super().__init__();self.net=base.net
    def forward(self,x):return self.net(x).sigmoid()

def native(device):
    node=next(n for n in ast.parse(NATIVE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='fit_model_dp')
    class Device(ast.NodeTransformer):
        n=0
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':
                assert not node.args and not node.keywords
                node.func.attr='to';node.args=[ast.Constant(device)];self.n+=1
            return node
    transform=Device();node=transform.visit(node);assert transform.n==4
    env=dict(torch=torch,np=np,tqdm=lambda x:x,pprint=lambda *a:None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(NATIVE),'exec'),env)
    return env['fit_model_dp']

def checkpoint(base,cfg,transform,epoch,out):
    torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in base.state_dict().items()},
                    config=cfg,transform=transform,epoch=epoch),out/'selected.pt')

def run(task,seed,arch,arm):
    device='cuda' if arch=='ft' else 'cpu';out=directory(task,seed,arch,arm)
    if (out/'DONE.json').exists():return
    if seed<2000:
        freeze=json.loads((STUDY/'FROZEN_RECOVERY.json').read_text())
        for f,h in freeze['code_hashes'].items():assert c.sha(f)==h,f
    out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    (out/'RUNNER_SNAPSHOT.py').write_bytes(Path(__file__).read_bytes())
    d,banks=c.data(task,seed);c.seed_all(seed);base=c.model(d,arch).to(device);m=base
    initial=c.state_hash(base);tr=d['splits']['train'];vi=d['splits']['val']
    x=d['x'][tr].copy();y=d['y'][tr].copy();w=np.ones(len(tr),np.float32);transform=None;extra={};src=None
    adaptations=[]
    if arm=='ltdd':
        sl,it,records=c.ltdd_fit(x,d['encoder'].columns,());m=c.LTDDPipeline(base,sl,it).to(device)
        transform=dict(slope=sl.tolist(),intercept=it.tolist());extra['transform_records']=records
        if arch=='ft':
            with torch.no_grad():
                tx=m.transform(torch.as_tensor(x,device=device)).cpu().numpy()
            changes={name:float((x[:,a:b].argmax(1)!=tx[:,a:b].argmax(1)).mean()) for name,(a,b) in zip(d['encoder'].categories,base.blocks)}
            extra['training_argmax_category_change_rate']=changes
            adaptations.append('LTDD residualizes encoded inputs before native FT argmax tokenization; protected token fixed to zero.')
    elif arm in ('fairsmote','fairsmote_reference'):
        if arm=='fairsmote':x,y,w=fairsmote(d,seed,out)
        adaptations.append('Verbatim typed Fair-SMOTE preprocessing; native downstream architecture protocol.' if arch=='ft' else 'Existing Fair-SMOTE MLP recipe: lr .0007, gradient clip 5, RNG seed+991, 40 epochs.')
    elif arm=='reweighing':
        group=d['s'][tr]
        for g in (0,1):
            for label in (0,1):
                keep=(group==g)&(y==label);assert keep.any();w[keep]=(group==g).mean()*(y==label).mean()/keep.mean()
        assert abs(float(w.mean())-1)<1e-6
    elif arm=='cot_phi':
        fav=int(y[x[:,0]==1].mean()>y[x[:,0]==0].mean());oriented=x.copy()
        if fav:oriented[:,0]=1-oriented[:,0]
        x,extra=c.cot_transform(oriented,y,seed+401)
        if fav:x[:,0]=1-x[:,0]
        extra['training_favored_group']=fav
    elif arm=='dralign':
        src=source(task,seed,arch);base.load_state_dict(torch.load(src,map_location='cpu',weights_only=False)['state_dict']);c.seed_all(seed)
        adaptations.append('Literal native FAIRER DP epoch, device substitution only; sigmoid wrapper around downstream logits.')
        if arch=='mlp':adaptations.append('Expose base.net only, exactly as the existing native MLP adapter; unused Predictor offsets are excluded from rationale gradients.')
    else:assert arm in ('hifi_shared','erm')
    dr=arm=='dralign';fs_mlp=arch=='mlp' and arm in ('fairsmote','fairsmote_reference')
    epochs=20 if dr else 100 if arch=='ft' else 40
    lr=.0007 if fs_mlp else .001
    cfg=dict(task=task,seed=seed,architecture=arch,arm=arm,device=device,epochs=epochs,batch=512,
      optimizer='Adam lr=.001' if dr else 'native FT default optimizer' if arch=='ft' else f'AdamW lr={lr} wd=.0001',
      selector='minimum validation soft-DP at zero-based epochs 3..19' if dr else 'validation AUROC then BCE; FT patience16',
      initial_state_sha256=initial,initialization='source ERM' if dr else 'random',source_checkpoint=str(src) if src else None,
      source_checkpoint_sha256=c.sha(src) if src else None,source_state_sha256=c.state_hash(base) if src else None,
      data_hashes={k:c.r.array_hash(d[k]) for k in ('x','y','s')},split_hashes={k:c.r.array_hash(v) for k,v in d['splits'].items()},
      code_hashes={str(f):c.sha(f) for f in CODE_FILES},adaptations=adaptations,selection_uses_audit=False,
      training_only_preprocessing=True,training_rows=len(y),train_x_sha256=c.r.array_hash(x),train_y_sha256=c.r.array_hash(y),
      eta=.1 if arm=='hifi_shared' else None,gradient_clip=5. if fs_mlp else None,
      rationale_weight=.03 if dr else None,endpoint_weight=.3 if dr else None,batch_per_group=1000 if dr else None)
    c.write(out/'CONFIG.json',cfg);c.write(out/'TRANSFORM.json',extra)
    history=[];best=None;batch_hash=hashlib.sha256()
    if arch=='ft':torch.cuda.reset_peak_memory_stats()
    if dr:
        view=c.Probability(base) if arch=='ft' else NativeMLP(base);opt=torch.optim.Adam(view.parameters(),lr=.001)
        groups=[tr[d['s'][tr]==g] for g in (0,1)];steps=min(len(g)//1000 for g in groups);assert steps>0
        cfg['steps_per_epoch']=steps;c.write(out/'CONFIG.json',cfg)
        rng=np.random.default_rng(seed+98710);epoch_fn=native(device);writer=Writer()
        for ep in range(20):
            items=[]
            for g in groups:
                use=rng.permutation(g)[:steps*1000];batch_hash.update(use.tobytes())
                items.append([(torch.tensor(d['x'][jj]),torch.tensor(d['y'][jj])) for jj in use.reshape(steps,1000)])
            def validate(**kwargs):
                nonlocal best
                prob=c.logits(base,d['x'][vi],device);v=c.behavior(d['y'][vi],d['s'][vi],c.expit(prob))
                pp=c.expit(prob);gap=abs(float(pp[d['s'][vi]==0].mean()-pp[d['s'][vi]==1].mean()))
                row=dict(epoch=ep,soft_dp=gap,validation=v);history.append(row)
                if ep>=3 and (best is None or gap<best['soft_dp']):
                    best=row.copy();checkpoint(base,cfg,None,ep,out)
                return row.copy()
            epoch_fn(ep,view,Loader(items[0]),Loader(items[0]),Loader(items[1]),mode='CAIGA',lam=.03,lam2=.3,
                args=SimpleNamespace(epochs=20,pruning=False),criterion=nn.BCELoss(),writer=writer,optimizer=opt,pretest_call=validate)
            assert all(np.isfinite(v) for v in writer.last.values())
            c.write(out/'HISTORY.json',history)
            print(task,arch,arm,seed,ep,'seconds',round(time.monotonic()-started,1),flush=True)
    else:
        opt=base.optimizer() if arch=='ft' else torch.optim.AdamW(base.parameters(),lr=lr,weight_decay=.0001)
        x,y,w=[torch.as_tensor(v,device=device) for v in (x,y,w)]
        rng=np.random.default_rng(seed+(991 if fs_mlp else 201));view=c.Probability(base)
        for ep in range(1,epochs+1):
            m.train();perm=rng.permutation(len(x));batch_hash.update(perm.tobytes());losses=[]
            for off in range(0,len(x),512):
                jj=perm[off:off+512];xx,yy=x[jj],y[jj];opt.zero_grad(set_to_none=True)
                loss=c.native_hifi('dl',view,xx,view(xx),yy[:,None],[0],.1) if arm=='hifi_shared' else (F.binary_cross_entropy_with_logits(m(xx),yy,reduction='none')*w[jj]).mean()
                assert torch.isfinite(loss);loss.backward()
                if fs_mlp:torch.nn.utils.clip_grad_norm_(m.parameters(),5.)
                opt.step();losses.append(float(loss.detach()))
            v=c.behavior(d['y'][vi],d['s'][vi],c.expit(c.logits(m,d['x'][vi],device)));key=(-v['auc'],v['bce'])
            row=dict(epoch=ep,validation=v,key=key,mean_loss=float(np.mean(losses)));history.append(row)
            if best is None or key<tuple(best['key']):best=row.copy();checkpoint(base,cfg,transform,ep,out)
            c.write(out/'HISTORY.json',history)
            print(task,arch,arm,seed,ep,round(v['auc'],5),'seconds',round(time.monotonic()-started,1),flush=True)
            if arch=='ft' and ep-best['epoch']>=16:break
    c.write(out/'SELECTION.json',dict(selected=best,selection_finished_before_audit=True))
    fitted=c.load(out/'selected.pt',d,arch,device);audit=c.evaluate(fitted,d,device,out)
    reload_error=float(np.max(np.abs(c.logits(fitted,d['x'][vi[:64]],device)-c.logits(load(out,device),d['x'][vi[:64]],device))))
    assert reload_error<1e-6
    result=dict(status='complete',config=cfg,selected=best,epochs_completed=ep,checkpoint=str(out/'selected.pt'),
      checkpoint_sha256=c.sha(out/'selected.pt'),arrays_sha256=c.sha(out/'audit.npz'),audit=audit,
      seconds=time.monotonic()-started,batch_sha256=batch_hash.hexdigest(),reload_max_error=reload_error,
      selection_finished_before_audit=True,peak_gpu_bytes=torch.cuda.max_memory_allocated() if arch=='ft' else 0)
    c.write(out/'DONE.json',result);print('COMPLETE',task,arch,arm,seed,flush=True)

def validate_existing():
    path=ROOT/'training/shared/runs/confirmation/main/ft/hmda_oh/1000/reweighing/selected.pt'
    d,_=c.data('hmda_oh',1000);m=c.load(path,d,'ft','cuda');a=c.evaluate(m,d,'cuda')
    b=json.loads((path.parent/'AUDIT.json').read_text());metrics=['auc','aod','L_R','violation_005','decision_disagreement']
    errors={k:abs(a[k]-b[k]) for k in metrics};assert max(errors.values())<1e-6,errors
    c.write(P/'VALIDATION.json',dict(status='PASS',checkpoint=str(path),checkpoint_sha256=c.sha(path),errors=errors))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--validate',action='store_true');ap.add_argument('--job',nargs=4);a=ap.parse_args()
    if a.validate:validate_existing()
    if a.job:run(a.job[0],int(a.job[1]),a.job[2],a.job[3])
