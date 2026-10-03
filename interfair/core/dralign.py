"""Pinned FAIRER DP epoch with device-only adaptation and matched controls."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[1]
import argparse,ast,copy,hashlib,json,math,subprocess,time
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from core import ROOT,Predictor,behavior,evaluate,predict,save_json,sha,seed_all
from maintenance import data_for,erm_path

VENDOR=_PACKAGE / 'vendor/fairer';SOURCE=_PACKAGE / 'vendor/fairer/celeba/utils_train_alexnet.py';BASE=_PACKAGE / 'core/runs/dralign';PROTOCOL=_PACKAGE / 'core/reports/DRALIGN_PROTOCOL.md'

class DeviceOnly(ast.NodeTransformer):
    def __init__(self):self.n=0
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':
            assert not node.args and not node.keywords;node.func.attr='to';node.args=[ast.Constant('cpu')];self.n+=1
        return node

def native_function():
    tree=ast.parse(SOURCE.read_text());node=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='fit_model_dp');change=DeviceOnly();node=change.visit(node);assert change.n==4
    ns=dict(torch=torch,np=np,tqdm=lambda x:x,pprint=lambda *a:None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),ns)
    return ns['fit_model_dp'],ast.unparse(node)

class MLP(nn.Module):
    def __init__(self,source):super().__init__();self.net=copy.deepcopy(source.net)
    def forward(self,x):return self.net(x).sigmoid()
    def score(self,x,scale):
        z=self.net(x).squeeze(-1);return z.sigmoid() if scale=='P' else z

class Writer:
    def __init__(self):self.last={}
    def add_scalar(self,name,value,step):self.last[name]=float(value.detach()) if torch.is_tensor(value) else float(value)

class Iterator:
    def __init__(self,items):self.it=iter(items)
    def __iter__(self):return self
    def __next__(self):return next(self.it)
    def next(self):return next(self.it)

class Loader:
    def __init__(self,items):self.items=items
    def __len__(self):return len(self.items)
    def __iter__(self):return Iterator(self.items)

def objective(model,xs,ys,endpoint=.3,rationale=.03):
    outs=[model(x).squeeze(-1) for x in xs];sup=nn.functional.binary_cross_entropy(torch.cat(outs),torch.cat(ys));gap=abs(outs[0].mean()-outs[1].mean())
    pars=list(model.parameters());grads=[torch.autograd.grad(nn.functional.binary_cross_entropy(o,y),pars,create_graph=True,retain_graph=True) for o,y in zip(outs,ys)];cos=[]
    for g0,g1,(name,p) in zip(*grads,model.named_parameters()):
        if 'weight' in name and 'gate' not in name:
            a=(p*g0).square().reshape(p.shape[0],-1).sum(1);b=(p*g1).square().reshape(p.shape[0],-1).sum(1);cos.append(torch.cosine_similarity(a,b,dim=0))
    alignment=torch.stack(cos).sum();return sup+endpoint*gap-rationale*alignment,dict(supervised=sup,gap=gap,alignment=alignment)

def fidelity():
    torch.set_num_threads(1);native,transformed=native_function();checks=[];seed_all(98711)
    for dtype in (torch.float32,torch.float64):
        src=Predictor(5,'mlp','erm',width=8);a=MLP(src).to(dtype);b=copy.deepcopy(a);xs=[torch.randn(12,5,dtype=dtype),torch.randn(12,5,dtype=dtype)+.4];ys=[torch.randint(0,2,(12,)).to(dtype) for _ in range(2)];writer=Writer()
        opta=torch.optim.Adam(a.parameters(),lr=.001);optb=torch.optim.Adam(b.parameters(),lr=.001)
        # Targets are binary. Cast criterion target to prediction dtype only for double verification.
        if dtype in (torch.float32,torch.float64):
            native(0,a,Loader([(xs[0],ys[0])]),Loader([(xs[0],ys[0])]),Loader([(xs[1],ys[1])]),mode='CAIGA',lam=.03,lam2=.3,args=SimpleNamespace(epochs=1,pruning=False),criterion=lambda p,y:nn.functional.binary_cross_entropy(p,y.to(p.dtype)),writer=writer,optimizer=opta,pretest_call=lambda **kw:{})
            loss,terms=objective(b,xs,ys);optb.zero_grad();loss.backward();optb.step()
            error=max(float(abs(p-q).max().detach()) for p,q in zip(a.parameters(),b.parameters()));gerror=max(float(abs(p.grad-q.grad).max()) for p,q in zip(a.parameters(),b.parameters()));lerror=abs(writer.last['Loss/loss']-float(loss.detach()));assert max(error,gerror,lerror)<=2e-6
            checks.append(dict(name='literal_native_epoch_update',dtype=str(dtype),parameter_error=error,gradient_error=gerror,loss_error=lerror,passed=True))
        if dtype==torch.float64:
            loss,terms=objective(a,xs,ys);alignment=terms['alignment'];grads=torch.autograd.grad(alignment,list(a.parameters()));errs=[]
            # Direct finite difference of the full gradient-through-gradient objective.
            for p,g in zip(a.parameters(),grads):
                flat=p.view(-1);idx=len(flat)//2;old=float(flat[idx].detach());step=1e-5
                with torch.no_grad():flat[idx]=old+step
                plus=float(objective(a,xs,ys)[1]['alignment'].detach())
                with torch.no_grad():flat[idx]=old-step
                minus=float(objective(a,xs,ys)[1]['alignment'].detach())
                with torch.no_grad():flat[idx]=old
                fd=(plus-minus)/(2*step);errs.append(abs(fd-float(g.reshape(-1)[idx]))/(1+abs(fd)))
            assert max(errs)<2e-6,errs
            _,same=objective(a,[xs[0],xs[0]],[ys[0],ys[0]]);expected=len([p for n,p in a.named_parameters() if 'weight' in n]);same_error=abs(float(same['alignment'].detach())-expected);assert same_error<2e-6,same_error
            checks.extend([dict(name='second_order_finite_difference',coordinates=len(errs),max_scaled_error=max(errs),passed=True),dict(name='identical_group_alignment',error=same_error,passed=True)])
    dest=BASE/'fidelity';dest.mkdir(parents=True,exist_ok=True);(dest/'device_adapted_native_epoch.py').write_text(transformed+'\n');save_json(ROOT/'reports/DRALIGN_FIDELITY.json',dict(status='PASS',checks=checks,upstream_commit='bee321df01851f339ff7f9b30384a817c23db040',source_sha256=sha(SOURCE),adapter_sha256=sha(__file__),scope='native DP computation on fixed synthetic MLP; not full CelebA reproduction',device_transform_count=4));print('Fidelity PASS',checks,flush=True)

def run(task,seed,arm):
    assert seed in (2000,2001) and arm in ('balanced_task','dp_only','dralign');torch.set_num_threads(1);assert json.loads((ROOT/'reports/DRALIGN_FIDELITY.json').read_text())['status']=='PASS'
    dest=BASE/task/str(seed)/arm;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'DONE.json').exists():return
    data=data_for(task,seed);path=erm_path(task,seed);source=Predictor(data['x'].shape[1],'mlp','erm');source.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict']);model=MLP(source);seed_all(seed);native,_=native_function();writer=Writer();args=SimpleNamespace(epochs=20,pruning=False)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);ix=data['splits']['train'];groups=[ix[data['s'][ix]==g] for g in (0,1)];batch=1000;steps=min(len(g)//batch for g in groups);assert steps>0
    rng=np.random.default_rng(seed+98710);batch_hash=hashlib.sha256();history=[];selected=None;started=time.monotonic()
    config=dict(task=task,seed=seed,arm=arm,source=str(path),source_sha256=sha(path),upstream_commit='bee321df01851f339ff7f9b30384a817c23db040',code_sha256=sha(__file__),upstream_sha256=sha(SOURCE),protocol_sha256=None,batch_per_group=batch,steps_per_epoch=steps,epochs=20,lr=.001,endpoint_weight=0 if arm=='balanced_task' else .3,rationale_weight=.03 if arm=='dralign' else 0,training_mode='post-training common-MLP adaptation',selector='minimum validation soft DP gap over zero-based epoch3..19',audit_used_for_selection=False)
    save_json(dest/'CONFIG.json',config)
    for epoch in range(20):
        items=[]
        for g in groups:
            order=rng.permutation(g);use=order[:steps*batch];batch_hash.update(use.tobytes());items.append([(torch.tensor(data['x'][jj]),torch.tensor(data['y'][jj])) for jj in use.reshape(steps,batch)])
        mode='CAIGA' if arm=='dralign' else 'GapReg' if arm=='dp_only' else 'van'
        def validation(**kwargs):
            nonlocal selected
            j=data['splits']['val'];p=predict(model,data['x'][j],'cpu');b=behavior(data['y'][j],data['s'][j],p);gap=abs(float(p[data['s'][j]==0].mean()-p[data['s'][j]==1].mean()));row=dict(epoch=epoch,soft_dp=gap,behavior=b)
            history.append(row)
            if epoch>2 and (selected is None or gap<selected['soft_dp']):selected=dict(epoch=epoch,soft_dp=gap,state_dict=copy.deepcopy(model.state_dict()),validation=row)
            return row.copy()
        native(epoch,model,Loader(items[0]),Loader(items[0]),Loader(items[1]),mode=mode,lam=.03 if arm=='dralign' else .3,lam2=.3,args=args,criterion=nn.BCELoss(),writer=writer,optimizer=optimizer,pretest_call=validation)
        save_json(dest/'progress.json',dict(epoch=epoch+1,seconds=time.monotonic()-started,selected_epoch=None if selected is None else selected['epoch']))
    assert selected is not None
    torch.save(dict(state_dict=model.state_dict(),config=config,epoch=19),dest/'last.pt');torch.save(dict(state_dict=selected['state_dict'],config=config,epoch=selected['epoch']),dest/'selected.pt');save_json(dest/'SELECTION.json',dict(epoch=selected['epoch'],soft_dp=selected['soft_dp'],selection_finished_before_audit=True))
    outputs={}
    for label in ('selected','last'):
        ck=torch.load(dest/(label+'.pt'),map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict']);outputs[label]=evaluate(model,data,'audit','cpu',save_path=dest/(label+'_audit.npz'))
    save_json(dest/'DONE.json',dict(status='completed',task=task,seed=seed,arm=arm,config=config,history=history,selected_epoch=selected['epoch'],batch_schedule_sha256=batch_hash.hexdigest(),training_rows_forwarded=2*batch*steps*20,seconds=time.monotonic()-started,audit=outputs));print(task,seed,arm,'DONE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fidelity',action='store_true');p.add_argument('--job',nargs=3);a=p.parse_args()
    if a.fidelity:fidelity()
    if a.job:run(a.job[0],int(a.job[1]),a.job[2])
