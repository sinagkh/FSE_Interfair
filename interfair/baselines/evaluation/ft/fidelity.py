"""Independent one-update check of the literal FAIRER epoch on native FT."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[3]
from runner import *

def check():
    d,_=c.data('acs_income',2000);c.seed_all(301)
    base=c.model(d,'ft').double()
    # Removing dropout randomness permits comparison with an independent compact
    # expression. Actual runs execute the literal native epoch with native dropout.
    for module in base.modules():
        if isinstance(module,nn.Dropout):module.p=0.
    a=c.Probability(base);b=copy.deepcopy(a)
    tr=d['splits']['train'];ids=[tr[d['s'][tr]==g][:16] for g in (0,1)]
    xs=[torch.as_tensor(d['x'][j],dtype=torch.float64) for j in ids]
    ys=[torch.as_tensor(d['y'][j],dtype=torch.float64) for j in ids]
    opta=torch.optim.Adam(a.parameters(),lr=.001);optb=torch.optim.Adam(b.parameters(),lr=.001);writer=Writer()
    criterion=lambda p,y:F.binary_cross_entropy(p,y.to(p.dtype))
    native('cpu')(0,a,Loader([(xs[0],ys[0])]),Loader([(xs[0],ys[0])]),Loader([(xs[1],ys[1])]),mode='CAIGA',lam=.03,lam2=.3,
      args=SimpleNamespace(epochs=1,pruning=False),criterion=criterion,writer=writer,optimizer=opta,pretest_call=lambda **kw:{})
    outs=[b(x)[:,0] for x in xs];sup=F.binary_cross_entropy(torch.cat(outs),torch.cat(ys));gap=abs(outs[0].mean()-outs[1].mean())
    grads=[torch.autograd.grad(F.binary_cross_entropy(o,y),list(b.parameters()),create_graph=True,retain_graph=True) for o,y in zip(outs,ys)]
    terms=[]
    for g0,g1,(name,p) in zip(*grads,b.named_parameters()):
        if 'weight' in name and 'gate' not in name:
            u=(p*g0).square().reshape(p.shape[0],-1).sum(1);v=(p*g1).square().reshape(p.shape[0],-1).sum(1)
            terms.append(torch.cosine_similarity(u,v,dim=0))
    loss=sup+.3*gap-.03*sum(terms);optb.zero_grad();loss.backward();optb.step()
    errs=dict(loss=abs(float(loss.detach())-writer.last['Loss/loss']),
      parameter=max(float(abs(p-q).max().detach()) for p,q in zip(a.parameters(),b.parameters())),
      gradient=max(float(abs(p.grad-q.grad).max()) for p,q in zip(a.parameters(),b.parameters())))
    assert max(errs.values())<1e-8,errs
    c.write(P/'DRALIGN_FIDELITY.json',dict(status='PASS',errors=errs,native_sha256=c.sha(NATIVE),weight_tensors=len(terms),
       scope='literal native FAIRER CAIGA versus independent objective, native FT architecture, one full optimizer update, float64, dropout disabled only for numerical check'))
    print(errs,flush=True)
if __name__=='__main__':check()
