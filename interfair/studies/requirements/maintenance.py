"""Task-update regression and matched full-objective maintenance."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
from engine import *

def run(seed):
    task='hmda_oh';out=P/'maintenance'/c.phase(seed)/str(seed)
    if (out/'DONE.json').exists():return
    if seed<2000:
        for f,h in read(P/'FREEZE.json')['files'].items():assert c.sha(f)==h,f
    d,b,db=q.data(task,seed);names=list(b);src=source(task,seed,'equality_direction')
    ref=read(source(task,seed).parent/'DONE.json')['selected']['validation']
    cfg=c.r.recipe(task,'mlp',1.);tr=d['splits']['train'];x=torch.as_tensor(d['x']);y=torch.as_tensor(d['y']);s=torch.as_tensor(d['s'])
    def make_pools(bs):return {n:dict(ids=v['indices'][v['supported']],c=torch.as_tensor(v['corners'][:,v['supported']])) for n,v in bs.items()}
    ordinary=make_pools(b);signed=make_pools({n:db['train'][n] for n in q.MAP[task]})
    rows=[];records=[];out.mkdir(parents=True,exist_ok=True)
    def audit(model,stage,path):
        v,edits=q.evaluate(model,d,names,db,'audit',out/(stage+'.npz'))
        rows.append(dict(task=task,architecture='mlp',seed=seed,arm=stage,stage=stage,checkpoint=str(path),**v))
        pd.DataFrame(edits).to_csv(out/(stage+'_edits.csv'),index=False)
    m=c.load(src,d,'mlp','cpu');audit(m,'before',src);del m
    for stage in ('update','task_control','rerepair'):
        path=src if stage=='update' else out/'update/selected.pt'
        dest=out/stage;dest.mkdir(exist_ok=True)
        m=c.load(path,d,'mlp','cpu');initial=c.state_hash(m)
        opt=torch.optim.AdamW(m.parameters(),lr=.0001,weight_decay=.0001)
        brng=np.random.default_rng(seed+(801 if stage=='update' else 802));arng=np.random.default_rng(seed+803)
        bh=hashlib.sha256();ah=hashlib.sha256();steps=0;queries=0;history=[];best=None;start=time.monotonic()
        config=dict(task=task,architecture='mlp',seed=seed,stage=stage,epochs=10,lr=.0001,
          initial_checkpoint=str(path),initial_state_sha256=initial,pretrained_weights_loaded=True,
          initialization='explicit maintenance continuation',selection_uses_audit=False,
          direction_weight=1. if stage=='rerepair' else 0.,decision_rate_weight=16. if stage=='rerepair' else 0.,
          decision_response_weight=4. if stage=='rerepair' else 0.,soft_recipe=cfg,
          trainer_sha256=c.sha(__file__),protocol_sha256=None)
        (dest/'TRAINER.py').write_bytes(Path(__file__).read_bytes());write(dest/'CONFIG.json',config)
        for epoch in range(1,11):
            order=brng.permutation(tr)
            for off in range(0,len(order),512):
                ii=order[off:off+512];bh.update(ii.tobytes());steps+=1
                m.train();opt.zero_grad(set_to_none=True);F.binary_cross_entropy_with_logits(m(x[ii]),y[ii]).backward()
                if stage!='update':
                    m.eval();ii=tr[c.r.balanced_positions(tr,d,arng,512)];ah.update(ii.tobytes());z=m(x[ii]);queries+=len(ii)
                    active=1. if stage=='rerepair' else 0.
                    loss=c.r.score_loss(z,s[ii],y[ii],cfg)+rb.rate_loss(z,y[ii],{'protected':s[ii]},16.,(.25,.5,1.))
                    (active*loss).backward()
                    name=names[int(arng.integers(len(names)))];pool=ordinary[name];jj=c.r.balanced_positions(pool['ids'],d,arng,256);ii=pool['ids'][jj]
                    cc=pool['c'][:,jj];ah.update(name.encode());ah.update(ii.tobytes());z=m(cc.reshape(-1,cc.shape[-1])).reshape(4,-1);queries+=len(ii)*4
                    with torch.no_grad():nz=m(x[ii]);queries+=len(ii)
                    loss=c.r.pair_effect_loss(z,nz,s[ii],y[ii],cfg)+4.*rb.transition_loss(z,False,(.15,.35,.7));(active*loss).backward()
                    name=list(signed)[int(arng.integers(len(signed)))];pool=signed[name];jj=c.r.balanced_positions(pool['ids'],d,arng,256);ii=pool['ids'][jj]
                    cc=pool['c'][:,jj];ah.update(name.encode());ah.update(ii.tobytes());z=m(cc.reshape(-1,cc.shape[-1])).reshape(4,-1);queries+=len(ii)*4
                    e=torch.stack([z[1]-z[0],z[3]-z[2]]);rv=e[1]-e[0];ad=F.relu(-q.MAP[task][name]*e)
                    loss=cfg['pair']*(rv.square().mean()+.25*c.r.relations.topk_square(rv,.1))+ad.square().mean()+.25*c.r.relations.topk_square(ad.flatten(),.1)
                    (active*loss).backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step()
            if epoch%5==0:
                v,_=q.evaluate(m,d,names,db,'val');key,admitted=primary.select_key(v,ref)
                item=dict(epoch=epoch,validation=v,key=key,admitted=admitted);history.append(item)
                if stage=='update' or best is None or key<tuple(best['key']):
                    best=copy.deepcopy(item);torch.save(dict(state_dict={k:v.detach().clone() for k,v in m.state_dict().items()},config=config,epoch=epoch),dest/'selected.pt')
                print(seed,stage,epoch,v['L_R'],v['direction_adverse_005'],flush=True)
        j=dict(status='complete',config=config,selected=best,checkpoint=str(dest/'selected.pt'),checkpoint_sha256=c.sha(dest/'selected.pt'),
          initial_state_sha256=initial,task_batch_sha256=bh.hexdigest(),anchor_sha256=ah.hexdigest(),optimizer_updates=steps,
          regularizer_model_rows=queries,seconds=time.monotonic()-start,selection_completed_before_audit=True)
        write(dest/'HISTORY.json',history);write(dest/'DONE.json',j);records.append(j)
        m=c.load(dest/'selected.pt',d,'mlp','cpu');audit(m,stage,dest/'selected.pt');del m
    for field in ('initial_state_sha256','task_batch_sha256','anchor_sha256','optimizer_updates','regularizer_model_rows'):
        assert records[1][field]==records[2][field],field
    pd.DataFrame(rows).to_csv(out/'per_seed.csv',index=False)
    write(out/'DONE.json',dict(status='complete',source=str(src),source_sha256=c.sha(src),matched_followup_budgets=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.seed)
