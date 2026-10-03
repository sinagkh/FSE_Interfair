"""Native default FT-Transformer with typed pipeline and deterministic rule gradients."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[1]
import argparse,copy,hashlib,json,os,pickle,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from core import ROOT,behavior,effects,summarize_effects,guard_loss,save_json,seed_all,sha,state_hash
from training_base import prepare,feasible,task_key
sys.path.insert(0,str(_PACKAGE / 'vendor/rtdl'))
from rtdl_revisiting_models import FTTransformer

class FTPipeline(nn.Module):
    def __init__(self,encoder,mode='erm'):
        super().__init__();self.mode=mode;self.ncont=2*len(encoder.continuous);self.blocks=[];start=1+self.ncont
        for name in encoder.categories:
            size=len(encoder.vocab[name])+1;self.blocks.append((start,start+size));start+=size
        assert start==len(encoder.columns)
        self.model=FTTransformer(n_cont_features=self.ncont,cat_cardinalities=[2]+[b-a for a,b in self.blocks],d_out=1,**FTTransformer.get_default_kwargs())
    def typed(self,x):
        protected=x[:,0].long() if self.mode!='removed' else torch.zeros(len(x),device=x.device,dtype=torch.long)
        return x[:,1:1+self.ncont],torch.stack([protected]+[x[:,a:b].argmax(1) for a,b in self.blocks],dim=1)
    def forward(self,x):return self.model(*self.typed(x)).squeeze(-1)
    def probability(self,x):return self(x).sigmoid()
    def score(self,x,scale='P'):return self.probability(x) if scale=='P' else self(x)
    def optimizer(self):return self.model.make_default_optimizer()

@torch.no_grad()
def logits(model,x,batch=512):
    model.eval();device=next(model.parameters()).device;out=[]
    for i in range(0,len(x),batch):out.append(model(torch.as_tensor(x[i:i+batch],device=device)).cpu().numpy())
    return np.concatenate(out)

@torch.no_grad()
def evaluate_ft(model,data,split,task_only=False,save_path=None):
    ids=data['splits'][split];l=logits(model,data['x'][ids]);p=torch.tensor(l).sigmoid().numpy();r=dict(behavior=behavior(data['y'][ids],data['s'][ids],p),rules={});arrays={}
    if task_only:return r
    for name,b in data['banks'][split].items():
        n=len(b['indices']);v=logits(model,b['corners'].reshape(4*n,-1)).reshape(4,n);values={'L':v,'P':torch.tensor(v).sigmoid().numpy()};keep=b['supported'];rule=dict(anchors=n,valid=int(b['valid'].sum()),supported=int(keep.sum()),training=b['training'])
        for scale,a in values.items():
            arrays[name+'_'+scale]=a
            for label,mask in [('valid',b['valid']),('supported',keep)]:rule[scale+'_'+label]=summarize_effects(a[:,mask],.01 if scale=='L' else .001,.05 if scale=='L' else .01) if mask.any() else None
        dec=(values['P']>=.5).astype(int);a,c,_=effects(dec[:,keep]);rule['decision_transition_disagreement']=float((a!=c).mean());r['rules'][name]=rule
        arrays[name+'_supported']=keep;arrays[name+'_valid']=b['valid'];arrays[name+'_indices']=b['indices']
    for scale in ('L','P'):
        values=[v[scale+'_supported']['mean_abs'] for v in r['rules'].values() if v['training'] and v[scale+'_supported'] is not None]
        r['primary_'+scale]=float(np.mean(values)) if values else None
    if save_path:Path(save_path).parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(save_path,**arrays)
    return r

def load_ft(path,data,device='cuda'):
    d=torch.load(path,map_location='cpu',weights_only=False);m=FTPipeline(data['encoder'],d['config']['mode']);m.load_state_dict(d['state_dict']);return m.to(device).eval()

def data_for(task,seed):
    with (ROOT/'cache_v2'/f'{task}_selection_{seed}_all.pkl').open('rb') as f:return pickle.load(f)

def train(task,seed,arm,output_base="runs/ft_default",data_override=None,source_base=None):
    assert seed in (2000,2001,2002);torch.set_num_threads(1);out=ROOT/output_base/task/str(seed)/arm
    if (out/'DONE.json').exists():return
    out.mkdir(parents=True,exist_ok=True);data=data_for(task,seed) if data_override is None else data_override;seed_all(seed);model=FTPipeline(data['encoder'],'removed' if arm=='removed' else 'erm').cuda();repair=arm not in ('erm','removed');source_root=(ROOT/source_base/task/str(seed)) if source_base else out.parent;source=source_root/'erm/selected.pt';reference=None
    if repair:
        model.load_state_dict(torch.load(source,map_location='cuda',weights_only=False)['state_dict']);reference=json.loads((source_root/'erm/DONE.json').read_text())['validation']
    interaction=arm in ('all','all_preserve');preserve=arm in ('control_preserve','all_preserve');initial=state_hash(model);optimizer=model.optimizer();teacher=copy.deepcopy(model).eval() if preserve else None
    if teacher is not None:
        for p in teacher.parameters():p.requires_grad_(False)
    config=dict(task=task,seed=seed,arm=arm,architecture='official default FT-Transformer',mode=model.mode,epochs=20 if repair else 100,patience=None if repair else 16,batch=512,anchor_batch=128,lr=1e-4,weight_decay=1e-5,optimizer='official make_default_optimizer',default_kwargs=FTTransformer.get_default_kwargs(),source=str(source) if repair else None,source_sha256=sha(source) if repair else None,initial_state_sha256=initial,interaction_weight=160 if interaction else 0,preserve_weight=160 if preserve else 0,guard_weight=.1 if repair else 0,main_effect_penalty=0,rule_forward='eval mode with gradients enabled',code_sha256=sha(__file__),vendor=json.loads((_PACKAGE / 'vendor/rtdl/MANIFEST.json').read_text()),concurrency=os.environ.get('INTERFAIR_CONCURRENCY','1'),audit_used_for_selection=False)
    config['training_bank_sha256']=hashlib.sha256(data['banks']['train']['selected_mixture']['corners'].tobytes()).hexdigest();config['sampling_weight_sha256']=hashlib.sha256(data['banks']['train']['selected_mixture']['sampling_weights'].tobytes()).hexdigest();config['requirement']=data['metadata'].get('active_requirement');save_json(out/'CONFIG.json',config);ids=data['splits']['train'];x=torch.tensor(data['x'][ids],device='cuda');y=torch.tensor(data['y'][ids],device='cuda');rng=np.random.default_rng(seed+201);arng=np.random.default_rng(seed+301);history=[];best=None;bestkey=None;bestepoch=0;batchhash=hashlib.sha256();anchorhash=hashlib.sha256();start=time.time();torch.cuda.reset_peak_memory_stats()
    bank=data['banks']['train']['selected_mixture'];corners=torch.tensor(bank['corners'],device='cuda');weights=bank['sampling_weights'];weights=weights/weights.sum()
    def consider(epoch):
        nonlocal best,bestkey,bestepoch
        r=evaluate_ft(model,data,'val',task_only=not repair);ok=not repair or feasible(r,reference,.005,.99);key=((r['primary_P'],)+task_key(r)) if repair else task_key(r);history.append(dict(epoch=epoch,validation=r,feasible=ok))
        if ok and (bestkey is None or key<bestkey):
            bestkey=key;bestepoch=epoch;best=dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},validation=r,epoch=epoch)
    if repair:consider(0)
    for epoch in range(1,config['epochs']+1):
        model.train();perm=rng.permutation(len(x));batchhash.update(perm.tobytes());running=[];t=time.time()
        for i in range(0,len(x),512):
            ix=torch.tensor(perm[i:i+512],device='cuda');z=model(x[ix]);p=z.sigmoid();loss=F.binary_cross_entropy_with_logits(z,y[ix])
            if repair:
                loss=loss+.1*guard_loss(p,y[ix],x[ix,0]);j=arng.choice(corners.shape[1],128,p=weights);anchorhash.update(j.tobytes())
                if interaction or preserve:
                    joined=corners[:,torch.tensor(j,device='cuda')].reshape(-1,x.shape[1]);model.eval();q=model.probability(joined).reshape(4,128);model.train();a=q[1]-q[0];b=q[3]-q[2]
                    if interaction:loss=loss+160*min(epoch/5,1)*(b-a).square().mean()
                    if preserve:
                        with torch.no_grad():v=teacher.probability(joined).reshape(4,128);target=(v[1]-v[0]+v[3]-v[2])/2
                        loss=loss+160*((a+b)/2-target).square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();running.append(float(loss.detach()))
        consider(epoch);history[-1].update(mean_loss=float(np.mean(running)),seconds=time.time()-t);save_json(out/'HISTORY.json',history)
        torch.save(dict(state_dict=best['state_dict'],config=config,epoch=bestepoch),out/'candidate.pt')
        print(json.dumps(dict(task=task,seed=seed,arm=arm,epoch=epoch,bestepoch=bestepoch,auc=history[-1]['validation']['behavior']['auc'],seconds=round(time.time()-t,2))),flush=True)
        if not repair and epoch-bestepoch>=16:break
    assert best is not None
    save_json(out/'SELECTION.json',dict(epoch=bestepoch,validation=best['validation'],selection_finished_before_audit=True,fallback_to_erm=repair and bestepoch==0,selector='feasible P residual then AUROC/BCE' if repair else 'AUROC then BCE'))
    torch.save(dict(state_dict=best['state_dict'],config=config,epoch=bestepoch),out/'selected.pt');model.load_state_dict(best['state_dict']);audit=evaluate_ft(model,data,'audit',save_path=out/'audit.npz')
    save_json(out/'DONE.json',dict(config=config,epoch=bestepoch,validation=best['validation'],audit=audit,epochs_completed=epoch,total_seconds=time.time()-start,batch_sha256=batchhash.hexdigest(),anchor_sha256=anchorhash.hexdigest(),parameter_count=sum(p.numel() for p in model.parameters()),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),status='completed',scope='two-seed development; not final confirmation'))
    (out/'candidate.pt').unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',nargs=3,required=True);args=p.parse_args();train(args.job[0],int(args.job[1]),args.job[2])
