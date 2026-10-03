"""Load full frozen prediction pipelines with explicit output semantics."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.special import expit, logit
from partitions import HERE, DEV, sha
from core import Predictor, predict


class Pipeline:
    output_kind='probability'
    queries_per_row=1
    def __init__(self,model,paths,device='cpu',batch=4096):
        self.model=model.to(device).eval();self.paths=[Path(p) for p in paths];self.device=device;self.batch=batch
    def __call__(self,x,scale='P'):
        return predict(self.model,np.asarray(x),self.device,scale,self.batch).astype(np.float64)
    def provenance(self):
        return dict(output_kind=self.output_kind,queries_per_row=self.queries_per_row,
                    sources={str(p):sha(p) for p in self.paths})


class Uniform:
    output_kind='probability'
    def __init__(self,source,sex_columns=None,sex_ids=None):
        assert source.output_kind=='probability'
        self.source=source;self.sex_columns=sex_columns;self.sex_ids=sex_ids
        self.queries_per_row=source.queries_per_row*(4 if sex_columns is not None else 2)
    def __call__(self,x,scale='P'):
        scores=[]
        for race in (0,1):
            for sx in ((0,1) if self.sex_columns is not None else (None,)):
                xx=np.array(x,copy=True);xx[:,0]=race
                if sx is not None:xx[:,self.sex_columns]=0;xx[:,self.sex_ids[sx]]=1
                scores.append(self.source(xx,'P'))
        p=np.mean(scores,axis=0,dtype=np.float64)
        return p if scale=='P' else logit(np.clip(p,1e-7,1-1e-7))
    def provenance(self):
        return dict(kind='uniform protected-input probability mixture',source=self.source.provenance(),
                    queries_per_row=self.queries_per_row,sex_columns=self.sex_columns,sex_ids=self.sex_ids,
                    transformed_logit='logit of mixture, clipped to [1e-7,1-1e-7] for finite reporting')


class Calibrated:
    output_kind='probability'
    def __init__(self,source,temperature):
        assert source.output_kind=='probability' and temperature>0
        self.source=source;self.T=float(temperature);self.queries_per_row=source.queries_per_row
    def __call__(self,x,scale='P'):
        z=self.source(x,'L')/self.T
        return expit(z) if scale=='P' else z
    def provenance(self):return dict(kind='positive temperature after source selection',T=self.T,source=self.source.provenance())


class Mirror:
    output_kind='hard_decision'
    queries_per_row=2
    def __init__(self,source,mirror,scenario,path):
        from mirrorfair_source import literal
        self.source,self.mirror,self.scenario,self.path=source,mirror,int(scenario),Path(path)
        _,self.combine=literal()
    def __call__(self,x,scale='P'):
        if scale!='P':raise ValueError('MirrorFair native pipeline has no logit output')
        p,q=self.source(x),self.mirror(x);g=1-np.asarray(x)[:,0]
        _,decision=self.combine(np.c_[1-p,p],np.c_[1-q,q],SimpleNamespace(protected_attributes=g[:,None]),self.scenario,False)
        return decision.astype(np.float64)
    def provenance(self):
        return dict(output_kind=self.output_kind,queries_per_row=2,scenario=self.scenario,
                    selection_sha256=sha(self.path),source=self.source.provenance(),mirror=self.mirror.provenance(),
                    native_combiner_sha256=sha(_PACKAGE / 'vendor/mirrorfair/code/MirrorFair.py'))


def plain(path,d,device='cpu',mode=None):
    path=Path(path);ck=torch.load(path,map_location='cpu',weights_only=False)
    model=Predictor(d['x'].shape[1],'mlp',mode or ck.get('config',{}).get('mode','erm'))
    model.load_state_dict(ck['state_dict']);return Pipeline(model,[path],device)


def load_pipeline(root,study,task,seed,arm,d,architecture='mlp',device='cpu'):
    root=Path(root);main=root/'main'/architecture/task/str(seed)
    if study=='main':
        if arm.startswith('uniform_'):
            return Uniform(load_pipeline(root,study,task,seed,arm[len('uniform_'):],d,architecture,device))
        native='all_preserve' if architecture=='ft' and arm=='interfair_preserve' else arm
        # Native archives keep their own arm spelling; the public label is stable.
        directory=main/native
        if architecture=='ft':
            from ft_architecture import load_ft
            path=directory/'selected.pt';return Pipeline(load_ft(path,d,device),[path],device,512)
        path=directory/('task.pt' if arm in ('erm','removed') else 'P.pt')
        return plain(path,d,device)
    if study=='trees':
        if arm=='uniform_erm':return Uniform(load_pipeline(root,study,task,seed,'erm',d,architecture,'cpu'))
        from tree_models import TreeScore
        path=root/'tree_models'/task/str(seed)/(arm+'.pkl')
        with path.open('rb') as f:tree=pickle.load(f)
        return Pipeline(TreeScore(tree,arm=='removed'),[path],'cpu')
    if study=='named':
        if arm in ('ltdd','cot_phi'):
            from preprocessing_baselines import load_pipeline as preprocessed
            path=root/'preprocessing_baselines'/task/str(seed)/arm
            return Pipeline(preprocessed(path,device),[path/'selected.pt'],device)
        if arm in ('balanced_task','dp_only','dralign'):
            from dralign import MLP
            path=root/'dralign'/task/str(seed)/arm/'selected.pt'
            model=MLP(Predictor(d['x'].shape[1]));model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict'])
            return Pipeline(model,[path],device)
        if arm=='mirrorfair':
            base=plain(root/'main/mlp'/task/str(seed)/'erm/task.pt',d,device)
            directory=root/'mirrorfair_common'/task/str(seed);mirror=plain(directory/'selected.pt',d,device)
            selected=json.loads((directory/'PAIR_SELECTION.json').read_text())
            return Mirror(base,mirror,selected['scenario'],directory/'PAIR_SELECTION.json')
        if arm=='neufair':
            from neufair import MaskedMLP
            path=root/'neufair'/task/str(seed)/'selected.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
            source=Predictor(d['x'].shape[1]);source.load_state_dict(ck['state_dict']);model=MaskedMLP(source);model.state=int(ck['mask'])
            return Pipeline(model,[path],device)
        raise ValueError(arm)
    if study=='learning':
        mode,kind=arm.split('_',1);path=root/'learning_modes'/task/str(seed)/mode/kind/'selected.pt'
    elif study=='gradients':path=root/'input_gradient_study'/task/str(seed)/'P'/arm/'selected.pt'
    elif study=='maintenance':path=root/'maintenance'/task/str(seed)/arm/'selected.pt'
    elif study=='selection' or study.startswith('direction_'):path=root/study/task/str(seed)/arm/'P.pt'
    elif study=='vector':
        from protected_vector import load
        path=root/'protected_vector'/str(seed)/arm/'selected.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
        return Pipeline(load(path,d,ck['config']['mask_columns']),[path],device)
    else:raise ValueError(study)
    return plain(path,d,device)
