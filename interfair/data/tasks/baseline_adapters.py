"""Enumerated path/device/orientation changes to admitted native producers."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
from task_context import *
from task_data import data_for
from task_pipelines import favored_group
from run_core import source_path
from method_adapters import verify, load as original_load

def native_groups(d,s):
    favored,_=favored_group(d)
    return (np.asarray(s)==favored).astype(int)

def cot_oriented(native,x,y,seed):
    rates=[float(y[x[:,0]==g].mean()) for g in (0,1)];favored=int(rates[1]>rates[0])
    xx=x.copy()
    if favored==1:xx[:,0]=1-xx[:,0]
    out,record=native(xx,y,seed)
    if favored==1:out[:,0]=1-out[:,0]
    changed=out[:,0]!=x[:,0]
    assert np.array_equal(out[:,1:],x[:,1:]) and np.all(y[changed]==1)
    assert np.all(x[changed,0]==favored) and np.all(out[changed,0]==1-favored)
    record.update(favored_common_group=favored,orientation_source='training favorable-label rates only; common group0 on tie',favorable_training_rates=rates)
    return out,record

def derive(name):
    source=verify(name);text=Path(source['derived_path']).read_text();changes=[]
    def replace(old,new,n,reason):
        nonlocal text
        assert text.count(old)==n,(name,old,text.count(old))
        text=text.replace(old,new);changes.append(dict(old=old,new=new,count=n,reason=reason))
    if name=='preprocessing_baselines':
        replace('.cuda()',".to('cpu')",2,'CPU device only')
        replace("'cuda'","'cpu'",text.count("'cuda'"),'CPU device only')
        replace("('race',) if task=='adult' else ('derived_sex',)",'()',1,'Declared all-other-coordinate LTDD adaptation; no Adult-specific secondary-race exemption')
        replace('raw,extra=cot_transform(raw,y,seed+401)','raw,extra=oriented_cot(raw,y,seed+401)',1,'Native orientation from training labels only')
    elif name=='mirrorfair_common':
        replace("g=1-d['s'][vi]","g=native_groups(d,d['s'][vi])",1,'Training-favored native group mapping')
        replace("native_group_code='1-common_protected'","native_group_code='training-favored-common-group maps to 1'",1,'Accurate provenance')
        replace("source_erm_device='cuda'","source_erm_device='cpu'",1,'Actual source-device record')
    elif name=='ft_architecture':
        replace('.cuda()',".to('cpu')",text.count('.cuda()'),'CPU FT execution; same native architecture and optimizer')
        replace("'cuda'","'cpu'",text.count("'cuda'"),'CPU FT tensor/load device')
        replace('torch.cuda.reset_peak_memory_stats()','None',1,'GPU allocator is inapplicable to CPU execution')
        replace('torch.cuda.max_memory_allocated()','0',1,'No GPU memory in CPU execution')
    else:raise ValueError(name)
    path=HERE/'adapters'/f'{name}.py';path.parent.mkdir(exist_ok=True)
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
    compile(text,str(path),'exec')
    return dict(source=source,path=str(path),sha256=sha(path),changes=changes)

def load(name,seed):
    root=HERE/'runs'/phase(seed)/'baselines'
    if name=='dralign':
        return original_load(name,root,data_provider=data_for,erm_provider=source_path,scope='bounded completion; native FAIRER procedure unchanged')
    record=derive(name);m=load_module('_completion_'+name,record['path'])
    m.campaign_base=root/name;m.BASE=m.campaign_base;m.partitions=data_for;m.data_for=data_for
    m.campaign_erm=source_path;m.erm_path=source_path;m.campaign_scope='native procedure with explicit task/device adapter'
    m.native_groups=native_groups
    if name=='preprocessing_baselines':m.oriented_cot=lambda x,y,s:cot_oriented(m.cot_transform,x,y,s)
    return m

if __name__=='__main__':
    write(HERE/'BASELINE_ADAPTERS.json',dict(status='derived; interface admission required',builder_sha256=sha(__file__),producers={n:derive(n) for n in ['preprocessing_baselines','mirrorfair_common','ft_architecture']}))
