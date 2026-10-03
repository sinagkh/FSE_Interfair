"""R1: outcome-independent typed banks; preserve compatible Credit predictors."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import argparse
import copy
import itertools
import time
import pandas as pd
from task_context import *
from core import Encoder, Support, categorical
from partitions import data_for as previous_data
from credit import prepare as credit_original, MONEY, CATS

RAW_ACS = _PACKAGE / 'data/raw/acs/2018/1-Year/psam_p24.csv'
ACS_FIELDS = ['AGEP','COW','SCHL','MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P']

def categories_by_frequency(frame, columns):
    counts=frame.groupby(columns,dropna=False).size()
    pairs=[(list(key) if isinstance(key,tuple) else [key],int(n)) for key,n in counts.items()]
    return sorted(pairs,key=lambda v:(-v[1],tuple(map(str,v[0]))))

def category_spec(name, columns, train, kind='nominal', rank=1, training=True):
    levels=categories_by_frequency(train,columns)
    assert len(levels)>rank,(name,len(levels))
    return dict(name=name,columns=columns,low=levels[0][0],high=levels[rank][0],
                kind=kind,training=training,rule=f'most frequent versus frequency rank {rank+1}; training rows only',
                observed_levels=len(levels),endpoint_training_mass=sum(levels[j][1] for j in (0,rank))/len(train))

def acs_frame_and_splits():
    raw=pd.read_csv(RAW_ACS,usecols=ACS_FIELDS+['SERIALNO','SPORDER','PINCP','PWGTP'])
    raw['row_id']=np.arange(len(raw))
    f=raw[(raw.AGEP>16)&(raw.PINCP>100)&(raw.WKHP>0)&(raw.PWGTP>=1)].copy().reset_index(drop=True)
    assert len(f)==33042 and not f.duplicated(['SERIALNO','SPORDER']).any()
    assert not f[ACS_FIELDS].isna().any().any()
    f['label']=(f.PINCP>50000).astype(np.float32);f['protected']=(f.SEX==2).astype(int)
    for c in ACS_FIELDS:
        if c not in ['AGEP','WKHP']:f[c]=f[c].map(lambda x:str(int(x)))
    # Reuse the already inspected household assignment wherever it exists.
    old=read_pickle(BREADTH/'cache/acs_employment_sex_2000.pkl')
    groups={k:set(old['frame'].SERIALNO.iloc[ix]) for k,ix in old['splits'].items()}
    lookup={g:k for k,gg in groups.items() for g in gg}
    missing=sorted(set(f.SERIALNO)-set(lookup))
    for g in missing:
        u=int(hashlib.sha256(('73404:'+str(g)).encode()).hexdigest()[:16],16)/2**64
        lookup[g]='train' if u<.65 else 'val' if u<.77 else 'calibration' if u<.80 else 'audit'
    role=f.SERIALNO.map(lookup)
    parts={k:np.flatnonzero(role==k) for k in ('train','val','calibration','audit')}
    provenance=dict(reused_household_assignment=str(BREADTH/'cache/acs_employment_sex_2000.pkl'),
                    reused_source_hash=sha(BREADTH/'cache/acs_employment_sex_2000.pkl'),
                    additional_households=len(missing),additional_rule='SHA256(73404:SERIALNO), 65/12/3/20 percent intervals',
                    previous_source_inspection=True,label='PINCP > 50000',eligible_rows=len(f),raw_rows=len(raw))
    return f,parts,provenance

def acs_specs(f,parts):
    tr=f.iloc[parts['train']];family=tr[tr.RELP.astype(int)<16]
    specs=[category_spec('work_class',['COW'],tr),
           category_spec('household',['MAR','RELP'],family,'linked_nominal'),
           category_spec('occupation',['OCCP'],tr)]
    lo,hi=tr.SCHL.astype(int).quantile([.2,.8],interpolation='nearest').to_numpy()
    specs.append(dict(name='education',columns=['SCHL'],low=[str(lo)],high=[str(hi)],kind='ordinal_category',training=True,rule='training observed p20/p80'))
    # str(np.int64) is stable; a float quantile is converted explicitly.
    specs[-1]['low']=[str(int(lo))];specs[-1]['high']=[str(int(hi))]
    lo,hi=tr.WKHP.quantile([.2,.8],interpolation='nearest').to_numpy()
    assert lo<hi
    specs.append(dict(name='hours',columns=['WKHP'],low=[float(lo)],high=[float(hi)],kind='numeric',training=True,rule='training p20/p80'))
    for name,cols,data,kind in [('work_class',['COW'],tr,'nominal'),('household',['MAR','RELP'],family,'linked_nominal'),('occupation',['OCCP'],tr,'nominal')]:
        for rank in ([2,4] if name=='occupation' else [2]):
            specs.append(category_spec(name+f'_edge_rank{rank+1}',cols,data,kind,rank,False))
    for a,b in [(16,21),(21,22)]:
        specs.append(dict(name=f'education_{a}_{b}',columns=['SCHL'],low=[str(a)],high=[str(b)],kind='ordinal_category',training=False,rule='prespecified observed education levels'))
    for delta in [5,10]:
        specs.append(dict(name=f'hours_plus{delta}',columns=['WKHP'],delta=delta,kind='numeric_increment',training=False,rule='prespecified weekly-hour increment'))
    return specs

def compile_rectangles(raw,spec,enc,task):
    frames=[]
    for s in (0,1):
        for side in ('low','high'):
            q=raw.copy();q['protected']=s
            if 'delta' in spec:
                q[spec['columns'][0]]=raw[spec['columns'][0]]+(spec['delta'] if side=='high' else 0)
            else:
                for c,v in zip(spec['columns'],spec[side]):q[c]=v
            if task=='acs_income':q['SEX']=str(s+1)
            else:q['SEX']=s+1
            frames.append(q)
    valid=np.ones(len(raw),bool)
    if task=='acs_income':
        for q in frames:
            valid &= q.AGEP.gt(16).to_numpy() & q.WKHP.between(1,99).to_numpy()
            valid &= ((q.RELP!='1')|(q.MAR=='1')).to_numpy()
        if spec['name'].startswith('household'):valid &= raw.RELP.astype(int).lt(16).to_numpy()
    if 'delta' in spec:
        c=spec['columns'][0];st=enc.stats[c]
        valid &= raw[c].between(st['q01'],st['q99']).to_numpy()
        valid &= (raw[c]+spec['delta']).between(st['q01'],st['q99']).to_numpy()
    c=np.stack([enc.transform(q) for q in frames])
    assert np.array_equal(c[0,:,1:],c[2,:,1:]) and np.array_equal(c[1,:,1:],c[3,:,1:])
    assert (c[:2,:,0]==0).all() and (c[2:,:,0]==1).all()
    allowed={j for j,col in enumerate(enc.columns) if any(col.startswith(n+'=') or col.startswith(n+'__') for n in spec['columns'])}
    changed=set(np.flatnonzero((c[0]!=c[1]).any(0)));assert changed and changed<=allowed,(spec,changed,allowed)
    start=1+2*len(enc.continuous)
    for name in enc.categories:
        width=len(enc.vocab[name])+1;assert (c[:,:,start:start+width].sum(2)==1).all();start+=width
    assert start==c.shape[2]
    # Every affine additive oracle has zero mixed difference, including linked edits.
    w=np.linspace(-.2,.3,c.shape[2]);z=c.astype(np.float64)@w
    assert np.max(abs(z[3]-z[2]-z[1]+z[0]))<1e-10
    assert np.max(abs(np.ones_like(z)[3]-np.ones_like(z)[2]-np.ones_like(z)[1]+np.ones_like(z)[0]))==0
    return c,valid,sorted(changed)

def build(task):
    destination=HERE/'cache'/f'{task}.pkl'
    if destination.exists():
        record=json.loads(destination.with_suffix('.json').read_text());assert record['compiler_sha256']==sha(__file__)
        print('existing prepared data',task,flush=True);return
    start=time.monotonic();checks=[]
    if task=='acs_income':
        f,parts,provenance=acs_frame_and_splits()
        enc=Encoder(['AGEP','WKHP'],[c for c in ACS_FIELDS if c not in ['SEX','AGEP','WKHP']]).fit(f.iloc[parts['train']])
        x=enc.transform(f);specs=acs_specs(f,parts)
        anchors={'train':parts['train']}
        for split,size,offset in [('val',768,1),('audit',2048,2)]:
            anchors[split]=np.sort(np.random.default_rng(93410+offset).choice(parts[split],min(size,len(parts[split])),replace=False))
        banks={k:{} for k in anchors};metadata=dict(task=task,raw_sha256=sha(RAW_ACS),target_source_sha256=sha(HERE/'sources/folktables_acs.py'),source_provenance=provenance,
                 additional_protected_context=['AGEP','RAC1P','POBP'],ordinary_blocks=['COW','SCHL','MAR+RELP','OCCP','WKHP'])
    elif task=='credit_broad':
        d=previous_data('default_credit',2000);original=credit_original()
        f=d['frame'];parts=d['splits'];enc=d['encoder'];x=d['x']
        anchors={k:original['banks']['train']['LIMIT_BAL']['indices'] if k=='train' else d['banks'][k]['LIMIT_BAL']['indices'] for k in ('train','val','audit')}
        banks={'train':{n:b for n,b in original['banks']['train'].items() if b['training']},'val':copy.deepcopy(d['banks']['val']),'audit':copy.deepcopy(d['banks']['audit'])}
        specs=[dict(name=s['name'],columns=[s['column']],low=[s['low']],high=[s['high']],kind='numeric',training=True,rule=s['rule'],reused_bank=True) for s in original['metadata']['specs']]
        for c in CATS:
            specs.append(category_spec(c,[c],f.iloc[parts['train']]))
            specs.append(category_spec(c+'_edge_rank3',[c],f.iloc[parts['train']],rank=2,training=False))
        metadata=dict(task=task,raw_sha256=original['metadata']['raw_sha256'],source_provenance={'compatible_predictor_task':'default_credit'},additional_protected_context=['AGE'],ordinary_blocks=MONEY+CATS,
                      special_codes='EDUCATION 0/5/6, MARRIAGE 0, PAY histories -2/0 are retained as observed nominal levels; no undocumented favorable ordering assigned',
                      original_data_hashes={k:array_hash(d[k]) for k in ('x','y','s')})
        for seed in DEVELOPMENT+FINAL:
            other=previous_data('default_credit',seed)
            assert all(array_hash(other[k])==array_hash(d[k]) for k in ('x','y','s'))
            assert all(np.array_equal(other['splits'][k],v) for k,v in parts.items())
            assert other['encoder'].metadata()==enc.metadata()
        checks.append(dict(check='all 12 archived Credit seed inputs/labels/encoders/partitions identical',passed=True))
    else:raise ValueError(task)
    for a,b in itertools.combinations(parts,2):
        assert not np.intersect1d(parts[a],parts[b]).size
        if task=='acs_income':assert not np.intersect1d(f.SERIALNO.iloc[parts[a]],f.SERIALNO.iloc[parts[b]]).size
    assert np.array_equal(np.sort(np.concatenate(list(parts.values()))),np.arange(len(f)))
    checks.append(dict(check='complete disjoint partitions and household disjointness where applicable',passed=True))
    counts={k:{} for k in anchors};support=Support(x[parts['train']]);edges=[]
    for spec in specs:
        spec.update(target=0.,scalar='favorable-class logit',authority='developer-supplied equal-effect requirement',direction=None)
        if task=='acs_income' and spec['name'].startswith(('hours','education')):spec['direction']='positive orientation for diagnostics; not a training constraint'
        for split,ix in anchors.items():
            if split=='train' and not spec['training']:continue
            name=spec['name']
            if spec.get('reused_bank'):
                bank=banks[split][name];assert np.array_equal(bank['indices'],ix)
                counts[split][name]=dict(anchors=len(ix),valid=int(bank['valid'].sum()),supported=int(bank['supported'].sum()),reused=True)
                continue
            corners,valid,changed=compile_rectangles(f.iloc[ix],spec,enc,task)
            keep=np.zeros(len(ix),bool)
            if valid.any():keep[valid],_=support.mask(corners[:,valid])
            counts[split][name]=dict(anchors=len(ix),valid=int(valid.sum()),supported=int(keep.sum()),support_by_group={str(s):int((keep&(f.protected.iloc[ix].to_numpy()==s)).sum()) for s in (0,1)})
            if split=='train':
                candidates=np.flatnonzero(keep);assert len(candidates)>=64,('insufficient support; diagnose before fitting',task,name,len(candidates))
                selected=np.sort(np.random.default_rng(93420+specs.index(spec)).permutation(candidates)[:8192])
                corners=corners[:,selected];ix_used=ix[selected];valid=valid[selected];keep=keep[selected]
            else:ix_used=ix
            banks[split][name]=dict(corners=corners,indices=ix_used,valid=valid,supported=keep,training=spec['training'])
            checks.append(dict(check='typed edits, protected isolation, allowed columns, constant/additive null',split=split,edit=name,passed=True,changed=[enc.columns[j] for j in changed]))
            print(task,split,name,counts[split][name],flush=True)
        if spec['kind'] in ('nominal','linked_nominal','ordinal_category'):
            cols=spec['columns'];tr=f.iloc[parts['train']];lv=categories_by_frequency(tr,cols)
            covered=[v for v,n in lv if v in [spec['low'],spec['high']]]
            edges.append(dict(edit=name,columns='+'.join(cols),training=spec['training'],observed_levels=len(lv),endpoint_levels=len(covered),possible_unordered_edges=len(lv)*(len(lv)-1)//2,
                              low=str(spec['low']),high=str(spec['high']),endpoint_mass=sum(n for v,n in lv if v in covered)/len(tr)))
    labels=f.label.to_numpy(np.float32);protected=f.protected.to_numpy()
    assert min(int(((labels[ix]==y)&(protected[ix]==s)).sum()) for ix in parts.values() for s in (0,1) for y in (0,1))>=10
    ordinary=[s for s in specs if s['training']];assert len(ordinary)==(21 if task=='credit_broad' else 5)
    metadata.update(rows=len(f),encoded_width=x.shape[1],split_rows={k:len(v) for k,v in parts.items()},split_hashes={k:array_hash(v) for k,v in parts.items()},encoder=enc.metadata(),specs=specs,bank_counts=counts,
                    active_requirement=f'zero favorable-logit interaction across {len(ordinary)} declared ordinary semantic blocks',support_thresholds=support.thresholds,support_counts=support.counts,
                    checks=checks,compiler_sha256=sha(__file__),model_outcomes_used=False,preparation_seconds=time.monotonic()-start,
                    label_definition='non-default' if task=='credit_broad' else 'income > 50000 under published ACSIncome eligibility; descriptive prediction, not a recorded allocation decision')
    data=dict(frame=f,x=x,y=labels,s=protected,splits=parts,encoder=enc,banks=banks,metadata=metadata,edits=[],education_map={})
    save_pickle(destination,data)
    write(destination.with_suffix('.json'),dict(status='PASS',task=task,cache_sha256=sha(destination),compiler_sha256=sha(__file__),metadata=metadata))
    write(HERE/'reports'/f'{task}_DATA.json',metadata)
    pd.DataFrame(edges).to_csv(HERE/'reports'/f'{task}_CATEGORY_COVERAGE.csv',index=False)
    print('DATA PASS',task,flush=True)

def data_for(task,seed=None):
    assert task in TASKS
    path=HERE/'cache'/f'{task}.pkl';meta=json.loads(path.with_suffix('.json').read_text())
    assert meta['compiler_sha256']==sha(__file__)
    return read_pickle(path)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=TASKS);a=p.parse_args();build(a.task)
