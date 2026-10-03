"""Capture changed-model queries once; all downstream analyses reuse them."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import json,sys
import numpy as np
from repair import HERE,OLD,DEV,write,sha,read_pickle,array_hash
from calibration_audit import fit_temperature


def scores(pipeline,x):
    return {sc:pipeline(x,sc) for sc in ('L','P')}


def capture(pipeline,d,out,task,seed,arch):
    dest=out/'queries';dest.mkdir(parents=True,exist_ok=True);cal=d['splits']['calibration'];z=pipeline(d['x'][cal],'L');fit=fit_temperature(z,d['y'][cal])
    write(dest/'CALIBRATION.json',dict(**fit,indices_sha256=array_hash(cal),checkpoint_fixed_before_fit=True,calibration_only=True,pipeline=pipeline.provenance()))
    ix=d['splits']['audit'];x=d['x'][ix];natural={}
    for key,xx in [('natural',x),('A0',x.copy()),('A1',x.copy())]:
        if key!='natural':xx[:,0]=int(key[1])
        natural.update({key+'_'+sc:v for sc,v in scores(pipeline,xx).items()})
    np.savez_compressed(dest/'natural_protected.npz',**natural)
    provenance=dict(checkpoint=pipeline.provenance(),fit=fit,calibration_rows=array_hash(cal),natural_rows=array_hash(ix),typed=None,transfer={})
    if task in ('adult','hmda_oh'):
        verification=json.loads((OLD/'reports/toggle_results/ASSET_VERIFICATION.json').read_text());asset=next(a for a in verification['banks'] if a['task']==task)
        assert sha(asset['bank_path'])==asset['bank_sha256'];banks=read_pickle(asset['bank_path']);a={}
        for name,b in banks.items():
            n=len(b['indices']);a.update({name+'_'+sc:v.reshape(4,n) for sc,v in scores(pipeline,b['corners'].reshape(4*n,-1)).items()})
        np.savez_compressed(dest/'typed_toggles.npz',**a);provenance['typed']=asset
    if task=='hmda_oh':
        direction={}
        for kind in ('fixed','joint'):
            path=DEV/'cache_v2'/f'hmda_direction_{kind}.pkl';bank=read_pickle(path)
            for name,b in bank['banks']['audit'].items():
                if not name.startswith(kind+'_income_plus'):continue
                n=len(b['indices']);direction.update({name+'_'+sc:v.reshape(4,n) for sc,v in scores(pipeline,b['corners'].reshape(4*n,-1)).items()})
        np.savez_compressed(dest/'income_edits.npz',**direction)
        sys.path.insert(0,str(OLD/'analysis'))
        from transfer_queries import raw_scores,variants
        for block in ('hmda_2019_mi','hmda_2020_oh'):
            path=OLD/'external'/f'{block}.pkl';target=read_pickle(path);a=raw_scores(pipeline,target['data']);variants(a,fit['T'],'probability')
            np.savez_compressed(dest/(block+'.npz'),**a);provenance['transfer'][block]=dict(path=str(path),sha256=sha(path),target_used_for_fitting=False,scope='previously inspected fixed target; retrospective source-selected evaluation')
    write(dest/'DONE.json',dict(status='completed',provenance=provenance,code_sha256=sha(__file__),model_selection_changed=False))
