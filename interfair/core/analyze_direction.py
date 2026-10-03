"""Factorial analysis, including crossed edit policies and ERM-first references."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[1]
import csv,json,pickle
import numpy as np
import torch
from core import ROOT,Predictor,evaluate,save_json


def load_model(path,data):
    ckpt=torch.load(path,map_location='cpu',weights_only=False);m=Predictor(data['x'].shape[1],'mlp',ckpt['config']['mode']).cuda();m.load_state_dict(ckpt['state_dict']);return m


def main():
    torch.set_num_threads(2);rows=[];checks=[]
    data={}
    for p in ('fixed','joint'):
        with (ROOT/'cache_v2'/f'hmda_direction_{p}.pkl').open('rb') as f:data[p]=pickle.load(f)
    for seed in (2000,2001):
        source=ROOT/'runs/pilot_v3/hmda_oh/mlp'/str(seed)
        er={p:evaluate(load_model(source/'erm/task.pt',data[p]),data[p],'audit','cuda') for p in data}
        reference=er['fixed']['behavior']
        for score in ('L','P'):
            for arm in ('erm','removed','structural_L','structural_P'):
                selector='task' if arm=='erm' else score;m=load_model(source/arm/f'{selector}.pt',data['fixed'])
                for ap in data:
                    result=er[ap] if arm=='erm' else evaluate(m,data[ap],'audit','cuda')
                    for name,rule in result['rules'].items():
                        e=er[ap]['rules'][name][score+'_supported'];v=rule[score+'_supported'];b=result['behavior']
                        rows.append(dict(seed=seed,score=score,training_policy='reference',arm=arm,audit_policy=ap,edit=name,n=v['n'],residual=v['mean_abs'],reduction=1-v['mean_abs']/e['mean_abs'],negative_magnitude=v['negative_effect_mean'],opposed=v['opposed_rate'],negative_both=v['negative_both_rate'],near_flat=v['near_flat_both_rate'],common_retention=v['mean_common_abs']/max(e['mean_common_abs'],1e-12),individual_retention=v['mean_effect_abs']/max(e['mean_effect_abs'],1e-12),auc=b['auc'],f1=b['f1'],feasible=b['auc']>=reference['auc']-.01 and b['f1']>=.98*reference['f1']))
            for tp in data:
                group={}
                for arm in ('control','invariance','direction','both'):
                    directory=ROOT/'runs/direction_factorial'/tp/str(seed)/score/arm;done=json.loads((directory/'DONE.json').read_text());group[arm]=done
                    m=load_model(directory/f'{score}.pt',data[tp])
                    for ap in data:
                        result=done['selections'][score]['audit'] if ap==tp else evaluate(m,data[ap],'audit','cuda')
                        for name,rule in result['rules'].items():
                            e=er[ap]['rules'][name][score+'_supported'];v=rule[score+'_supported'];b=result['behavior']
                            rows.append(dict(seed=seed,score=score,training_policy=tp,arm=arm,audit_policy=ap,edit=name,n=v['n'],residual=v['mean_abs'],reduction=1-v['mean_abs']/e['mean_abs'],negative_magnitude=v['negative_effect_mean'],opposed=v['opposed_rate'],negative_both=v['negative_both_rate'],near_flat=v['near_flat_both_rate'],common_retention=v['mean_common_abs']/max(e['mean_common_abs'],1e-12),individual_retention=v['mean_effect_abs']/max(e['mean_effect_abs'],1e-12),auc=b['auc'],f1=b['f1'],feasible=b['auc']>=reference['auc']-.01 and b['f1']>=.98*reference['f1']))
                for key in ('batch_sha256','anchor_sha256'):
                    checks.append(dict(seed=seed,score=score,policy=tp,check=key,passed=len({v[key] for v in group.values()})==1))
                for key in ('initial_state_sha256','steering_bank_sha256'):
                    checks.append(dict(seed=seed,score=score,policy=tp,check=key,passed=len({v['config'][key] for v in group.values()})==1))
    assert all(c['passed'] for c in checks)
    with (ROOT/'reports/per_seed.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save_json(ROOT/'reports/factorial_results.json',dict(rows=rows,checks=checks))
    for policy in data:
        for score in ('L','P'):
            print(policy,score)
            for arm in ('erm','removed','control','invariance','direction','both'):
                tp='reference' if arm in ('erm','removed') else policy
                v=[x for x in rows if (x['score'],x['training_policy'],x['arm'],x['audit_policy'],x['edit'])==(score,tp,arm,policy,f'{policy}_income_plus10k')]
                print(arm,{k:round(float(np.mean([x[k] for x in v])),5) for k in ('residual','reduction','negative_both','opposed','near_flat','common_retention','auc','f1','feasible')},flush=True)


if __name__=='__main__':main()
