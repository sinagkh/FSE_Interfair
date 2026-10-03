"""Literal upstream prediction/selection branch audit; no comparative training."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[1]
import ast,copy,json
from types import SimpleNamespace
import numpy as np
from core import ROOT,save_json,sha

SOURCE=_PACKAGE / 'vendor/mirrorfair/code/MirrorFair.py'

def literal():
    tree=ast.parse(SOURCE.read_text());outer=next(n for n in tree.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='r')
    select=next(n for n in outer.body if isinstance(n,ast.If) and ast.unparse(n.test)=='r == 0')
    nodes=copy.deepcopy(select.body);nodes=[n for n in nodes if not isinstance(n,ast.Expr)]
    # The source resets dif after assigning scenario; execute exactly, return scenario.
    args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=v) for v in ['pred_de','pred_de2','dataset_orig_test']],kwonlyargs=[],kw_defaults=[],defaults=[])
    init=ast.parse('dif = []').body;fn=ast.FunctionDef(name='select',args=args,body=init+nodes+[ast.Return(ast.Name('scenario',ast.Load()))],decorator_list=[])
    loop=copy.deepcopy(next(n for n in outer.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='i'))
    # One observer append; all branching/decisions remain literal.
    loop.body+=ast.parse('scores.append(prob_t)').body
    args2=ast.arguments(posonlyargs=[],args=[ast.arg(arg=v) for v in ['pred_de','pred_de2','dataset_orig_test','scenario','flag']],kwonlyargs=[],kw_defaults=[],defaults=[])
    fn2=ast.FunctionDef(name='combine',args=args2,body=ast.parse('res=[]\nscores=[]').body+[loop]+ast.parse('return np.array(scores), np.array(res)').body,decorator_list=[])
    ns=dict(np=np,mean=np.mean,std=np.std);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn,fn2],type_ignores=[])),str(SOURCE),'exec'),ns)
    return ns['select'],ns['combine']

def vector(p,q,g,scenario,flag):
    score=(p+q)/2;near=~(((p>=.55)&(q>=.55))|((p<.45)&(q<.45)));keep=near&(g==0)
    if flag and scenario in (0,1):score[keep]=np.minimum(p,q)[keep]
    elif not flag and scenario==1:score[keep]=np.maximum(p,q)[keep]
    elif not flag and scenario==2:score[keep]=1.
    return score,(score>=.5).astype(int)

def run():
    select,combine=literal();values=np.array([0.,.1,.44,.45,.46,.49,.5,.51,.54,.55,.56,.9,1.]);p,q,g=np.meshgrid(values,values,[0,1],indexing='ij');p,q,g=[v.reshape(-1) for v in (p,q,g)];checks=[]
    for scenario in (0,1,2):
        for flag in (0,1):
            a,b=combine(np.c_[1-p,p],np.c_[1-q,q],SimpleNamespace(protected_attributes=g[:,None]),scenario,flag);c,d=vector(p,q,g,scenario,flag);error=float(abs(a-c).max());assert error==0 and np.array_equal(b,d)
            checks.append(dict(scenario=scenario,bank_flag=flag,rows=len(p),max_score_error=error,decision_mismatch=0,passed=True))
    selection_cases=[]
    for label,p0,p1,group,expected in [('insensitive',[.50,.51],[.51,.52],[0,0],2),('regular',[.50,.51],[.70,.71],[0,0],0),('irregular',[.5,.5],[.5,.9],[0,0],1),('empty_near_boundary',[.2,.8],[.2,.8],[0,0],1),('no_unprivileged',[.5,.5],[.5,.5],[1,1],1)]:
        p0=np.array(p0);p1=np.array(p1)
        with np.errstate(all='ignore'):actual=select(np.c_[1-p0,p0],np.c_[1-p1,p1],SimpleNamespace(protected_attributes=np.array(group)[:,None]))
        assert actual==expected;selection_cases.append(dict(case=label,scenario=actual,passed=True))
    # An exact protected-independent pair of monotone scores. Inspect complete combination.
    inp=np.array([.46,.56,.46,.56]);groups=np.array([0,0,1,1]);score,decision=combine(np.c_[1-inp,inp],np.c_[1-inp,inp],SimpleNamespace(protected_attributes=groups[:,None]),2,0)
    residual=lambda v:float((v[3]-v[2])-(v[1]-v[0]))
    example=dict(label='constructed source-code composition example; not observed population frequency',original=inp.tolist(),mirror=inp.tolist(),combined_score=score.tolist(),decisions=decision.tolist(),source_score_residual=residual(inp),combined_score_residual=residual(score),decision_effects=[int(decision[1]-decision[0]),int(decision[3]-decision[2])])
    assert abs(example['combined_score_residual']-.54)<1e-12 and example['decision_effects']==[0,1]
    save_json(ROOT/'reports/MIRRORFAIR_GATE.json',dict(status='PASS semantics gate; no comparative efficacy results',grid_checks=checks,selection_checks=selection_cases,constructed_example=example,source_sha256=sha(SOURCE),code_sha256=sha(__file__),upstream_commit='b9359f341d525993e4ea7c4d7830b2005fc9265d',native_output='hard decisions; intermediate combined score observed separately',common_training_complete=False))
    print('PASS',len(p)*6,'prediction combinations;',len(selection_cases),'selection cases;',example)

if __name__=='__main__':run()
