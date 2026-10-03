"""fixed-specification ordinary update and re-repair development study."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[1]
import argparse
import copy
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.nn import functional as F

from core import ROOT, Predictor, behavior, evaluate, guard_loss, predict, save_json, seed_all, sha, state_hash
from training_base import train

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE = _PACKAGE / 'core/runs/maintenance'
PROTOCOL = _PACKAGE / 'core/reports/MAINTENANCE_PROTOCOL.md'


def data_for(task, seed):
    with (ROOT / 'cache_v2' / f'{task}_selection_{seed}_all.pkl').open('rb') as f:
        return pickle.load(f)


def erm_path(task, seed):
    return ROOT / 'runs/pilot_v3' / task / 'mlp' / str(seed) / 'erm/task.pt'


def initial_path(task, seed, arm='all'):
    if task == 'hmda_oh':
        return ROOT / 'runs/selection_preservation' / str(seed) / arm / 'P.pt'
    return BASE / task / str(seed) / ('initial_' + arm) / 'P.pt'


def load_model(path, data):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model = Predictor(data['x'].shape[1], 'mlp', 'erm').to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()


def val_behavior(model, data):
    ix = data['splits']['val']
    return behavior(data['y'][ix], data['s'][ix], predict(model, data['x'][ix], DEVICE))


def feasible(b, references, auc_budget=.005, f1_fraction=.99):
    return all(b['auc'] >= r['auc'] - auc_budget and b['f1'] >= f1_fraction*r['f1'] for r in references)


def initial(task, seed, arm):
    assert task == 'adult' and arm in ('all', 'control')
    data = data_for(task, seed)
    src = erm_path(task, seed)
    reference = json.loads((src.parent / 'DONE.json').read_text())['selections']['task']['validation']
    train(data, 'mlp', seed, arm, initial_path(task, seed, arm).parent,
          source=src, reference=reference, weight=160, preserve_weight=160,
          preserve_target='signed', selection_auc_budget=.005, selection_f1_ratio=.99,
          score_scale='P', interaction_enabled=arm == 'all')


def worker(task, seed, stage, arm):
    torch.set_num_threads(1)
    assert seed in (2000, 2001)
    if stage == 'initial':
        return initial(task, seed, arm)
    repair = stage == 'rerepair'
    assert stage in ('update', 'rerepair')
    assert arm in (('all', 'control') if repair else ('erm', 'repaired'))
    output = BASE / task / str(seed) / (stage + '_' + arm)
    if (output / 'DONE.json').exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    data = data_for(task, seed)
    source = (BASE / task / str(seed) / 'update_repaired/selected.pt') if repair else (
        erm_path(task, seed) if arm == 'erm' else initial_path(task, seed))
    seed_all(seed)
    model = load_model(source, data)
    start_hash = state_hash(model)
    teacher_path = erm_path(task, seed)
    teacher = load_model(teacher_path, data)
    for p in teacher.parameters():
        p.requires_grad_(False)
    references = [val_behavior(teacher, data)]
    if repair:
        updated_erm = load_model(BASE / task / str(seed) / 'update_erm/selected.pt', data)
        references.append(val_behavior(updated_erm, data))
        del updated_erm
    epochs, lr = (20, .001) if repair else (10, .0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.0001)
    ids = data['splits']['train']
    x = torch.as_tensor(data['x'][ids], device=DEVICE)
    y = torch.as_tensor(data['y'][ids], device=DEVICE)
    bank = data['banks']['train']['selected_mixture']
    anchors = torch.as_tensor(bank['corners'][:, bank['supported']], device=DEVICE)
    weights = bank['sampling_weights'][bank['supported']]
    assert abs(weights.sum() - 1) < 1e-12
    rng = np.random.default_rng(seed + (1201 if repair else 1101))
    arng = np.random.default_rng(seed + 1301)
    batch_hash, anchor_hash = hashlib.sha256(), hashlib.sha256()
    bank_file = ROOT / 'cache_v2' / f'{task}_selection_{seed}_all.pkl'
    config = dict(task=task, seed=seed, stage=stage, arm=arm, mode='erm', architecture='mlp',
                  source=str(source), source_sha256=sha(source), initial_state_sha256=start_hash,
                  teacher=str(teacher_path), teacher_sha256=sha(teacher_path),
                  epochs=epochs, lr=lr, weight_decay=.0001, batch=512,
                  interaction_weight=160 if repair and arm == 'all' else 0,
                  preserve_weight=160 if repair else 0, guard_weight=.1 if repair else 0,
                  bank_file_sha256=sha(bank_file), specification_features=data['metadata']['eligible_features'],
                  protocol_sha256=None, code_sha256={p.name: sha(p) for p in (ROOT/'maintenance.py', ROOT/'core.py')},
                  validation_references=references, selection='P_under_both_utility_references' if repair else 'AUROC_then_BCE',
                  audit_used_for_selection=False, concurrency=4, runtime_is_isolated=False)
    save_json(output / 'CONFIG.json', config)
    history, selected = [], None

    def consider(epoch):
        nonlocal selected
        result = evaluate(model, data, 'val', DEVICE, primary_only=True) if repair else dict(behavior=val_behavior(model, data))
        b = result['behavior']
        ok = feasible(b, references) if repair else True
        key = ((result['primary_P'],) if repair else ()) + (-b['auc'], b['bce'])
        history.append(dict(epoch=epoch, validation=result, feasible=ok))
        if ok and (selected is None or key < selected['key']):
            selected = dict(key=key, epoch=epoch, validation=result,
                            state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    start = time.time()
    train_seconds = 0.
    consider(0)
    for epoch in range(1, epochs+1):
        model.train()
        perm = rng.permutation(len(x))
        batch_hash.update(perm.tobytes())
        torch.cuda.synchronize() if DEVICE == 'cuda' else None
        epstart = time.time()
        losses = []
        for i in range(0, len(x), 512):
            ix = torch.as_tensor(perm[i:i+512], device=DEVICE)
            logits = model(x[ix])
            loss = F.binary_cross_entropy_with_logits(logits, y[ix])
            if repair:
                loss = loss + .1*guard_loss(logits.sigmoid(), y[ix], x[ix, 0])
                jj = arng.choice(anchors.shape[1], size=128, p=weights)
                anchor_hash.update(jj.tobytes())
                probe = anchors[:, torch.as_tensor(jj, device=DEVICE)].reshape(-1, x.shape[1])
                v = model.probability(probe).reshape(4, 128)
                a, b = v[1]-v[0], v[3]-v[2]
                with torch.no_grad():
                    t = teacher.probability(probe).reshape(4, 128)
                    common = (t[1]-t[0]+t[3]-t[2])/2
                loss = loss + 160*((a+b)/2-common).square().mean()
                if arm == 'all':
                    loss = loss + 160*min(epoch/5, 1.)*(b-a).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        torch.cuda.synchronize() if DEVICE == 'cuda' else None
        train_seconds += time.time()-epstart
        consider(epoch)
        history[-1]['mean_training_loss'] = float(np.mean(losses))
    last = dict(epoch=epochs, validation=history[-1]['validation'],
                state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    selections = {'last': last}
    if selected is not None:
        selections['selected'] = selected
    save_json(output/'SELECTION.json', dict(selection_finished_before_audit=True, no_feasible_selector=selected is None,
              selectors={k: {kk: vv for kk, vv in v.items() if kk not in ('key','state_dict')} for k, v in selections.items()}))
    save_json(output/'HISTORY.json', history)
    results = {}
    for name, item in selections.items():
        torch.save(dict(state_dict=item['state_dict'], config=config, epoch=item['epoch']), output/(name+'.pt'))
        model.load_state_dict(item['state_dict'])
        audit = evaluate(model, data, 'audit', DEVICE, save_path=output/('audit_'+name+'.npz'))
        results[name] = dict(epoch=item['epoch'], validation=item['validation'], audit=audit, state_sha256=state_hash(model))
    save_json(output/'DONE.json', dict(config=config, selections=results, training_seconds=train_seconds,
              total_seconds=time.time()-start, batch_sha256=batch_hash.hexdigest(), anchor_sha256=anchor_hash.hexdigest(),
              status='completed', claim_scope='same-data development maintenance, not external/final confirmation'))
    print(json.dumps(dict(task=task, seed=seed, stage=stage, arm=arm,
                         epochs={k:v['epoch'] for k,v in results.items()}, seconds=time.time()-start)), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job', nargs=4)
    args = p.parse_args()
    if args.job:
        task, seed, stage, arm = args.job
        worker(task, int(seed), stage, arm)
        return
    logs = ROOT/'job_logs/maintenance'
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
           'INTERFAIR_CONCURRENCY':'4', 'INTERFAIR_THREADS':'1'}
    groups = [[('adult',s,'initial',a) for s in (2000,2001) for a in ('all','control')]]
    groups += [[(t,s,stage,a) for t in ('adult','hmda_oh') for s in (2000,2001) for a in arms]
               for stage,arms in [('update',('erm','repaired')),('rerepair',('all','control'))]]
    start = time.time()
    def launch(j):
        with (logs/('_'.join(map(str,j))+'.log')).open('w') as f:
            result = subprocess.run([sys.executable,__file__,'--job',*map(str,j)], stdout=f, stderr=subprocess.STDOUT, env=env)
        assert result.returncode == 0, j
        print(json.dumps(dict(completed=j)), flush=True)
    for jobs in groups:
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(launch, jobs))
    save_json(ROOT/'reports/MAINTENANCE_EXECUTION.json', dict(status='completed', new_trajectories=20,
              seeds=[2000,2001], concurrency=4, seconds=time.time()-start, protocol_sha256=None))


if __name__ == '__main__':
    main()
