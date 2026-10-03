"""Additive protected-offset models and their training adapters."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import argparse
import json
import sys
import time
from pathlib import Path

HERE = _PACKAGE / 'baselines/additive'
ROOT = _PACKAGE
COMPLETION = _PACKAGE / 'data/tasks'
CORRECTED = _PACKAGE / 'training/equality'
OLD = _PACKAGE / 'data/partitions'
DEV = _PACKAGE / 'core'
sys.path[:0] = list(map(str, (COMPLETION, CORRECTED, OLD, DEV)))

import numpy as np
import torch
import repair
from core import Predictor, seed_all, state_hash, sha
from task_data import data_for as task_data
from run_core import source_path as completion_source
from pipeline_audit import audit
from model_interfaces import Pipeline, plain

TASKS = ('hmda_oh', 'credit_broad', 'acs_income')
SEEDS = (2000, 2001)
ORIGINAL_GET_DATA = repair.get_data
ORIGINAL_SOURCE_PATH = repair.source_path
ORIGINAL_WRITE = repair.write


def data_and_banks(task, seed, study='additive', arm='structural'):
    if task == 'hmda_oh':
        return ORIGINAL_GET_DATA(task, seed)
    d = task_data(task, seed)
    return d, {n: b for n, b in d['banks']['train'].items() if b['training']}


def source_path(task, seed, arch='mlp'):
    return (ORIGINAL_SOURCE_PATH if task == 'hmda_oh' else completion_source)(task, seed, arch)


def reference_done(task, seed):
    if task == 'hmda_oh':
        return CORRECTED / 'runs/development/main/mlp' / task / str(seed) / 'interfair/DONE.json'
    p = COMPLETION / 'runs/development/primary/mlp' / task / str(seed) / 'METHOD_SELECTION.json'
    return Path(json.loads(p.read_text())['paths']['interfair']) / 'DONE.json'


def install(arm, device):
    repair.HERE = HERE
    repair.get_data = data_and_banks
    repair.source_path = source_path
    mode = 'structural_L' if arm == 'structural' else 'erm'
    repair.make_model = lambda d, arch: Predictor(d['x'].shape[1], arch, mode)

    def annotated_write(path, value):
        if Path(path).name == 'CONFIG.json':
            value.update(
                mode=mode,
                method='additive protected-offset model with current InterFair population penalties'
                if arm == 'structural' else 'unchanged current InterFair replay',
                additive=True,
                structural_formula='q(s,x)=g_theta(x_without_direct_s)+b_s' if arm == 'structural' else None,
                architectural_change='zero encoded protected column before network, add two learned offsets'
                if arm == 'structural' else None,
                restriction_scope='all direct protected interactions, including otherwise held-fixed identity context'
                if arm == 'structural' else 'existing finite supported edit requirement',
                pilot_adapter_sha256=sha(__file__),
                canonical_trainer_sha256=sha(CORRECTED / 'repair.py'),
                predictor_source_sha256=sha(DEV / 'core.py'),
                pilot_protocol_sha256=None,
                numerical_device=device,
                direct_penalty='retained unchanged; algebraically zero for structural logits',
                unchanged_components=['data', 'splits', 'typed edits', 'support masks',
                    'initial parameter tensors', 'optimizer', 'population score and effect penalties',
                    'boundary penalties', 'anchor sampler', 'checkpoint selector', 'admission rules'],
                speed_scope='generic four-corner trainer retained; no optimized structural implementation',
            )
        ORIGINAL_WRITE(path, value)

    repair.write = annotated_write


def preflight():
    torch.set_num_threads(1)
    checks = []
    for task in TASKS:
        for seed in SEEDS:
            d, banks = data_and_banks(task, seed)
            seed_all(seed)
            model = Predictor(d['x'].shape[1], 'mlp', 'structural_L')
            init_hash = state_hash(model)
            saved = json.loads(reference_done(task, seed).read_text())
            assert saved['config']['initial_state_sha256'] == init_hash
            assert saved['config']['data_hashes'] == {k: repair.array_hash(d[k]) for k in ('x', 'y', 's')}
            assert saved['config']['split_hashes'] == {k: repair.array_hash(v) for k, v in d['splits'].items()}
            assert saved['config']['bank_hashes'] == {n: repair.array_hash(b['corners']) for n, b in banks.items()}
            assert saved['config']['recipe'] == repair.recipe(task, 'mlp', 1.)
            model.double().eval()
            with torch.no_grad():
                model.offsets.copy_(torch.tensor([-.7, .4], dtype=torch.float64))
            maxima = []
            for name, b in d['banks']['audit'].items():
                c = b['corners'][:, :64]
                assert np.array_equal(c[0, :, 1:], c[2, :, 1:])
                assert np.array_equal(c[1, :, 1:], c[3, :, 1:])
                v = model(torch.tensor(c.reshape(-1, c.shape[-1]), dtype=torch.float64)).reshape(4, -1)
                residual = v[3] - v[2] - v[1] + v[0]
                maximum = float(residual.detach().abs().max())
                assert maximum < 1e-12, (task, seed, name, maximum)
                maxima.append(maximum)
            # Check that the allowed main effect remains trainable.
            xx = torch.tensor(d['x'][:32], dtype=torch.float64)
            lo, hi = xx.clone(), xx.clone()
            lo[:, 0], hi[:, 0] = 0, 1
            gap = model(hi) - model(lo)
            assert torch.allclose(gap, torch.full_like(gap, 1.1), atol=1e-12, rtol=0)
            gap.mean().backward()
            assert torch.allclose(model.offsets.grad, torch.tensor([-1., 1.], dtype=torch.float64))
            checks.append(dict(task=task, seed=seed, same_initial_tensors=True,
                same_inputs_splits_training_banks_recipe=True, protected_offset_trainable=True,
                max_double_residual=max(maxima), parameters=sum(p.numel() for p in model.parameters()),
                primary_features=list(banks), baseline_source=str(reference_done(task, seed)),
                baseline_source_sha256=sha(reference_done(task, seed))))
    ORIGINAL_WRITE(HERE / 'PREFLIGHT.json', dict(status='PASS', checks=checks,
        pilot_sha256=sha(__file__), protocol_sha256=None))
    print(json.dumps(dict(status='PASS', settings=len(checks))), flush=True)


def run(task, seed, device='cpu', arm='structural'):
    assert task in TASKS and seed in SEEDS
    assert json.loads((HERE / 'PREFLIGHT.json').read_text())['status'] == 'PASS'
    install(arm, device)
    base = HERE / 'runs/development/additive/mlp' / task / str(seed)
    selection_path = base / f'{arm.upper()}_SELECTION.json'
    if selection_path.exists():
        print('Already completed', task, seed, arm, flush=True)
        return
    attempts = []
    for strength in (1., .3, .1):
        result = repair.train(task, seed, 'mlp', 'additive', arm, device, strength)
        attempts.append(dict(strength=strength, admitted=result['admitted'],
            checkpoint=result['checkpoint'], checkpoint_sha256=result['checkpoint_sha256'],
            seconds=result['seconds'], epochs=result['epochs_completed'],
            done=str(Path(result['checkpoint']).parent / 'DONE.json')))
        if result['admitted']:
            break
    chosen = next((a for a in attempts if a['admitted']), attempts[0])
    ORIGINAL_WRITE(selection_path, dict(status='completed', task=task, seed=seed, arm=arm,
        admitted=any(a['admitted'] for a in attempts), attempts=attempts, chosen=chosen,
        audit_used_for_selection=False, development_only=True))
    print('PILOT COMPLETE', task, seed, arm, chosen, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', nargs='?', choices=TASKS)
    parser.add_argument('seed', nargs='?', type=int, choices=SEEDS)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--arm', default='structural', choices=('structural', 'interfair'))
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        preflight()
    else:
        assert args.task is not None and args.seed is not None
        run(args.task, args.seed, args.device, args.arm)
