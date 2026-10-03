"""Complete Fair-SMOTE using its existing trainer and frozen state audit banks."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time

P = _PACKAGE / 'baselines/state_resampling'
ROOT = _PACKAGE
STATE = _PACKAGE / 'training/states'
TASKS = ('hmda_md', 'hmda_va', 'hmda_pa')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def phase(seed):
    return 'development' if seed >= 2000 else 'confirmation'


def directory(task, seed, arm):
    return P / 'runs' / phase(seed) / task / str(seed) / arm


def dependencies():
    return [Path(__file__), P / 'collect.py',
            ROOT / 'studies/learning/controlled_training.py',
            ROOT / 'studies/learning/controls_common.py',
            ROOT / 'baselines/resampling/resampling.py',
            ROOT / 'vendor/fair_smote/Generate_Samples.py',
            STATE / 'data.py', STATE / 'is_trainer.py',
            ROOT / 'training/shared/common.py',
            ROOT / 'training/transformer/ft_trainer.py',
            ROOT / 'training/direction/direction.py']


def freeze():
    assert (P / 'exports/development/VERIFICATION.json').exists()
    check = json.loads((P / 'exports/development/VERIFICATION.json').read_text())
    assert check['status'] == 'PASS' and check['rows'] == 12
    write(P / 'FREEZE.json', dict(files={str(p): sha(p) for p in dependencies()},
                                   confirmation_seeds=list(range(1000, 1010)),
                                   development_checked=True, tuning=False))


def worker(task, seed):
    assert task in TASKS
    if seed < 2000:
        for path, digest in json.loads((P / 'FREEZE.json').read_text())['files'].items():
            assert sha(path) == digest, path
    # The upstream trainers use different modules called common.py. Keep their
    # original import environments isolated rather than replacing those modules.
    sys.path.insert(0, str(ROOT / 'studies/learning'))
    import controlled_training as canonical
    spec = importlib.util.spec_from_file_location('fairsmote_state_data', STATE / 'data.py')
    data_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(data_module)
    d, banks, _ = data_module.data(task, seed)
    canonical.HERE = P
    canonical.data = lambda unused: (d, banks)
    canonical.run_dir = lambda task, arm, seed: directory(task, seed, 'erm' if arm == 'real_erm' else arm)
    source_hashes = {str(p): sha(p) for p in dependencies()}
    data_hashes = {k: canonical.r.array_hash(d[k]) for k in ('x', 'y', 's')}
    split_hashes = {k: canonical.r.array_hash(v) for k, v in d['splits'].items()}
    for arm in ('erm', 'fairsmote'):
        out = directory(task, seed, arm)
        out.mkdir(parents=True, exist_ok=True)
        if (out / 'EVALUATION.json').exists():
            continue
        write(out / 'ADAPTER.json', dict(task=task, seed=seed, arm=arm,
            device='cpu', source_hashes=source_hashes, data_hashes=data_hashes,
            split_hashes=split_hashes, data_cache_sha256=sha(STATE / 'cache' / f'{task}.pkl'),
            paired_reference='erm', new_hyperparameter_selection=False))
        print('START', task, seed, arm, flush=True)
        canonical.run(task, 'real_erm' if arm == 'erm' else arm, seed)
    subprocess.run([sys.executable, __file__, 'query', '--task', task, '--seed', str(seed)], check=True)


def query(task, seed):
    import numpy as np
    sys.path.insert(0, str(STATE))
    import is_trainer as state
    c = state.c
    d, banks, db = state.state_data(task, seed)
    for arm in ('erm', 'fairsmote'):
        out = directory(task, seed, arm)
        if (out / 'EVALUATION.json').exists(): continue
        fit = json.loads((out / 'DONE.json').read_text())
        cfg = fit['config']
        assert cfg['recipe']['epochs'] == 40 and cfg['recipe']['lr'] == .0007
        assert cfg['recipe']['gradient_clip'] == 5 and not cfg['audit_used_for_selection']
        model = c.load(out / 'selected.pt', d, 'mlp', 'cpu')
        values, edits, features = state.ee.evaluate(model, d, list(banks), db, 'audit',
                                                    device='cpu', save=out / 'audit_current.npz')
        np.testing.assert_allclose(values['L_R'], fit['summary']['L_R'], atol=1e-10)
        np.testing.assert_allclose(values['aod'], fit['audit_behavior']['aod'], atol=1e-10)
        if arm == 'fairsmote':
            reference = json.loads((directory(task, seed, 'erm') / 'DONE.json').read_text())
            assert cfg['initial_state_sha256'] == reference['config']['initial_state_sha256']
            assert cfg['recipe'] == reference['config']['recipe']
            assert cfg['selection'] == reference['config']['selection']
            assert cfg['fairsmote']['fit_partition'] == 'train only'
            assert cfg['fairsmote']['original_encoding_check']
        c.write(out / 'EVALUATION.json', dict(task=task, seed=seed, arm=arm, architecture='mlp',
            phase=phase(seed), metrics=values, ordinary_features=features, direction_features=edits,
            checkpoint=str(out / 'selected.pt'), checkpoint_sha256=c.sha(out / 'selected.pt'),
            arrays=str(out / 'audit_current.npz'), arrays_sha256=c.sha(out / 'audit_current.npz'),
            direction_arrays_sha256=c.sha(out / 'DIRECTION_audit_current.npz'),
            data_cache_sha256=c.sha(STATE / 'cache' / f'{task}.pkl'),
            optimizer_updates=40 * math.ceil(cfg['training_rows'] / cfg['recipe']['batch']),
            selected_epoch=fit['selected_epoch'], training_seconds=fit['seconds'],
            selection_finished_before_audit=True))
        print('VERIFIED', task, seed, arm, flush=True)


def campaign(phase, workers):
    seeds = (2000, 2001) if phase == 'development' else tuple(range(1000, 1010))
    logs = P / 'logs'; logs.mkdir(exist_ok=True)
    jobs = [(task, seed) for task in TASKS for seed in seeds]
    def launch(cell):
        task, seed = cell
        path = logs / f'{task}_{seed}.log'
        started = time.monotonic()
        with path.open('a') as stream:
            done = subprocess.run([sys.executable, __file__, 'worker', '--task', task,
                '--seed', str(seed)], stdout=stream, stderr=subprocess.STDOUT)
        return dict(task=task, seed=seed, returncode=done.returncode,
                    seconds=time.monotonic()-started, log=str(path))
    completed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(launch, job) for job in jobs]):
            result = future.result(); completed.append(result)
            write(P / f'JOBS_{phase}.json', completed)
            print('FINISHED', len(completed), '/', len(jobs), json.dumps(result), flush=True)
    assert all(j['returncode'] == 0 for j in completed), 'Inspect failed worker logs'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['worker', 'query', 'campaign', 'freeze'])
    parser.add_argument('--task'); parser.add_argument('--seed', type=int)
    parser.add_argument('--phase', choices=['development', 'confirmation'], default='development')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if args.command == 'worker': worker(args.task, args.seed)
    elif args.command == 'query': query(args.task, args.seed)
    elif args.command == 'freeze': freeze()
    else: campaign(args.phase, args.workers)
