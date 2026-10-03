"""Final-campaign data interface; only development seeds until launch freeze.

Do not modify the development producers. This adapter reconstructs their admitted
bank recipe for a supplied optimizer seed and retains their split/calibration rule.
"""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import copy
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = _PACKAGE / 'data/partitions'
DEV = _PACKAGE / 'core'
sys.path.insert(0, str(DEV))
from core import save_json, sha
from splits import reserve

FINAL_SEEDS = tuple(range(1000, 1010))
DEVELOPMENT_SEEDS = (2000, 2001)
TASKS = ('adult', 'hmda_oh', 'default_credit')


def read_pickle(path):
    with Path(path).open('rb') as f:
        return pickle.load(f)


def array_hash(a):
    a = np.asarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(json.dumps(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def reconstruct(task, seed):
    """Pure reconstruction apart from reading immutable admitted input files."""
    assert task in TASKS and seed in FINAL_SEEDS + DEVELOPMENT_SEEDS
    if task == 'default_credit':
        from credit import prepare, train_data
        d = train_data(prepare())
    else:
        # E10 all-feature data are identical across seeds. Use a single canonical
        # copy; seed-specific E14 samples are built below, never borrowed.
        d = read_pickle(DEV / 'cache_v2' / f'{task}_selection_2000_all.pkl')
        master = read_pickle(DEV / 'cache_v2' / f'{task}_anchor_master.pkl')
        rng = np.random.default_rng(seed + 95103)
        chunks, indices, selected = [], [], {}
        for name, b in sorted(master['banks'].items()):
            chosen = rng.permutation(len(b['indices']))[:8192]
            assert len(chosen) == 8192
            chunks.append(b['corners'][:, chosen])
            indices.append(b['indices'][chosen])
            selected[name] = b['indices'][chosen].tolist()
        c = np.concatenate(chunks, axis=1)
        ix = np.concatenate(indices)
        mask = np.ones(len(ix), dtype=bool)
        d['banks']['train'] = {'selected_mixture': dict(
            corners=c, indices=ix, valid=mask, supported=mask, training=True,
            sampling_weights=np.full(len(ix), 1 / len(ix), dtype=np.float64))}
        d['metadata']['active_requirement'] = (
            'E14 supported steering budget 8192 per feature; broad P invariance '
            'with signed-response preservation')
        d['metadata']['anchor_budget_per_feature'] = 8192
        d['metadata']['anchor_indices'] = selected
    d, reservation, parts = reserve(d, task)
    return d, reservation, parts


def cache_path(task, seed):
    return HERE / 'cache' / f'{task}_{seed}.pkl'


def data_signature(d):
    """Hash numerical inputs/roles, not pickle layout or prose metadata."""
    record = dict(arrays={k: array_hash(d[k]) for k in ('x', 'y', 's')},
                  splits={k: array_hash(v) for k, v in d['splits'].items()},
                  encoder=d['encoder'].metadata(), banks={})
    for split, banks in d['banks'].items():
        record['banks'][split] = {}
        for name, b in banks.items():
            record['banks'][split][name] = {
                k: array_hash(v) if isinstance(v, np.ndarray) else v
                for k, v in b.items()
                if k in ('corners', 'indices', 'valid', 'supported',
                         'sampling_weights', 'training')}
    return record


def data_for(task, seed):
    p = cache_path(task, seed)
    if p.exists():
        record = json.loads(p.with_suffix('.json').read_text())
        assert record['data_adapter_sha256'] == sha(__file__)
        assert record['cache_sha256'] == sha(p)
        return read_pickle(p)
    d, reservation, parts = reconstruct(task, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
    tmp.replace(p)
    save_json(p.with_suffix('.json'), dict(
        task=task, seed=seed, data_adapter_sha256=sha(__file__),
        cache_sha256=sha(p), reservation=reservation, signature=data_signature(d)))
    return d


def gate():
    checks = []
    for task in TASKS:
        for seed in DEVELOPMENT_SEEDS:
            d = data_for(task, seed)
            reference = read_pickle(DEV / 'cache_v2' / f'{task}_final_preflight_{seed}.pkl')
            sig = data_signature(d)
            assert sig == data_signature(reference), (task, seed)
            checks.append(dict(task=task, seed=seed,
                               check='all numerical inputs, banks, masks, weights, encoder and partitions equal preflight', passed=True))
            bank = d['banks']['train']['selected_mixture']
            assert bank['supported'].all() and np.isin(bank['indices'], d['splits']['train']).all()
            assert abs(bank['sampling_weights'].sum() - 1) < 1e-12
            if task != 'default_credit':
                names = sorted(d['metadata']['anchor_indices'])
                for name in names:
                    assert len(set(d['metadata']['anchor_indices'][name])) == 8192
                w = bank['sampling_weights'].reshape(len(names), 8192)
                assert np.allclose(w.sum(1), 1 / len(names), atol=1e-12)
            checks.append(dict(task=task, seed=seed, check='supported train-only anchors and equal feature mass', passed=True))
        assert data_signature(data_for(task, 2000))['splits'] == data_signature(data_for(task, 2001))['splits']
        checks.append(dict(task=task, check='optimizer seeds share every partition', passed=True))
    # Verify a truly new seed's sampling recipe without training or querying any
    # final model. Bank construction uses only training inputs/support.
    for task in TASKS[:2]:
        d, _, _ = reconstruct(task, 1000)
        before = data_for(task, 2000)
        assert data_signature(d)['splits'] == data_signature(before)['splits']
        assert not np.array_equal(d['banks']['train']['selected_mixture']['indices'],
                                  before['banks']['train']['selected_mixture']['indices'])
        checks.append(dict(task=task, check='final-seed bank is new; partitions fixed; no predictions queried', passed=True))
    result = dict(status='PASS', checks=checks, source_sha256=sha(__file__),
                  final_model_training_started=False, external_outcomes_opened=False)
    save_json(HERE / 'reports/DATA_ADMISSION.json', result)
    print('data admission PASS', len(checks), flush=True)


if __name__ == '__main__':
    gate()
