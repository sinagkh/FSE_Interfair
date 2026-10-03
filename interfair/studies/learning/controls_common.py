"""Shared data, model loaders, and measurements for the learning controls."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = _PACKAGE / 'studies/learning'
ROOT = _PACKAGE
sys.path[:0] = [str(ROOT / 'baselines/additive'), str(ROOT / 'core'),
                str(ROOT.parent / 'scripts')]
import structural_models as u
from inputs import load as prepared_data
import additive as p
import numpy as np
import torch
import pandas as pd
from scipy.special import expit

r = p.repair
TASKS = ('hmda_oh', 'acs_income', 'credit_broad')
SETTINGS = [('hmda_oh', 'mlp'), ('credit_broad', 'mlp'), ('acs_income', 'mlp'), ('hmda_oh', 'ft'), ('acs_income', 'ft')]
SEEDS = list(range(1000, 1010))
INDEX = _PACKAGE.parent / 'results/main/paper_primary_per_seed.csv'
_DATA = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else
                              (x.tolist() if isinstance(x, np.ndarray) else str(x))) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def data(task, seed=1000):
    """Data and training banks. Inputs, labels, splits and audit banks are seed-invariant (checked)."""
    key = (task, seed)
    if key not in _DATA:
        _DATA[key] = prepared_data(task, seed)[:2]
    return _DATA[key]


def primary_features(d):
    return [n for n, b in d['banks']['audit'].items() if b['training']]


def support_for(task, d):
    """Exact reproduction of the group-specific 5-NN support test used to build the banks."""
    from core import Support
    if task == 'hmda_oh':
        from support import fitted_support
        return fitted_support(d)
    return Support(d['x'][d['splits']['train']])


def block_columns(d, names):
    A = d['banks']['audit']; nat = d['x'][A[names[0]]['indices']]; out = {}
    for n in names:
        c = A[n]['corners']
        ch = np.any(c[0] != c[1], axis=0) | np.any(c[0] != nat, axis=0) | np.any(c[1] != nat, axis=0)
        ch[0] = False; out[n] = np.flatnonzero(ch)
    return out


# ---------------------------------------------------------------- model loaders
def _index():
    return pd.read_csv(INDEX)


def load(task, arch, seed, arm, d, device='cpu'):
    """Full prediction pipeline for a primary-panel model. Returns callable f(x, scale)."""
    from task_pipelines import pipeline as completion_pipeline
    from model_interfaces import load_pipeline, plain
    if arm == 'erm':
        r.source_path = p.source_path
        return r.reference_pipeline(task, seed, arch, d, device)
    if arm in ('soft', 'structural'):
        row = _index().query('task==@task and architecture==@arch and arm==@arm and seed==@seed').iloc[0]
        ck = Path(row.checkpoint)
        state = torch.load(ck, map_location='cpu', weights_only=False)['state_dict']
        if arm == 'soft':
            try:
                m = r.make_model(d, arch); m.load_state_dict(state)
            except RuntimeError:
                m = u.model(d, arch, 'soft'); m.load_state_dict(state)
        else:
            m = u.model(d, arch, 'structural'); m.load_state_dict(state)
        m.to(device).eval()
        return p.Pipeline(m, [ck], device, batch=512 if arch == 'ft' else 4096)
    if arm == 'removed':
        return load_pipeline(p.OLD / 'runs', 'main', task, seed, 'removed', d, arch, device)
    if arm in ('ltdd', 'cot_phi', 'dralign', 'mirrorfair', 'neufair'):
        assert arch == 'mlp'
        if task == 'hmda_oh':
            return load_pipeline(p.OLD / 'runs', 'named', task, seed, arm, d, 'mlp', device)
        return completion_pipeline(task, seed, arm, d, 'mlp', device)
    raise ValueError(arm)


def score(f, x, scale):
    """Query in chunks; MirrorFair and EO expose decisions/acceptance only."""
    out = [np.asarray(f(x[i:i + 200000], scale), dtype=np.float64) for i in range(0, len(x), 200000)]
    return np.concatenate(out)


def replay_check(task, arch, seed, arm, f, d):
    """The loaded pipeline must reproduce its saved audit probabilities."""
    row = _index().query('task==@task and architecture==@arch and arm==@arm and seed==@seed')
    if row.empty or not isinstance(row.iloc[0].gap_source_arrays, str):
        return dict(status='no saved arrays')
    with np.load(row.iloc[0].gap_source_arrays) as a:
        name = primary_features(d)[0]
        key = name + '_decision' if name + '_decision' in a else name + '_P'
        saved = a[key].astype(float)
        c = d['banks']['audit'][name]['corners']
        got = score(f, c.reshape(-1, c.shape[-1]).astype(np.float32), 'P').reshape(saved.shape)
        err = float(np.nanmax(np.abs(got - saved)))
    return dict(status='PASS' if err < 1e-5 else 'FAIL', feature=name, max_abs_error=err)
