"""Re-score the matched-study Fair-SMOTE and reweighing checkpoints (Ohio, Credit, ACSIncome MLP) on the primary audit
and direction banks. No training; checks that each saved audit uses the primary banks and replays from its checkpoint."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[3]
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.special import expit
sys.path.insert(0, str(_PACKAGE / 'baselines/evaluation/rescore'))
import score as S
from multiprocessing import Pool
SRC = S.ROOT / 'studies/learning/runs'
OUT = _PACKAGE / 'baselines/evaluation/rescore/classic'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def one(job):
    task, arm, seed = job
    run = SRC / task / arm / str(seed); dest = OUT / task / arm / str(seed); dest.mkdir(parents=True, exist_ok=True)
    d = S.ex.taskdata(task); saved = np.load(run / 'audit.npz')
    checks = {}
    # same tests as the primary audit
    for name, b in d['banks']['audit'].items():
        if not b['training']: continue
        assert np.array_equal(saved[name + '_indices'], b['indices']), (task, name, 'indices')
        assert np.array_equal(saved[name + '_supported'], b['supported']), (task, name, 'supported')
    assert np.array_equal(saved['natural_indices'], d['splits']['audit'])
    checks['primary_banks_identical'] = True
    # checkpoint replay
    m = S.ex.load(run / 'selected.pt', d, 'mlp', 'cpu')
    nat = S.query(m, d['x'][d['splits']['audit']])
    checks['natural_replay_max_error'] = float(abs(expit(nat) - (saved['natural'] if 'natural_L' not in saved else expit(saved['natural_L']))).max())
    name0 = next(n for n, b in d['banks']['audit'].items() if b['training'])
    c0 = d['banks']['audit'][name0]['corners'][:, :64]
    z0 = S.query(m, c0.reshape(-1, c0.shape[-1])).reshape(4, -1)
    checks['corner_replay_max_error'] = float(abs(z0 - saved[name0 + '_L'][:, :64]).max())
    assert checks['natural_replay_max_error'] < 1e-5 and checks['corner_replay_max_error'] < 1e-4, checks
    row, ff, pe = S.score(task, run / 'audit.npz', m)
    ident = dict(task=task, architecture='mlp', arm=arm, seed=seed)
    row = dict(**ident, **row, checkpoint=str(run / 'selected.pt'), checkpoint_sha256=sha(run / 'selected.pt'),
               arrays=str(run / 'audit.npz'), arrays_sha256=sha(run / 'audit.npz'),
               protocol='matched preprocessing study (studies/learning): AdamW .0007, per-epoch validation AUROC')
    ff.insert(0, 'seed', seed); ff.insert(0, 'arm', arm); ff.insert(0, 'architecture', 'mlp'); ff.insert(0, 'task', task)
    pe.insert(0, 'seed', seed); pe.insert(0, 'arm', arm); pe.insert(0, 'architecture', 'mlp'); pe.insert(0, 'task', task)
    pd.DataFrame([row]).to_csv(dest / 'per_seed.csv', index=False); ff.to_csv(dest / 'per_feature.csv', index=False)
    pe.to_csv(dest / 'per_edit.csv', index=False); (dest / 'CHECKS.json').write_text(json.dumps(checks, indent=1))
    return row

if __name__ == '__main__':
    jobs = [(t, a, s) for t in ('hmda_oh', 'credit_broad', 'acs_income') for a in ('fairsmote', 'reweighing') for s in range(1000, 1010)]
    with Pool(12) as p: rows = p.map(one, jobs)
    df = pd.DataFrame(rows); df.to_csv(OUT / 'per_seed_all.csv', index=False)
    print(df.groupby(['task', 'arm'])[['auc', 'aod', 'L_R', 'violation_005', 'decision_disagreement', 'direction_adverse_005']].mean().round(4))
