"""Apply the Ohio 2019 Fair-SMOTE and reweighing checkpoints, unchanged, to Michigan 2019 and Ohio 2020, using the same
evaluator as the transfer table's ERM and InterFair rows (training/transformer/auxiliary.py, adopted path)."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[3]
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import numpy as np, pandas as pd
from multiprocessing import Pool
ROOT = _PACKAGE
sys.path.insert(0, str(ROOT / 'training/transformer'))
import auxiliary as A
e, c = A.e, A.c
OUT = _PACKAGE / 'baselines/evaluation/rescore/transfer'
SRC = _PACKAGE / 'studies/learning/runs/hmda_oh'

def one(job):
    block, arm, seed = job
    out = OUT / block / 'mlp' / str(seed) / arm; out.mkdir(parents=True, exist_ok=True)
    d, names, db = A.target_data(block)
    path = (e.reference_path('hmda_oh', seed, 'mlp', True) if arm == 'erm_check' else SRC / arm / str(seed) / 'selected.pt')
    src, _, _ = e.q.data('hmda_oh', seed); m = c.load(path, src, 'mlp', 'cpu')
    metrics, edits, features = e.evaluate(m, d, names, db, 'audit', False, 'cpu', out / 'audit.npz')
    row = dict(task=block, architecture='mlp', seed=seed, arm=arm, block=block, checkpoint=str(path),
               checkpoint_sha256=c.sha(path), arrays=str(out / 'audit.npz'), **metrics)
    pd.DataFrame([row]).to_csv(out / 'per_seed.csv', index=False)
    pd.DataFrame([{**dict(task=block, seed=seed, arm=arm), **f} for f in features]).to_csv(out / 'per_feature.csv', index=False)
    return row

if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'check':
        jobs = [(b, 'erm_check', 1000) for b in ('hmda_2019_mi', 'hmda_2020_oh')]
    else:
        jobs = [(b, a, s) for b in ('hmda_2019_mi', 'hmda_2020_oh') for a in ('fairsmote', 'reweighing') for s in range(1000, 1010)]
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 8) as p: rows = p.map(one, jobs)
    df = pd.DataFrame(rows); df.to_csv(OUT / f'per_seed_{which}.csv', index=False)
    print(df.groupby(['block', 'arm']).mean(numeric_only=True)[['auc', 'aod', 'L_R', 'violation_005', 'decision_disagreement', 'direction_adverse_005']].round(4))
