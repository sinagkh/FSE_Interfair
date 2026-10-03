"""Controls for the AUROC split: noisy-label subsets, ERM refits, published methods, k and threshold sensitivity."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
HERE = _PACKAGE / 'studies/labels'
sys.path.insert(0, str(HERE))
import situation_test as ST  # noqa (re-runs main analysis quickly; outputs identical)

rows = []
for task, arch in ST.SETTINGS:
    d = ST.Q.dataset(task)
    tr, au = d['splits']['train'], d['splits']['audit']
    X = d['x'][:, 1:].astype(float); s = d['s'].astype(int); y = d['y'].astype(int)
    sa, ya = s[au], y[au]
    cache = {}
    for k in ST.KS:
        rate = {}
        for g in (0, 1):
            idx = tr[s[tr] == g]
            _, ind = NearestNeighbors(n_neighbors=k).fit(X[idx]).kneighbors(X[au])
            rate[g] = y[idx][ind].mean(1)
        same = np.where(sa == 1, rate[1], rate[0]); other = np.where(sa == 1, rate[0], rate[1])
        cache[k] = ((2 * ya - 1) * (same - other), np.abs(ya - same))
    arms = sorted(p.name for p in (ST.U / 'queries' / task / arch / str(ST.SEEDS[0])).iterdir() if p.is_dir())
    arms = [a for a in arms if a not in ('erm', 'hifi_erm', 'mirrorfair')]
    for sd in ST.SEEDS:
        pe, _ = ST.load(task, arch, sd, 'erm')
        comps = [(a, ST.load(task, arch, sd, a)) for a in arms] + [('erm_refit', ST.load(task, arch, (sd - 1000 + 1) % 10 + 1000, 'erm'))]
        for arm, got in comps:
            if got is None: continue
            pm = got[0]
            for k in ST.KS:
                S, N = cache[k]
                for tau in (0.1, 0.2, 0.3):
                    gp = S >= tau
                    q = gp.mean()
                    noisy = N >= np.quantile(N, 1 - q) if q > 0 else np.zeros_like(gp)
                    f = lambda m: ST.auc_safe(ya[m], pm[m]) - ST.auc_safe(ya[m], pe[m])
                    rows.append(dict(task=task, arch=arch, seed=sd, arm=arm, k=k, tau=tau, share=float(q),
                                     dauc_full=f(np.ones_like(gp)), dauc_gp=f(gp), dauc_rest=f(~gp),
                                     dauc_noisy=f(noisy), dauc_not_noisy=f(~noisy)))
    print('done', task, arch, flush=True)
R = pd.DataFrame(rows)
R.to_csv(HERE / 'exports' / 'controls_per_seed.csv', index=False)
R.groupby(['task', 'arch', 'arm', 'k', 'tau']).mean(numeric_only=True).reset_index().to_csv(HERE / 'exports' / 'controls_summary.csv', index=False)
print('wrote', len(R))
