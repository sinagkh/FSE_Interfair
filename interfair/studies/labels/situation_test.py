"""k-NN situation testing of recorded labels vs. new label disagreements; no training, no model queries."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import os, sys, json, itertools
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

ROOT = _PACKAGE
U = _PACKAGE / 'archive'
sys.path.insert(0, str(U))
import query as Q  # noqa: E402  (provides dataset(); main is guarded)

OUT = _PACKAGE / 'studies/labels/exports'
OUT.mkdir(exist_ok=True)
SETTINGS = [('hmda_oh', 'mlp'), ('hmda_md', 'mlp'), ('hmda_va', 'mlp'), ('hmda_pa', 'mlp'), ('credit_broad', 'mlp'),
            ('acs_income', 'mlp'), ('acs_employment_sex', 'mlp'), ('acs_employment_age', 'mlp'),
            ('hmda_oh', 'ft'), ('acs_income', 'ft')]
SEEDS = list(range(1000, 1010))
KS = [5, 10, 25]
MAIN_K, TAU = 10, 0.2
rng = np.random.default_rng(20261001)


def situation(task):
    d = Q.dataset(task)
    tr, au = d['splits']['train'], d['splits']['audit']
    people = np.load(U / 'queries' / task / 'PEOPLE.npz')
    assert np.array_equal(people['ids'], au)
    X = d['x'][:, 1:].astype(float)
    s, y = d['s'].astype(int), d['y'].astype(int)
    res = {}
    for k in KS:
        rate = {}
        for g in (0, 1):
            idx = tr[s[tr] == g]
            nn = NearestNeighbors(n_neighbors=k).fit(X[idx])
            _, ind = nn.kneighbors(X[au])
            rate[g] = y[idx][ind].mean(1)
        sa, ya = s[au], y[au]
        same = np.where(sa == 1, rate[1], rate[0])
        other = np.where(sa == 1, rate[0], rate[1])
        res[k] = (2 * ya - 1) * (same - other)
    return sa, ya, res


def load(task, arch, seed, arm):
    f = U / 'queries' / task / arch / str(seed) / arm / 'QUERIES.npz'
    if not f.exists():
        return None
    z = np.load(f)
    return z['frozen_p'], z['frozen_y']


def matched_mean(S, new, stable, strata):
    """Mean S among stable people, reweighted to the new-error strata distribution."""
    num, den = 0.0, 0
    for key in np.unique(strata[new]):
        m_new = new & (strata == key)
        m_st = stable & (strata == key)
        if m_st.sum() == 0:
            continue
        num += m_new.sum() * S[m_st].mean()
        den += m_new.sum()
    return num / den if den else np.nan


def auc_safe(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


rows, auc_rows = [], []
if __name__ != "__main__":
    SETTINGS_RUN = []
else:
    SETTINGS_RUN = SETTINGS
for task, arch in SETTINGS_RUN:
    sa, ya, Sk = situation(task)
    arms = sorted(p.name for p in (U / 'queries' / task / arch / str(SEEDS[0])).iterdir() if p.is_dir())
    erm = {sd: load(task, arch, sd, 'erm') for sd in SEEDS}
    for sd in SEEDS:
        pe, ye = erm[sd]
        assert np.array_equal(ye, ya)
        de = pe >= .5
        ce = de == ya
        strata = sa * 1000 + ya * 100 + np.minimum((pe / .05).astype(int), 19)
        comparisons = [(arm, load(task, arch, sd, arm)) for arm in arms if arm not in ('erm', 'hifi_erm')]
        comparisons += [(f'erm_refit_{b}', erm[b]) for b in SEEDS if b != sd]
        for arm, got in comparisons:
            if got is None:
                continue
            pm, _ = got
            dm = pm >= .5
            cm = dm == ya
            new, stable = ce & ~cm, ce & cm
            if new.sum() == 0:
                continue
            for k in KS:
                S = Sk[k]
                rows.append(dict(task=task, arch=arch, seed=sd, arm=arm, k=k, n_new=int(new.sum()),
                                 new_rate=float(new.mean()), S_new=float(S[new].mean()),
                                 S_matched=float(matched_mean(S, new, stable, strata)),
                                 flag_new_01=float((S[new] >= .1).mean()), flag_new_02=float((S[new] >= .2).mean()),
                                 flag_new_03=float((S[new] >= .3).mean()),
                                 flag_matched_02=float(matched_mean((S >= .2).astype(float), new, stable, strata)),
                                 new_denied_positive=float((new & (ya == 1)).sum() / new.sum()),
                                 new_protected=float((new & (sa == 1)).sum() / new.sum())))
            if arm.startswith('erm_refit') or 'mirrorfair' in arm:
                continue
            S = Sk[MAIN_K]
            gp = S >= TAU
            full = auc_safe(ya, pm) - auc_safe(ya, pe)
            sub_gp = auc_safe(ya[gp], pm[gp]) - auc_safe(ya[gp], pe[gp])
            sub_rest = auc_safe(ya[~gp], pm[~gp]) - auc_safe(ya[~gp], pe[~gp])
            rnd = []
            for _ in range(20):
                pick = rng.choice(len(ya), size=int((~gp).sum()), replace=False)
                rnd.append(auc_safe(ya[pick], pm[pick]) - auc_safe(ya[pick], pe[pick]))
            acc_full = float((dm == ya).mean() - ce.mean())
            acc_rest = float((dm[~gp] == ya[~gp]).mean() - ce[~gp].mean())
            acc_gp = float((dm[gp] == ya[gp]).mean() - ce[gp].mean())
            auc_rows.append(dict(task=task, arch=arch, seed=sd, arm=arm, share_group_patterned=float(gp.mean()),
                                 dauc_full=full, dauc_group_patterned=sub_gp, dauc_rest=sub_rest,
                                 dauc_random_same_size=float(np.nanmean(rnd)),
                                 dacc_full=acc_full, dacc_group_patterned=acc_gp, dacc_rest=acc_rest))
    print('done', task, arch, flush=True)

if __name__ == '__main__':
    R = pd.DataFrame(rows)
    A = pd.DataFrame(auc_rows)
    R.to_csv(OUT / 'situation_per_seed.csv', index=False)
    A.to_csv(OUT / 'auc_split_per_seed.csv', index=False)
    R['group'] = np.where(R.arm.str.startswith('erm_refit'), 'erm_refit', R.arm)
    summ = (R.groupby(['task', 'arch', 'group', 'k']).mean(numeric_only=True).reset_index())
    summ.to_csv(OUT / 'situation_summary.csv', index=False)
    A.groupby(['task', 'arch', 'arm']).mean(numeric_only=True).reset_index().to_csv(OUT / 'auc_split_summary.csv', index=False)
    print('wrote', len(R), len(A))
