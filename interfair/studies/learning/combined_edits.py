"""Part C: combined typed edits and third-order protected terms (query only).

build TASK        -> cubes/TASK.npz  (per feature pair: supported anchor positions)
eval TASK ARCH    -> combined/TASK_ARCH.csv (per method, seed, pair)
"""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import itertools
import sys
import time
from controls_common import *

ARMS = {('hmda_oh', 'mlp'): ['erm', 'soft', 'structural', 'ltdd', 'cot_phi', 'dralign', 'neufair', 'mirrorfair', 'removed'],
        ('credit_broad', 'mlp'): ['erm', 'soft', 'structural', 'ltdd', 'cot_phi', 'dralign', 'mirrorfair'],
        ('acs_income', 'mlp'): ['erm', 'soft', 'structural', 'ltdd', 'cot_phi', 'dralign', 'mirrorfair'],
        ('hmda_oh', 'ft'): ['erm', 'soft', 'structural', 'removed'],
        ('acs_income', 'ft'): ['erm', 'soft', 'structural']}
DECISION_ONLY = {'mirrorfair'}


def cube(A, cols, B, C, pos):
    """x(s,b,c) at index 4s+2b+c; B-corners with C's columns taken from C-corners at the same anchor."""
    cb, cc = A[B]['corners'][:, pos], A[C]['corners'][:, pos]
    out = np.empty((8,) + cb.shape[1:], dtype=np.float32)
    for s in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                x = cb[2 * s + b].copy(); x[:, cols[C]] = cc[2 * s + c][:, cols[C]]; out[4 * s + 2 * b + c] = x
    return out


def build(task):
    d, _ = data(task); A = d['banks']['audit']; names = primary_features(d); cols = block_columns(d, names)
    assert all(np.array_equal(A[names[0]]['indices'], A[n]['indices']) for n in names)
    sup = support_for(task, d)
    for n in names[:3]:   # exact support reproduction on the existing single-edit banks
        b = A[n]; v = b['valid']; m = np.zeros(len(v), bool); m[v], _ = sup.mask(b['corners'][:, v])
        assert np.array_equal(m, b['supported']), n
    out = {}; counts = []
    t = time.time()
    for B, C in itertools.combinations(names, 2):
        pos = np.flatnonzero(A[B]['supported'] & A[C]['supported'])
        X = cube(A, cols, B, C, pos)
        m0, _ = sup.mask(X[[0, 1, 4, 5]]); m1, _ = sup.mask(X[[2, 3, 6, 7]])
        keep = pos[m0 & m1]; out[f'{B}|{C}'] = keep
        counts.append(dict(pair=f'{B}|{C}', both_single_supported=len(pos), cube_supported=len(keep)))
    np.savez_compressed(HERE / 'cubes' / f'{task}.npz', **out)
    write(HERE / 'cubes' / f'{task}_MANIFEST.json', dict(task=task, pairs=len(out), features=names, counts=counts,
          seconds=time.time() - t, anchors=int(len(A[names[0]]['indices'])), support='exact existing 5-NN test, all 8 corners'))
    print('BUILT', task, len(out), 'pairs', round(time.time() - t, 1), flush=True)


def metrics(v, scale):
    """v: (8, n). Returns dict of magnitudes for one pair."""
    G = lambda b, c: v[4 + 2 * b + c] - v[2 * b + c]
    T = G(1, 1) - G(1, 0) - G(0, 1) + G(0, 0); J = G(1, 1) - G(0, 0)
    RB = G(1, 0) - G(0, 0); RC = G(0, 1) - G(0, 0)
    tol = .05 if scale == 'L' else .01
    return {f'{scale}_T': np.abs(T).mean(), f'{scale}_T_rate': (np.abs(T) > tol).mean(), f'{scale}_J': np.abs(J).mean(),
            f'{scale}_J_rate': (np.abs(J) > tol).mean(), f'{scale}_R_single': (np.abs(RB).mean() + np.abs(RC).mean()) / 2,
            f'{scale}_R_single_rate': ((np.abs(RB) > tol).mean() + (np.abs(RC) > tol).mean()) / 2,
            f'{scale}_offset': np.abs(G(0, 0)).mean()}


def decision_metrics(dec):
    dec = dec.astype(np.int8)
    D = lambda s, b, c: dec[4 * s + 2 * b + c]
    joint = (D(1, 1, 1) - D(1, 0, 0)) != (D(0, 1, 1) - D(0, 0, 0))
    single = ((D(1, 1, 0) - D(1, 0, 0)) != (D(0, 1, 0) - D(0, 0, 0))).mean() / 2 + ((D(1, 0, 1) - D(1, 0, 0)) != (D(0, 0, 1) - D(0, 0, 0))).mean() / 2
    return dict(D_joint=joint.mean(), D_single=single)


def evaluate(task, arch):
    d, _ = data(task); A = d['banks']['audit']; names = primary_features(d); cols = block_columns(d, names)
    cubes = np.load(HERE / 'cubes' / f'{task}.npz'); pairs = list(cubes.files)
    Xs = [cube(A, cols, *pr.split('|'), cubes[pr]) for pr in pairs]
    sizes = [X.shape[1] for X in Xs]; flat = np.concatenate([X.reshape(-1, X.shape[-1]) for X in Xs])
    dest = HERE / 'combined' / f'{task}_{arch}.csv'; rows = []; checks = []
    device = 'cuda' if arch == 'ft' and torch.cuda.is_available() else 'cpu'
    for arm in ARMS[(task, arch)]:
        for seed in SEEDS:
            t = time.time(); f = load(task, arch, seed, arm, d, device)
            if arch == 'ft' and hasattr(f, 'batch'): f.batch = 8192   # query-only speed; replay check below still applies
            chk = replay_check(task, arch, seed, arm, f, d)
            checks.append(dict(task=task, architecture=arch, arm=arm, seed=seed, **chk))
            assert chk['status'] != 'FAIL', (task, arch, arm, seed, chk)
            outs = {}
            for scale in (('P',) if arm in DECISION_ONLY else ('L', 'P')):
                outs[scale] = score(f, flat, scale)
            start = 0
            for pr, n in zip(pairs, sizes):
                row = dict(task=task, architecture=arch, arm=arm, seed=seed, pair=pr, n=n)
                for scale, o in outs.items():
                    v = o[start * 8:(start + n) * 8].reshape(8, n)
                    if arm in DECISION_ONLY:
                        row.update(decision_metrics(v >= .5))
                    else:
                        row.update(metrics(v, scale))
                        if scale == 'P': row.update(decision_metrics(v >= .5))
                rows.append(row); start += n
            print('EVAL', task, arch, arm, seed, chk['status'], round(time.time() - t, 1), flush=True)
    pd.DataFrame(rows).to_csv(dest, index=False)
    write(HERE / 'combined' / f'{task}_{arch}_REPLAY.json', checks)


if __name__ == '__main__':
    torch.set_num_threads(4)
    if sys.argv[1] == 'build': build(sys.argv[2])
    else: evaluate(sys.argv[2], sys.argv[3])
