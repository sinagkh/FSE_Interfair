"""Part E1: soft repair with and without an explicit third-order (S x B x C) penalty.

usage: higher_order.py TASK ARM SEED     arms: soft40 | soft_ho
Both arms use the corrected soft loss under the fixed 40-epoch budget of the teacher/N6 studies
(fixed_budget.py), identical data, anchors, admission and selection. soft_ho adds
pair_weight * mean(T^2) over sampled supported combined training edits.
"""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import copy
import itertools
import sys
import time
from controls_common import *
from torch.nn import functional as F
from pipeline_audit import audit
import combined_edits as ce

sys.path.insert(0, str(ROOT / 'baselines/resampling'))
from fixed_budget import nested, restrict
PAIRS_PER_UPDATE, ANCHORS_PER_PAIR = 4, 128


def train_cubes(task, d, banks):
    """Supported cubes built from pairs of training banks at shared anchor rows (cached)."""
    path = HERE / 'cubes' / f'train_{task}.npz'
    if path.exists():
        z = np.load(path); return {k: z[k] for k in z.files}
    names = list(banks); cols = ce.block_columns(d, [n for n in primary_features(d)]); sup = support_for(task, d)
    out = {}
    for B, C in itertools.combinations(names, 2):
        rowC = {int(v): j for j, v in enumerate(banks[C]['indices'])}
        pb = np.array([j for j, v in enumerate(banks[B]['indices']) if int(v) in rowC]); pc = np.array([rowC[int(banks[B]['indices'][j])] for j in pb])
        if len(pb) < 64: continue
        X = build(banks, cols, B, C, pb, pc)
        m0, _ = sup.mask(X[[0, 1, 4, 5]]); m1, _ = sup.mask(X[[2, 3, 6, 7]]); k = m0 & m1
        if k.sum() >= 64: out[f'{B}|{C}'] = np.stack([pb[k], pc[k]])
    np.savez_compressed(path, **out); return out


def build(banks, cols, B, C, pb, pc):
    cb, cc = banks[B]['corners'][:, pb], banks[C]['corners'][:, pc]
    X = np.empty((8,) + cb.shape[1:], np.float32)
    for s in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                x = cb[2 * s + b].copy(); x[:, cols[C]] = cc[2 * s + c][:, cols[C]]; X[4 * s + 2 * b + c] = x
    return X


def audit_cubes(task, d, f):
    A = d['banks']['audit']; names = primary_features(d); cols = ce.block_columns(d, names)
    cubes = np.load(HERE / 'cubes' / f'{task}.npz'); res = []
    for pr in cubes.files:
        X = ce.cube(A, cols, *pr.split('|'), cubes[pr]); v = score(f, X.reshape(-1, X.shape[-1]), 'L').reshape(8, -1)
        res.append(ce.metrics(v, 'L'))
    return {k: float(np.mean([x[k] for x in res])) for k in res[0]}


def run(task, arm, seed):
    dest = HERE / 'runs_ho' / task / arm / str(seed)
    if (dest / 'DONE.json').exists(): return
    torch.set_num_threads(1); t0 = time.monotonic()
    d, banks = data(task, seed); r.source_path = p.source_path
    cubes = train_cubes(task, d, banks) if arm == 'soft_ho' else {}
    cols = ce.block_columns(d, primary_features(d))
    cube_t = {k: torch.as_tensor(build(banks, cols, *k.split('|'), v[0], v[1])) for k, v in cubes.items()}
    native_ref = read(p.source_path(task, seed, 'mlp').parent / 'DONE.json'); ref = native_ref['selections']['task']['validation']['behavior']
    cfg = r.recipe(task, 'mlp', 1.); cfg.update(epochs=40, patience=None)
    p.seed_all(seed); m = p.Predictor(d['x'].shape[1], 'mlp', 'erm'); initial = p.state_hash(m)
    assert initial == native_ref['config']['initial_state_sha256']
    opt = torch.optim.AdamW(m.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    x = torch.as_tensor(d['x']); y = torch.as_tensor(d['y']); s = torch.as_tensor(d['s'])
    pools = {n: restrict(b, d, 75001 + j, 0) for j, (n, b) in enumerate(banks.items())}
    for b in pools.values(): b['c'] = torch.as_tensor(b['c'])
    score_ids = nested(d['splits']['train'], d, 76001, 0)
    config = dict(task=task, seed=seed, arm=arm, recipe=cfg, initial_state_sha256=initial, higher_order_pairs=len(cube_t),
                  pairs_per_update=PAIRS_PER_UPDATE if cube_t else 0, anchors_per_pair=ANCHORS_PER_PAIR, protocol_sha256=None,
                  code_sha256=sha(__file__), audit_used_for_selection=False)
    write(dest / 'CONFIG.json', config)
    taskrng = np.random.default_rng(seed + 991); arng = np.random.default_rng(seed + 992); hrng = np.random.default_rng(seed + 993)
    keys = list(cube_t); history = []; best = None; native = None
    for epoch in range(1, 41):
        warm = min(epoch / cfg['warmup'], 1.)
        for ii in r.relations.iter_minibatches(d['splits']['train'], cfg['batch'], taskrng):
            opt.zero_grad(set_to_none=True); m.train()
            F.binary_cross_entropy_with_logits(m(x[ii]), y[ii]).backward()
            ai = score_ids[r.balanced_positions(score_ids, d, arng, cfg['anchors'])]; m.eval()
            (warm * r.score_loss(m(x[ai]), s[ai], y[ai], cfg)).backward()
            for name, b in pools.items():
                j = r.balanced_positions(b['ids'], d, arng, cfg['anchors']); ids = b['ids'][j]
                v = m(b['c'][:, j].reshape(-1, x.shape[1])).reshape(4, -1)
                with torch.no_grad(): zz = m(x[ids])
                (warm * r.pair_effect_loss(v, zz, s[ids], y[ids], cfg) / len(pools)).backward()
            if keys:
                for k in hrng.choice(len(keys), min(PAIRS_PER_UPDATE, len(keys)), replace=False):
                    X = cube_t[keys[k]]; j = hrng.choice(X.shape[1], min(ANCHORS_PER_PAIR, X.shape[1]), replace=False)
                    v = m(X[:, j].reshape(-1, x.shape[1])).reshape(8, -1)
                    T = (v[7] - v[3]) - (v[6] - v[2]) - (v[5] - v[1]) + (v[4] - v[0])
                    (warm * cfg['pair'] * T.square().mean() / min(PAIRS_PER_UPDATE, len(keys))).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), cfg['gradient_clip']); opt.step()
        result = r.validation(m, d, list(banks), 'cpu', cfg); b = result['behavior']; ok = r.eligible(b, ref, cfg)
        floor = ref['auc'] - cfg['credit_auc_budget']
        sc = result['score'] if b['auc'] >= floor else 1000 + floor - b['auc'] + result['score'] - .001 * b['bce']
        history.append(dict(epoch=epoch, auc=b['auc'], aod=b['aod'], eligible=ok, score=result['score']))
        state = None
        if native is None or sc < native[0] - cfg['selection_min_delta']:
            native = (sc, epoch); state = copy.deepcopy(m.state_dict()); torch.save(dict(state_dict=state, epoch=epoch), dest / 'native_selected.pt')
        if ok and (best is None or result['score'] < best[0]):
            best = (result['score'], epoch); torch.save(dict(state_dict=state or copy.deepcopy(m.state_dict()), epoch=epoch), dest / 'selected.pt')
        print(task, arm, seed, epoch, round(b['auc'], 4), ok, flush=True)
    chosen = dest / ('selected.pt' if best else 'native_selected.pt')
    m.load_state_dict(torch.load(chosen, map_location='cpu', weights_only=False)['state_dict'])
    pipe = p.Pipeline(m, [chosen]); report, arrays = audit(pipe, d)
    from controlled_training import l_scale_summary
    summary = l_scale_summary(arrays, d); summary.update({f'combined_{k}': v for k, v in audit_cubes(task, d, pipe).items()})
    write(dest / 'DONE.json', dict(status='completed', config=config, admitted=best is not None, selected_epoch=(best or native)[1], checkpoint=str(chosen),
          checkpoint_sha256=sha(chosen), summary=summary, audit_behavior=report['behavior'], history=history, seconds=time.monotonic() - t0))
    print('DONE', task, arm, seed, {k: round(v, 4) for k, v in summary.items()}, round(time.monotonic() - t0, 1), flush=True)


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]))
