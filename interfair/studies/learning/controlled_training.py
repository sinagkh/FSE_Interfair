"""Parts D and E2: controlled training studies with one fixed MLP recipe.

usage: controlled_training.py TASK ARM SEED
arms: teacher (seed 7000) | real_erm | real_additive | perm_erm | teacher_erm_d{1,2,3}
      | teacher_minority_f{50,25,10} | real_minority_f{50,25,10} | reweighing | fairsmote
"""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import copy
import hashlib
import sys
import time
from controls_common import *
from torch.nn import functional as F
from pipeline_audit import audit
from core import behavior

TEACHER_SEED = 7000
LABEL_SEEDS = {1: 360001, 2: 360002, 3: 360003}


def run_dir(task, arm, seed):
    return HERE / 'runs' / task / arm / str(seed)


def teacher_labels(task, d, draw):
    done = read(run_dir(task, 'teacher', TEACHER_SEED) / 'DONE.json')
    m = p.Predictor(d['x'].shape[1], 'mlp', 'structural_L')
    m.load_state_dict(torch.load(done['checkpoint'], map_location='cpu', weights_only=False)['state_dict']); m.eval()
    z = r.predict_logits(m, d['x'], 'cpu'); prob = expit(z)
    y = (np.random.default_rng(LABEL_SEEDS[draw]).random(len(prob)) < prob).astype(np.float32)
    return y, dict(teacher=done['checkpoint'], teacher_sha256=sha(done['checkpoint']), label_seed=LABEL_SEEDS[draw],
                   labels_sha256=r.array_hash(y), mean_label=float(y.mean()))


def minority_subsample(d, train, frac, seed):
    s = d['s'][train]; minority = int(np.argmin([(s == g).sum() for g in (0, 1)]))
    ids_min = train[s == minority]; rng = np.random.default_rng(6060 + seed)
    keep = rng.choice(ids_min, int(round(frac * len(ids_min))), replace=False)
    out = np.sort(np.concatenate([train[s != minority], keep]))
    return out, dict(minority_group=minority, minority_rows_before=int(len(ids_min)), minority_rows_after=int(len(keep)),
                     training_rows=int(len(out)))


def l_scale_summary(arrays, d):
    """Feature-macro logit residual over supported anchors (main-table definition), plus rates."""
    A = d['banks']['audit']; rs, rates, dd = [], [], []
    for n in primary_features(d):
        keep = A[n]['supported']
        if not keep.any(): continue
        v = arrays[n + '_L'][:, keep].astype(float); res = (v[3] - v[2]) - (v[1] - v[0])
        rs.append(np.abs(res).mean()); rates.append((np.abs(res) > .05).mean())
        q = arrays[n + '_P'][:, keep] >= .5; dd.append(((q[3].astype(int) - q[2]) != (q[1].astype(int) - q[0])).mean())
    return dict(L_R=float(np.mean(rs)), L_rate=float(np.mean(rates)), decision_disagreement=float(np.mean(dd)))


def run(task, arm, seed):
    dest = run_dir(task, arm, seed)
    if (dest / 'DONE.json').exists(): return
    torch.set_num_threads(1); t0 = time.monotonic()
    d0, banks = data(task); d = dict(d0); train = np.asarray(d['splits']['train']); extra = {}
    mode = 'structural_L' if arm in ('teacher', 'real_additive') else 'erm'
    if arm.startswith('teacher_'):
        draw = int(arm[-1]) if arm.startswith('teacher_erm_d') else 1
        y, extra['labels'] = teacher_labels(task, d, draw); d['y'] = y
    x_tr, y_tr, w_tr = d['x'][train].copy(), d['y'][train].astype(np.float32).copy(), np.ones(len(train), np.float32)
    if '_minority_f' in arm:
        frac = int(arm.rsplit('_f', 1)[1]) / 100
        sub, extra['subsample'] = minority_subsample(d, train, frac, seed)
        x_tr, y_tr, w_tr = d['x'][sub].copy(), d['y'][sub].astype(np.float32).copy(), np.ones(len(sub), np.float32)
    if arm == 'perm_erm':
        x_tr[:, 0] = np.random.default_rng(5150 + seed).permutation(x_tr[:, 0])
        extra['permutation'] = 'protected column permuted among training rows only'
    if arm == 'reweighing':
        s = d['s'][train]
        for a in (0, 1):
            for b in (0, 1):
                mask = (s == a) & (y_tr == b); w_tr[mask] = (s == a).mean() * (y_tr == b).mean() / mask.mean()
        extra['reweighing'] = 'Kamiran-Calders cell weights, mean one'
    if arm == 'fairsmote':
        sys.path.insert(0, str(ROOT / 'baselines/resampling'))
        from resampling import fair_smote
        x_tr, y_tr, w_tr = fair_smote(d, seed, dest)
        extra['fairsmote'] = read(dest / 'PREPROCESSING.json')
    cfg = r.recipe(task, 'mlp', 1.); cfg.update(epochs=40, patience=None)
    p.seed_all(seed); m = p.Predictor(d['x'].shape[1], 'mlp', mode); initial = p.state_hash(m)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    X, Y, W = map(torch.as_tensor, (x_tr, y_tr, w_tr)); rng = np.random.default_rng(seed + 991)
    vi = d['splits']['val']; history = []; best = None; bh = hashlib.sha256()
    config = dict(task=task, arm=arm, seed=seed, mode=mode, recipe=cfg, initial_state_sha256=initial, training_rows=len(y_tr),
                  train_x_sha256=r.array_hash(x_tr), train_y_sha256=r.array_hash(y_tr), selection='validation AUROC (BCE tie-break)',
                  audit_used_for_selection=False, protocol_sha256=None, code_sha256=sha(__file__), **extra)
    write(dest / 'CONFIG.json', config)
    for epoch in range(1, 41):
        m.train()
        for ii in r.relations.iter_minibatches(np.arange(len(y_tr)), cfg['batch'], rng):
            bh.update(ii.tobytes()); opt.zero_grad(set_to_none=True)
            loss = (F.binary_cross_entropy_with_logits(m(X[ii]), Y[ii], reduction='none') * W[ii]).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), cfg['gradient_clip']); opt.step()
        v = behavior(d['y'][vi], d['s'][vi], expit(r.predict_logits(m, d['x'][vi], 'cpu')))
        history.append(dict(epoch=epoch, auc=v['auc'], bce=v['bce'])); key = (-v['auc'], v['bce'])
        if best is None or key < best[0]:
            best = (key, epoch); torch.save(dict(state_dict=copy.deepcopy(m.state_dict()), config=config, epoch=epoch), dest / 'selected.pt')
    m.load_state_dict(torch.load(dest / 'selected.pt', map_location='cpu', weights_only=False)['state_dict'])
    pipe = p.Pipeline(m, [dest / 'selected.pt']); report, arrays = audit(pipe, d)
    keep = {k: v.astype(np.float32) if v.dtype == np.float64 else v for k, v in arrays.items()}
    np.savez_compressed(dest / 'audit.npz', **keep)
    summary = l_scale_summary(arrays, d); b = report['behavior']
    write(dest / 'DONE.json', dict(status='completed', config=config, history=history, selected_epoch=best[1], checkpoint=str(dest / 'selected.pt'),
          checkpoint_sha256=sha(dest / 'selected.pt'), summary=summary, audit_behavior=b, audit_macro=report['macro'],
          batch_sha256=bh.hexdigest(), seconds=time.monotonic() - t0))
    print('DONE', task, arm, seed, {k: round(v, 4) for k, v in summary.items()}, round(b['auc'], 4), round(time.monotonic() - t0, 1), flush=True)


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]))
