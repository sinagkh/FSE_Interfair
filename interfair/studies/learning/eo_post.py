"""Part E2: equalized-odds post-processing of the main ERM MLP (Fairlearn ThresholdOptimizer, native settings)."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import sys
from controls_common import *
sys.path[:0] = [str(ROOT / 'vendor/python_deps'), str(ROOT / 'vendor/fairlearn')]
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import ClassifierMixin, BaseEstimator


class FrozenScore(ClassifierMixin, BaseEstimator):
    # Same frozen-score adapter as the breadth additive: the ERM probability is the only input.
    def __init__(self):
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        raise RuntimeError('Frozen probabilities cannot be refitted')

    def predict_proba(self, X):
        v = np.asarray(X)[:, 0]; return np.column_stack([1 - v, v])


def eo_expected(pp, scores, protected):
    # Native randomized-threshold expectation, independently replayed from the fitted rules.
    a = pp._pmf_predict(np.asarray(scores)[:, None], sensitive_features=protected.astype(int))[:, 1]
    independent = np.zeros(len(a))
    for g, rule in pp.interpolated_thresholder_.interpolation_dict.items():
        keep = protected == g
        mix = rule.p0 * rule.operation0(scores[keep]) + rule.p1 * rule.operation1(scores[keep])
        independent[keep] = rule.p_ignore * rule.prediction_constant + (1 - rule.p_ignore) * mix
    np.testing.assert_allclose(a, independent, rtol=0, atol=1e-12)
    return a


def four(a, keep):
    a = a[:, keep]; return (a[3] - a[2]) - (a[1] - a[0])


def expected_behavior(y, s, acc):
    rates = {}
    for g in (0, 1):
        k = s == g
        rates[g] = dict(tpr=float(acc[k & (y == 1)].mean()), fpr=float(acc[k & (y == 0)].mean()), positive=float(acc[k].mean()))
    dt = rates[1]['tpr'] - rates[0]['tpr']; df = rates[1]['fpr'] - rates[0]['fpr']
    return dict(accuracy=float((acc * y + (1 - acc) * (1 - y)).mean()), aod=.5 * (abs(dt) + abs(df)),
                dp=abs(rates[1]['positive'] - rates[0]['positive']))


def main():
    rows = []
    for task in TASKS:
        d, _ = data(task); A = d['banks']['audit']; vi = d['splits']['val']; ai = d['splits']['audit']
        for seed in SEEDS:
            f = load(task, 'mlp', seed, 'erm', d)
            pp = ThresholdOptimizer(estimator=FrozenScore(), constraints='equalized_odds', objective='accuracy_score',
                                    prefit=True, predict_method='predict_proba', grid_size=1000, flip=False)
            pp.fit(score(f, d['x'][vi], 'P')[:, None], d['y'][vi].astype(int), sensitive_features=d['s'][vi].astype(int))
            nat = score(f, d['x'][ai], 'P'); acc = eo_expected(pp, nat, d['s'][ai].astype(int))
            erm_b = expected_behavior(d['y'][ai], d['s'][ai], (nat >= .5).astype(float)); eo_b = expected_behavior(d['y'][ai], d['s'][ai], acc)
            ra, rd = [], []
            for n in primary_features(d):
                keep = A[n]['supported']; c = A[n]['corners']; q = score(f, c.reshape(-1, c.shape[-1]), 'P')
                groups = c[..., 0].reshape(-1).astype(int)
                a = eo_expected(pp, q, groups).reshape(4, -1); h = (q >= .5).astype(float).reshape(4, -1)
                ra.append(np.abs(four(a, keep)).mean()); rd.append(np.abs(four(h, keep)).mean())
            rows.append(dict(task=task, seed=seed, erm_decision_residual=float(np.mean(rd)), eo_acceptance_residual=float(np.mean(ra)),
                             **{f'erm_{k}': v for k, v in erm_b.items()}, **{f'eo_{k}': v for k, v in eo_b.items()}))
            print('EO', task, seed, round(rows[-1]['erm_decision_residual'], 4), round(rows[-1]['eo_acceptance_residual'], 4), flush=True)
    pd.DataFrame(rows).to_csv(HERE / 'classic' / 'EO_PER_SEED.csv', index=False)


if __name__ == '__main__':
    torch.set_num_threads(2); main()
