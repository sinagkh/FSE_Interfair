"""Parts A-E summaries and the single consistent inference pass (analysis only)."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import glob
import hashlib
from controls_common import *
from scipy import stats
from statsmodels.stats.multitest import multipletests

OUT = _PACKAGE / 'studies/learning/results'; OUT.mkdir(exist_ok=True)
TESTS = []


def paired(family, label, ref, meth, metric, higher_is='worse'):
    """ref/meth: Series indexed by seed. Primary: paired t on differences; secondary: Wilcoxon."""
    k = ref.index.intersection(meth.index); a, b = ref[k].astype(float), meth[k].astype(float); diff = b - a
    ok = (a != 0).all()
    ratio = (b / a) if ok else pd.Series(np.nan, index=k)
    rng = np.random.default_rng(int(hashlib.sha256(f'{family}|{label}|{metric}'.encode()).hexdigest()[:8], 16))   # reproducible
    boot = ratio.values[rng.integers(0, len(k), (20000, len(k)))].mean(1) if ok else np.array([np.nan])
    t = stats.ttest_rel(b, a).pvalue if diff.std() > 0 else (0. if diff.abs().sum() > 0 else 1.)
    w = stats.wilcoxon(b, a).pvalue if (diff != 0).any() else 1.
    TESTS.append(dict(family=family, comparison=label, metric=metric, n=len(k), reference_mean=a.mean(), method_mean=b.mean(),
                      mean_difference=diff.mean(), mean_ratio=ratio.mean(), ratio_lo=np.quantile(boot, .025), ratio_hi=np.quantile(boot, .975),
                      seeds_higher=int((diff > 0).sum()), seeds_lower=int((diff < 0).sum()), p_t=t, p_wilcoxon=w))


def holm():
    x = pd.DataFrame(TESTS)
    for fam, g in x.groupby('family'):
        x.loc[g.index, 'p_holm_t'] = multipletests(g.p_t.fillna(1), method='holm')[1]
        x.loc[g.index, 'p_holm_wilcoxon'] = multipletests(g.p_wilcoxon.fillna(1), method='holm')[1]
    return x


def by_seed(df, arm, metric):
    return df[df.arm == arm].set_index('seed')[metric]


# ------------------------------------------------------------------ Part A: existing ten-seed panel
def part_a():
    allm = pd.read_csv(ROOT / 'archive/ALL_METHODS_PER_SEED.csv')
    for (task, arch), g in allm.groupby(['task', 'architecture']):
        for arm in sorted(set(g.arm) - {'erm'}):
            fam = 'F1 repairs vs ERM: requirement' if arm in ('soft', 'structural') else 'F2 published vs ERM: requirement'
            for met in ('L_R', 'P_R', 'P_decision_disagreement'):
                if arm == 'structural' and met == 'L_R': continue          # exact zero by construction, not a test
                if g[g.arm == arm][met].isna().all(): continue
                paired(fam, f'{task}/{arch}/{arm}', by_seed(g, 'erm', met), by_seed(g, arm, met), met)
            if arm in ('soft', 'structural'):
                for met in ('auc', 'f1', 'aod', 'dp'):
                    paired('F4 repairs vs ERM: utility and group metrics', f'{task}/{arch}/{arm}', by_seed(g, 'erm', met), by_seed(g, arm, met), met)
    prim = pd.read_csv(ROOT / 'data/tasks/exports/PRIMARY_PER_SEED.csv')
    for (task, arch), g in prim.groupby(['task', 'architecture']):
        if 'control' not in set(g.arm): continue
        for met in ('L_R', 'P_R', 'P_decision_disagreement', 'aod', 'dp', 'auc'):
            paired('F3 interaction term ablation (full soft vs penalties only)', f'{task}/{arch}', by_seed(g, 'control', met), by_seed(g, 'interfair', met), met)


# ------------------------------------------------------------------ Part D: bug establishment
def load_runs():
    rows = []
    for f in glob.glob(str(HERE / 'runs/*/*/*/DONE.json')):
        j = read(f); c = j['config']; b = j['audit_behavior']
        rows.append(dict(task=c['task'], arm=c['arm'], seed=c['seed'], **j['summary'], auc=b['auc'], bce=b['bce'], f1=b['f1'],
                         aod=b['aod'], dp=b['dp'], training_rows=c['training_rows']))
    return pd.DataFrame(rows)


def sign_consistency(task, arm, seeds):
    d, _ = data(task); A = d['banks']['audit']; signs = []
    for s in seeds:
        a = np.load(HERE / f'runs/{task}/{arm}/{s}/audit.npz'); row = []
        for n in primary_features(d):
            keep = A[n]['supported']; v = a[n + '_L'][:, keep].astype(float); row.append(np.mean((v[3] - v[2]) - (v[1] - v[0])))
        signs.append(np.sign(row))
    S = np.array(signs); agree = np.maximum((S > 0).sum(0), (S < 0).sum(0))
    return dict(features=S.shape[1], same_sign_all=int((agree == len(seeds)).sum()), same_sign_9plus=int((agree >= len(seeds) - 1).sum()))


def part_d(runs):
    tab = []
    for task, g in runs.groupby('task'):
        g = g[g.seed < 7000]
        teach = g[g.arm.str.startswith('teacher_erm_d')].groupby('seed')[['L_R', 'L_rate', 'decision_disagreement']].mean()
        base = dict(task=task)
        for arm in ('real_erm', 'teacher_erm_d1', 'teacher_erm_d2', 'teacher_erm_d3', 'perm_erm', 'real_additive'):
            x = g[g.arm == arm]
            base.update({f'{arm}_L_R': x.L_R.mean(), f'{arm}_rate': x.L_rate.mean(), f'{arm}_auc': x.auc.mean(), f'{arm}_bce': x.bce.mean()})
        tab.append(base)
        fam = 'F6 bug establishment'
        paired(fam, f'{task}: teacher-label ERM (3 draws) vs permuted-S noise floor', by_seed(g, 'perm_erm', 'L_R'), teach.L_R, 'L_R')
        paired(fam, f'{task}: teacher-label ERM vs real-label ERM', by_seed(g, 'real_erm', 'L_R'), teach.L_R, 'L_R')
        paired(fam, f'{task}: real-label ERM vs permuted-S noise floor', by_seed(g, 'perm_erm', 'L_R'), by_seed(g, 'real_erm', 'L_R'), 'L_R')
        for met in ('bce', 'auc'):
            paired(fam, f'{task}: ERM (with interactions) vs additive g(x)+b_s, held-out', by_seed(g, 'real_additive', met), by_seed(g, 'real_erm', met), met)
        for kind, full in (('teacher', 'teacher_erm_d1'), ('real', 'real_erm')):
            slopes = []; means = {}
            for s in range(1000, 1005):
                pts = [(1., g[(g.arm == full) & (g.seed == s)].L_R.iloc[0])]
                pts += [(f / 100, g[(g.arm == f'{kind}_minority_f{f}') & (g.seed == s)].L_R.iloc[0]) for f in (50, 25, 10)]
                xs, ys = np.log([q[0] for q in pts]), np.log([q[1] for q in pts]); slopes.append(np.polyfit(xs, ys, 1)[0])
            for f in (100, 50, 25, 10):
                arm = full if f == 100 else f'{kind}_minority_f{f}'
                means[f] = g[(g.arm == arm) & (g.seed < 1005)].L_R.mean()
            TESTS.append(dict(family=fam, comparison=f'{task}: {kind}-label L_R vs smaller-group training fraction (log-log slope)', metric='slope',
                              n=5, mean_difference=float(np.mean(slopes)), p_t=stats.ttest_1samp(slopes, 0).pvalue,
                              p_wilcoxon=stats.wilcoxon(slopes).pvalue, reference_mean=means[100], method_mean=means[10],
                              seeds_lower=int((np.array(slopes) < 0).sum()), seeds_higher=int((np.array(slopes) > 0).sum())))
            tab[-1].update({f'{kind}_minority_{f}': v for f, v in means.items()})
        for arm in ('real_erm', 'teacher_erm_d1', 'perm_erm'):
            tab[-1].update({f'{arm}_sign_{k}': v for k, v in sign_consistency(task, arm, range(1000, 1010)).items()})
    pd.DataFrame(tab).to_csv(OUT / 'BUG_STUDY.csv', index=False)


# ------------------------------------------------------------------ Part C: combined edits
def part_c():
    rows = []
    for f in sorted(glob.glob(str(HERE / 'combined/*.csv'))):
        x = pd.read_csv(f); task, arch = x.task.iloc[0], x.architecture.iloc[0]
        per = x.groupby(['arm', 'seed']).mean(numeric_only=True).reset_index()
        for arm, g in per.groupby('arm'):
            rows.append(dict(task=task, architecture=arch, arm=arm, **g.mean(numeric_only=True).drop('seed').to_dict()))
            if arm == 'erm': continue
            for met in ('L_T', 'L_J', 'D_joint'):
                if met in g and g[met].notna().all() and not (arm in ('structural', 'removed') and met != 'D_joint'):
                    paired('F5 combined edits vs ERM', f'{task}/{arch}/{arm}', by_seed(per, 'erm', met), g.set_index('seed')[met], met)
    pd.DataFrame(rows).to_csv(OUT / 'COMBINED_EDITS.csv', index=False)


# ------------------------------------------------------------------ Part E: classic baselines and higher-order repair
def part_e(runs):
    for task, g in runs.groupby('task'):
        for arm in ('reweighing', 'fairsmote'):
            if arm not in set(g.arm): continue
            for met in ('L_R', 'decision_disagreement', 'auc', 'aod', 'dp'):
                paired('F7 classic baselines vs matched ERM', f'{task}/{arm}', by_seed(g, 'real_erm', met), by_seed(g, arm, met), met)
    eo = pd.read_csv(HERE / 'classic/EO_PER_SEED.csv')
    for task, g in eo.groupby('task'):
        g = g.set_index('seed')
        paired('F7 classic baselines vs matched ERM', f'{task}/eo_postprocessing (decision scale)', g.erm_decision_residual, g.eo_acceptance_residual, 'decision_residual')
        for met in ('accuracy', 'aod', 'dp'):
            paired('F7 classic baselines vs matched ERM', f'{task}/eo_postprocessing', g[f'erm_{met}'], g[f'eo_{met}'], met)
    ho = []
    for f in glob.glob(str(HERE / 'runs_ho/*/*/*/DONE.json')):
        j = read(f); c = j['config']; ho.append(dict(task=c['task'], arm=c['arm'], seed=c['seed'], admitted=j['admitted'], **j['summary'],
                                                    auc=j['audit_behavior']['auc'], aod=j['audit_behavior']['aod'], dp=j['audit_behavior']['dp']))
    if ho: pd.DataFrame(ho).sort_values(['task', 'arm', 'seed']).to_csv(OUT / 'HIGHER_ORDER_REPAIR.csv', index=False)


if __name__ == '__main__':
    part_a(); runs = load_runs(); runs.to_csv(OUT / 'TRAINING_RUNS.csv', index=False)
    part_d(runs); part_c(); part_e(runs)
    h = holm(); h.to_csv(OUT / 'INFERENCE.csv', index=False)
    old = pd.read_csv(ROOT / 'archive/POST_HOC_TESTS.csv') if (ROOT / 'archive/POST_HOC_TESTS.csv').exists() else None
    print('tests', len(h), 'families', h.family.nunique())
