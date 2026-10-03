"""Render main-comparison table rows from per-seed results and Scott-Knott ESD ranks.

Cell format follows the paper: 3 decimals for utility, group and residual metrics; percentages with 2 decimals for
V, D, W; superscript = ESD group; bold = best ESD group among the displayed methods (best observed mean when the
family has no defined grouping)."""
import math
import pandas as pd

METRICS = ['auc', 'accuracy', 'f1', 'aod', 'dp', 'eomax', 'L_R', 'violation_005', 'decision_disagreement', 'direction_adverse_005']
PERCENT = {'violation_005', 'decision_disagreement', 'direction_adverse_005'}
HIGH = {'auc', 'accuracy', 'f1'}
LABEL = {'erm': 'ERM', 'ltdd': 'LTDD', 'cot_phi': 'CoT-Phi', 'dralign': 'DRAlign', 'fairsmote': 'Fair-SMOTE',
         'reweighing': 'Reweighing', 'hifi_shared': 'HIFI', 'soft': r'\textbf{\method}'}
ORDER = ['erm', 'ltdd', 'cot_phi', 'dralign', 'fairsmote', 'reweighing', 'hifi_shared', 'soft']

def fmt(metric, v):
    return f'{100 * v:.2f}' if metric in PERCENT else f'{v:.3f}'

def block_rows(per_seed, ranks, task, arch, arms=None, metrics=METRICS):
    """per_seed: rows task/architecture/arm/seed/metrics; ranks: task/architecture/metric/arm/rank."""
    g = per_seed[(per_seed.task == task) & (per_seed.architecture == arch)]
    present = [a for a in ORDER if a in set(g.arm)] if arms is None else arms
    means = {a: g[g.arm == a][metrics].mean() for a in present}
    for a in present: assert (g.arm == a).sum() == 10, (task, arch, a, (g.arm == a).sum())
    rk = ranks[(ranks.task == task) & (ranks.architecture == arch)]
    rank = {(r.metric, r.arm): r['rank'] for _, r in rk.iterrows()}
    cells = {a: [] for a in present}
    for m in metrics:
        rs = {a: rank.get((m, a)) for a in present}
        defined = all(r is not None and not (isinstance(r, float) and math.isnan(r)) for r in rs.values())
        if defined:
            best = min(rs.values()); bold = {a: rs[a] == best for a in present}
        else:
            vals = {a: means[a][m] for a in present}   # unrounded, as in the paper's undefined-ESD cells
            target = max(vals.values()) if m in HIGH else min(vals.values()); bold = {a: vals[a] == target for a in present}
        for a in present:
            v = fmt(m, means[a][m]); v = rf'\mathbf{{{v}}}' if bold[a] else v
            sup = f'^{{{int(rs[a])}}}' if defined else ''
            cells[a].append(f'${v}{sup}$')
    return [f'{LABEL[a]} & ' + ' & '.join(cells[a]) + r'\\' for a in present]
