"""Reproduce the attribute-flip table and comparison counts from saved test counts."""
from pathlib import Path
import argparse
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARMS = ['erm', 'ltdd', 'cot_phi', 'dralign', 'fairsmote', 'reweighing', 'hifi_shared', 'soft']
LABELS = ['ERM', 'LTDD', 'CoT-Phi', 'DRAlign', 'Fair-SMOTE', 'Reweighing', 'HIFI', r'\method']
SETTINGS = [
    ('hmda_oh', 'mlp', 'HMDA Ohio / MLP'),
    ('hmda_md', 'mlp', 'HMDA Maryland'),
    ('hmda_va', 'mlp', 'HMDA Virginia'),
    ('hmda_pa', 'mlp', 'HMDA Pennsylvania'),
    ('hmda_oh', 'ft', 'HMDA Ohio / FT'),
    ('credit_broad', 'mlp', 'Credit / MLP'),
    ('acs_income', 'mlp', 'ACSIncome / MLP'),
    ('acs_employment_sex', 'mlp', 'Employment: sex'),
    ('acs_employment_age', 'mlp', 'Employment: age'),
    ('acs_income', 'ft', 'ACSIncome / FT'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'reproduced/testing')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / 'results/testing/per_seed.csv')
    keys = ['task', 'arch', 'arm', 'seed']
    assert len(data) == 800 and not data.duplicated(keys).any()
    for task, arch, _ in SETTINGS:
        for arm in ARMS:
            rows = data[(data.task == task) & (data.arch == arch) & (data.arm == arm)]
            assert sorted(rows.seed.tolist()) == list(range(1000, 1010)), (task, arch, arm)
    for metric in ['eq', 'dec', 'flipdec_x', 'flipdec_any', 'flipscore_x', 'flipscore_any']:
        assert ((data[metric] / data.n - data[metric + '_rate']).abs() < 1e-12).all(), metric
    means = data.groupby(keys[:-1]).mean(numeric_only=True)
    comparisons = []
    lines = [r'\begin{table}[tbp]\centering\footnotesize',
             r'\caption{Equality failures and attribute-flip flags on the same tests (ten-seed means; percentages of tests pooled over features). Flips are evaluated at the unedited profile.}\label{tab:c-flip}']
    for block in (SETTINGS[:5], SETTINGS[5:]):
        if block == SETTINGS[5:]:
            lines.append(r'\hfill')
        lines.extend([r'\begin{minipage}[t]{.49\linewidth}\centering',
                      r'\begin{tabular}[t]{@{}lrrr@{}}\toprule',
                      r'Model & Eq. & Dec. & Score\\\midrule'])
        for task, arch, title in block:
            lines.append(r'\multicolumn{4}{@{}l}{\textit{' + title + r'}}\\')
            reference = means.loc[(task, arch, 'erm')]
            for arm, label in zip(ARMS, LABELS):
                row = means.loc[(task, arch, arm)]
                values = [100 * row[m] for m in ['eq_rate', 'flipdec_x_rate', 'flipscore_x_rate']]
                lines.append(label + ' & ' + ' & '.join(f'{v:.1f}' for v in values) + r'\\')
                if arm not in ['erm', 'soft']:
                    comparisons.append(dict(setting=f'{task}/{arch}', arm=arm,
                                            eFD=reference.flipdec_x_rate, FD=row.flipdec_x_rate,
                                            eV=reference.eq_rate, V=row.eq_rate))
        lines.append(r'\bottomrule\end{tabular}\end{minipage}')
    lines.append(r'\end{table}')
    comparison = pd.DataFrame(comparisons)
    lower = comparison.FD < comparison.eFD
    retained = comparison.V >= comparison.eV - .05
    checks = dict(per_seed_rows=len(data), model_settings=len(means),
                  published_comparisons=len(comparison), lower_decision_flips=int(lower.sum()),
                  lower_flips_without_5pp_equality_improvement=int((lower & retained).sum()))
    assert list(checks.values()) == [800, 80, 60, 38, 30], checks
    supplied = pd.read_csv(ROOT / 'results/testing/mitigation_first_order.csv')
    index = ['setting', 'arm']
    pd.testing.assert_frame_equal(comparison.set_index(index).sort_index(),
                                  supplied.set_index(index).sort_index(), atol=1e-12, rtol=1e-12)
    comparison.to_csv(out / 'mitigation_first_order.csv', index=False)
    (out / 'attribute_flip_table.tex').write_text('\n'.join(lines) + '\n')
    (out / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
