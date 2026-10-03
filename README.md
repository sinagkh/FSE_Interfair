# InterFair

Code, prepared data, configurations, and results for the paper.

## Reproduce

Python 3.10, Linux:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/verify.py
python scripts/reproduce.py
python scripts/testing.py
python scripts/figures.py
```

These commands use the supplied per-seed measurements and write to `reproduced/`: `reproduce.py` rebuilds the main comparison tables (HMDA, Credit and census tasks, FT-Transformer, and transfer); `testing.py` rebuilds the attribute-flip table and its comparison counts; `figures.py` renders the running example, the per-feature heatmap, and the maintenance figure. The data behind the remaining tables are in `results/` (see `results/index.json`). Together the commands take under a minute on a CPU.

## Train

```bash
python scripts/check_inputs.py
python scripts/train.py --task credit_broad --seed 1000 --smoke
```

Omit `--smoke` for full ERM and InterFair training. See `--help` for tasks and architectures. Baseline adapters: LTDD and CoT-Phi in `interfair/core/preprocessing_baselines.py`, DRAlign in `interfair/core/dralign.py`, HIFI in `interfair/training/shared/common.py`, Fair-SMOTE and reweighing in `interfair/baselines/`. LTDD, CoT-Phi, and MirrorFair need their pinned upstream files: `python scripts/fetch_upstream.py --source ltdd` (or `cot`, `mirrorfair`). Run settings and selected epochs are in `configs/`. Checkpoints are not included.

## Statistical groups

Tested with R 4.1.2; the ScottKnottESD and effsize packages are bundled in `environment/R_library/`. Install the remaining R dependencies with `Rscript -e 'install.packages(c("reshape2", "car", "forecast", "effsize"), repos="https://cloud.r-project.org")'`, then run:

```bash
python scripts/rank.py
```

## Contents

- `interfair/`: core functions, data preparation, training, baselines, auditing, and studies.
- `scripts/`: reproduction and training commands.
- `configs/`: run settings and selected epochs.

- `results/main/`: primary and transfer comparisons; ten seeds per setting and method.
- `results/studies/`: supporting experiments, indexed in `results/index.json`.
- `data/`: prepared inputs and edit specifications.
- `third_party/`: source records and licenses.
- `supplement.pdf`: supplementary methods and results.

`soft` identifies InterFair in result files. Stored rates are fractions; V, D, and W are displayed as percentages.
