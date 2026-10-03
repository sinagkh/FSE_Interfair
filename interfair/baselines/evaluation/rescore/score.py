"""Score saved audits with the paper's primary metric code; directions as in baseline_directions.py.

score(task, audit_npz, model=None) -> (row, per_feature, per_edit)
  row: utility, group, residual and decision metrics (training/shared/export.saved_metrics)
       plus direction_* metrics (training/transformer/baseline_directions.metrics), averaged over signed edits.
  model: logits module used only for the linked HMDA edit, which saved ordinary audits do not contain.
"""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[3]
import os, sys
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[k] = '1'
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.special import expit
ROOT = _PACKAGE
sys.path.insert(0, str(ROOT / 'training/shared'))
import export as ex                                   # saved_metrics, taskdata (frozen primary metric code)
sys.path.insert(0, str(ROOT / 'training/direction'))
import direction as q                                 # direction specification MAP and direction banks
sys.path.insert(0, str(ROOT / 'training/transformer'))
import baseline_directions as bd                      # metrics()

_DB = {}
def direction_banks(task):
    if task not in _DB: _DB[task] = q.data(task, 2000)[2]
    return _DB[task]

@torch.no_grad()
def query(model, x, batch=4096):
    model.eval()
    return np.concatenate([model(torch.as_tensor(x[i:i + batch], dtype=torch.float32)).numpy().astype(float)
                           for i in range(0, len(x), batch)])

def directions(task, saved, model=None):
    db = direction_banks(task); recs = []
    for name, sign in q.MAP[task].items():
        bank = db['audit'][name]; keep = bank['supported']; ids = bank['indices'][keep]
        if name.startswith('linked_'):
            assert model is not None, 'linked edit needs the model'
            xx = bank['corners'][:, keep]; z = query(model, xx.reshape(-1, xx.shape[-1])).reshape(4, -1); p = expit(z)
        else:
            lookup = {int(v): i for i, v in enumerate(saved[name + '_indices'])}
            jj = np.asarray([lookup[int(i)] for i in ids]); z = saved[name + '_L'][:, jj]
            p = saved[name + '_P'][:, jj] if name + '_P' in saved else expit(z)
        recs.append(dict(edit=name, **bd.metrics(p, sign, z, False)))
    per_edit = pd.DataFrame(recs)
    means = per_edit.drop(columns=['edit', 'n', 'sign']).mean().to_dict()
    return {'direction_' + k: v for k, v in means.items()}, per_edit

def score(task, audit_npz, model=None):
    row, ff = ex.saved_metrics(task, audit_npz)
    saved = np.load(audit_npz)
    dirs, per_edit = directions(task, saved, model)
    row.update(dirs)
    return row, pd.DataFrame(ff), per_edit
