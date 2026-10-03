"""Wall-clock cost of one full audit (HMDA Ohio, MLP ERM, seed 1000) on one CPU thread."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import sys, time, json
import numpy as np, torch
ROOT = str(_PACKAGE)
sys.path.insert(0, f"{ROOT}/training/direction"); sys.path.insert(0, f"{ROOT}/training/behavior")
import direction as q
c = q.c; torch.set_num_threads(1)
d, b, db = q.data("hmda_oh", 1000)
feats = json.load(open("features.json"))["hmda_oh"]
m = c.load(f"{ROOT}/training/mlp/runs/confirmation/hmda_oh/1000/erm/selected.pt", d, "mlp", "cpu")
banks = d["banks"]["audit"]; X = []
for f in feats:
    bb = banks[f]; xx = bb["corners"][:, np.asarray(bb["supported"]).astype(bool)]; X.append(xx.reshape(-1, xx.shape[-1]))
X = torch.as_tensor(np.concatenate(X)); t = time.perf_counter()
with torch.no_grad():
    for i in range(0, len(X), 4096): m(X[i:i + 4096])
el = time.perf_counter() - t
print(json.dumps(dict(tests=len(X) // 4, forward_passes=len(X), seconds=round(el, 3))))
json.dump(dict(tests=len(X) // 4, forward_passes=len(X), seconds=el), open("timing.json", "w"))
