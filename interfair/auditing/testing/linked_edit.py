"""What the reporting-code mapping of the linked income edit changes (HMDA Ohio, paper ERM models).

Compares the +$10k linked edit with DTI recomputed at fixed debt and left continuous (unmapped)
against the same edit mapped back to HMDA's reported DTI codes (the edit the paper uses).
"""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import sys, json, pickle
import numpy as np, torch
ROOT = str(_PACKAGE); OUT = f"{ROOT}/auditing/testing"
sys.path.insert(0, f"{ROOT}/training/direction"); sys.path.insert(0, f"{ROOT}/training/behavior")
import direction as q
c = q.c
d, b, db = q.data("hmda_oh", 1000)
mapped = db["audit"]["linked_income_plus10k"]
orig = pickle.load(open(f"{ROOT}/core/cache_v2/hmda_direction_joint.pkl", "rb"))["banks"]["audit"]["joint_income_plus10k"]
assert np.array_equal(orig["indices"], mapped["indices"])
enc = d["encoder"]; j = enc.columns.index("debt_to_income_ratio__scaled"); st = enc.stats["debt_to_income_ratio"]
raw = lambda col: np.round(col.astype(float) * st["scale"] + st["median"], 4)
dti0 = raw(orig["corners"][0, :, j]); dti_cont = raw(orig["corners"][1, :, j]); dti_map = raw(mapped["corners"][1, :, j])
grid = np.array([19, 25, 33, *range(36, 50), 55, 65], float)
valid = np.asarray(orig["valid"]).astype(bool)
on_grid = np.isin(np.round(dti_cont, 4), grid)
from support import fitted_support
support = fitted_support(d)
sup_cont = support.mask(orig["corners"])[0] & valid
sup_map = np.asarray(mapped["supported"]).astype(bool)
res = dict(candidates=int(len(valid)), schema_valid=int(valid.sum()), supported_mapped=int(sup_map.sum()),
           unmapped_off_grid_share_of_valid=float((~on_grid[valid]).mean()), supported_unmapped=int(sup_cont.sum()),
           example_dti=[float(dti0[0]), float(dti_cont[0]), float(dti_map[0])])
both = sup_map & sup_cont
per = []
for seed in range(1000, 1010):
    m = c.load(f"{ROOT}/training/mlp/runs/confirmation/hmda_oh/{seed}/erm/selected.pt", d, "mlp", "cpu")
    out = {}
    for name, bank in (("mapped", mapped), ("unmapped", orig)):
        xx = bank["corners"][:, both]
        with torch.no_grad():
            L = m(torch.as_tensor(xx.reshape(-1, xx.shape[-1]))).numpy().reshape(4, -1).astype(float)
        P = 1 / (1 + np.exp(-L)); R = (L[3] - L[2]) - (L[1] - L[0])
        wrong = (P[1] - P[0] < -.005) | (P[3] - P[2] < -.005); eq = np.abs(R) > .05
        out[name] = dict(R=R, eq=eq, wrong=wrong)
    a, u = out["mapped"], out["unmapped"]
    per.append(dict(seed=seed, n=int(both.sum()), eq_mapped=int(a["eq"].sum()), eq_unmapped=int(u["eq"].sum()),
                    wrong_mapped=int(a["wrong"].sum()), wrong_unmapped=int(u["wrong"].sum()),
                    eq_verdict_changes=int((a["eq"] != u["eq"]).sum()), wrong_verdict_changes=int((a["wrong"] != u["wrong"]).sum()),
                    mean_abs_R_mapped=float(np.abs(a["R"]).mean()), mean_abs_R_unmapped=float(np.abs(u["R"]).mean()),
                    mean_abs_R_difference=float(np.abs(a["R"] - u["R"]).mean())))
    print(per[-1], flush=True)
res["per_seed"] = per
json.dump(res, open(f"{OUT}/linked_edit.json", "w"), indent=1)
print({k: v for k, v in res.items() if k != "per_seed"})
