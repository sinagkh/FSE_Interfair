"""Test generation and validity counts per task (candidates, schema-valid, supported), from the audit banks."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import sys, json, csv
import numpy as np
ROOT = str(_PACKAGE); OUT = f"{ROOT}/auditing/testing"
sys.path.insert(0, f"{ROOT}/training/shared")
feats = json.load(open(f"{OUT}/features.json"))
rows = []
def load(task):
    if task in ("hmda_md", "hmda_va", "hmda_pa"):
        sys.path.insert(0, f"{ROOT}/training/states"); import data as sd
        out = sd.data(task, 1000); return out[0] if isinstance(out, tuple) else out
    import common as c
    out = c.data(task, 1000); return out[0] if isinstance(out, tuple) else out
for task in ["hmda_oh", "hmda_md", "hmda_va", "hmda_pa", "credit_broad", "acs_income", "acs_employment_sex", "acs_employment_age"]:
    try:
        d = load(task)
    except Exception as e:
        print(task, "LOAD FAIL", repr(e)[:200]); continue
    banks = d["banks"]["audit"]; tot = dict(candidates=0, valid=0, supported=0)
    for f in feats[task if task in feats else "acs_employment_sex"]:
        b = banks[f]; n = len(b["indices"]); v = int(np.asarray(b["valid"]).sum()); s = int(np.asarray(b["supported"]).sum())
        tot["candidates"] += n; tot["valid"] += v; tot["supported"] += s
    profiles = len(banks[feats[task][0]]["indices"]) if task in feats else None
    rows.append(dict(task=task, features=len(feats.get(task, [])), profiles=profiles, **tot))
    print(rows[-1], flush=True)
with open(f"{OUT}/generation.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
