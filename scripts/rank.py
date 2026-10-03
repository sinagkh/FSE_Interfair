"""Re-run the paper's complete Scott–Knott ESD families, including retained undefined groups."""
from pathlib import Path
import subprocess,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'reproduced/ranks';out.mkdir(parents=True,exist_ok=True)
checks=[]
for panel in ['primary','states','transfer']:
    subprocess.run(['Rscript',str(ROOT/'scripts/scott_knott.R'),str(ROOT/'environment/R_library'),str(ROOT/f'results/main/{panel}_sk_input.csv'),str(out/f'{panel}_sk_ranks.csv'),str(out/f'{panel}_sk_errors.csv'),str(out/f'{panel}_R_session.txt')],check=True)
    cols=['family','task','architecture','metric','arm','rank']
    a=pd.read_csv(ROOT/f'results/main/{panel}_sk_ranks.csv')[cols].sort_values(cols[:-1]).reset_index(drop=True)
    b=pd.read_csv(out/f'{panel}_sk_ranks.csv')[cols].sort_values(cols[:-1]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b)
    ae=pd.read_csv(ROOT/f'results/main/{panel}_sk_errors.csv');be=pd.read_csv(out/f'{panel}_sk_errors.csv')
    assert sorted(ae.family)==sorted(be.family)
    checks.append(dict(panel=panel,rank_rows=len(a),undefined_families=len(ae),match=True))
(out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks)
