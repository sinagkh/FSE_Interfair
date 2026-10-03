"""Job queue for Part D/E training. usage: launch.py QUEUE WORKERS"""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = _PACKAGE / 'studies/learning'
PY = '.venv/bin/python'
TASKS = ('hmda_oh', 'acs_income', 'credit_broad')
S10 = range(1000, 1010); S5 = range(1000, 1005)


def jobs(queue):
    out = []
    if queue == 'bug':
        for t in TASKS:
            out += [(t, a, s) for a in ('real_erm', 'real_additive', 'perm_erm') for s in S10]
            out += [(t, f'teacher_erm_d{k}', s) for k in (1, 2, 3) for s in S10]
            out += [(t, f'{kind}_minority_f{f}', s) for kind in ('teacher', 'real') for f in (50, 25, 10) for s in S5]
    if queue == 'classic':
        out += [(t, a, s) for t in TASKS for a in ('reweighing', 'fairsmote') for s in S10]
    return [j for j in out if not (HERE / 'runs' / j[0] / j[1] / str(j[2]) / 'DONE.json').exists()]


def run(job):
    t, a, s = job
    if shutil.disk_usage(HERE).free < 2 * 1024 ** 3: raise RuntimeError('less than 2 GiB free')
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''}
    log = HERE / 'logs' / f'train_{t}_{a}_{s}.log'
    with log.open('w') as f:
        code = subprocess.call([PY, str(HERE / 'controlled_training.py'), t, a, str(s)], stdout=f, stderr=subprocess.STDOUT, env=env)
    print(t, a, s, 'exit', code, flush=True)
    return dict(task=t, arm=a, seed=s, exit=code)


if __name__ == '__main__':
    q, w = sys.argv[1], int(sys.argv[2]); todo = jobs(q); print('jobs', len(todo), flush=True)
    with concurrent.futures.ThreadPoolExecutor(w) as ex:
        res = list(ex.map(run, todo))
    (HERE / f'EXECUTION_{q}.json').write_text(json.dumps(dict(finished=time.time(), results=res), indent=2))
