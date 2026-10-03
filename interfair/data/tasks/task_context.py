"""Shared task-loading and pipeline helpers."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import hashlib
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

HERE = _PACKAGE / 'data/tasks'
ROOT = _PACKAGE
CORRECTED = _PACKAGE / 'training/equality'
OLD = _PACKAGE / 'data/partitions'
DEV = _PACKAGE / 'core'
BREADTH = _PACKAGE / 'data/employment'
PLAN = _PACKAGE / 'archive/REMAINING_EMPIRICAL_WORK_20260925.md'
sys.path[:0] = [str(CORRECTED), str(OLD), str(DEV)]
import numpy as np
from core import sha
from repair import write
from partitions import array_hash

DEVELOPMENT = (2000, 2001)
FINAL = tuple(range(1000, 1010))
TASKS = ('credit_broad', 'acs_income')

def phase(seed):
    assert seed in DEVELOPMENT + FINAL + (2002,)
    return 'development' if seed >= 2000 else 'confirmation'

def read_pickle(path):
    with Path(path).open('rb') as f:
        return pickle.load(f)

def save_pickle(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        pickle.dump(value, f, pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
