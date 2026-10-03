"""Reconstruct exactly the additive's train-only support reference/calibration fit."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[1]
import numpy as np
from sklearn.model_selection import train_test_split
from core import SPLIT_SEED,Support


def ordered_training_indices(data):
    ids=np.arange(len(data['frame']));labels=2*data['y']+data['s']
    tv,_=train_test_split(ids,test_size=.2,random_state=SPLIT_SEED,stratify=labels)
    train,_=train_test_split(tv,test_size=.1875,random_state=SPLIT_SEED+1,stratify=labels[tv])
    assert np.array_equal(np.sort(train),data['splits']['train'])
    return train


def fitted_support(data):
    support=Support(data['x'][ordered_training_indices(data)])
    assert all(abs(support.thresholds[g]-data['metadata']['support']['thresholds'][g])<1e-9 for g in (0,1))
    return support
