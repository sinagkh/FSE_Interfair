"""Load any completed published-method pipeline as an encoded-input logits module."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[3]
from pathlib import Path
import sys
sys.path.insert(0,str(_PACKAGE / 'baselines/evaluation/ft'))
from runner import load
