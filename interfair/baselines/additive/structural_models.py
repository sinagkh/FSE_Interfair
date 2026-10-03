"""Protected-offset model adapters for the additive and resampling baselines."""
from pathlib import Path
import hashlib
import json
import sys

_PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PACKAGE / 'baselines/additive'))
import additive as p
import numpy as np
import torch
from ft_architecture import FTPipeline

r = p.repair
write = p.ORIGINAL_WRITE

def read(path):
    return json.loads(Path(path).read_text())

class StructuralFT(FTPipeline):
    def __init__(self, encoder, structural=True):
        super().__init__(encoder, 'erm')
        self.structural = structural
        self.offsets = torch.nn.Parameter(torch.zeros(2))

    def typed(self, x):
        cont, cats = super().typed(x)
        if self.structural:
            cats = cats.clone()
            cats[:, 0] = 0
        return cont, cats

    def forward(self, x):
        z = super().forward(x)
        return z + self.offsets[x[:, 0].long()] if self.structural else z

    def optimizer(self):
        opt = super().optimizer()
        opt.add_param_group({'params': [self.offsets], 'weight_decay': 0.})
        return opt

class StructuralVector(p.Predictor):
    def __init__(self, d, policy):
        from protected_vector import sex_columns
        super().__init__(d['x'].shape[1], 'mlp', 'erm')
        self.sex_columns, _ = sex_columns(d)
        self.policy = policy
        k = len(self.sex_columns)
        self.sex_offsets = torch.nn.Parameter(torch.zeros((2, k) if policy == 'joint' else (k,)))

    def forward(self, x):
        r = x[:, 0].long()
        s = x[:, self.sex_columns].argmax(1)
        xx = x.clone()
        xx[:, [0] + self.sex_columns] = 0
        z = self.net(xx).squeeze(-1) + self.offsets[r]
        return z + (self.sex_offsets[r, s] if self.policy == 'joint' else self.sex_offsets[s])

    def probability(self, x):
        return self(x).sigmoid()

def backbone_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        if isinstance(model, StructuralFT) and key == 'offsets':
            continue
        if isinstance(model, StructuralVector) and key == 'sex_offsets':
            continue
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def phase(seed): return 'development' if seed >= 2000 else 'confirmation'

def soft_done(task, seed, arch):
    if task == 'hmda_oh':
        base=p.CORRECTED/f'runs/{phase(seed)}/main/{arch}/{task}/{seed}'
        path=base/'interfair/DONE.json'
        if not path.exists(): path=base/'interfair_strength0.3/DONE.json'
        return path
    base=p.COMPLETION/f'runs/{phase(seed)}/primary/{arch}/{task}/{seed}'
    return Path(read(base/'METHOD_SELECTION.json')['paths']['interfair'])/'DONE.json'

def model(d,arch,arm):
    if arch=='ft':return StructuralFT(d['encoder'],arm!='soft')
    return p.Predictor(d['x'].shape[1],arch,'erm' if arm=='soft' else 'structural_L')
