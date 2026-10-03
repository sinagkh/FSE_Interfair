"""Tied-offset model adapter for the additive baseline comparison."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import argparse
import sys
from pathlib import Path

HERE = _PACKAGE / 'baselines/resampling'
sys.path.insert(0, str(_PACKAGE / 'baselines/additive'))
import structural_models as u
from ft_architecture import FTPipeline
p, r, torch, np = u.p, u.r, u.torch, u.np
c = u
write, read = u.write, u.read


class TiedMLP(p.Predictor):
    def __init__(self, d):
        super().__init__(d['x'].shape[1], 'mlp', 'structural_L')

    def forward(self, x):
        return self.base(x) + self.offsets.mean()

    def probability(self, x):
        return self(x).sigmoid()


class TiedFT(c.StructuralFT):
    def forward(self, x):
        return FTPipeline.forward(self, x) + self.offsets.mean()


def model(d, arch, tied=True):
    if not tied:
        return u.model(d, arch, 'structural')
    return TiedMLP(d) if arch == 'mlp' else TiedFT(d['encoder'])


def install(arch, tied=True):
    r.HERE = HERE / 'N5'
    r.get_data, r.source_path = p.data_and_banks, p.source_path
    r.make_model = lambda d, a: model(d, a, tied)
    r.state_hash = c.backbone_hash if arch == 'ft' else p.state_hash
    def annotated(path, obj):
        if Path(path).name == 'CONFIG.json':
            obj.update(mode='tied_structural' if tied else 'structural_L',
                       method='N5 tied-offset attribution' if tied else 'N5 free-offset replay',
                       structural_formula='q(s,x)=g(x_without_s)+mean(b)' if tied else 'q(s,x)=g(x_without_s)+b_s',
                       protocol_sha256=None,study_code_sha256=p.sha(__file__),
                       tied_offsets=tied,development_only=True)
        write(path, obj)
    r.write = annotated


def run(task, seed, arch, device, tied=True):
    assert seed in (2000, 2001)
    install(arch, tied)
    torch.set_num_threads(1)
    d, banks = p.data_and_banks(task, seed)
    p.seed_all(seed)
    m = model(d, arch, tied)
    ref = read(u.soft_done(task, seed, arch))
    assert r.state_hash(m) == ref['config']['initial_state_sha256']
    assert {k:r.array_hash(d[k]) for k in ('x','y','s')} == ref['config']['data_hashes']
    assert {k:r.array_hash(v) for k,v in d['splits'].items()} == ref['config']['split_hashes']
    assert {n:r.array_hash(b['corners']) for n,b in banks.items()} == ref['config']['bank_hashes']
    m.double().eval()
    with torch.no_grad():
        m.offsets.copy_(torch.tensor([-.4,.7],dtype=torch.float64))
        x = torch.as_tensor(d['x'][:32],dtype=torch.float64)
        lo, hi = x.clone(), x.clone()
        lo[:,0], hi[:,0] = 0, 1
        gap = (m(hi)-m(lo)).abs().max().item()
        assert (gap < 1e-12) if tied else (abs(gap-1.1)<1e-12)
    if tied:
        m.zero_grad();m(x).sum().backward()
        assert torch.equal(m.offsets.grad[0],m.offsets.grad[1])
    out=HERE/'N5'/f'preflight/{arch}/{task}/{seed}'
    write(out/('tied.json' if tied else 'free.json'),dict(status='PASS',task=task,seed=seed,architecture=arch,tied=tied,
          matched_inputs_initialization=True,protected_gap_test=gap,protocol_sha256=None,code_sha256=p.sha(__file__)))
    strength=.3 if task=='hmda_oh' and arch=='ft' else 1.
    done=r.train(task,seed,arch,'offset_attribution','tied' if tied else 'free',device,strength)
    print('N5 COMPLETE',task,seed,arch,tied,done['admitted'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('task');ap.add_argument('seed',type=int)
    ap.add_argument('--arch',default='mlp');ap.add_argument('--device',default='cpu');ap.add_argument('--free',action='store_true')
    a=ap.parse_args();run(a.task,a.seed,a.arch,a.device,not a.free)
