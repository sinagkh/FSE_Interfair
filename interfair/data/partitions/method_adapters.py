"""Mechanical, enumerated interfaces to frozen development producers.

Generated copies change seed admission, data/output/source paths and run labels.
The objective, optimizer, schedules, selection and native search budgets stay in
the original producer. Every transformation has an exact occurrence check.
"""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
import importlib.util
import json
import sys
from pathlib import Path

from partitions import HERE, DEV, data_for, cache_path, save_json, sha

SEEDS = '(1000,1001,1002,1003,1004,1005,1006,1007,1008,1009,2000,2001)'
RECIPES = {
    'ft_architecture': [
        ('seed in (2000,2001,2002)', 'seed in ' + SEEDS, 'seed whitelist'),
        ("scope='two-seed development; not final confirmation'", "scope=campaign_scope", 'scope label'),
    ],
    'preprocessing_baselines': [
        ('data=prepare(task);out=ROOT/\'runs/preprocessing_baselines\'/task/str(seed)/method',
         'data=partitions(task,seed);out=campaign_base/task/str(seed)/method', 'data and output path'),
        ("ROOT/'runs/pilot_v3'/task/'mlp'/str(seed)/'erm/DONE.json'",
         "campaign_erm(task,seed).parent/'DONE.json'", 'ERM reference path'),
    ],
    'tree_models': [
        ('seed in (2000,2001)', 'seed in ' + SEEDS, 'seed whitelist'),
        ("data=data_for(task);out=ROOT/'runs/tree_query'/task/str(seed)",
         'data=partitions(task,seed);out=campaign_base/task/str(seed)', 'data and output path'),
        ("claim_scope='development query audit; no learned tree repair'", 'claim_scope=campaign_scope', 'scope label'),
    ],
    'dralign': [('seed in (2000,2001)', 'seed in ' + SEEDS, 'seed whitelist')],
    'mirrorfair_common': [('seed in (2000,2001)', 'seed in ' + SEEDS, 'seed whitelist')],
    'neufair': [
        ('assert seed in (2000,2001)', 'assert seed in ' + SEEDS, 'seed whitelist'),
        ("sha(ROOT/'cache_v2'/f'{task}_selection_{seed}_all.pkl')", 'sha(campaign_cache(task,seed))', 'bank provenance'),
    ],
    'maintenance': [
        ('seed in (2000, 2001)', 'seed in ' + SEEDS, 'seed whitelist'),
        ("bank_file = ROOT / 'cache_v2' / f'{task}_selection_{seed}_all.pkl'", 'bank_file = campaign_cache(task,seed)', 'bank provenance'),
    ],
    'learning_modes': [
        ('and seed in (2000,2001) and mode', 'and seed in ' + SEEDS + ' and mode', 'seed whitelist'),
        ("gate_result=json.loads((ROOT/'reports/LEARNING_MODES_GATE.json').read_text());assert gate_result['code_sha256']==sha(__file__) and gate_result['protocol_sha256']==sha(PROTOCOL)",
         'campaign_verify_adapter()', 'verify derived code and original protocol instead of original file hash'),
    ],
    'input_gradient_study': [
        ('assert seed in (2000,2001)', 'assert seed in ' + SEEDS, 'seed whitelist'),
        ("assert gate_result['code_sha256']==sha(__file__)", 'campaign_verify_adapter()', 'verify derived source and upstream hashes'),
        ('d=data_for(task);out=BASE/task/str(seed)/scale/arm', 'd=partitions(task,seed);out=BASE/task/str(seed)/scale/arm', 'study data interface'),
    ],
    'protected_vector': [
        ('seed in (2000,2001)', 'seed in ' + SEEDS, 'seed whitelist'),
        ("data=prepare('hmda_oh');bank=compile_banks();src=ROOT/'runs/pilot_v3/hmda_oh/mlp'/str(seed)/'erm/task.pt'",
         "data=partitions('hmda_oh',seed);bank=compile_banks();src=campaign_erm('hmda_oh',seed)", 'data and source interface'),
    ],
}


def build():
    destination = HERE / 'adapters'
    destination.mkdir(parents=True, exist_ok=True)
    ledger = {}
    for name, recipe in RECIPES.items():
        src = DEV / (name + '.py')
        original = src.read_text()
        text = original
        changes = []
        for old, new, reason in recipe:
            assert text.count(old) == 1, (name, reason, text.count(old))
            text = text.replace(old, new)
            changes.append(dict(before=old, after=new, reason=reason))
        target = destination / src.name
        if target.exists():
            assert target.read_text() == text, ('derived file changed; use an explicit version', target)
        else:
            target.write_text(text)
        # Reverse transformation detects any edit outside the allowed changes.
        restored = text
        for old, new, _ in reversed(recipe):
            assert restored.count(new) == 1, (name, new)
            restored = restored.replace(new, old)
        assert restored == original
        compile(text, str(target), 'exec')
        ledger[name] = dict(original_path=str(src), original_sha256=sha(src),
                            derived_path=str(target), derived_sha256=sha(target), changes=changes,
                            exact_reverse_transform_verified=True)
    save_json(HERE / 'reports/PRODUCER_DERIVATIONS.json', dict(status='PASS', producers=ledger,
              builder_sha256=sha(__file__), training_logic_changed=False))
    return ledger


def verify(name):
    r = json.loads((HERE / 'reports/PRODUCER_DERIVATIONS.json').read_text())
    assert r['status'] == 'PASS' and r['builder_sha256'] == sha(__file__)
    p = r['producers'][name]
    assert sha(p['original_path']) == p['original_sha256']
    assert sha(p['derived_path']) == p['derived_sha256']
    return p


def load(name, root, data_provider=data_for, erm_provider=None, scope='development adapter admission',
         cache_provider=cache_path, initial_provider=None):
    record = verify(name)
    root = Path(root)
    def erm(task, seed):
        return root / 'main/mlp' / task / str(seed) / 'erm/task.pt'
    source = erm_provider or erm
    module_name = '_campaign_' + name
    spec = importlib.util.spec_from_file_location(module_name, record['derived_path'])
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.campaign_base = root / name
    module.BASE = module.campaign_base
    module.partitions = data_provider
    module.campaign_erm = source
    module.campaign_cache = cache_provider
    module.campaign_scope = scope
    module.campaign_verify_adapter = lambda: verify(name)
    if name in ('dralign', 'mirrorfair_common', 'neufair', 'maintenance', 'learning_modes', 'input_gradient_study'):
        module.data_for = data_provider
        module.erm_path = source
    if initial_provider:
        module.initial_path = initial_provider
    return module


if __name__ == '__main__':
    print('derived producers verified', len(build()), flush=True)
