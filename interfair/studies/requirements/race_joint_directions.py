"""Audit signed edits on the joint cube of saved race-trained pipelines."""
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[2]
from engine import *
sys.path.insert(0, str(ROOT / 'training/transformer'))
import baseline_directions as bd
import ft_trainer as extension

ARMS = ('ltdd', 'cot_phi', 'dralign', 'neufair', 'mirrorfair')

def one(seed, arm):
    out = P / 'race_trained_joint_directions' / str(seed) / arm
    if (out / 'DONE.json').exists():
        return
    d, _, banks = extension.data('hmda_oh', seed, True)
    table = bd.primary()
    row = table[(table.task == 'hmda_oh') & (table.architecture == 'mlp') &
                (table.seed == seed) & (table.arm == arm)].iloc[0]
    pipeline = bd.pipeline(row, d)
    hard = arm == 'mirrorfair'
    records, arrays = [], {}
    for name, bank in banks['audit'].items():
        keep = bank['supported']
        ids = bank['indices'][keep]
        corners = bank['corners'][:, keep]
        assert corners.shape[0] == 8
        flat = corners.reshape(-1, corners.shape[-1])
        probability = pipeline(flat, 'P').reshape(8, -1)
        sign = q.MAP['hmda_oh'].get(name, 1)
        if hard:
            assert np.isin(probability, [0, 1]).all()
            decisions = probability.reshape(4, 2, -1)
            delta = decisions[:, 1] - decisions[:, 0]
            metrics = dict(n=len(ids), sign=sign,
                adverse_decision=float((sign * delta < 0).any(0).mean()))
        else:
            logits = pipeline(flat, 'L').reshape(8, -1)
            np.testing.assert_allclose(expit(logits), probability, atol=2e-6, rtol=2e-6)
            metrics = extension.direction_metrics(logits, sign, True)
            arrays[name + '_L'] = logits
        arrays[name + ('_decision' if hard else '_P')] = probability
        arrays[name + '_indices'] = ids
        records.append(dict(task='hmda_oh', architecture='mlp', seed=seed, arm=arm,
            edit=name, training_edit=name in q.MAP['hmda_oh'],
            output_kind='hard_decision' if hard else 'probability', **metrics))
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out / 'per_edit.csv', index=False)
    metadata = ['task', 'architecture', 'seed', 'arm', 'edit', 'training_edit', 'output_kind', 'n', 'sign']
    macro = frame[frame.training_edit].drop(columns=metadata).mean().to_dict()
    pd.DataFrame([dict(task='hmda_oh', architecture='mlp', seed=seed, arm=arm,
        **{'direction_' + k: v for k, v in macro.items()})]).to_csv(out / 'per_seed.csv', index=False)
    np.savez_compressed(out / 'corners.npz', **arrays)
    write(out / 'DONE.json', dict(status='complete', pipeline=pipeline.provenance(),
        trained_protected_axes=['race'], audited_protected_axes=['race', 'sex'],
        training_performed=False, selection_performed=False,
        query_sha256=c.sha(out / 'corners.npz'), script_sha256=c.sha(__file__),
        specification_sha256=c.sha(q.P / 'DIRECTION_SPECIFICATIONS.json')))

if __name__ == '__main__':
    import concurrent.futures, subprocess
    if len(sys.argv) == 3:
        one(int(sys.argv[1]), sys.argv[2])
    else:
        jobs = [(s, a) for s in range(1000, 1010) for a in ARMS]
        def run(job):
            seed, arm = job
            log = P / 'logs' / f'race_joint_direction_{seed}_{arm}.log'
            with log.open('w') as stream:
                result = subprocess.run([sys.executable, __file__, str(seed), arm], stdout=stream, stderr=subprocess.STDOUT)
            return dict(seed=seed, arm=arm, returncode=result.returncode, log=str(log))
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for result in executor.map(run, jobs):
                results.append(result)
                write(P / 'RACE_JOINT_DIRECTION_JOBS.json', results)
                if result['returncode'] or len(results) % 10 == 0:
                    print(result if result['returncode'] else f'Complete {len(results)}/{len(jobs)}', flush=True)
        assert all(row['returncode'] == 0 for row in results)
