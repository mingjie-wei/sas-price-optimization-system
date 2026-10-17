"""Re-evaluate the baseline policies, then fit offline DQN on their original data."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .cql import TrainingConfig, make_network, policy_table
from .diagnostics import paired_difference
from .environment import load_config
from .evaluation import solve, rollouts
from .offline_data import load_dataset
from .run_cql import run

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(plan_path, output):
    plan = json.loads(plan_path.read_text())
    source = ROOT / plan['cql_run']
    data = ROOT / plan['dataset']
    old = json.loads((source / 'results.json').read_text())
    config = load_config(ROOT / plan['config'])
    assert json.loads(json.dumps(asdict(config))) == old['config']
    assert digest(data / 'manifest.json') == old['dataset_manifest_sha256']
    assert asdict(TrainingConfig(updates=plan['updates'])) == old['training']
    assert plan['training_seeds'] == old['training_seeds']
    # Compare against recorded baseline source hashes, without a duplicate package.
    provenance = json.loads((ROOT / 'docs/provenance/baseline.json').read_text())
    for name, checksum in provenance['learning_source_sha256'].items():
        assert digest(Path(__file__).with_name(name)) == checksum
    saved_seeds = set()
    for split in ['train', 'validation', 'test']:
        records = load_dataset(data, split)
        saved_seeds.update(records['environment_seed'].tolist())
    seeds = np.arange(plan['evaluation_seed'], plan['evaluation_seed'] + plan['evaluation_episodes'])
    assert not saved_seeds.intersection(seeds.tolist())
    old_start = old['evaluation']['first_environment_seed']
    assert not set(range(old_start, old_start+old['evaluation']['episodes'])).intersection(seeds.tolist())
    with np.load(source / 'policies.npz', allow_pickle=False) as saved:
        policies = dict(saved)
    # Load the original checkpoints and confirm that their decisions match the saved tables.
    for seed in plan['training_seeds']:
        checkpoint = torch.load(source / f'cql_seed_{seed}.pt', map_location='cpu', weights_only=True)
        model = make_network(len(config.prices))
        model.load_state_dict(checkpoint['state_dict'])
        np.testing.assert_array_equal(policy_table(model.eval(), config), policies[f'CQL seed {seed}'])
    provenance = {str(p.relative_to(ROOT)): digest(p) for p in [
        data / 'manifest.json', source / 'results.json', source / 'policies.npz',
        *[source / f'cql_seed_{seed}.pt' for seed in plan['training_seeds']]]}
    frozen = {**plan, 'source_sha256': provenance, 'original_training_settings': old['training']}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'protocol.json').write_text(json.dumps(frozen, indent=2)+'\n')
    baseline = output / 'cql'
    baseline.mkdir()
    optimum, _ = solve(config)
    results, episodes = [], {'environment_seeds': seeds}
    for name, policy in policies.items():
        exact, _ = solve(config, policy)
        revenue, occupancy = rollouts(config, policy, seeds, action_seed=plan['action_seed'])
        episodes[name+' revenue'] = revenue
        episodes[name+' occupancy'] = occupancy
        results.append({'policy': name, 'exact_revenue': exact,
                        'optimality_gap_percent': 100*(optimum-exact)/optimum,
                        'mean_revenue': float(revenue.mean()),
                        'mean_revenue_ci95_half_width': float(1.96*revenue.std(ddof=1)/np.sqrt(len(seeds))),
                        'mean_occupancy': float(occupancy.mean())})
        print(f'Original {name}: exact {exact:.2f}', flush=True)
    np.savez_compressed(baseline/'policies.npz', **policies)
    np.savez_compressed(baseline/'evaluation_episodes.npz', **episodes)
    (baseline/'results.json').write_text(json.dumps({'results': results, 'source': plan['cql_run'],
        'retrained': False, 'evaluation': frozen}, indent=2)+'\n')
    dqn = run(data, output/'offline_dqn', updates=plan['updates'],
              seeds=tuple(plan['training_seeds']), config_path=ROOT/plan['config'],
              evaluation_episodes=plan['evaluation_episodes'], evaluation_seed=plan['evaluation_seed'],
              action_seed=plan['action_seed'], algorithm='Offline DQN', alpha=0)
    paired = []
    with np.load(output/'offline_dqn/evaluation_episodes.npz') as other:
        np.testing.assert_array_equal(seeds, other['environment_seeds'])
        for seed in plan['training_seeds']:
            paired.append({'seed': seed, **paired_difference(other[f'Offline DQN seed {seed} revenue'],
                                                             episodes[f'CQL seed {seed} revenue'])})
    report = {'protocol': frozen, 'cql': results, 'offline_dqn': dqn['results'],
              'paired_dqn_minus_cql': paired,
              'code_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}}
    (output/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    for path, checksum in provenance.items():
        assert digest(ROOT/path) == checksum
    print(f'Completed: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT/'configs/baseline_comparison.json')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/baseline/comparison')
    args = parser.parse_args()
    compare(args.plan, args.output)
