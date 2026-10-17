"""Run one pre-specified offline CQL experiment; save every training seed."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .behavior import probability_table
from .cql import TrainingConfig, diagnostics, policy_table, tensors, train
from .environment import load_config
from .evaluation import rollouts, solve
from .offline_data import load_dataset


def run(data_path, output, updates=20000, evaluation_episodes=1000, seeds=(0, 1, 2),
        config_path=None, evaluation_seed=1000000, action_seed=2000000, algorithm="CQL", alpha=0.1):
    if algorithm not in ['CQL', 'Offline DQN']:
        raise ValueError('Choose CQL or Offline DQN')
    if algorithm == 'Offline DQN' and alpha != 0:
        raise ValueError('Offline DQN has no CQL penalty: use alpha=0')
    if evaluation_episodes < 2:
        raise ValueError('Use at least two evaluation episodes')
    config = load_config(config_path)
    # Give the learner only dimensions and numerical scales, not demand parameters.
    learning_config = SimpleNamespace(horizon=config.horizon, capacity=config.capacity,
                                      reference_price=config.reference_price, prices=config.prices)
    manifest_path = data_path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['config'] != json.loads(json.dumps(asdict(config))):
        raise ValueError('Simulator config differs from the fixed dataset')
    records = {name: load_dataset(data_path, name) for name in ['train', 'validation']}
    output.mkdir(parents=True, exist_ok=False)
    settings = TrainingConfig(updates=updates, alpha=alpha)
    histories, policies, models = {}, {}, {}
    # Fit every seed on the same saved training set.
    for seed in seeds:
        model, history = train(records['train'], records['validation'], learning_config, settings, seed)
        name = f'{algorithm} seed {seed}'
        policies[name] = policy_table(model, learning_config)
        models[name] = model
        histories[name] = history
        torch.save({'state_dict': model.state_dict(), 'config': asdict(config),
                    'settings': asdict(settings), 'seed': seed}, output / f'{algorithm.lower().replace(" ", "_")}_seed_{seed}.pt')
        print(f'Finished {name}: {updates:,} offline updates', flush=True)

    # All final checkpoints are now fixed. No best-seed or test-based selection.
    test = load_dataset(data_path, 'test')
    test_metrics = {name: diagnostics(model, tensors(test, learning_config)) for name, model in models.items()}
    # These benchmarks are for evaluation, after training has finished.
    shape = (config.horizon + 1, config.capacity + 1)
    policies['Behavior policy'] = probability_table(config, manifest['behavior'])
    policies[f'Reference price {config.reference_price}'] = np.full(shape, config.prices.index(config.reference_price))
    fixed_values = [solve(config, np.full(shape, a))[0] for a in range(len(config.prices))]
    best_action = int(np.argmax(fixed_values))
    policies[f'Best fixed {config.prices[best_action]} (model-informed)'] = np.full(shape, best_action)
    optimal_value, policies['Optimum (model-informed)'] = solve(config)

    evaluation_seeds = np.arange(evaluation_seed, evaluation_seed + evaluation_episodes)
    saved_seeds = set(np.concatenate([records['train']['environment_seed'],
                                     records['validation']['environment_seed'], test['environment_seed']]).tolist())
    assert not saved_seeds.intersection(evaluation_seeds.tolist())
    results, episode_results = [], {}
    for name, policy in policies.items():
        exact_value, _ = solve(config, policy)
        revenue, occupancy = rollouts(config, policy, evaluation_seeds, action_seed=action_seed)
        episode_results[name + ' revenue'] = revenue
        episode_results[name + ' occupancy'] = occupancy
        sd = float(revenue.std(ddof=1))
        results.append({'policy': name, 'exact_revenue': exact_value,
                        'optimality_gap_percent': 100 * (optimal_value - exact_value) / optimal_value,
                        'mean_revenue': float(revenue.mean()), 'episode_revenue_sd': sd,
                        'mean_revenue_ci95_half_width': 1.96 * sd / np.sqrt(evaluation_episodes),
                        'mean_occupancy': float(occupancy.mean())})
        print(f'{name}: exact revenue {exact_value:.2f}, simulated {revenue.mean():.2f}', flush=True)
    np.savez_compressed(output / 'policies.npz', **policies)
    np.savez_compressed(output / 'evaluation_episodes.npz', environment_seeds=evaluation_seeds, **episode_results)
    result = {'algorithm': algorithm, 'config': asdict(config), 'training': asdict(settings), 'training_seeds': list(seeds),
              'checkpoint_rule': 'final update for every seed; no hyperparameter or seed selection',
              'dataset_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'dataset_files': manifest['files'], 'dataset': str(data_path.resolve()),
              'versions': {'torch': torch.__version__, 'numpy': np.__version__},
              'evaluation': {'episodes': evaluation_episodes, 'first_environment_seed': evaluation_seed,
                             'behavior_action_seed': action_seed, 'matched_sale_randomness': True},
              'code_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ['cql.py', 'evaluation.py', 'run_cql.py', 'environment.py', 'behavior.py']},
              'history': histories, 'test_diagnostics': test_metrics, 'results': results}
    (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/simulated/baseline'))
    parser.add_argument('--output', type=Path, default=Path('outputs/baseline/cql'))
    parser.add_argument('--updates', type=int, default=20000)
    parser.add_argument('--evaluation-episodes', type=int, default=1000)
    args = parser.parse_args()
    run(args.data, args.output, args.updates, args.evaluation_episodes)
