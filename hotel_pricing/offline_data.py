"""Collect a fixed synthetic dataset. This module does not train a policy."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from .behavior import probability_table
from .environment import HotelPricingEnv, load_config


BEHAVIOR_PROBABILITIES = (0.10, 0.15, 0.50, 0.15, 0.10)


def validate_probabilities(probabilities, actions):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (actions,) or not np.isfinite(probabilities).all():
        raise ValueError('Provide one finite probability for each price')
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError('Action probabilities must be nonnegative and sum to 1')
    return probabilities


def collect_episodes(config, episodes=2000, seed=42, probabilities=BEHAVIOR_PROBABILITIES, behavior=None):
    """Collect with independent sale and action RNGs, using observed state only."""
    if type(episodes) is not int or episodes < 10:
        raise ValueError('Use at least 10 episodes for train/validation/test splits')
    behavior = behavior or {'probabilities': probabilities}
    table = probability_table(config, behavior)
    env = HotelPricingEnv(config)
    records = {name: [] for name in [
        'observations', 'actions', 'rewards', 'next_observations', 'terminated',
        'truncated', 'episode_id', 'step', 'action_probability', 'environment_seed', 'action_seed',
    ]}
    for episode_id in range(episodes):
        # Separate random streams: choosing a price should not change the sale draw.
        env_seed, action_seed = np.random.SeedSequence([seed, episode_id]).generate_state(2)
        rng = np.random.default_rng(int(action_seed))
        observation, _ = env.reset(seed=int(env_seed))
        done = False
        step = 0
        while not done:
            # Choose a price, run one day, then save exactly what happened.
            probabilities = table[tuple(observation)]
            action = int(rng.choice(len(config.prices), p=probabilities))
            next_observation, reward, terminated, truncated, _ = env.step(action)
            transition = {
                'observations': observation, 'actions': action, 'rewards': reward,
                'next_observations': next_observation, 'terminated': terminated,
                'truncated': truncated, 'episode_id': episode_id, 'step': step,
                'action_probability': probabilities[action],
                'environment_seed': env_seed, 'action_seed': action_seed,
            }
            for name, value in transition.items():
                records[name].append(value)
            observation = next_observation
            done = terminated or truncated
            step += 1
    env.close()
    # Lists are convenient while collecting; arrays are convenient for learning.
    for name in records:
        records[name] = np.asarray(records[name])
    return records


def split_episodes(records, seed=42):
    """Split whole episodes 80/10/10; never mix steps from one episode."""
    ids = np.unique(records['episode_id'])
    shuffled = np.random.default_rng(seed).permutation(ids)
    train_end = int(len(ids) * .8)
    validation_end = int(len(ids) * .9)
    groups = [shuffled[:train_end], shuffled[train_end:validation_end], shuffled[validation_end:]]
    splits = {}
    for name, group in zip(['train', 'validation', 'test'], groups):
        mask = np.isin(records['episode_id'], group)
        splits[name] = {key: value[mask] for key, value in records.items()}
    return splits


def describe(records, config):
    """Coverage is measured over structurally reachable nonterminal states."""
    t, c = records['observations'].T
    visits = np.zeros((config.horizon + 1, config.capacity + 1, len(config.prices)), dtype=int)
    np.add.at(visits, (t, c, records['actions']), 1)
    reachable = np.zeros(visits.shape[:2], dtype=bool)
    for time_left in range(1, config.horizon + 1):
        for rooms_left in range(1, config.capacity + 1):
            reachable[time_left, rooms_left] = config.capacity - rooms_left <= config.horizon - time_left
    ids = np.unique(records['episode_id'])
    return {
        'episodes': int(len(ids)),
        'transitions': int(len(t)),
        'action_counts': np.bincount(records['actions'], minlength=len(config.prices)).tolist(),
        'no_sale_fraction': float(np.mean(records['rewards'] == 0)),
        'mean_episode_revenue': float(records['rewards'].sum() / len(ids)),
        'mean_final_occupancy': float(np.sum(records['rewards'] > 0) / (len(ids) * config.capacity)),
        'state_action_coverage': float(np.mean(visits[reachable] > 0)),
        'reachable_state_action_pairs': int(reachable.sum() * len(config.prices)),
    }


def save_dataset(directory, config, episodes=2000, seed=42,
                 probabilities=BEHAVIOR_PROBABILITIES, behavior_name='fixed random price mix',
                 dataset_version='hotel-v1', behavior=None):
    """Write once. Use a new directory for a different dataset version."""
    behavior = behavior or {'name': behavior_name, 'probabilities': list(probabilities)}
    probability_table(config, behavior)  # Validate before creating files.
    directory = Path(directory)
    # An existing destination is never silently regenerated or overwritten.
    directory.mkdir(parents=True, exist_ok=False)
    records = collect_episodes(config, episodes, seed, probabilities, behavior)
    splits = split_episodes(records, seed)
    metadata = {
        'dataset_version': dataset_version, 'synthetic': True,
        'config': asdict(config), 'episodes': episodes, 'seed': seed,
        'behavior': behavior,
        'split_rule': '80/10/10 by shuffled episode ID; no transition crosses splits',
        'observation_order': ['selling_periods_remaining', 'rooms_remaining'],
        'reward': 'raw sales revenue; no normalization',
        'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'gymnasium']},
        'files': {},
    }
    for name, data in splits.items():
        path = directory / f'{name}.npz'
        np.savez_compressed(path, **data)
        metadata['files'][name] = {
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            **describe(data, config),
        }
    # Snapshot the exact transition/collection code used with the data.
    metadata['code_sha256'] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ['environment.py', 'offline_data.py', 'behavior.py']
    }
    (directory / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def load_dataset(directory, split):
    """Check integrity before returning arrays for one split."""
    directory = Path(directory)
    metadata = json.loads((directory / 'manifest.json').read_text())
    path = directory / f'{split}.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['files'][split]['sha256']:
        raise ValueError(f'{split} data changed after collection')
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/simulated/baseline'))
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    metadata = save_dataset(args.output, load_config(), args.episodes, args.seed)
    for split, info in metadata['files'].items():
        print(f"{split}: {info['episodes']} episodes, {info['transitions']:,} transitions")
    print(f'Saved fixed dataset: {args.output.resolve()}')


if __name__ == '__main__':
    main()
