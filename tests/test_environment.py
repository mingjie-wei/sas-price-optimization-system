from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from gymnasium.utils.env_checker import check_env

from hotel_pricing.environment import HotelPricingEnv, load_config, sale_probability
from hotel_pricing.offline_data import collect_episodes, split_episodes, save_dataset, load_dataset


class EnvironmentTests(unittest.TestCase):
    def test_gymnasium_interface(self):
        check_env(HotelPricingEnv(), skip_render_check=True)

    def test_reference_probability_and_price_response(self):
        config = load_config()
        self.assertAlmostEqual(sale_probability(config, 90), 12 / 330)
        probabilities = [sale_probability(config, price) for price in config.prices]
        self.assertTrue(all(0 < p < 1 for p in probabilities))
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))

    def test_horizon_and_sellout_are_terminal(self):
        for probability in [0, 1]:
            env = HotelPricingEnv(replace(load_config(), horizon=3, capacity=2))
            state, _ = env.reset(seed=4)
            revenue = 0
            with patch('hotel_pricing.environment.sale_probability', return_value=probability):
                for step in range(3):
                    next_state, reward, terminated, truncated, info = env.step(2)
                    self.assertEqual(next_state[0], state[0] - 1)
                    self.assertEqual(next_state[1], state[1] - info['sold'])
                    self.assertEqual(reward, info['sold'] * 90)
                    self.assertFalse(truncated)
                    self.assertTrue(env.observation_space.contains(next_state))
                    revenue += reward
                    state = next_state
                    if terminated:
                        break
            self.assertEqual(step + 1, 2 if probability else 3)
            self.assertEqual(revenue, 180 if probability else 0)
            with self.assertRaises(RuntimeError):
                env.step(0)

    def test_seed_and_invalid_action(self):
        env = HotelPricingEnv()
        paths = []
        for _ in range(2):
            env.reset(seed=19)
            paths.append([env.step(2)[1] for _ in range(50)])
        self.assertEqual(paths[0], paths[1])
        for action in [-1, 5, 1.5]:
            with self.assertRaises(ValueError):
                env.step(action)

    def test_unsupported_assumptions_fail(self):
        for change in [{'cancellations': True}, {'horizon': 0}, {'reference_sale_probability': 1}]:
            with self.assertRaises(ValueError):
                replace(load_config(), **change)


class OfflineDataTests(unittest.TestCase):
    def test_contiguous_episodes_and_independent_splits(self):
        config = replace(load_config(), horizon=8, capacity=2)
        records = collect_episodes(config, episodes=20, seed=9)
        splits = split_episodes(records, seed=9)
        ids = [set(data['episode_id']) for data in splits.values()]
        self.assertEqual([len(s) for s in ids], [16, 2, 2])
        self.assertEqual(len(set.union(*ids)), 20)
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        for episode_id in range(20):
            mask = records['episode_id'] == episode_id
            states = records['observations'][mask]
            next_states = records['next_observations'][mask]
            done = records['terminated'][mask]
            np.testing.assert_array_equal(states[0], [8, 2])
            np.testing.assert_array_equal(next_states[:-1], states[1:])
            self.assertTrue(done[-1])
            self.assertEqual(int(done.sum()), 1)
        self.assertTrue(np.any(records['rewards'] == 0))
        self.assertFalse(records['truncated'].any())
        replay = collect_episodes(config, episodes=20, seed=9)
        for name in records:
            np.testing.assert_array_equal(records[name], replay[name])

    def test_frozen_files_and_integrity_check(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'dataset'
            config = replace(load_config(), horizon=4, capacity=2)
            save_dataset(target, config, episodes=10, seed=3)
            self.assertGreater(len(load_dataset(target, 'train')['actions']), 0)
            with self.assertRaises(FileExistsError):
                save_dataset(target, config, episodes=10, seed=3)
            with (target / 'train.npz').open('ab') as file:
                file.write(b'changed')
            with self.assertRaises(ValueError):
                load_dataset(target, 'train')


if __name__ == '__main__':
    unittest.main()
