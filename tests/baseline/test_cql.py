import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import torch

from hotel_pricing.cql import TrainingConfig, bellman_target, conservative_penalty, policy_table, train
from hotel_pricing.environment import HotelPricingEnv, load_config, sale_probability
from hotel_pricing.evaluation import rollouts, solve
from hotel_pricing.offline_data import collect_episodes, split_episodes


class CQLTests(unittest.TestCase):
    def test_terminal_targets_exclude_future_value(self):
        rewards = torch.tensor([2., 3.])
        next_q = torch.tensor([[100., 200.], [5., 8.]])
        target = bellman_target(rewards, torch.tensor([True, False]), next_q)
        torch.testing.assert_close(target, torch.tensor([2., 11.]))

    def test_penalty_pushes_down_unsupported_actions(self):
        q = torch.zeros((1, 3), requires_grad=True)
        penalty = conservative_penalty(q, torch.tensor([1]))
        self.assertAlmostEqual(penalty.item(), np.log(3), places=6)
        penalty.backward()
        self.assertLess(q.grad[0, 1], 0)
        self.assertGreater(q.grad[0, 0], 0)
        self.assertGreater(q.grad[0, 2], 0)

    def test_exact_values_match_two_day_enumeration(self):
        config = replace(load_config(), capacity=1, horizon=2)
        prices = np.array(config.prices)
        p = np.array([sale_probability(config, price) for price in prices])
        last_day = (p * prices).max()
        expected = (p * prices + (1 - p) * last_day).max()
        value, policy = solve(config)
        self.assertAlmostEqual(value, expected)
        self.assertAlmostEqual(solve(config, policy)[0], value)
        for a, price in enumerate(prices):
            fixed = np.full((3, 2), a)
            self.assertAlmostEqual(solve(config, fixed)[0], price * (1 - (1 - p[a]) ** 2))
        mix = np.broadcast_to([.1, .15, .5, .15, .1], (3, 2, 5))
        daily_reward = np.dot(mix[1, 1], p * prices)
        expected_mix = daily_reward * (2 - np.dot(mix[1, 1], p))
        self.assertAlmostEqual(solve(config, mix)[0], expected_mix)

    def test_training_is_offline_and_repeatable(self):
        config = replace(load_config(), capacity=2, horizon=5)
        splits = split_episodes(collect_episodes(config, episodes=20))
        settings = TrainingConfig(updates=8, batch_size=16, log_every=4, target_update=2)
        original = {k: v.copy() for k, v in splits['train'].items()}
        with patch.object(HotelPricingEnv, 'step', side_effect=AssertionError('Online interaction')):
            model1, history1 = train(splits['train'], splits['validation'], config, settings, seed=9)
            model2, history2 = train(splits['train'], splits['validation'], config, settings, seed=9)
        self.assertEqual(history1, history2)
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            torch.testing.assert_close(p1, p2, rtol=0, atol=0)
        for k, v in original.items():
            np.testing.assert_array_equal(splits['train'][k], v)
        policy = policy_table(model1, config)
        self.assertEqual(policy.shape, (6, 3))
        self.assertTrue(np.all((policy >= 0) & (policy < 5)))

    def test_rollout_accounting_and_repeatability(self):
        config = replace(load_config(), capacity=2, horizon=5)
        policy = np.full((6, 3), 2)
        first = rollouts(config, policy, range(10))
        second = rollouts(config, policy, range(10))
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_allclose(first[0], first[1] * config.capacity * 90)
        self.assertTrue(np.all((first[1] >= 0) & (first[1] <= 1)))


if __name__ == '__main__':
    unittest.main()
