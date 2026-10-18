import unittest
from unittest.mock import patch
from dataclasses import replace
import numpy as np
import torch

from hotel_pricing.cql import TrainingConfig, train, losses
from hotel_pricing.diagnostics import paired_difference
from hotel_pricing.environment import load_config, HotelPricingEnv
from hotel_pricing.offline_data import collect_episodes, split_episodes


class AlgorithmComparisonTests(unittest.TestCase):
    def test_dqn_training_is_offline_and_ignores_penalty_gradient(self):
        config = replace(load_config(), horizon=5, capacity=2)
        data = split_episodes(collect_episodes(config, episodes=20))
        settings = TrainingConfig(updates=5, alpha=0, batch_size=16, log_every=5)
        with patch.object(HotelPricingEnv, 'step', side_effect=AssertionError('Online interaction')):
            first, _ = train(data['train'], data['validation'], config, settings, seed=4)
            # Changing the finite penalty must have no effect when alpha is zero.
            def changed_penalty(network, target, batch):
                td, penalty = losses(network, target, batch)
                return td, 37 * penalty
            with patch('hotel_pricing.cql.losses', side_effect=changed_penalty):
                second, _ = train(data['train'], data['validation'], config, settings, seed=4)
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_paired_intervals_preserve_matching(self):
        first = np.array([1, 50, 100, 500])
        result = paired_difference(first + 3, first)
        self.assertEqual(result['mean'], 3)
        self.assertEqual(result['ci95_half_width'], 0)
