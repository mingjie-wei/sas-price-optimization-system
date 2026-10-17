from dataclasses import replace
import unittest
import numpy as np

from hotel_pricing.environment import load_config, sale_probability
from hotel_pricing.offline_data import collect_episodes, split_episodes
from hotel_pricing.diagnostics import coverage, optimal_state_visits, verify_environment
from hotel_pricing.evaluation import solve
import hashlib
import json
from pathlib import Path


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(load_config(), horizon=8, capacity=2)

    def test_default_collection_preserves_baseline(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/baseline_collection.json').read_text())
        config = replace(load_config(), horizon=fixture['horizon'], capacity=fixture['capacity'])
        records = collect_episodes(config, episodes=fixture['episodes'], seed=fixture['seed'])
        self.assertEqual(set(records), set(fixture['arrays']))
        for key, expected in fixture['arrays'].items():
            values = records[key]
            self.assertEqual(list(values.shape), expected['shape'])
            self.assertEqual(values.dtype.str, expected['dtype'])
            self.assertEqual(hashlib.sha256(values.tobytes()).hexdigest(), expected['sha256'], key)

    def test_policy_is_configurable_and_recorded(self):
        data = collect_episodes(self.config, episodes=20, probabilities=[0, 0, 1, 0, 0])
        self.assertTrue(np.all(data['actions'] == 2))
        self.assertTrue(np.all(data['action_probability'] == 1))
        for bad in [[.2, .2], [-1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [np.nan, 0, 0, 0, 0]]:
            with self.assertRaises(ValueError):
                collect_episodes(self.config, episodes=20, probabilities=bad)

    def test_matched_collection_seeds_and_split_ids(self):
        datasets = [collect_episodes(self.config, episodes=20, probabilities=p)
                    for p in [[.1, .15, .5, .15, .1], [.2] * 5]]
        for key in ['environment_seed', 'action_seed']:
            a = datasets[0][key][datasets[0]['step'] == 0]
            b = datasets[1][key][datasets[1]['step'] == 0]
            np.testing.assert_array_equal(a, b)
        a, b = [split_episodes(d) for d in datasets]
        for split in a:
            np.testing.assert_array_equal(np.unique(a[split]['episode_id']), np.unique(b[split]['episode_id']))

    def test_exact_visit_weights(self):
        config = replace(self.config, capacity=1, horizon=2)
        policy = np.full((3, 2), 2)
        visits = optimal_state_visits(config, policy)
        p = sale_probability(config, 90)
        self.assertAlmostEqual(visits[2, 1], 1)
        self.assertAlmostEqual(visits[1, 1], 1 - p)
        self.assertAlmostEqual(visits.sum(), 2 - p)
        self.assertAlmostEqual((visits.sum() * p * 90), solve(config, policy)[0])

    def test_coverage_counts_and_thresholds(self):
        data = collect_episodes(self.config, episodes=20)
        stats, visits = coverage(data, self.config)
        self.assertEqual(visits.sum(), len(data['actions']))
        self.assertGreaterEqual(stats['coverage_at_least_1'], stats['coverage_at_least_5'])
        self.assertGreaterEqual(stats['coverage_at_least_5'], stats['coverage_at_least_20'])
        for threshold in [1, 5, 20]:
            self.assertTrue(0 <= stats[f'optimal_path_support_at_least_{threshold}'] <= 1)

    def test_environment_verification_reproduces_benchmark(self):
        result = verify_environment(load_config())
        self.assertEqual(result['best_fixed_price'], 90)
        self.assertAlmostEqual(result['optimal_revenue'], 888.4103406704032)
