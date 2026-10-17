import unittest
from dataclasses import replace
import numpy as np
from hotel_pricing.environment import load_config
from hotel_pricing.evaluation import solve
from hotel_pricing.policy_gap import explain_gap

class PolicyGapTests(unittest.TestCase):
    def test_decomposition_and_training_counts(self):
        config = replace(load_config(), horizon=5, capacity=2)
        data = {'observations': np.array([[5,2],[5,2],[4,1]]), 'actions': np.array([2,2,1])}
        for seed in range(4):
            policy = np.random.default_rng(seed).integers(0,5,size=(6,3))
            result = explain_gap(config,policy,data)
            self.assertAlmostEqual(result['contribution'].sum(),solve(config)[0]-solve(config,policy)[0])
            self.assertTrue((result['local_loss'] >= -1e-10).all())
            self.assertEqual(result['visits'][5,2],1)
            self.assertEqual(result['counts'][5,2,2],2)
            self.assertEqual(result['counts'][4,1,1],1)
            self.assertEqual(result['visits'][:,0].sum(),0)
            self.assertEqual(result['visits'][0,:].sum(),0)
            self.assertEqual(result['visits'][5,1],0)

    def test_optimal_policy_has_zero_loss(self):
        config=replace(load_config(),horizon=5,capacity=2)
        _, policy=solve(config)
        data={'observations':np.empty((0,2),dtype=int),'actions':np.array([],dtype=int)}
        result=explain_gap(config,policy,data)
        self.assertAlmostEqual(result['gap'],0)
        np.testing.assert_allclose(result['contribution'],0,atol=1e-10)
