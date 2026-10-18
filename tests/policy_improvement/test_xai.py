"""Checks for explanation and intervention validity, not plot styling."""
from dataclasses import replace
import unittest
import numpy as np
import torch
from hotel_pricing.cql import make_network

from hotel_pricing.environment import load_config, HotelPricingEnv, sale_probability
from hotel_pricing.xai_audit import reachable, shapley, counterfactual, simulator_values, model_values
from hotel_pricing.xai_experiment import collect_extra, collection_probabilities


class XAITests(unittest.TestCase):
    def setUp(self):
        self.config=replace(load_config(),horizon=8,capacity=2)
        self.plan={'additional_episodes':10,'target_days':3,
                   'ordinary_probabilities':[.1,.15,.5,.15,.1],'target_probabilities':[.2]*5,
                   'collection_environment_seed':21000000,'collection_action_seed':22000000}

    def test_reachable_grid_and_shapley_interaction(self):
        self.assertEqual(reachable(load_config()).sum(),3255)
        t,r=np.indices((9,3)); f=2*t+3*r+t*r
        reference=np.array([[1,1],[7,2]])
        base,pt,pr,error=shapley(f,[[4,2]],reference,self.config)
        self.assertAlmostEqual(base+pt[0]+pr[0],f[4,2])
        self.assertLess(error,1e-12)
        with self.assertRaises(ValueError):shapley(f,[[8,1]],reference,self.config)

    def test_counterfactual_full_argmax_and_no_solution(self):
        policy=np.full((9,3),2)
        self.assertIsNone(counterfactual(policy,[4,2],self.config))
        policy[3,2]=0
        found=counterfactual(policy,[4,2],self.config)
        self.assertEqual(found['state'],[3,2])
        self.assertEqual(found['price'],55)
        self.assertIsNone(counterfactual(policy,[1,1],self.config,future_only=True))
        with self.assertRaises(ValueError):counterfactual(policy,[8,1],self.config)

    def test_one_day_continuation_values(self):
        q_star,_=simulator_values(self.config)
        q_policy,_=simulator_values(self.config,np.full((9,3),2))
        expected=[p*sale_probability(self.config,p) for p in self.config.prices]
        np.testing.assert_allclose(q_star[1,1],expected)
        np.testing.assert_allclose(q_policy[1,1],expected)

    def test_collector_rule_and_episode_integrity(self):
        np.testing.assert_array_equal(collection_probabilities([4,2],self.plan,True),self.plan['ordinary_probabilities'])
        np.testing.assert_array_equal(collection_probabilities([3,2],self.plan,True),self.plan['target_probabilities'])
        ordinary=collect_extra(self.config,self.plan,False,2000)
        targeted=collect_extra(self.config,self.plan,True,2400)
        for records in [ordinary,targeted]:
            self.assertEqual(records['terminated'].sum(),10)
            self.assertFalse(records['truncated'].any())
            for episode in np.unique(records['episode_id']):
                i=np.flatnonzero(records['episode_id']==episode)
                np.testing.assert_array_equal(records['observations'][i[0]],[8,2])
                np.testing.assert_array_equal(records['next_observations'][i[:-1]],records['observations'][i[1:]])
                np.testing.assert_array_equal(records['step'][i],np.arange(len(i)))
                self.assertTrue(records['terminated'][i[-1]])
        # Outside target, the paired collectors are identical up to divergence.
        for records in [ordinary,targeted]:
            for i,state in enumerate(records['observations']):
                p=collection_probabilities(state,self.plan,records is targeted)
                self.assertEqual(records['action_probability'][i],p[records['actions'][i]])
        a=ordinary['observations'][:,0]>3;b=targeted['observations'][:,0]>3
        np.testing.assert_array_equal(ordinary['observations'][a],targeted['observations'][b])
        np.testing.assert_array_equal(ordinary['actions'][a],targeted['actions'][b])

    def test_sale_randomness_is_time_aligned_and_separate(self):
        first,second=HotelPricingEnv(self.config),HotelPricingEnv(self.config)
        first.reset(seed=77);second.reset(seed=77)
        draws=np.random.default_rng(77).random(8)
        for step in range(8):
            for env,action in [(first,0),(second,4)]:
                if env.ended:continue
                _,reward,_,_,_=env.step(action)
                expected=int(draws[step]<sale_probability(self.config,self.config.prices[action]))
                self.assertEqual(reward,self.config.prices[action]*expected)

    def test_margin_matches_independent_direct_inference(self):
        torch.manual_seed(12)
        model=make_network(5).eval()
        q=model_values(model,self.config)
        state=[4,2]
        with torch.no_grad():
            direct=model(torch.tensor([[state[0]/8,state[1]/2]],dtype=torch.float32)).numpy()[0]
        np.testing.assert_allclose(q[4,2],direct,atol=1e-6,rtol=0)
        self.assertAlmostEqual(q[4,2,0]-q[4,2,2],float(direct[0]-direct[2]),places=6)
