"""Numerical guardrails for follow-up targets, replay, continuation, and probes."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.policy_improvement.learning import (draw_indices,fit_trajectory,probe_records,plan_from_empirical_probabilities,
                               tiny_fit,transform,q_grid)
from hotel_pricing.cql import bellman_target,train,TrainingConfig,make_network
from hotel_pricing.environment import load_config,HotelPricingEnv
from hotel_pricing.offline_data import collect_episodes,split_episodes


class LearningFollowupTests(unittest.TestCase):
    def test_log_time_training_and_inference_match(self):
        config=load_config()
        records=probe_records(config,[[1,1],[30,4],[330,10]],[0,1,2],8)
        batch=transform(records,config,'log_time')
        np.testing.assert_allclose(batch['states'][:,0],np.log1p([1,30,330])/np.log1p(330),rtol=1e-6)
        model=make_network(5)
        grid=q_grid(model,config,'log_time')
        with torch.no_grad():
            np.testing.assert_allclose(model(batch['states']).numpy(),grid[[1,30,330],[1,4,10]],atol=1e-6)

    def test_terminal_target_ignores_extreme_next_values(self):
        rewards=torch.tensor([0.,1.,55/90,.5]);terminated=torch.tensor([True,True,True,False])
        actual=bellman_target(rewards,terminated,torch.full((4,5),1e6))
        torch.testing.assert_close(actual[:3],rewards[:3])
        self.assertEqual(actual[3],1000000.5)

    def test_replay_has_exact_strata_and_no_reward_filter(self):
        records=collect_episodes(replace(load_config(),horizon=40,capacity=10),episodes=10)
        data=transform(records,load_config())
        late=torch.tensor(np.flatnonzero(records['observations'][:,0]<=30));early=torch.tensor(np.flatnonzero(records['observations'][:,0]>30))
        idx=draw_indices(data,late,early,'late_32_224')
        self.assertEqual(int((records['observations'][idx,0]<=30).sum()),32)
        self.assertEqual(len(idx),256)
        with self.assertRaises(ValueError):draw_indices(data,torch.tensor([],dtype=torch.long),early,'late_32_224')

    def test_uniform_matches_original_and_resume_is_exact(self):
        config=replace(load_config(),horizon=8,capacity=2)
        records=collect_episodes(config,episodes=10)
        spec={'seed':4,'preprocessing':'linear','alpha':.1,'replay_rule':'uniform'}
        original,_=train(records,records,config,TrainingConfig(updates=200),seed=4)
        with tempfile.TemporaryDirectory() as temp:
            direct,target,_=fit_trajectory(records,config,spec,Path(temp)/'direct',200)
            fit_trajectory(records,config,spec,Path(temp)/'resume',100)
            resumed,_,_=fit_trajectory(records,config,spec,Path(temp)/'resume',200)
            for a,b,c in zip(original.parameters(),direct.parameters(),resumed.parameters()):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                torch.testing.assert_close(b,c,rtol=0,atol=0)
            for p,t in zip(direct.parameters(),target.parameters()):
                self.assertNotEqual(p.data_ptr(),t.data_ptr())
                self.assertFalse(t.requires_grad)
                torch.testing.assert_close(p,t,rtol=0,atol=0)

    def test_probe_uses_ordinary_kernel_and_correct_termination(self):
        config=load_config();seed=61000000
        samples=probe_records(config,[[330,10]],[2],seed)
        env=HotelPricingEnv(config);before,_=env.reset(seed=seed)
        after,reward,ended,truncated,_=env.step(2)
        np.testing.assert_array_equal(samples['next_observations'][0],after)
        self.assertEqual(samples['rewards'][0],reward)
        self.assertEqual(samples['terminated'][0],ended)
        terminal=probe_records(config,[[1,1],[1,10]],[0,4],seed)
        self.assertTrue(terminal['terminated'].all())
        nonterminal=probe_records(config,[[30,10]],[2],seed)
        self.assertFalse(nonterminal['terminated'][0])

    def test_tiny_objective_and_missing_support(self):
        record={'actions':np.repeat(np.arange(5),10),'rewards':np.tile([0,90],25)}
        fit=tiny_fit(record,0)
        np.testing.assert_allclose(fit['q'],[.5]*5,atol=1e-9)
        missing={'actions':np.array([0,1]),'rewards':np.array([0.,1.])}
        self.assertEqual(tiny_fit(missing,.1)['status'],'missing_action_support')

    def test_learned_planner_uses_supplied_estimates(self):
        config=replace(load_config(),horizon=1,capacity=1)
        policy=plan_from_empirical_probabilities(config,[0,0,0,0,1])
        self.assertEqual(policy[1,1],4)
