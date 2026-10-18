"""Staged follow-up runner. Every stage is frozen before its outcomes are read."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.policy_improvement.learning import *
from scripts.study_layout import training_source_digest, historical_source_path
from hotel_pricing.cql import conservative_penalty
from hotel_pricing.xai_audit import load_model

PLAN=json.loads((ROOT/'configs/policy_improvement/training.json').read_text())
OUT=ROOT/PLAN['output']
CONFIG=load_config(ROOT/PLAN['environment'])


def initialize():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    sources={historical_source_path(p):training_source_digest(p) for p in [*sorted((ROOT/'hotel_pricing').glob('*.py')),ROOT/'scripts/policy_improvement/learning.py',ROOT/'scripts/policy_improvement/experiments.py']}
    if path.exists():
        prior=json.loads(path.read_text());assert prior['plan']==PLAN
        assert prior['sources']==sources,'Frozen follow-up source changed'
        return prior
    protected={}
    for folder in ['outputs/baseline','data/simulated/baseline','outputs/policy_improvement/coverage','data/simulated/policy_improvement/coverage']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and not any(part.startswith('.') for part in p.relative_to(ROOT).parts):protected[str(p.relative_to(ROOT))]=sha(p)
    # Old checkpoints lack optimizer/target/RNG state, so E1 uses new resumable trajectories.
    ckpt=torch.load(ROOT/'outputs/policy_improvement/coverage/models/original/cql_seed_0.pt',weights_only=True)
    existing_seeds=set()
    for p in (ROOT/'outputs').rglob('cql_seed_*.pt'):
        existing_seeds.add(int(p.stem.split('_')[-1]))
    assert not existing_seeds.intersection(PLAN['confirmation_seeds'])
    original=load_dataset(ROOT/PLAN['original_data'],'train')
    used=set()
    for folder in ['data/simulated/baseline','data/simulated/policy_improvement/coverage','outputs/baseline','outputs/policy_improvement/coverage']:
        for p in (ROOT/folder).rglob('*.npz'):
            with np.load(p) as a:
                for k in ['environment_seed','environment_seeds','action_seed']:
                    if k in a:used.update(map(int,a[k]))
    namespaces={k:PLAN[k] for k in PLAN if k.endswith('_seed')}
    assert len(set(namespaces.values()))==len(namespaces)
    for key,seed in namespaces.items():
        if 'evaluation_environment' in key:assert not used.intersection(range(seed,seed+1000))
        else:assert seed not in used
    protocol={'plan':PLAN,'sources':sources,'protected_sha256':protected,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'starting_working_tree':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        'versions':{p:importlib.metadata.version(p) for p in ['torch','numpy','gymnasium']},
        'old_checkpoint_keys':list(ckpt),'old_checkpoints_support_exact_continuation':False,
        'unused_confirmation_seeds_verified':True,'seed_namespaces':namespaces,
        'dataset_manifest_sha256':sha(ROOT/PLAN['original_data']/'manifest.json')}
    dump(path,protocol);return protocol


def protect(protocol):
    for p,h in protocol['protected_sha256'].items():assert sha(ROOT/p)==h,p


def stage_diagnostics():
    folder=OUT/'diagnostics';folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():return
    records=load_dataset(ROOT/PLAN['original_data'],'train')
    all_records={split:load_dataset(ROOT/PLAN['original_data'],split) for split in ['train','validation','test']}
    transition_checks={}
    for split,x in all_records.items():
        s=x['observations'];n=x['next_observations'];sold=s[:,1]-n[:,1]
        assert np.all(n[:,0]==s[:,0]-1) and np.all((sold==0)|(sold==1))
        np.testing.assert_array_equal(x['rewards'],np.array(CONFIG.prices)[x['actions']]*sold)
        np.testing.assert_array_equal(x['terminated'],(n[:,0]==0)|(n[:,1]==0))
        assert x['terminated'].dtype==np.bool_ and not x['truncated'].any()
        continuation=np.flatnonzero(~x['terminated'])
        np.testing.assert_array_equal(n[continuation],s[continuation+1])
        np.testing.assert_array_equal(x['episode_id'][continuation],x['episode_id'][continuation+1])
        transition_checks[split]={'rows':len(s),'terminal_dtype':str(x['terminated'].dtype),'accounting':'passed','episode_boundaries':'passed'}
    masks=[(records['observations'][:,0]==1)&(records['rewards']>0),
           (records['observations'][:,0]==1)&(records['rewards']==0),
           (records['observations'][:,0]>1)&records['terminated'],
           ~records['terminated']]
    examples=[]
    for label,mask in zip(['last_day_sale','last_day_no_sale','early_sellout','continuing'],masks):
        i=int(np.flatnonzero(mask)[0]);one={k:v[i:i+1] for k,v in records.items()};batch=tensors(one,CONFIG)
        y=bellman_target(batch['rewards'],batch['terminated'],torch.full((1,5),1e6))
        expected=batch['rewards'] if records['terminated'][i] else batch['rewards']+1e6
        torch.testing.assert_close(y,expected)
        examples.append({'case':label,'state':records['observations'][i].tolist(),'action':int(records['actions'][i]),
            'raw_reward':float(records['rewards'][i]),'scaled_reward':float(batch['rewards'][0]),'next_state':records['next_observations'][i].tolist(),
            'terminated':bool(records['terminated'][i]),'target_with_next_q_1e6':float(y[0])})
    q=torch.tensor([[.2,.7,1.1,-.3,.9],[.5,.4,.3,.2,.1]])
    acts=torch.tensor([2,0]);shift=17.
    torch.testing.assert_close(conservative_penalty(q+shift,acts),conservative_penalty(q,acts),atol=1e-6,rtol=1e-5)
    manual=(torch.logsumexp(q,1)-q[torch.arange(2),acts]).mean()
    torch.testing.assert_close(conservative_penalty(q,acts),manual)
    paths=['outputs/baseline/cql/cql_seed_0.pt','outputs/policy_improvement/coverage/models/original/cql_seed_0.pt']
    model_rows=[]
    for path in paths:
        model,checkpoint=load_model(ROOT/path,CONFIG)
        q=model_values(model,CONFIG);examples_q=q[1,1]
        raw=np.array([[1,1]],dtype=np.int32)
        batch=tensors({'observations':raw,'next_observations':raw,'actions':np.array([2]),'rewards':np.array([90]),'terminated':np.array([True])},CONFIG)
        np.testing.assert_allclose(batch['states'].numpy(),[[1/330,1/10]],atol=1e-8)
        with torch.no_grad():np.testing.assert_allclose(model(batch['states']).numpy()[0],examples_q,atol=1e-6)
        rows=residual_groups(model,model,records,CONFIG)
        tag='historical' if 'baseline/cql' in path else 'fresh_original'
        pd.DataFrame(rows).to_csv(folder/f'{tag}_group_residuals.csv',index=False)
        model_rows.append({'checkpoint':path,'sha256':sha(ROOT/path),'q_at_1_1':examples_q.tolist(),
            'q90_revenue_equivalent':float(examples_q[2]*90),'final_day_action_upper_bounds':(np.array(CONFIG.prices)/90).tolist(),
            'residual_bootstrap_source':'online model: historical target-network state was not saved'})
    result={'classification':'Mechanics correct, values poorly learned','transitions':transition_checks,'numerical_targets':examples,
        'models':model_rows,'feature_order':['days','rooms'],'normalization':[330,10],'reward_scale':90,
        'terminal_mask':'torch.bool before bitwise inversion','target_rule':'DQN max(target(next_state)), gamma 1',
        'target_network':'deepcopy, requires_grad=False, not in optimizer; synchronization after every 100 updates',
        'td_loss':'half mean squared error','cql':'mean(logsumexp over actions minus recorded-action value), alpha 0.1',
        'source_locations':{'features_and_rewards':'hotel_pricing/cql.py::tensors','targets':'hotel_pricing/cql.py::bellman_target',
                            'objective':'hotel_pricing/cql.py::losses','inference':'hotel_pricing/xai_audit.py::model_values'}}
    dump(folder/'pipeline_audit.json',result)
    # Local empirical objectives include all no-sales and select pre-action t=1.
    terminal=[]
    for label,mask in [('state_1_1',(records['observations']==[1,1]).all(1)),('pooled_t1',records['observations'][:,0]==1)]:
        subset={k:v[mask] for k,v in records.items()}
        terminal.append({'subset':label,'counts':outcomes(records,mask),'fits':[tiny_fit(subset,a) for a in [0.,.1,.01]]})
    dump(folder/'terminal_logged_fits.json',terminal)
    # Sparse terminal observations trigger the prespecified adequately supported control.
    stages={}
    for phase,n,seed in [('train',PLAN['terminal_train_per_action'],PLAN['terminal_collection_seed']),('evaluation',PLAN['terminal_eval_per_action'],PLAN['terminal_evaluation_seed'])]:
        actions=np.repeat(np.arange(5),n);states=np.tile([1,1],(len(actions),1))
        stages[phase]=probe_records(CONFIG,states,actions,seed)
        np.savez_compressed(folder/f'terminal_{phase}.npz',**stages[phase])
    fits=[]
    for alpha in [0.,.1]:
        tiny=tiny_fit(stages['train'],alpha)
        deterministic=fit_empirical_network(stages['train'],alpha)
        model,history=original_train(stages['train'],stages['evaluation'],CONFIG,
            TrainingConfig(updates=PLAN['terminal_updates'],alpha=alpha),PLAN['terminal_training_seed'])
        q=model_values(model,CONFIG)[1,1]
        fitted=np.array(tiny['q']);det_error=float(np.max(np.abs(np.array(deterministic)-fitted)))
        fits.append({'alpha':alpha,'tiny_fit':tiny,'deterministic_network_q':deterministic,'deterministic_max_error':det_error,
            'deterministic_fit_tolerance':.001,'deterministic_pass':det_error<=.001,
            'minibatch_q':q.tolist(),'minibatch_max_error':float(np.max(np.abs(q-fitted))),
            'minibatch_updates':PLAN['terminal_updates'],'evaluation_outcomes':outcomes(stages['evaluation'],np.ones(len(stages['evaluation']['actions']),bool))})
        torch.save({'state_dict':model.state_dict(),'alpha':alpha},folder/f'terminal_network_alpha_{alpha}.pt')
    dump(folder/'terminal_positive_control.json',fits)
    # Price-only stationary demand is verified by the environment's sale_probability(config,price) call.
    estimates=outcomes(records,np.ones(len(records['actions']),bool))
    probabilities=np.array([r['sale_probability'] for r in estimates])
    learned_policy=plan_from_empirical_probabilities(CONFIG,probabilities)
    validation=all_records['validation'];truth=(validation['rewards']>0).astype(float)
    predicted=probabilities[validation['actions']]
    reference=model_values(load_model(ROOT/paths[1],CONFIG)[0],CONFIG).argmax(2)
    revenue=solve(CONFIG,learned_policy)[0]
    learned={'label':'learned-data price-only demand planning; not the oracle','counts_and_probabilities':estimates,
        'structural_assumption':'Sale probability depends on price only, pooled over all time/inventory; inventory/time dynamics known.',
        'validation_brier':float(np.mean((predicted-truth)**2)),
        'validation_log_loss':float(-np.mean(truth*np.log(predicted)+(1-truth)*np.log(1-predicted))),
        'exact_revenue':revenue,'minus_fixed90':revenue-solve(CONFIG,np.full((331,11),2))[0],
        'minus_verified_cql':revenue-solve(CONFIG,reference)[0],'oracle_gap':solve(CONFIG)[0]-revenue,
        'validation_observed_by_action':outcomes(validation,np.ones(len(truth),bool))}
    dump(folder/'learned_demand.json',learned);np.save(folder/'learned_demand_policy.npy',learned_policy)
    dump(folder/'complete.json',{'mechanics':'passed','classification':result['classification'],'positive_control_complete':True})
    print('Mechanics verified. Learned-demand exact revenue:',revenue,flush=True)


def train_stage(trajectory,rule,updates,seeds,phase='development',alpha=.1,preprocessing='linear',dataset='original'):
    original=ROOT/PLAN['original_data']
    if dataset=='original':records=load_dataset(original,'train');dataset_sha=sha(original/'manifest.json')
    else:
        path=ROOT/'data/simulated/policy_improvement/probes'/dataset
        with np.load(path/'train.npz') as a:records=dict(a)
        dataset_sha=sha(path/'manifest.json')
    sources=json.loads((OUT/'protocol.json').read_text())['sources']
    results=[]
    for seed in seeds:
        spec={'trajectory_id':trajectory,'stage':phase,'dataset_id':dataset,'collection_seed':None if dataset=='original' else PLAN['probe_development_seed'],
            'seed':seed,'training_seed':seed,'source_commit':json.loads((OUT/'protocol.json').read_text())['source_commit'],
            'dataset_sha256':dataset_sha,'source_sha256':sources,'preprocessing':preprocessing,'alpha':alpha,'replay_rule':rule,
            'primary_reference_id':'E1_U20','primary_reference_revenue':None,'matched_reference_id':None,'matched_reference_revenue':None,
            'planned_endpoints':PLAN['endpoints']}
        folder=OUT/'models'/trajectory/f'seed_{seed}'
        print('Starting',trajectory,'seed',seed,'endpoint',updates,flush=True)
        model,target,runtime=fit_trajectory(records,CONFIG,spec,folder,updates)
        results.append(assess(model,target,records,CONFIG,spec,updates,runtime,OUT))
    return results


def collect_probes():
    folder=ROOT/'data/simulated/policy_improvement/probes'
    original=load_dataset(ROOT/PLAN['original_data'],'train')
    late=np.flatnonzero(original['observations'][:,0]<=30)
    count=30*10*5*PLAN['probe_outcomes_per_pair']
    for label in ['familiar','balanced']:
        dest=folder/label
        if (dest/'manifest.json').exists():continue
        dest.mkdir(parents=True,exist_ok=False)
        rng=np.random.default_rng(PLAN['probe_development_seed'])
        if label=='familiar':
            selected=rng.choice(late,size=count,replace=True)
            states=original['observations'][selected];actions=original['actions'][selected]
        else:
            pairs=np.array([(t,r,a) for t in range(1,31) for r in range(1,11) for a in range(5)])
            pairs=np.repeat(pairs,PLAN['probe_outcomes_per_pair'],axis=0)
            states=pairs[:,:2];actions=pairs[:,2]
        start=time.perf_counter();extra=probe_records(CONFIG,states,actions,PLAN['probe_development_seed']+1)
        # Only fields required for one-step training; probe IDs are distinct from episode IDs.
        combined={k:np.concatenate([original[k],extra[k]]) for k in extra}
        np.savez_compressed(dest/'train.npz',**combined);np.savez_compressed(dest/'probes.npz',probe_id=np.arange(count)+(0 if label=='familiar' else count),**extra)
        support=[]
        for source,x in [('original',original),('added',extra),('combined',combined)]:
            tab=pd.DataFrame({'days':x['observations'][:,0],'rooms':x['observations'][:,1],'action':x['actions'],'sale':x['rewards']>0})
            tab=tab[tab.days<=30].groupby(['days','rooms','action']).sale.agg(['size','sum']).reset_index()
            tab['no_sales']=tab['size']-tab['sum'];tab[['wilson_low','wilson_high']]=[wilson(s,n) for s,n in zip(tab['sum'],tab['size'])]
            tab.to_csv(dest/f'{source}_support.csv',index=False)
            support.append({'source':source,'distinct_pairs':len(tab),'rows':int(tab['size'].sum()),'sales':int(tab['sum'].sum())})
        dump(dest/'manifest.json',{'collector':label,'seed':PLAN['probe_development_seed'],'outcome_seed':PLAN['probe_development_seed']+1,
            'transitions':count,'outcomes_per_pair':PLAN['probe_outcomes_per_pair'] if label=='balanced' else None,'support':support,
            'kernel':'HotelPricingEnv.step; explicit reachable interior states','terminal_rule':'original environment only, not probe cutoff',
            'runtime_seconds':time.perf_counter()-start,'files':{p.name:sha(p) for p in dest.iterdir() if p.is_file()}})
        print('Frozen probes:',label,count,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['diagnose','replay20','replay100','probes','probe_train','method','confirmation'])
    parser.add_argument('--alpha',type=float,default=.1)
    parser.add_argument('--preprocessing',choices=['linear','log_time'],default='linear')
    parser.add_argument('--dataset',choices=['original','familiar','balanced'],default='original')
    parser.add_argument('--updates',type=int,choices=[20000,100000],default=100000)
    parser.add_argument('--replay',choices=['uniform','late_32_224'],default='late_32_224')
    args=parser.parse_args();protocol=initialize()
    if args.stage=='diagnose':stage_diagnostics()
    elif args.stage in ['replay20','replay100']:
        assert (OUT/'diagnostics/complete.json').exists()
        updates=20000 if args.stage=='replay20' else 100000
        for trajectory,rule in [('E1_U','uniform'),('E1_R','late_32_224')]:train_stage(trajectory,rule,updates,PLAN['development_seeds'])
    elif args.stage=='probes':collect_probes()
    elif args.stage=='probe_train':
        for label in ['familiar','balanced']:train_stage('E2_'+label,'late_32_224',args.updates,PLAN['development_seeds'],dataset=label)
    elif args.stage=='method':
        assert (args.alpha!=.1) != (args.preprocessing!='linear'),'Change exactly one method factor'
        train_stage('E3_method',args.replay,args.updates,PLAN['development_seeds'],alpha=args.alpha,preprocessing=args.preprocessing,dataset=args.dataset)
    elif args.stage=='confirmation':
        from scripts.policy_improvement.confirmation import main as confirm
        confirm()
    protect(protocol)


if __name__=='__main__':main()
