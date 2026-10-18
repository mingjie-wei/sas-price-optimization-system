"""Reproducible policy audit and frozen three-arm coverage experiment."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from scripts.study_layout import training_source_digest

import numpy as np
import pandas as pd
import torch

from .cql import TrainingConfig, train, diagnostics, tensors
from .diagnostics import optimal_state_visits, paired_difference
from .environment import HotelPricingEnv, load_config
from .evaluation import solve, rollouts
from .offline_data import load_dataset
from .policy_gap import explain_gap
from .xai_audit import (reachable, load_model, model_values, simulator_values,
                        sample_reference, shapley, counterfactual)

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def check_protected(protected):
    for name, checksum in protected.items():
        if digest(ROOT/name) != checksum:
            raise ValueError(f'Protected baseline changed: {name}')


def freeze(plan, config):
    out = ROOT/plan['output']
    protected = json.loads((ROOT/'docs/provenance/baseline.json').read_text())['artifact_sha256']
    protected = {**protected, plan['environment']:digest(ROOT/plan['environment'])}
    check_protected(protected)
    source = {str(p.relative_to(ROOT)):training_source_digest(p) for p in sorted((ROOT/'hotel_pricing').glob('*.py'))}
    if (out/'protocol.json').exists():
        saved = json.loads((out/'protocol.json').read_text())
        if saved['plan'] != plan or saved['source_sha256'] != source:
            raise ValueError('Protocol/source changed; use a new run identity')
        check_protected(saved['protected_sha256'])
        return saved
    out.mkdir(parents=True, exist_ok=False)
    original = {s:load_dataset(ROOT/plan['original_data'],s) for s in ['train','validation','test']}
    used = set()
    # Inspect historical dataset/rollout seed arrays, including comparison runs.
    for folder in ['data/simulated/baseline','outputs/baseline']:
        for path in (ROOT/folder).rglob('*.npz'):
            with np.load(path, allow_pickle=False) as saved:
                for key in ['environment_seed','environment_seeds','action_seed']:
                    if key in saved: used.update(map(int, saved[key]))
    collection = set(range(plan['collection_environment_seed'],plan['collection_environment_seed']+plan['additional_episodes']))
    collection_actions = set(range(plan['collection_action_seed'],plan['collection_action_seed']+plan['additional_episodes']))
    evaluation = set(range(plan['evaluation_environment_seed'],plan['evaluation_environment_seed']+plan['evaluation_episodes']))
    evaluation_actions = set(range(plan['evaluation_action_seed'],plan['evaluation_action_seed']+plan['evaluation_episodes']))
    namespaces = [collection,collection_actions,evaluation,evaluation_actions,set(plan['reference_seeds']),set(plan['training_seeds'])]
    for i, group in enumerate(namespaces):
        if i < 4: assert not used.intersection(group), 'Historical seed overlap'
        for other in namespaces[:i]: assert not group.intersection(other), 'New seed namespace overlap'
    saved = {'plan':plan,'environment':asdict(config),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'source_sha256':source,'protected_sha256':protected,
             'versions':{p:importlib.metadata.version(p) for p in ['numpy','torch','gymnasium','pandas']},
             'seed_overlap_check':'passed; historical NPZ seed arrays checked',
             'pairing':'One sale RNG draw per active day in unchanged HotelPricingEnv; action RNG separate. Same collection sale/action streams for B and C. Evaluation sale draws align by days elapsed; stopped episodes need no further draws.',
             'units':{'q_model':'reward divided by reference price','q_star':'raw revenue','q_policy':'raw revenue'},
             'attribution':'64 original-training transition states, sampled without replacement within t <= horizon-capacity+1; duplicates of state coordinates are permitted.',
             'counterfactual_distance':'abs(delta_days)/horizon + abs(delta_rooms)/capacity; ties: absolute days, absolute rooms, time, rooms',
             'case_rule':'Four motivating states; highest contribution among states with oracle-action support >=20; lowest local loss control, ties by descending fixed-90 visits then state coordinates.'}
    write_json(out/'protocol.json',saved)
    inventory = []
    for role, name in [('environment',plan['environment']),('dataset manifest',plan['original_data']+'/manifest.json'),
                       ('source metadata','data/source.json'),('raw bookings','data/raw/h2.csv'),
                       ('original results',plan['original_models']+'/results.json'),
                       ('original policies',plan['original_models']+'/policies.npz'),
                       ('comparison','outputs/baseline/comparison/comparison.json'),
                       *[('checkpoint',plan['original_models']+f'/cql_seed_{s}.pt') for s in plan['training_seeds']],
                       *[('implementation','hotel_pricing/'+n) for n in ['environment.py','offline_data.py','cql.py','policy_gap.py']]]:
        p=ROOT/name; inventory.append({'role':role,'path':name,'exists':p.exists(),'sha256':digest(p) if p.exists() else None})
    inventory.append({'role':'rubric','path':'/Users/daisy/Downloads/Project 1.pdf','exists':Path('/Users/daisy/Downloads/Project 1.pdf').exists()})
    write_json(out/'inventory.json',inventory)
    return saved


def audit_model(model, config, records, reference, sensitivity, cases, plan, name, directory):
    q = model_values(model,config)
    policy = q.argmax(axis=2)
    gap = explain_gap(config,policy,records)
    mask = reachable(config)
    states = np.argwhere(mask)
    t,r=states.T
    q_star,_ = simulator_values(config)
    q_policy,_ = simulator_values(config,policy)
    fixed = np.full(mask.shape,config.prices.index(90))
    weights=optimal_state_visits(config,fixed)
    rect = states[states[:,0] <= config.horizon-config.capacity+1]
    frame=pd.DataFrame({'days':t,'rooms':r,'price':np.array(config.prices)[policy[t,r]],
                        'optimal_price':np.array(config.prices)[gap['optimal_actions'][t,r]],
                        'local_regret':gap['local_loss'][t,r],'visits':gap['visits'][t,r],
                        'gap_contribution':gap['contribution'][t,r],
                        'common_weight':weights[t,r]})
    for a,price in enumerate(config.prices):
        frame[f'q_model_{price}']=q[t,r,a]
        frame[f'support_{price}']=gap['counts'][t,r,a]
    recon=[]
    attrs={}
    for price,base_price in plan['contrasts']:
        a,b=config.prices.index(price),config.prices.index(base_price)
        margin=q[:,:,a]-q[:,:,b]
        base,pt,pr,error=shapley(margin,rect,reference,config)
        recon.append(error)
        key=str(price)
        attrs[key]=(base,pt,pr)
        frame[f'margin_{price}']=margin[t,r]
        frame[f'phi_days_{price}']=np.nan
        frame[f'phi_rooms_{price}']=np.nan
        selection=frame.days <= config.horizon-config.capacity+1
        frame.loc[selection,f'phi_days_{price}']=pt
        frame.loc[selection,f'phi_rooms_{price}']=pr
    assert max(recon) <= plan['tolerances']['shapley']
    frame.to_csv(directory/f'{name}_states.csv',index=False)
    examples=[]
    for state in cases:
        t,r=state
        item={'state':state,'price':float(config.prices[policy[t,r]]),
              'q_model':q[t,r].tolist(),'q_star':q_star[t,r].tolist(),'q_policy':q_policy[t,r].tolist(),
              'support':gap['counts'][t,r].tolist(),'counterfactual':counterfactual(policy,state,config),
              'local_regret':float(gap['local_loss'][t,r]),'attributions':{}}
        for price,base_price in plan['contrasts']:
            margin=q[:,:,config.prices.index(price)]-q[:,:,config.prices.index(base_price)]
            first=shapley(margin,[state],reference,config)
            second=shapley(margin,[state],sensitivity,config)
            item['attributions'][str(price)]={'margin':float(margin[t,r]),'base':first[0],
                'days':float(first[1][0]),'rooms':float(first[2][0]),
                'sensitivity_base':second[0],'sensitivity_days':float(second[1][0]),'sensitivity_rooms':float(second[2][0])}
        item['explanation']=f'At {t} days and {r} rooms, the model selects {item["price"]:g}. '+(
            'No reachable operating condition changes its selected action.' if item['counterfactual'] is None else
            f'The nearest different decision is {item["counterfactual"]["price"]:g} at {item["counterfactual"]["state"]}.')
        examples.append(item)
    write_json(directory/f'{name}_cases.json',examples)
    target=(np.indices(mask.shape)[0]<=plan['target_days'])&mask
    scarcity=(np.indices(mask.shape)[1]<=2)&mask
    oracle_support=np.take_along_axis(gap['counts'],gap['optimal_actions'][...,None],axis=2)[...,0]
    revenue=solve(config,policy)[0]
    return {'model':name,'exact_revenue':revenue,'gap':gap['gap'],
            'gap_percent':100*gap['gap']/solve(config)[0], 'minus_fixed90':revenue-solve(config,fixed)[0],
            'target_gap':float(gap['contribution'][target].sum()),
            'scarcity_gap':float(gap['contribution'][scarcity].sum()),
            'support_at_most4_gap':float(gap['contribution'][mask&(oracle_support<=4)].sum()),
            'common_target_regret':float((weights*gap['local_loss'])[target].sum()),
            'common_total_regret':float((weights*gap['local_loss']).sum()),
            'max_shapley_reconstruction_error':max(recon),
            'prices_on_reachable_states':{str(price):int(np.sum(policy[mask]==a)) for a,price in enumerate(config.prices)},
            'case_prices':[x['price'] for x in examples]},policy


def audit(plan,config,out):
    if (out/'audit/complete.json').exists(): return
    folder=out/'audit';folder.mkdir(exist_ok=True)
    records=load_dataset(ROOT/plan['original_data'],'train')
    models=[load_model(ROOT/plan['original_models']/f'cql_seed_{s}.pt',config) for s in plan['training_seeds']]
    for (_,checkpoint),seed in zip(models,plan['training_seeds']):
        assert checkpoint['seed']==seed
        assert checkpoint['settings']==plan['training']
        assert json.loads(json.dumps(checkpoint['config']))==json.loads(json.dumps(asdict(config)))
    refs=[sample_reference(records,config,plan['reference_size'],seed) for seed in plan['reference_seeds']]
    first_policy=model_values(models[0][0],config).argmax(axis=2)
    gap=explain_gap(config,first_policy,records)
    states=np.argwhere(reachable(config))
    states=states[states[:,0]<=config.horizon-config.capacity+1]
    supported=[tuple(s) for s in states if gap['counts'][*s,gap['optimal_actions'][tuple(s)]]>=20]
    high=min(supported,key=lambda s:(-gap['contribution'][s],s))
    weights=optimal_state_visits(config,np.full(first_policy.shape,config.prices.index(90)))
    low=min(map(tuple,states),key=lambda s:(gap['local_loss'][s],-weights[s],s))
    cases=[list(s) for s in plan['initial_cases']]+[list(map(int,high)),list(map(int,low))]
    write_json(out/'references.json',{'primary':refs[0].tolist(),'sensitivity':refs[1].tolist(),'cases':cases})
    historical=json.loads((ROOT/plan['original_models']/'results.json').read_text())
    metrics=[]
    with np.load(ROOT/plan['original_models']/'policies.npz') as saved:
        stored={v['policy']:v['exact_revenue'] for v in historical['results']}
        for name,policy in saved.items():
            assert abs(solve(config,policy)[0]-stored[name])<=plan['tolerances']['revenue']
        for seed,(model,_) in zip(plan['training_seeds'],models):
            result,policy=audit_model(model,config,records,*refs,cases,plan,f'historical_{seed}',folder)
            np.testing.assert_array_equal(policy,saved[f'CQL seed {seed}'])
            metrics.append(result)
    write_json(folder/'metrics.json',metrics)
    write_json(folder/'complete.json',{'reachable_states':int(reachable(config).sum()),'attribution_states':len(states),
        'excluded_attribution_states':int(reachable(config).sum()-len(states)),
        'checkpoint_agreement':True,'stored_revenues_reproduced':True,
        'historical_source_match':{n:historical['code_sha256'].get(n)==digest(ROOT/'hotel_pricing'/n) for n in ['cql.py','evaluation.py']},
        'one_day_expected_revenue':{str(p):float(simulator_values(config)[0][1,1,i]) for i,p in enumerate(config.prices)}})
    print('Historical audit verified; representative cases and references frozen.',flush=True)


def collection_probabilities(state,plan,targeted):
    return np.asarray(plan['target_probabilities'] if targeted and 1<=state[0]<=plan['target_days'] and state[1]>=1
                      else plan['ordinary_probabilities'],float)


def collect_extra(config,plan,targeted,episode_offset):
    env=HotelPricingEnv(config)
    rows={k:[] for k in ['observations','actions','rewards','next_observations','terminated','truncated','episode_id','step','action_probability','environment_seed','action_seed']}
    for i in range(plan['additional_episodes']):
        env_seed=plan['collection_environment_seed']+i
        action_seed=plan['collection_action_seed']+i
        rng=np.random.default_rng(action_seed)
        state,_=env.reset(seed=env_seed)
        done=False;step=0
        while not done:
            probabilities=collection_probabilities(state,plan,targeted)
            action=int(rng.choice(len(config.prices),p=probabilities))
            after,reward,done,truncated,_=env.step(action)
            assert not truncated
            assert after[0]==state[0]-1 and state[1]-after[1] in [0,1]
            assert reward==config.prices[action]*(state[1]-after[1])
            assert done==(after[0]==0 or after[1]==0)
            values=[state,action,reward,after,done,truncated,episode_offset+i,step,probabilities[action],env_seed,action_seed]
            for k,v in zip(rows,values):rows[k].append(v)
            state=after;step+=1
    env.close()
    return {k:np.asarray(v) for k,v in rows.items()}


def dataset_summary(records,config,plan):
    target=(records['observations'][:,0]<=plan['target_days'])&(records['observations'][:,1]>0)
    counts=np.zeros((config.horizon+1,config.capacity+1,len(config.prices)),int)
    t,r=records['observations'].T
    np.add.at(counts,(t,r,records['actions']),1)
    return {'episodes':len(np.unique(records['episode_id'])),'transitions':len(t),
            'sales':int(np.sum(records['rewards']>0)),'target_transitions':int(target.sum()),
            'target_episodes':len(np.unique(records['episode_id'][target])),
            'action_counts':np.bincount(records['actions'],minlength=len(config.prices)).tolist(),
            'target_action_counts':np.bincount(records['actions'][target],minlength=len(config.prices)).tolist(),
            'target_state_action_pairs_observed':int((counts[1:plan['target_days']+1,1:]>0).sum())}


def collect(plan,config,out):
    marker=out/'datasets.json'
    if marker.exists():
        for arm,info in json.loads(marker.read_text()).items():
            if arm=='original':continue
            for name,sha in info['checksums'].items():assert digest(ROOT/plan['data_output']/arm/name)==sha
        return
    records={s:load_dataset(ROOT/plan['original_data'],s) for s in ['train','validation','test']}
    offset=1+max(int(v['episode_id'].max()) for v in records.values())
    summaries={'original':dataset_summary(records['train'],config,plan)}
    for i,arm in enumerate(['ordinary','targeted']):
        folder=ROOT/plan['data_output']/arm;folder.mkdir(parents=True,exist_ok=False)
        extra=collect_extra(config,plan,arm=='targeted',offset+i*plan['additional_episodes'])
        assert len(np.unique(extra['episode_id']))==plan['additional_episodes']
        assert np.sum(extra['terminated'])==plan['additional_episodes']
        for split in records:assert not np.intersect1d(extra['episode_id'],records[split]['episode_id']).size
        combined={k:np.concatenate([records['train'][k],extra[k]]) for k in extra}
        np.savez_compressed(folder/'extra.npz',**extra)
        np.savez_compressed(folder/'train.npz',**combined)
        for split in ['validation','test']:shutil.copyfile(ROOT/plan['original_data']/f'{split}.npz',folder/f'{split}.npz')
        checksums={p.name:digest(p) for p in folder.glob('*.npz')}
        manifest={'config':asdict(config),'environment_sha256':digest(ROOT/plan['environment']),
            'collector':'uniform in final target days' if arm=='targeted' else 'original fixed mixture',
            'plan':plan,'summary':dataset_summary(extra,config,plan),'combined_summary':dataset_summary(combined,config,plan),
            'episode_id_range':[int(extra['episode_id'].min()),int(extra['episode_id'].max())],
            'checksums':checksums,'files':{s:{'sha256':checksums[f'{s}.npz']} for s in ['train','validation','test']}}
        write_json(folder/'manifest.json',manifest)
        summaries[arm]={'extra':manifest['summary'],'combined':manifest['combined_summary'],'checksums':{**checksums,'manifest.json':digest(folder/'manifest.json')}}
    write_json(marker,summaries)
    print('Both 400-episode datasets frozen before training.',flush=True)


def train_models(plan,config,out):
    collect(plan,config,out)
    learning=SimpleNamespace(horizon=config.horizon,capacity=config.capacity,reference_price=config.reference_price,prices=config.prices)
    validation=load_dataset(ROOT/plan['original_data'],'validation')
    # A short pipeline check does not enter reported model comparisons.
    if not (out/'smoke.json').exists():
        smoke_records=load_dataset(ROOT/plan['data_output']/'targeted','train')
        smoke,_=train(smoke_records,validation,learning,TrainingConfig(updates=5,log_every=5),seed=99)
        assert np.isfinite(model_values(smoke,config)).all()
        write_json(out/'smoke.json',{'updates':5,'status':'passed','used_for_results':False})
    for arm in ['original','ordinary','targeted']:
        data=ROOT/(plan['original_data'] if arm=='original' else plan['data_output']+'/'+arm)
        records=load_dataset(data,'train')
        folder=out/'models'/arm;folder.mkdir(parents=True,exist_ok=True)
        for seed in plan['training_seeds']:
            path=folder/f'cql_seed_{seed}.pt'
            if path.exists():
                _,checkpoint=load_model(path,config)
                assert checkpoint['settings']==plan['training'] and checkpoint['dataset_manifest_sha256']==digest(data/'manifest.json')
                continue
            print(f'Training {arm}, seed {seed}: {plan["training"]["updates"]} updates',flush=True)
            model,history=train(records,validation,learning,TrainingConfig(**plan['training']),seed)
            checkpoint={'state_dict':model.state_dict(),'config':asdict(config),'settings':plan['training'],'seed':seed,
                        'dataset_manifest_sha256':digest(data/'manifest.json')}
            write_json(folder/f'history_{seed}.json',history)
            temporary=path.with_suffix('.tmp');torch.save(checkpoint,temporary);temporary.rename(path)
    print('All nine final checkpoints saved.',flush=True)


def evaluate(plan,config,out):
    folder=out/'evaluation'
    if (folder/'complete.json').exists():return
    folder.mkdir(exist_ok=True)
    frozen=json.loads((out/'references.json').read_text())
    references=[np.array(frozen[k]) for k in ['primary','sensitivity']]
    cases=frozen['cases']
    seeds=np.arange(plan['evaluation_environment_seed'],plan['evaluation_environment_seed']+plan['evaluation_episodes'])
    metrics=[];episodes={};checkpoint_hashes={}
    for arm in ['original','ordinary','targeted']:
        for seed in plan['training_seeds']:
            p=out/'models'/arm/f'cql_seed_{seed}.pt'
            if not p.exists():raise FileNotFoundError('All checkpoints must be frozen before evaluation')
            checkpoint_hashes[str(p.relative_to(ROOT))]=digest(p)
    write_json(folder/'checkpoint_hashes.json',checkpoint_hashes)
    for arm in ['original','ordinary','targeted']:
        data=ROOT/(plan['original_data'] if arm=='original' else plan['data_output']+'/'+arm)
        records=load_dataset(data,'train')
        for seed in plan['training_seeds']:
            name=f'{arm}_{seed}'
            model,_=load_model(out/'models'/arm/f'cql_seed_{seed}.pt',config)
            result,policy=audit_model(model,config,records,*references,cases,plan,name,folder)
            revenue,occupancy=rollouts(config,policy,seeds,action_seed=plan['evaluation_action_seed'])
            episodes[name]=revenue
            result.update(arm=arm,seed=seed,simulation_mean=float(revenue.mean()),simulation_sd=float(revenue.std(ddof=1)),mean_occupancy=float(occupancy.mean()),
                          test_diagnostics=diagnostics(model,tensors(load_dataset(data,'test'),config)))
            metrics.append(result)
            print(f'{name}: exact revenue {result["exact_revenue"]:.6f}',flush=True)
    paired=[]
    for seed in plan['training_seeds']:
        for a,b in [('targeted','ordinary'),('ordinary','original'),('targeted','original')]:
            first=next(m for m in metrics if m['arm']==a and m['seed']==seed)
            second=next(m for m in metrics if m['arm']==b and m['seed']==seed)
            paired.append({'comparison':f'{a}_minus_{b}','seed':seed,'exact_difference':first['exact_revenue']-second['exact_revenue'],
                           **paired_difference(episodes[f'{a}_{seed}'],episodes[f'{b}_{seed}'])})
    write_json(folder/'metrics.json',metrics);write_json(folder/'paired.json',paired)
    pd.DataFrame(metrics).drop(columns=['test_diagnostics','prices_on_reachable_states','case_prices']).to_csv(folder/'metrics.csv',index=False)
    pd.DataFrame(paired).to_csv(folder/'paired.csv',index=False)
    np.savez_compressed(folder/'episodes.npz',environment_seeds=seeds,**episodes)
    write_json(folder/'complete.json',{'models':len(metrics),'evaluation_episodes':len(seeds),'checkpoint_sha256':checkpoint_hashes})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/policy_improvement/coverage.json')
    parser.add_argument('--stage',choices=['audit','collect','train','evaluate','all'],default='all')
    args=parser.parse_args()
    plan=json.loads((ROOT/args.config).read_text());config=load_config(ROOT/plan['environment'])
    protocol=freeze(plan,config);out=ROOT/plan['output']
    audit(plan,config,out)
    if args.stage in ['collect','train','evaluate','all']:collect(plan,config,out)
    if args.stage in ['train','all']:train_models(plan,config,out)
    if args.stage in ['evaluate','all']:evaluate(plan,config,out)
    check_protected(protocol['protected_sha256'])


if __name__=='__main__':main()
