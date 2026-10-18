"""Follow-up helpers: actual CQL objective, explicit replay, full resumable state."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from hotel_pricing.cql import make_network,tensors,losses,bellman_target,TrainingConfig,train as original_train
from hotel_pricing.environment import HotelPricingEnv,load_config
from hotel_pricing.evaluation import solve
from hotel_pricing.offline_data import load_dataset
from hotel_pricing.diagnostics import optimal_state_visits
from hotel_pricing.policy_gap import explain_gap
from hotel_pricing.xai_audit import reachable,model_values


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def transform(records,config,preprocessing='linear'):
    data=tensors(records,config)
    if preprocessing=='log_time':
        for key,source in [('states','observations'),('next_states','next_observations')]:
            data[key][:,0]=torch.tensor(np.log1p(records[source][:,0])/np.log1p(config.horizon),dtype=torch.float32)
    elif preprocessing!='linear':raise ValueError(preprocessing)
    return data


def q_grid(model,config,preprocessing='linear'):
    if preprocessing=='linear':return model_values(model,config)
    t,r=np.indices((config.horizon+1,config.capacity+1))
    x=np.stack([np.log1p(t.ravel())/np.log1p(config.horizon),r.ravel()/config.capacity],axis=1)
    with torch.no_grad():return model(torch.tensor(x,dtype=torch.float32)).numpy().reshape((*t.shape,5)).astype(float)


def draw_indices(data,late,early,rule):
    if rule=='uniform':return torch.randint(len(data['actions']),(256,))
    if rule!='late_32_224':raise ValueError(rule)
    if not len(late) or not len(early):raise ValueError('Replay stratum absent')
    idx=torch.cat([late[torch.randint(len(late),(32,))],early[torch.randint(len(early),(224,))]])
    return idx[torch.randperm(256)]


def residual_groups(model,target,records,config,alpha=.1,preprocessing='linear'):
    data=transform(records,config,preprocessing)
    rows=[]
    with torch.no_grad():
        for start in range(0,len(records['actions']),4096):
            end=start+4096;batch={k:v[start:end] for k,v in data.items()}
            q=model(batch['states']);nextq=target(batch['next_states'])
            target_value=bellman_target(batch['rewards'],batch['terminated'],nextq)
            pred=q.gather(1,batch['actions'][:,None]).squeeze(1)
            cql=torch.logsumexp(q,1)-pred
            obs=records['observations'][start:end];actions=records['actions'][start:end]
            bound=np.where(obs[:,0]==1,np.array(config.prices)[actions]/config.reference_price,
                           max(config.prices)*np.minimum(obs[:,0],obs[:,1])/config.reference_price)
            rows.append(pd.DataFrame({'days':obs[:,0],'rooms':obs[:,1],'action':actions,
                'sale':records['rewards'][start:end]>0,'predicted_q':pred.numpy(),
                'reward_scaled':batch['rewards'].numpy(),'bootstrap':(target_value-batch['rewards']).numpy(),
                'td_target':target_value.numpy(),'signed_residual':(pred-target_value).numpy(),
                'squared_residual':((pred-target_value)**2).numpy(),'weighted_cql':alpha*cql.numpy(),
                'above_physical_bound':pred.numpy()>bound}))
    frame=pd.concat(rows,ignore_index=True)
    frame['group']=pd.cut(frame.days,bins=[0,1,7,30,config.horizon],labels=['t1','t2_7','t8_30','t_gt30'])
    summary=[]
    for (group,action),f in frame.groupby(['group','action'],observed=True):
        row={'group':str(group),'action':int(action),'price':config.prices[action],'records':len(f),
             'distinct_pairs':len(f[['days','rooms','action']].drop_duplicates()),'sales':int(f.sale.sum()),'no_sales':int((~f.sale).sum())}
        for key in ['predicted_q','reward_scaled','bootstrap','td_target','signed_residual','squared_residual','weighted_cql']:row[key]=float(f[key].mean())
        row['td_loss']=.5*row['squared_residual'];row['bound_violations']=int(f.above_physical_bound.sum())
        summary.append(row)
    return summary


def gradient_norm(loss,model):
    gs=torch.autograd.grad(loss,model.parameters(),retain_graph=True,allow_unused=True)
    return float(torch.sqrt(sum(g.square().sum() for g in gs if g is not None)))


def fit_trajectory(records,config,spec,folder,stop_at):
    """Original objective/architecture; only specified sampling or input/alpha differs."""
    folder.mkdir(parents=True,exist_ok=True)
    spec_path=folder/'spec.json'
    if spec_path.exists():assert json.loads(spec_path.read_text())==spec
    else:dump(spec_path,spec)
    torch.set_num_threads(1);torch.manual_seed(spec['seed']);torch.use_deterministic_algorithms(True)
    model=make_network(5);target=deepcopy(model).requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
    data=transform(records,config,spec['preprocessing'])
    late=torch.tensor(np.flatnonzero(records['observations'][:,0]<=30))
    early=torch.tensor(np.flatnonzero(records['observations'][:,0]>30))
    counts=np.zeros(len(records['actions']),dtype=np.int64)
    history=[];start=0;elapsed=0.
    endpoints=[int(p.stem.split('_')[1]) for p in folder.glob('checkpoint_*.pt')]
    if endpoints:
        previous=max(e for e in endpoints if e<=stop_at)
        c=torch.load(folder/f'checkpoint_{previous}.pt',weights_only=True)
        model.load_state_dict(c['online']);target.load_state_dict(c['target']);optimizer.load_state_dict(c['optimizer'])
        torch.set_rng_state(c['torch_rng']);counts=c['replay_counts'].numpy().copy()
        start=c['updates'];elapsed=c['runtime_seconds'];history=c['history']
        if start==stop_at:return model.eval(),target,elapsed
    tick=time.perf_counter()
    for update in range(start+1,stop_at+1):
        idx=draw_indices(data,late,early,spec['replay_rule'])
        np.add.at(counts,idx.numpy(),1)
        batch={k:v[idx] for k,v in data.items()}
        td,cql=losses(model,target,batch)
        grad={}
        if update in [1,1000,20000,50000,100000]:
            grad={'td_gradient_norm':gradient_norm(td,model),'weighted_cql_gradient_norm':gradient_norm(spec['alpha']*cql,model)}
        objective=td+spec['alpha']*cql
        if not torch.isfinite(objective):raise ValueError('Nonfinite loss')
        optimizer.zero_grad();objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
        if update%100==0:target.load_state_dict(model.state_dict())
        if update==1 or update%1000==0:
            # Separate errors on the two portions of this same sampled batch.
            late_mask=records['observations'][idx.numpy(),0]<=30
            with torch.no_grad():
                preds=model(batch['states']).gather(1,batch['actions'][:,None]).squeeze(1)
                targets=bellman_target(batch['rewards'],batch['terminated'],target(batch['next_states']))
                errors=(preds-targets).square().numpy()
            history.append({'updates':update,'td_loss':float(td),'weighted_cql_loss':float(spec['alpha']*cql),
                'late_batch_mse':float(errors[late_mask].mean()) if late_mask.any() else None,
                'earlier_batch_mse':float(errors[~late_mask].mean()) if (~late_mask).any() else None,**grad})
        if update in [20000,50000,100000] or update==stop_at:
            runtime=elapsed+time.perf_counter()-tick
            checkpoint={'online':model.state_dict(),'target':target.state_dict(),'optimizer':optimizer.state_dict(),
                'torch_rng':torch.get_rng_state(),'updates':update,'replay_counts':torch.tensor(counts),
                'history':history,'runtime_seconds':runtime,'spec':spec}
            torch.save(checkpoint,folder/f'checkpoint_{update}.pt')
            dump(folder/'history.json',history)
            late_np=records['observations'][:,0]<=30;sales=records['rewards']>0
            dump(folder/f'replay_{update}.json',{'draws':int(counts.sum()),'unique_records':int((counts>0).sum()),
                'late_draws':int(counts[late_np].sum()),'earlier_draws':int(counts[~late_np].sum()),
                'sale_draws':int(counts[sales].sum()),'no_sale_draws':int(counts[~sales].sum()),
                'unique_late':int((counts[late_np]>0).sum()),'unique_earlier':int((counts[~late_np]>0).sum())})
    return model.eval(),target,elapsed+time.perf_counter()-tick


def assess(model,target,records,config,spec,updates,runtime,out):
    q=q_grid(model,config,spec['preprocessing']);policy=q.argmax(2)
    gap=explain_gap(config,policy,records)
    mask=reachable(config);t,r=np.indices(mask.shape);late=mask&(t<=30)
    weights=optimal_state_visits(config,np.full(mask.shape,2))
    revenue=solve(config,policy)[0];oracle=solve(config)[0];fixed=solve(config,np.full(mask.shape,2))[0]
    refs=json.loads((ROOT/'outputs/policy_improvement/coverage/references.json').read_text())
    summaries=residual_groups(model,target,records,config,spec['alpha'],spec['preprocessing'])
    terminal=q[1,1]
    result={**spec,'experiment_id':spec['trajectory_id']+str(updates//1000),'updates':updates,'exact_revenue':revenue,
      'fixed90_revenue':fixed,'minus_fixed90':revenue-fixed,'oracle_revenue':oracle,'oracle_gap':oracle-revenue,
      'oracle_gap_percent':100*(oracle-revenue)/oracle,'actual_total_regret':gap['contribution'].sum().item(),
      'late_common_regret':(weights*gap['local_loss'])[late].sum().item(),
      'earlier_common_regret':(weights*gap['local_loss'])[mask&~late].sum().item(),
      'terminal_q':terminal.tolist(),'terminal_bound_violations':int((terminal>np.array(config.prices)/90).sum()),
      'terminal_error_summary':summaries[:5],'case_prices':[int(config.prices[policy[tuple(s)]]) for s in refs['cases']],
      'runtime_seconds':runtime,'status':'complete','checkpoint_rule':'fixed endpoint','reward_scale':90,'gamma':1,
      'price_counts':{str(p):int((policy[mask]==i).sum()) for i,p in enumerate(config.prices)}}
    folder=out/'evaluations';folder.mkdir(exist_ok=True)
    key=f'{spec["trajectory_id"]}_{spec["seed"]}_{updates}'
    dump(folder/f'{key}.json',result)
    pd.DataFrame(summaries).to_csv(folder/f'{key}_residuals.csv',index=False)
    np.savez_compressed(folder/f'{key}_states.npz',q_model=q,policy=policy,local_regret=gap['local_loss'],visits=gap['visits'])
    print(f'{key}: revenue={revenue:.6f}; Q(1,1,90)={terminal[2]:.5f}; seconds={runtime:.1f}',flush=True)
    return result


def probe_records(config,states,actions,seed):
    """Explicit one-step interior probes through unchanged HotelPricingEnv.step."""
    env=HotelPricingEnv(config);env.reset(seed=seed)
    records={k:[] for k in ['observations','actions','rewards','next_observations','terminated','truncated']}
    mask=reachable(config)
    for state,action in zip(states,actions):
        t,r=map(int,state)
        if not mask[t,r]:raise ValueError('Unreachable probe state')
        env.time_left=t;env.rooms_left=r;env.ended=False
        after,reward,done,truncated,_=env.step(int(action))
        for k,v in zip(records,[np.array([t,r],np.int32),int(action),reward,after,done,truncated]):records[k].append(v)
    env.close()
    return {k:np.asarray(v) for k,v in records.items()}


def wilson(successes,trials):
    p=successes/trials;z=1.96;denom=1+z*z/trials
    mid=(p+z*z/(2*trials))/denom;half=z*np.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/denom
    return float(mid-half),float(mid+half)


def outcomes(records,mask):
    rows=[]
    for action in range(5):
        rewards=records['rewards'][mask&(records['actions']==action)]
        n=len(rewards);sales=int((rewards>0).sum());ci=wilson(sales,n) if n else (None,None)
        rows.append({'action':action,'trials':n,'sales':sales,'no_sales':n-sales,
                     'mean_scaled_reward':float(rewards.mean()/90) if n else None,
                     'sale_probability':sales/n if n else None,'wilson_low':ci[0],'wilson_high':ci[1]})
    return rows


def tiny_fit(records,alpha):
    counts=np.bincount(records['actions'],minlength=5)
    if np.any(counts==0):return {'status':'missing_action_support'}
    frequency=torch.tensor(counts/counts.sum(),dtype=torch.float64)
    means=torch.tensor([records['rewards'][records['actions']==a].mean()/90 for a in range(5)],dtype=torch.float64)
    q=torch.nn.Parameter(means.clone())
    optimizer=torch.optim.LBFGS([q],lr=1,max_iter=500,tolerance_grad=1e-12,tolerance_change=1e-14,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad();loss=.5*(frequency*(q-means).square()).sum()+alpha*(torch.logsumexp(q,0)-(frequency*q).sum());loss.backward();return loss
    optimizer.step(closure)
    grad=frequency*(q-means)+alpha*(torch.softmax(q,0)-frequency)
    return {'status':'complete','alpha':alpha,'q':q.detach().tolist(),'empirical_means':means.tolist(),
            'selected_action':int(q.argmax()),'maximum_stationarity_residual':float(grad.abs().max().detach())}


def fit_empirical_network(records,alpha):
    """Exact constant-state empirical squared-error objective; no oracle targets."""
    config=load_config();counts=np.bincount(records['actions'],minlength=5)
    weights=torch.tensor(counts/counts.sum(),dtype=torch.float64)
    means=torch.tensor([records['rewards'][records['actions']==a].mean()/90 for a in range(5)],dtype=torch.float64)
    torch.manual_seed(53000000);model=make_network(5).double()
    x=torch.tensor([[1/config.horizon,1/config.capacity]],dtype=torch.float64)
    optimizer=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=500,tolerance_grad=1e-12,tolerance_change=1e-14,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad();q=model(x)[0];loss=.5*(weights*(q-means).square()).sum()+alpha*(torch.logsumexp(q,0)-(weights*q).sum());loss.backward();return loss
    optimizer.step(closure)
    return model(x)[0].detach().tolist()


def plan_from_empirical_probabilities(config,probabilities):
    """Fitted price-only model: does not read generator probabilities or oracle Q."""
    p=np.asarray(probabilities);prices=np.array(config.prices)
    values=np.zeros((config.horizon+1,config.capacity+1));policy=np.zeros_like(values,dtype=int)
    for t in range(1,config.horizon+1):
        for r in range(1,config.capacity+1):
            q=p*(prices+values[t-1,r-1])+(1-p)*values[t-1,r]
            policy[t,r]=q.argmax();values[t,r]=q.max()
    return policy
