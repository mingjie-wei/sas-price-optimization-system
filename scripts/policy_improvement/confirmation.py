"""Confirm a frozen package on unused seeds, with fresh probe data when needed."""
from scripts.policy_improvement.experiments import *


def fresh_data(candidate):
    label=candidate['dataset']
    if label=='original':return load_dataset(ROOT/PLAN['original_data'],'train'),None
    dest=ROOT/'data/simulated/policy_improvement/probes'/('confirmation_'+label)
    if (dest/'manifest.json').exists():
        with np.load(dest/'train.npz') as a:return dict(a),dest
    dest.mkdir(parents=True,exist_ok=False)
    original=load_dataset(ROOT/PLAN['original_data'],'train')
    n=30*10*5*PLAN['probe_outcomes_per_pair'];seed=PLAN['probe_confirmation_seed']
    if label=='familiar':
        late=np.flatnonzero(original['observations'][:,0]<=30)
        selected=np.random.default_rng(seed).choice(late,size=n,replace=True)
        states=original['observations'][selected];actions=original['actions'][selected]
    elif label=='balanced':
        pairs=np.repeat(np.array([(t,r,a) for t in range(1,31) for r in range(1,11) for a in range(5)]),PLAN['probe_outcomes_per_pair'],axis=0)
        states=pairs[:,:2];actions=pairs[:,2]
    else:raise ValueError(label)
    extra=probe_records(CONFIG,states,actions,seed+1)
    combined={k:np.concatenate([original[k],extra[k]]) for k in extra}
    np.savez_compressed(dest/'probes.npz',probe_id=np.arange(n),**extra)
    np.savez_compressed(dest/'train.npz',**combined)
    tab=pd.DataFrame({'days':states[:,0],'rooms':states[:,1],'action':actions,'sale':extra['rewards']>0})
    tab=tab.groupby(['days','rooms','action']).sale.agg(['size','sum']).reset_index()
    tab['no_sales']=tab['size']-tab['sum'];tab[['wilson_low','wilson_high']]=[wilson(s,n) for s,n in zip(tab['sum'],tab['size'])]
    tab.to_csv(dest/'support.csv',index=False)
    dump(dest/'manifest.json',{'collector':label,'phase':'confirmation','collection_seed':seed,'outcome_seed':seed+1,
        'transitions':n,'distinct_pairs':len(tab),'sales':int(tab['sum'].sum()),'no_sales':int(tab.no_sales.sum()),
        'source_sha256':training_source_digest(Path(__file__)),'files':{p.name:sha(p) for p in dest.iterdir() if p.is_file()}})
    return combined,dest


def main():
    protocol=initialize();candidate=json.loads((OUT/'candidate.json').read_text())
    records,path=fresh_data(candidate)
    frozen={'candidate_sha256':sha(OUT/'candidate.json'),'confirmation_source_sha256':training_source_digest(Path(__file__)),
            'dataset_manifest_sha256':sha(path/'manifest.json') if path else sha(ROOT/PLAN['original_data']/'manifest.json'),
            'seeds':PLAN['confirmation_seeds'],
            'claim':'selected package versus ordinary replay; log-time versus linear-input control on the same fresh probe data'}
    manifest=OUT/'confirmation_protocol.json'
    if manifest.exists():assert json.loads(manifest.read_text())==frozen
    else:dump(manifest,frozen)
    budget=candidate['updates']
    train_stage('CONF_U','uniform',20000,PLAN['confirmation_seeds'],phase='confirmation')
    if budget>20000:train_stage('CONF_U','uniform',budget,PLAN['confirmation_seeds'],phase='confirmation')
    variants=[('CONF_C',candidate['preprocessing'])]
    if candidate.get('confirm_component_control'):variants.append(('CONF_L','linear'))
    for trajectory,preprocessing in variants:
      for seed in PLAN['confirmation_seeds']:
        spec={'trajectory_id':trajectory,'stage':'confirmation','dataset_id':candidate['dataset'],
            'collection_seed':PLAN['probe_confirmation_seed'] if path else None,'seed':seed,'training_seed':seed,
            'source_commit':protocol['source_commit'],'dataset_sha256':frozen['dataset_manifest_sha256'],
            'source_sha256':{**protocol['sources'],'scripts/confirm_learning_followup.py':training_source_digest(Path(__file__))},
            'preprocessing':preprocessing,'alpha':candidate['alpha'],'replay_rule':candidate['replay_rule'],
            'primary_reference_id':'CONF_U20','primary_reference_revenue':None,'matched_reference_id':None,
            'matched_reference_revenue':None,'planned_endpoints':PLAN['endpoints']}
        print('Confirming',trajectory,'seed',seed,flush=True)
        model,target,runtime=fit_trajectory(records,CONFIG,spec,OUT/'models'/trajectory/f'seed_{seed}',budget)
        assess(model,target,records,CONFIG,spec,budget,runtime,OUT)
    protect(protocol)


if __name__=='__main__':main()
