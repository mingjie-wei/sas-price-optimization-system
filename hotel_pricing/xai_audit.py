"""Exact two-input explanations of learned price contrasts and simulator outcomes."""
import numpy as np
import torch

from .cql import make_network
from .environment import sale_probability


def reachable(config):
    t, r = np.indices((config.horizon + 1, config.capacity + 1))
    return (t > 0) & (r > 0) & (config.capacity-r <= config.horizon-t)


def load_model(path, config):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = make_network(len(config.prices))
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval(), checkpoint


def model_values(model, config):
    """Network scores in scaled-reward units, preserving first-index argmax ties."""
    t, r = np.indices((config.horizon+1, config.capacity+1))
    states = np.stack([t.ravel()/config.horizon, r.ravel()/config.capacity], axis=1)
    with torch.no_grad():
        q = model(torch.tensor(states, dtype=torch.float32)).numpy()
    return q.reshape((*t.shape, len(config.prices))).astype(float)


def simulator_values(config, policy=None):
    """q_star if policy is None, otherwise q_policy; all in raw revenue units."""
    values = np.zeros((config.horizon+1, config.capacity+1))
    q = np.zeros((*values.shape, len(config.prices)))
    prices = np.array(config.prices)
    p = np.array([sale_probability(config, price) for price in prices])
    for t in range(1, config.horizon+1):
        for r in range(1, config.capacity+1):
            q[t,r] = p*(prices+values[t-1,r-1])+(1-p)*values[t-1,r]
            values[t,r] = q[t,r].max() if policy is None else q[t,r,policy[t,r]]
    return q, values


def sample_reference(records, config, size, seed):
    states = records['observations']
    limit = config.horizon-config.capacity+1
    eligible = states[(states[:,0] >= 1) & (states[:,0] <= limit) & (states[:,1] >= 1)]
    indices = np.random.default_rng(seed).choice(len(eligible), size=size, replace=False)
    return eligible[indices].astype(int)


def shapley(margin, states, reference, config):
    """Exact marginal reference game, evaluated on cached direct network outputs."""
    states, reference = np.asarray(states, int), np.asarray(reference, int)
    limit = config.horizon-config.capacity+1
    for points in [states, reference]:
        if not (np.all((points[:,0] >= 1) & (points[:,0] <= limit)) and
                np.all((points[:,1] >= 1) & (points[:,1] <= config.capacity))):
            raise ValueError('Attribution requires the reachable rectangular domain')
    t, r = states.T
    t0, r0 = reference.T
    f = margin[t,r]
    background = margin[t0,r0]
    cross_time = margin[t[:,None],r0[None,:]]
    cross_rooms = margin[t0[None,:],r[:,None]]
    phi_time = .5*(cross_time-background+f[:,None]-cross_rooms).mean(axis=1)
    phi_rooms = .5*(cross_rooms-background+f[:,None]-cross_time).mean(axis=1)
    base = float(background.mean())
    error = np.max(np.abs(base+phi_time+phi_rooms-f))
    return base, phi_time, phi_rooms, float(error)


def counterfactual(policy, state, config, future_only=False):
    """Nearest action-changing operating condition, not an actionable intervention."""
    t,r = map(int,state)
    mask = reachable(config)
    if not (0 <= t < mask.shape[0] and 0 <= r < mask.shape[1] and mask[t,r]):
        raise ValueError('State is not reachable and nonterminal')
    candidates = np.argwhere(mask & (policy != policy[t,r]))
    if future_only and len(candidates):
        dt,dr = t-candidates[:,0], r-candidates[:,1]
        candidates = candidates[(dt>=0)&(dr>=0)&(dr<=dt)]
    if not len(candidates):
        return None
    dt,dr = np.abs(candidates[:,0]-t), np.abs(candidates[:,1]-r)
    # Integer numerator avoids floating-point tie instability.
    numerator = dt*config.capacity+dr*config.horizon
    order = np.lexsort((candidates[:,1],candidates[:,0],dr,dt,numerator))
    chosen = candidates[order[0]]
    assert policy[tuple(chosen)] != policy[t,r]
    return {'state':chosen.tolist(), 'price':float(config.prices[policy[tuple(chosen)]]),
            'delta_days':int(chosen[0]-t), 'delta_rooms':int(chosen[1]-r),
            'distance':float(numerator[order[0]]/(config.horizon*config.capacity)),
            'mode':'future_only' if future_only else 'operating_condition'}
