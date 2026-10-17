"""Explain an exact policy gap using local decisions and policy visit probabilities."""
import numpy as np
from .environment import sale_probability
from .evaluation import solve
from .diagnostics import optimal_state_visits


def explain_gap(config, policy, records):
    """Weight optimal-advantage losses by visits under the policy being diagnosed.

    Their sum equals V*(start) - V_policy(start), without double counting.
    Training counts are descriptive support, never inputs to this calculation.
    """
    shape = (config.horizon + 1, config.capacity + 1)
    values = np.zeros(shape)
    q = np.zeros((*shape, len(config.prices)))
    prices = np.asarray(config.prices)
    probability = np.array([sale_probability(config, p) for p in prices])
    for t in range(1, config.horizon + 1):
        for c in range(1, config.capacity + 1):
            q[t, c] = probability * (prices + values[t-1, c-1]) + (1-probability) * values[t-1, c]
            values[t, c] = q[t, c].max()
    optimal_actions = q.argmax(axis=2)
    chosen_value = np.take_along_axis(q, policy[..., None], axis=2)[..., 0]
    local_loss = values - chosen_value
    # This forward calculation works for any deterministic policy, not only the optimum.
    visits = optimal_state_visits(config, policy)
    contribution = visits * local_loss
    counts = np.zeros_like(q, dtype=int)
    t, c = records['observations'].T
    np.add.at(counts, (t, c, records['actions']), 1)
    gap = values[-1, -1] - solve(config, policy)[0]
    np.testing.assert_allclose(contribution.sum(), gap, atol=1e-8, rtol=1e-9)
    return {'optimal_actions': optimal_actions, 'local_loss': local_loss,
            'visits': visits, 'contribution': contribution, 'counts': counts,
            'gap': float(gap)}
