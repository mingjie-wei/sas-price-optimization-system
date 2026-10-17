"""Model-informed benchmarks and fresh rollouts, separate from offline fitting."""
import numpy as np

from .environment import HotelPricingEnv, sale_probability


def solve(config, policy=None):
    """Exact expected revenue, with zero terminal value and gamma=1.

    policy=None finds the optimum; otherwise evaluate an action table or
    a probability distribution over actions at each state.
    """
    shape = (config.horizon + 1, config.capacity + 1)
    values = np.zeros(shape)
    choices = np.zeros(shape, dtype=int)
    prices = np.asarray(config.prices)
    p = np.array([sale_probability(config, price) for price in prices])
    for t in range(1, config.horizon + 1):
        for c in range(1, config.capacity + 1):
            revenue_if_sold = prices + values[t - 1, c - 1]
            revenue_if_not_sold = values[t - 1, c]
            q = p * revenue_if_sold + (1 - p) * revenue_if_not_sold
            if policy is None:
                choices[t, c] = q.argmax()
                values[t, c] = q.max()
            elif policy.ndim == 2:
                values[t, c] = q[policy[t, c]]
            else:
                values[t, c] = np.dot(policy[t, c], q)
    return float(values[-1, -1]), choices


def rollouts(config, policy, environment_seeds, action_seed=2000000):
    env = HotelPricingEnv(config)
    revenues, occupancies = [], []
    for index, seed in enumerate(environment_seeds):
        state, _ = env.reset(seed=int(seed))
        rng = np.random.default_rng(action_seed + index)
        revenue = 0
        done = False
        while not done:
            t, c = state
            if policy.ndim == 2:
                # A learned policy chooses one action for this state.
                action = int(policy[t, c])
            else:
                # The behavior policy randomly picks from its price mix.
                action = int(rng.choice(len(config.prices), p=policy[t, c]))
            state, reward, done, truncated, _ = env.step(action)
            assert not truncated
            revenue += reward
        revenues.append(revenue)
        occupancies.append((config.capacity - state[1]) / config.capacity)
    env.close()
    return np.array(revenues), np.array(occupancies)
