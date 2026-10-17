"""Check the assumed world and the experience recorded inside it."""
import numpy as np
from gymnasium.utils.env_checker import check_env

from .environment import HotelPricingEnv, sale_probability
from .evaluation import solve


def verify_environment(config):
    """Check the interface, probability curve, and a repeatable full rollout."""
    check_env(HotelPricingEnv(config), skip_render_check=True)
    prices = np.asarray(config.prices)
    probabilities = np.array([sale_probability(config, p) for p in prices])
    assert np.all((probabilities > 0) & (probabilities < 1))
    order = np.argsort(prices)
    assert np.all(np.diff(probabilities[order]) < 0)
    assert np.isclose(sale_probability(config, config.reference_price), config.reference_sale_probability)
    traces = []
    for _ in range(2):
        env = HotelPricingEnv(config)
        state, info = env.reset(seed=71)
        trace = []
        for day in range(config.horizon):
            action = day % len(prices)
            next_state, reward, ended, truncated, info = env.step(action)
            sold = int(state[1] - next_state[1])
            assert sold in [0, 1] and next_state[0] == state[0] - 1
            assert reward == prices[action] * sold
            assert ended == (next_state[0] == 0 or next_state[1] == 0)
            assert not truncated and env.observation_space.contains(next_state)
            assert set(info) == {'price', 'sold'}
            trace.append((state.tolist(), action, reward, next_state.tolist(), ended))
            state = next_state
            if ended:
                break
        assert ended
        env.close()
        traces.append(trace)
    assert traces[0] == traces[1]
    shape = (config.horizon + 1, config.capacity + 1)
    fixed = [solve(config, np.full(shape, a))[0] for a in range(len(prices))]
    optimal, _ = solve(config)
    assert optimal >= max(fixed) - 1e-8
    return {'interface': 'passed', 'probabilities': probabilities.tolist(),
            'accounting_and_endings': 'passed', 'seed_replay': 'passed',
            'observation': ['selling_periods_remaining', 'rooms_remaining'],
            'info': ['price', 'sold'], 'fixed_revenues': fixed,
            'best_fixed_price': float(prices[np.argmax(fixed)]), 'optimal_revenue': optimal}


def optimal_state_visits(config, policy):
    """Expected visits from full inventory, using known probabilities for diagnosis only."""
    mass = np.zeros((config.horizon + 1, config.capacity + 1))
    mass[config.horizon, config.capacity] = 1
    for t in range(config.horizon, 0, -1):
        for c in range(1, config.capacity + 1):
            p = sale_probability(config, config.prices[policy[t, c]])
            mass[t - 1, c - 1] += mass[t, c] * p
            mass[t - 1, c] += mass[t, c] * (1 - p)
    mass[0, :] = 0
    mass[:, 0] = 0
    return mass


def coverage(records, config):
    """Report training-only coverage, including support near the optimal policy's path."""
    visits = np.zeros((config.horizon + 1, config.capacity + 1, len(config.prices)), dtype=int)
    t, c = records['observations'].T
    np.add.at(visits, (t, c, records['actions']), 1)
    time, rooms = np.indices(visits.shape[:2])
    reachable = (time > 0) & (rooms > 0) & (config.capacity - rooms <= config.horizon - time)
    _, optimum = solve(config)
    optimal_visits = optimal_state_visits(config, optimum)
    selected_counts = np.take_along_axis(visits, optimum[..., None], axis=2)[..., 0]
    result = {'transitions': len(t), 'action_counts': visits.sum(axis=(0, 1)).tolist(),
              'reachable_state_action_pairs': int(reachable.sum() * len(config.prices)),
              'optimal_expected_decisions': float(optimal_visits.sum())}
    for threshold in [1, 5, 20]:
        result[f'coverage_at_least_{threshold}'] = float((visits[reachable] >= threshold).mean())
        result[f'optimal_path_support_at_least_{threshold}'] = float(
            np.sum(optimal_visits * (selected_counts >= threshold)) / optimal_visits.sum())
    return result, visits


def paired_difference(first, second):
    """Uncertainty for matched evaluation episodes, conditional on fixed policies."""
    differences = np.asarray(first) - np.asarray(second)
    if differences.ndim != 1 or len(differences) < 2:
        raise ValueError('Use at least two paired episode outcomes')
    return {'mean': float(differences.mean()),
            'ci95_half_width': float(1.96 * differences.std(ddof=1) / np.sqrt(len(differences)))}

