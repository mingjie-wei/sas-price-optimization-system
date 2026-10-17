"""Fixed price mixtures used to collect and evaluate offline data."""
import numpy as np


def probability_table(config, behavior):
    """Return price probabilities for each [time left, rooms left] state."""
    if behavior.get('type', 'fixed_mix') != 'fixed_mix':
        raise ValueError('Unknown collection rule')
    p = np.asarray(behavior['probabilities'], dtype=float)
    if p.shape != (len(config.prices),) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError('Provide one nonnegative probability per price, summing to 1')
    shape = (config.horizon + 1, config.capacity + 1, len(config.prices))
    return np.broadcast_to(p, shape)
