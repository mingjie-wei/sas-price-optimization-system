"""Read in order: network, learning targets, loss, training, price choices.

CQL learns from saved decisions. It never calls the hotel environment here.
"""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TrainingConfig:
    # One update means learning from one batch, not one hotel episode.
    updates: int = 20000
    batch_size: int = 256
    learning_rate: float = 0.0003
    alpha: float = 0.1  # Strength of the extra CQL penalty.
    target_update: int = 100
    log_every: int = 1000


def make_network(actions):
    # Two inputs: time left and rooms left. One output per offered price.
    return nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, actions),
    )


def tensors(records, config):
    """Scale inputs to [0,1] and revenue to units of the reference price."""
    scale = np.array([config.horizon, config.capacity], dtype=np.float32)
    return {
        'states': torch.tensor(records['observations'] / scale, dtype=torch.float32),
        'actions': torch.tensor(records['actions'], dtype=torch.long),
        'rewards': torch.tensor(records['rewards'] / config.reference_price, dtype=torch.float32),
        'next_states': torch.tensor(records['next_observations'] / scale, dtype=torch.float32),
        'terminated': torch.tensor(records['terminated'], dtype=torch.bool),
    }


def bellman_target(rewards, terminated, next_q):
    # Gamma=1: total finite-horizon revenue; no future value after termination.
    best_future_value = next_q.max(dim=1).values
    episode_continues = (~terminated).float()
    return rewards + episode_continues * best_future_value


def conservative_penalty(q, actions):
    # gather selects the value of the price actually recorded in each row.
    recorded_action_values = q.gather(1, actions[:, None]).squeeze(1)
    # logsumexp combines all price values, giving more weight to high ones.
    all_action_values = torch.logsumexp(q, dim=1)
    return (all_action_values - recorded_action_values).mean()


def losses(network, target, batch):
    action_values = network(batch['states'])
    recorded_actions = batch['actions'][:, None]
    predicted_values = action_values.gather(1, recorded_actions).squeeze(1)

    # The target is a fixed reference for this update; do not train through it.
    with torch.no_grad():
        next_action_values = target(batch['next_states'])
        target_values = bellman_target(
            batch['rewards'], batch['terminated'], next_action_values,
        )

    prediction_error = 0.5 * F.mse_loss(predicted_values, target_values)
    caution_penalty = conservative_penalty(action_values, batch['actions'])
    return prediction_error, caution_penalty


def diagnostics(network, data, batch_size=4096):
    """A held-out Bellman residual is not a policy-return estimate."""
    totals = np.zeros(2)
    with torch.no_grad():
        for start in range(0, len(data['actions']), batch_size):
            batch = {k: v[start:start + batch_size] for k, v in data.items()}
            terms = losses(network, network, batch)
            totals += np.array([x.item() for x in terms]) * len(batch['actions'])
    averages = totals / len(data['actions'])
    return {'td_loss': float(averages[0]), 'cql_penalty': float(averages[1])}


def train(train_records, validation_records, config, settings=TrainingConfig(), seed=0):
    """No environment or demand probabilities enter this function."""
    if settings.updates < 1 or settings.batch_size < 1 or settings.target_update < 1 or settings.log_every < 1:
        raise ValueError('Training counts must be positive')
    if settings.alpha < 0 or settings.learning_rate <= 0:
        raise ValueError('Use nonnegative alpha and positive learning rate')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    network = make_network(len(config.prices))
    # A slower-changing copy gives more stable targets than a moving guess.
    target = deepcopy(network).requires_grad_(False)
    optimizer = torch.optim.Adam(network.parameters(), lr=settings.learning_rate)
    data = tensors(train_records, config)
    validation = tensors(validation_records, config)
    # Same fixed validation subset at every checkpoint; never a training batch.
    rng = np.random.default_rng(2026)
    indices = rng.choice(len(validation['actions']), min(8192, len(validation['actions'])), replace=False)
    validation = {k: v[indices] for k, v in validation.items()}
    history = []
    for update in range(1, settings.updates + 1):
        # 1. Pick a batch of saved decisions. No new hotel experience.
        indices = torch.randint(len(data['actions']), (settings.batch_size,))
        batch = {k: v[indices] for k, v in data.items()}
        # 2. Measure prediction error and add the CQL penalty.
        td, cql = losses(network, target, batch)
        loss = td + settings.alpha * cql
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss')
        # 3. Adjust the network weights to reduce that combined loss.
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), 10)
        optimizer.step()
        # 4. Occasionally refresh the target copy.
        if update % settings.target_update == 0:
            target.load_state_dict(network.state_dict())
        if update == 1 or update % settings.log_every == 0 or update == settings.updates:
            validation_metrics = diagnostics(network, validation)
            history.append({'update': update, 'batch_td_loss': td.item(),
                            'batch_cql_penalty': cql.item(),
                            'validation_td_loss': validation_metrics['td_loss']})
    return network.eval(), history


def policy_table(network, config):
    t, c = np.meshgrid(np.arange(config.horizon + 1), np.arange(config.capacity + 1), indexing='ij')
    states = np.stack([t.ravel() / config.horizon, c.ravel() / config.capacity], axis=1)
    with torch.no_grad():
        actions = network(torch.tensor(states, dtype=torch.float32)).argmax(dim=1).numpy()
    return actions.reshape(t.shape)
