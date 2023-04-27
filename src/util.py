# Random functions / class definitions

import gym
import torch as th
import numpy as np

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")


def to_torch_tensor(t): return th.stack(t).to(device).detach()


def normalize(t, eps=1e-5):
    return (t - t.mean()) / (t.std() + eps)


def safe_reset(env):
    """
    Helper function that runs `env.reset()` until it succeeds
    """
    try:
        obs = env.reset()

    except Exception as e:
        print("Caught game crash! Trying again")
        env.close()
        env = gym.make(env.unwrapped.spec.id)
        return safe_reset(env)

    else:
        return obs, env


def hard_reset(env):
    env.close()
    env = gym.make(env.unwrapped.spec.id)
    obs = env.reset()
    return obs, env


def calculate_gae(rewards: list, values: list, masks: list, gamma: float, lam: float, next_obs_val=0):
    """
    Calculate the generalized advantage estimate
    """
    gae = 0
    returns = []

    for step in reversed(range(len(rewards))):
        next_value = values[step +
                            1] if step < len(rewards) - 1 else next_obs_val
        delta = rewards[step] + gamma * \
            next_value * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])

    return returns


def fix_initial_hidden_states(hidden_states):
    for i in range(len(hidden_states)):
        hidden_states[i] = list(hidden_states[i])
        hidden_states[i][0] = th.from_numpy(np.full(
            (1, 1, 128), False)).to(device)


def detach_hidden_states(hidden_states):
    """
    Detach the hidden states from the computation graph
    """
    for i in range(len(hidden_states)):
        hidden_states[i] = list(hidden_states[i])
        hidden_states[i][1] = list(hidden_states[i][1])
        hidden_states[i][1][0] = hidden_states[i][1][0].detach()
        hidden_states[i][1][1] = hidden_states[i][1][1].detach()


def squeeze_hidden_states(hidden_states):
    """
    Remove the extra dimension added by dataloader concatenation
    """
    for i in range(len(hidden_states)):
        hidden_states[i] = list(hidden_states[i])
        hidden_states[i][1] = list(hidden_states[i][1])
        hidden_states[i][0] = hidden_states[i][0].squeeze(1)
        hidden_states[i][1][0] = hidden_states[i][1][0].squeeze(1)
        hidden_states[i][1][1] = hidden_states[i][1][1].squeeze(1)
