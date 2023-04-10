# Random functions / class definitions

import gym
import torch as th

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


def calculate_gae(rewards: list, values: list, masks: list, gamma: float, lam: float):
    """
    Calculate the generalized advantage estimate
    """
    gae = 0
    returns = []

    for step in reversed(range(len(rewards))):
        next_value = values[step + 1] if step < len(rewards) - 1 else 0
        delta = rewards[step] + gamma * \
            next_value * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])

    return returns
