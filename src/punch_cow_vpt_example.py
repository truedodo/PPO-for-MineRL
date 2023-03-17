
import logging
import coloredlogs
from pprint import pprint
import gym
import minerl
import pickle

import numpy as np
from datetime import datetime


import matplotlib.pyplot as plt

# coloredlogs.install(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)

import sys  # nopep8
sys.path.insert(0, "vpt")  # nopep8

from agent import MineRLAgent  # nopep8


def safe_reset(env):
    """
    Helper function that runs `env.reset()` until it succeeds
    """
    try:
        obs = env.reset()

    except Exception as e:
        print("🛑 Caught game crash! Trying again\n")
        return safe_reset(env)

    else:
        return obs


env = gym.make("MineRLPunchCow-v0")
model = f"models/foundation-model-2x.model"
weights = f"weights/foundation-model-2x.weights"
agent_parameters = pickle.load(open(model, "rb"))

pprint(agent_parameters)

policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
agent = MineRLAgent(env, policy_kwargs=policy_kwargs,
                    pi_head_kwargs=pi_head_kwargs)
agent.load_weights(weights)

obs = safe_reset(env)
done = False

damage_dealt = 0
damage_taken = 0

while not done:
    action = agent.get_action(obs)
    # action = env.action_space.noop()
    obs, reward, done, info = env.step(action)
    env.render()

    _damage_dealt = int(obs["damage_dealt"]["damage_dealt"])
    _damage_taken = int(obs["damage_taken"]["damage_taken"])
    if _damage_dealt > damage_dealt:
        print(f"Damage dealt! ({_damage_dealt - damage_dealt})")
        damage_dealt = _damage_dealt

    if _damage_taken > damage_taken:
        print(f"Damage taken! ({_damage_taken - damage_taken})")
        damage_taken = _damage_taken


print(f"Total damage dealt: {damage_dealt}")

env.close()
