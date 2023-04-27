
import logging
import coloredlogs
import fire
import gym
import minerl
import pickle
import pandas as pd
import time

import numpy as np


import matplotlib.pyplot as plt

# coloredlogs.install(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)

import sys

from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival


sys.path.insert(0, "vpt")  # nopep8
from agent import MineRLAgent, ENV_KWARGS  # nopep8


def main(
        env: str = None,
        model: str = "foundation-model-1x",
        weights: str = "foundation-model-1x"

):
    env_name = env
    model_path = f"models/{model}.model"
    weights_path = f"weights/{weights}.weights"

    print(f"⭐️ Starting BattleCraft RL")
    print("============================")
    print(f"Environment: {env_name}")
    print(f"Model:       {model}")
    print(f"Weights:     {weights}")

    if env_name == None:
        env = HumanSurvival(**ENV_KWARGS).make()

    else:
        env = gym.make(env_name)

    agent_parameters = pickle.load(open(model_path, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

    agent = MineRLAgent(env, policy_kwargs=policy_kwargs,
                        pi_head_kwargs=pi_head_kwargs)

    agent.load_weights(weights_path)

    while True:

        agent.reset()
        obs = env.reset()

        done = False
        total_reward = 0
        while not done:
            action = agent.get_action(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            env.render()


if __name__ == "__main__":
    fire.Fire(main)
