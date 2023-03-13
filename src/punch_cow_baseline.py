
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


damages = []
for i in range(1, 4):

    env = gym.make("MineRLPunchCow-v0")

    model = f"models/foundation-model-{i}x.model"
    weights = f"weights/foundation-model-{i}x.weights"

    agent_parameters = pickle.load(open(model, "rb"))

    # pprint(agent_parameters)

    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs,
                        pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    n_trials = 100

    damages.append([])

    for k in range(n_trials):
        start = datetime.now()
        print(f"🚩 Starting trial {k+1}/{n_trials} for model {i}x")

        reset_status = False

        obs = safe_reset(env)

        done = False

        while not done:
            action = agent.get_action(obs)
            # action = env.action_space.noop()

            obs, reward, done, info = env.step(action)

            # env.render()

        dmg = obs["damage_dealt"]["damage_dealt"]

        print(
            f"🏁 Finished  with {dmg} damage (time: {datetime.now() - start})\n")

        damages[i-1].append(dmg)

    env.close()


plt.title(f"VPT Zero Shot Damage (n={n_trials})")

for i in range(1, 4):
    counts, bins = np.histogram(damages[i-1])
    plt.hist(counts, bins, alpha=0.5, label=f"foundation-{i}x")

plt.legend(loc='upper right')
plt.savefig("damage.png")
plt.show()

for i in range(1, 4):
    print(f"foundation-{i}x:")
    print(f"  {np.mean(damages[i-1])} avg damage")
    print(f"  {np.std(damages[i-1])} std damage\n")

print("All done 🎉")
