
from collections import namedtuple
import logging
from typing import List
import coloredlogs
from pprint import pprint
import fire
import gym
import minerl
import pickle

import torch as th

from tqdm import tqdm

import torch.nn.functional as F
from torch import nn

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.distributions import Categorical


import numpy as np
from datetime import datetime


import matplotlib.pyplot as plt

# coloredlogs.install(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)

import sys

from rewards import RewardsCalculator  # nopep8
sys.path.insert(0, "vpt")  # nopep8

from agent import MineRLAgent  # nopep8

# Inspired by:
# https://github.com/lucidrains/phasic-policy-gradient/blob/master/train.py

Memory = namedtuple(
    'Memory', ['state', 'action', 'action_log_prob', 'reward', 'done', 'value'])
AuxMemory = namedtuple('Memory', ['state', 'target_value', 'old_values'])


device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
# device = th.device("mps")


class ExperienceDataset(Dataset):
    # IDK what this is and why he didn't just use a list

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, ind):
        return tuple(map(lambda t: t[ind], self.data))


def create_shuffled_dataloader(data, batch_size):
    ds = ExperienceDataset(data)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def clipped_value_loss(values, rewards, old_values, clip):
    value_clipped = old_values + (values - old_values).clamp(-clip, clip)
    value_loss_1 = (value_clipped.flatten() - rewards) ** 2
    value_loss_2 = (values.flatten() - rewards) ** 2
    return th.mean(th.max(value_loss_1, value_loss_2))


def normalize(t, eps=1e-5):
    return (t - t.mean()) / (t.std() + eps)


def update_network_(loss, optimizer):
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()


class PhasicPolicyGradient():
    def __init__(
            self,
            env_name: str,
            model_path: str,
            weights_path: str,

            rc: RewardsCalculator,

            # Hyperparameters
            epochs: int = 8,
            epochs_aux: int = 8,
            minibatch_size: int = 10,
            lr: float = 1e-4,
            betas: tuple = (0.9, 0.999),

    ):
        self.env = gym.make(env_name)

        self.rc = rc

        agent_parameters = pickle.load(open(model_path, "rb"))
        policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

        self.agent = MineRLAgent(self.env, policy_kwargs=policy_kwargs,
                                 pi_head_kwargs=pi_head_kwargs)
        self.agent.load_weights(weights_path)

        # Just so we can access these in a cleaner way
        self.base = self.agent.policy.net
        self.actor = self.agent.policy.pi_head
        self.critic = self.agent.policy.value_head

        # Use Adam for both actor and critic
        # self.opt_actor = Adam(self.actor.parameters(), lr=lr, betas=betas)
        # self.opt_critic = Adam(self.critic.parameters(), lr=lr, betas=betas)

        # Memory buffers for the model during training
        self.memories: List[Memory] = []
        self.aux_memories: List[AuxMemory] = []
        return

    def run_base(self):
        pass

    def learn(self, next_state):
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        masks = []
        values = []

        for mem in self.memories:
            states.append(mem.state)
            actions.append(th.tensor(mem.action))
            old_log_probs.append(mem.action_log_prob)
            rewards.append(mem.reward)
            masks.append(1 - float(mem.done))
            values.append(mem.value)

        # calculate generalized advantage estimate
        next_state = th.from_numpy(next_state).to(device)
        next_value = self.critic(next_state).detach()
        values = values + [next_value]

        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * \
                values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.lam * masks[i] * gae
            returns.insert(0, gae + values[i])

        # Convert values to torch tensors
        def to_torch_tensor(t): return th.stack(t).to(device).detach()

        states = to_torch_tensor(states)
        actions = to_torch_tensor(actions)
        old_values = to_torch_tensor(values[:-1])
        old_log_probs = to_torch_tensor(old_log_probs)

        rewards = th.tensor(returns).float().to(device)

        # Store state and target values to auxiliary memory buffer for later training
        aux_memory = AuxMemory(states, rewards, old_values)
        self.aux_memories.append(aux_memory)

        # prepare dataloader for policy phase training
        dl = create_shuffled_dataloader(
            [states, actions, old_log_probs, rewards, old_values], self.minibatch_size)

        for _ in range(self.epochs):
            for states, actions, old_log_probs, rewards, old_values in dl:
                # pi_distribution, v_prediction, new_agent_state = policy.get_output_for_observation(
                #     agent_obs,
                #     initial_state,
                #     dummy_first
                # )
                print("do some training")
                pass


def main():
    rc = RewardsCalculator(
        damage_dealt=1,
        damage_taken=20,
    )
    ppg = PhasicPolicyGradient(
        "MineRLPunchCow-v0",
        "models/foundation-model-2x.model",
        "weights/foundation-model-2x.weights",
        rc)

    num_episodes = 10
    eps_per_learn = 1

    # Shorthand
    policy = ppg.agent.policy

    for eps in tqdm(range(num_episodes), desc='Episodes'):
        # Reset the environment
        obs = ppg.env.reset()

        # Reset the initial state
        # This may be redundant, but better to be safe
        ppg.agent.reset()
        damage_dealt = 0

        state = policy.initial_state(1)
        dummy_first = th.from_numpy(np.array((False,))).to(device)

        done = False
        while not done:
            # Preprocess image
            agent_obs = ppg.agent._env_obs_to_agent(obs)

            # Run the full model to get both heads and the
            # new hidden state
            pi_distribution, v_prediction, state = policy.get_output_for_observation(
                agent_obs,
                state,
                dummy_first
            )

            # print(pi_distribution)
            # print(policy.get_logprob_of_action(pi_distribution, None))

            # Get action sampled from policy distribution
            # If deterministic==True, this is just argmax
            action = ppg.actor.sample(
                pi_distribution, deterministic=False)

            # Get log probability of taking this action given pi
            action_log_prob = policy.get_logprob_of_action(
                pi_distribution, action)

            # Process this so the env can accept it
            minerl_action = ppg.agent._agent_action_to_env(action)

            # Take action step in the environment
            obs, reward, done, info = ppg.env.step(minerl_action)

            # Immediately disregard the reward function from the environment
            reward = rc.get_rewards(obs, True)

            memory = Memory(agent_obs, action, action_log_prob,
                            reward, done, v_prediction)

            ppg.memories.append(memory)

            ppg.env.render()

        # Do this at the end of each episode
        # TODO: pool multiple episodes and do this every 5 episodes or so
        # NOT SURE IF agent_obs IS CORRECT HERE, NEED TO PUT SOMETHING
        # ppg.learn(agent_obs)
        ppg.memories.clear()

    ppg.env.close()


if __name__ == "__main__":
    main()
