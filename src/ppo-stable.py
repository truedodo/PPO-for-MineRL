import pickle
import sys
import time
from typing import List
import gym
import torch as th
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy

from datetime import datetime
from tqdm import tqdm
from memory import Memory, MemoryDataset
from util import to_torch_tensor, normalize, safe_reset, hard_reset, calculate_gae
from vectorized_minerl import *

sys.path.insert(0, "vpt")  # nopep8

from agent import MineRLAgent  # nopep8
from lib.tree_util import tree_map  # nopep8

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor

from gym import spaces

# For debugging purposes
# th.autograd.set_detect_anomaly(True)
# device = th.device("mps" if th.backends.mps.is_available() else "cpu")
device = th.device("mps")  # apple silicon

TRAIN_WHOLE_MODEL = False


class MineRLPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        # Load agent parameters from the weight files
        model_path = f"models/foundation-model-3x.model"
        weights_path = f"weights/foundation-model-3x.weights"
        agent_parameters = pickle.load(open(model_path, "rb"))
        policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

        # Create the main agent with a policy and value head
        self.agent = MineRLAgent(None, policy_kwargs=policy_kwargs,
                                 pi_head_kwargs=pi_head_kwargs)
        self.agent.load_weights(weights_path)

        # Create the original agent which we will use in training
        self.orig_agent = MineRLAgent(None, policy_kwargs=policy_kwargs,
                                      pi_head_kwargs=pi_head_kwargs)
        self.orig_agent.load_weights(weights_path)
    
    def _build_mlp_extractor(self):
        # Define the architecture for the policy and value networks
        net_arch = [dict(pi=[64, 64], vf=[64, 64])]
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=net_arch,
            activation_fn=th.nn.Tanh
        )


# Update FlattenObsWrapper to flatten observations into a single Box space
class FlattenObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(FlattenObsWrapper, self).__init__(env)
        self.flatten_observation_space(env.observation_space)
    
    def flatten_observation_space(self, obs_space):
        if isinstance(obs_space, spaces.Dict):
            # Calculate the total size by summing all dimensions
            low = []
            high = []
            for key in sorted(obs_space.spaces.keys()):
                space = obs_space.spaces[key]
                if isinstance(space, spaces.Dict):
                    for sub_key in sorted(space.spaces.keys()):
                        sub_space = space.spaces[sub_key]
                        if isinstance(sub_space, spaces.Box):
                            low.append(sub_space.low.flatten())
                            high.append(sub_space.high.flatten())
                        elif isinstance(sub_space, spaces.Discrete):
                            low.append(np.array([0]))
                            high.append(np.array([sub_space.n - 1]))
                        else:
                            raise NotImplementedError(f"Space type {type(sub_space)} not supported.")
                elif isinstance(space, spaces.Box):
                    low.append(space.low.flatten())
                    high.append(space.high.flatten())
                elif isinstance(space, spaces.Discrete):
                    low.append(np.array([0]))
                    high.append(np.array([space.n - 1]))
                else:
                    raise NotImplementedError(f"Space type {type(space)} not supported.")
            
            self.observation_space = spaces.Box(
                low=np.concatenate(low),
                high=np.concatenate(high),
                dtype=np.float32
            )
        else:
            raise NotImplementedError("Only Dict observation spaces are supported by FlattenObsWrapper.")
    
    def observation(self, obs):
        flat_obs = []
        for key in sorted(self.env.observation_space.spaces.keys()):
            space = self.env.observation_space.spaces[key]
            if isinstance(space, spaces.Dict):
                for sub_key in sorted(space.spaces.keys()):
                    sub_space = space.spaces[sub_key]
                    value = obs[key][sub_key]
                    if isinstance(sub_space, spaces.Box):
                        flat_obs.append(value.flatten())
                    elif isinstance(sub_space, spaces.Discrete):
                        flat_obs.append(np.array([value], dtype=np.float32))
                    else:
                        raise NotImplementedError(f"Space type {type(sub_space)} not supported.")
            elif isinstance(space, spaces.Box):
                flat_obs.append(obs[key].flatten())
            elif isinstance(space, spaces.Discrete):
                flat_obs.append(np.array([obs[key]], dtype=np.float32))
            else:
                raise NotImplementedError(f"Space type {type(space)} not supported.")
        return np.concatenate(flat_obs).astype(np.float32)

# New: Custom ActionWrapper to handle mixed action spaces
class FlattenActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(FlattenActionWrapper, self).__init__(env)
        
        # Define the number of discrete actions
        discrete_action_keys = [
            'ESC', 'attack', 'back', 'drop', 'forward', 'hotbar.1', 'hotbar.2',
            'hotbar.3', 'hotbar.4', 'hotbar.5', 'hotbar.6', 'hotbar.7',
            'hotbar.8', 'hotbar.9', 'inventory', 'jump', 'left', 'pickItem',
            'right', 'sneak', 'sprint', 'swapHands', 'use'
        ]
        self.discrete_action_keys = discrete_action_keys
        self.num_discrete = len(discrete_action_keys)
        
        # Define the new action space
        # First num_discrete actions are for discrete actions (0 or 1)
        # Last 2 actions are for camera (continuous)
        low = np.array([0] * self.num_discrete + [-180.0, -180.0], dtype=np.float32)
        high = np.array([1] * self.num_discrete + [180.0, 180.0], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
    
    def action(self, action):
        # Split the action into discrete and continuous parts
        discrete_actions = np.clip(action[:self.num_discrete], 0, 1)
        discrete_actions = (discrete_actions > 0.5).astype(int)
        camera_actions = action[self.num_discrete:]
        
        # Map to Dict action
        dict_action = {}
        for idx, key in enumerate(self.discrete_action_keys):
            dict_action[key] = int(discrete_actions[idx])
        dict_action['camera'] = camera_actions  # Assuming camera expects a numpy array
        
        return dict_action

def make_minerl_env(env_name):
    def _init():
        env = gym.make(env_name)
        env = FlattenObsWrapper(env)
        env = FlattenActionWrapper(env)
        return env
    return _init

if __name__ == "__main__":
    num_envs = 4
    envs = [make_minerl_env("MineRLObtainDiamondShovel-v0") for _ in range(num_envs)]
    vec_env = DummyVecEnv(envs)

    # Create PPO model from stable-baselines3 using our custom policy
    model = PPO(
        policy=MineRLPolicy,
        env=vec_env,
        learning_rate=2.5e-5,
        n_steps=50,
        batch_size=50,  # Changed from 48 to 50 to be a factor of 200 (50*4)
        n_epochs=4,
        gamma=0.99,
        clip_range=0.2,
        tensorboard_log="logs",
        # ...other hyperparams...
    )

    # Train and save as before
    model.learn(total_timesteps=50 * 500)  # num_steps * num_rollouts
    model.save("weights/ppo-zombie-hunter-1x.weights")

    print("Training complete.")
