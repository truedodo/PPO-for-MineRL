from typing import List
import numpy as np
from dataclasses import dataclass
import torch as th
from torch.utils.data import Dataset


@dataclass
class Memory:
    """
    This class represents a single frame/step of the agent
    A full episode should be of type List[Memory]
    """

    # Raw pixel observation for this frame
    agent_obs: dict
    policy_hidden_state: list
    critic_hidden_state: list
    orig_pi_h: th.tensor
    pi_h: th.tensor
    v_h: th.tensor
    action: np.ndarray
    action_log_prob: np.ndarray
    reward: float
    total_reward: float
    done: bool
    value: float


@dataclass
class AuxMemory:
    """
    This class represents auxillary memory for PPG
    Only has an obs and the target value (return)
    Stored in `B` in paper
    """

    # Raw pixel observation for this frame
    agent_obs: dict
    v_targ: float
    done: bool


class MemoryDataset(Dataset):
    """
    This is a dataset of memory objects (potentially multiple episodes!)
    This is to be used with the PyTorch DataLoader
    """

    def __init__(self, memories: List[Memory]):
        self.memories: List[Memory] = memories

    def __len__(self):
        return len(self.memories)

    def __getitem__(self, idx):
        mem = self.memories[idx]

        # This needs to be returned as a tuple
        return mem.agent_obs, mem.policy_hidden_state, mem.critic_hidden_state, mem.orig_pi_h, mem.pi_h, mem.v_h, mem.action, mem.action_log_prob, mem.reward, mem.total_reward, mem.done, mem.value


# This is probably not needed, but might as well define this type so we have it
# Edit: this is almost certainly not useful since we shuffle memories anyway
Episode = List[Memory]
