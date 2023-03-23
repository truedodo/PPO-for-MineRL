from typing import List
import numpy as np
from dataclasses import dataclass


@dataclass
class Memory:
    """
    This class represents a single frame/step of the agent
    A full episode should be of type List[Memory]
    """
    obs: np.ndarray
    action: np.ndarray
    action_log_prob: np.ndarray
    reward: float
    done: bool
    value: float


Episode = List[Memory]
