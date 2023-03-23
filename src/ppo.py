
import pickle
import sys
from typing import List
import gym
import torch as th
import numpy as np

from datetime import datetime

from tqdm import tqdm

from rewards import RewardsCalculator
from memory import Episode, Memory

sys.path.insert(0, "vpt")  # nopep8

from agent import MineRLAgent  # nopep8

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")


class ProximalPolicyOptimizer:
    def __init__(
            self,
            env_name: str,
            model_path: str,
            weights_path: str,

            # A custom reward function to be used
            rc: RewardsCalculator = None,

            # Hyperparameters
            ppo_iterations=5,
            episodes: int = 3,
            epochs: int = 8,
            minibatch_size: int = 10,
            lr: float = 1e-4,

    ):
        self.env_name = env_name
        self.env = gym.make(env_name)

        # Set the rewards calcualtor
        if rc is None:
            # Basic default reward function
            rc = RewardsCalculator(
                damage_dealt=1,
                damage_taken=-1,
            )
        self.rc = rc

        # Load hyperparameters unchanged
        self.ppo_iterations = ppo_iterations
        self.episodes = episodes
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.lr = lr

        # Load the agent parameters from the weight files
        agent_parameters = pickle.load(open(model_path, "rb"))
        policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

        self.agent = MineRLAgent(self.env, policy_kwargs=policy_kwargs,
                                 pi_head_kwargs=pi_head_kwargs)
        self.agent.load_weights(weights_path)

        # Internal buffer of the most recent episode memories
        # This will be a relatively large chunk of data
        # Potential memory issues / optimizations around here...
        self.memories: List[Memory] = []

    def run_episode(self):
        """
        Runs a single episode and records the memories 
        """
        print(f"🚩 Running {self.env_name} episode")
        start = datetime.now()

        # episode: Episode = []

        # Initialize the hidden state vector
        # Note that batch size is 1 here because we are only running the agent
        # on a single episode
        # In learn(), we will use a variable batch size
        state = self.agent.policy.initial_state(1)

        # IDK what this is!
        dummy_first = th.from_numpy(np.array((False,))).to(device)

        # Start the episode with gym
        obs = self.env.reset()
        done = False

        while not done:
            # Preprocess image
            agent_obs = self.agent._env_obs_to_agent(obs)

            # Run the full model to get both heads and the
            # new hidden state
            pi_distribution, v_prediction, state = self.agent.policy.get_output_for_observation(
                agent_obs,
                state,
                dummy_first
            )

            # print(pi_distribution)
            # print(policy.get_logprob_of_action(pi_distribution, None))

            # Get action sampled from policy distribution
            # If deterministic==True, this is just argmax BTW
            action = self.agent.policy.pi_head.sample(
                pi_distribution, deterministic=False)

            # Get log probability of taking this action given pi
            action_log_prob = self.agent.policy.get_logprob_of_action(
                pi_distribution, action)

            # Process this so the env can accept it
            minerl_action = self.agent._agent_action_to_env(action)

            # Take action step in the environment
            obs, reward, done, info = self.env.step(minerl_action)

            # Immediately disregard the reward function from the environment
            reward = self.rc.get_rewards(obs, True)

            memory = Memory(agent_obs, action, action_log_prob,
                            reward, done, v_prediction)

            self.memories.append(memory)
            # Finally, render the environment to the screen
            # Comment this out if you are boring
            self.env.render()

        end = datetime.now()
        print(
            f"🏁 Episode finished (duration - {end - start} | reward - {reward})")
        # print(episode)

    def learn(self):
        # TODO: calcualte generalized advantage estimate
        for _ in tqdm(range(self.epochs)):
            # Shuffle the memories
            pass

    def run_train_loop(self):
        """
        Runs the basic PPO training loop
        """
        for i in range(self.ppo_iterations):
            for eps in range(self.episodes):
                print(f"Episode {eps + 1}/{self.episodes}")
                self.run_episode()
            self.learn()
            self.memories.clear()


if __name__ == "__main__":
    ppo = ProximalPolicyOptimizer(
        "MineRLPunchCow-v0",
        "models/foundation-model-2x.model",
        "weights/foundation-model-2x.weights",
    )

    ppo.run_episode()
