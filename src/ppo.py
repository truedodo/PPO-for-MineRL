
import pickle
import sys
from typing import List
import gym
import torch as th
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from tqdm import tqdm
from rewards import RewardsCalculator
from memory import Memory, MemoryDataset
from util import to_torch_tensor, normalize, safe_reset, hard_reset

sys.path.insert(0, "vpt")  # nopep8

from agent import MineRLAgent  # nopep8
from lib.tree_util import tree_map  # nopep8

# For debugging purposes
# th.autograd.set_detect_anomaly(True)
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
            betas: tuple = (0.9, 0.999),
            beta_s: float = 0.01,
            eps_clip: float = 0.2,
            value_clip: float = 0.4,

            # Optional: plot stuff
            plot: bool = False,

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
        self.betas = betas
        self.beta_s = beta_s
        self.eps_clip = eps_clip
        self.value_clip = value_clip

        self.plot = plot

        # Load the agent parameters from the weight files
        agent_parameters = pickle.load(open(model_path, "rb"))
        policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

        self.agent = MineRLAgent(self.env, policy_kwargs=policy_kwargs,
                                 pi_head_kwargs=pi_head_kwargs)
        self.agent.load_weights(weights_path)

        self.optim_pi = th.optim.Adam(
            self.agent.policy.pi_head.parameters(), lr=lr, betas=betas)

        self.optim_v = th.optim.Adam(
            self.agent.policy.value_head.parameters(), lr=lr, betas=betas)

        # Internal buffer of the most recent episode memories
        # This will be a relatively large chunk of data
        # Potential memory issues / optimizations around here...
        self.memories: List[Memory] = []

    def run_episode(self):
        """
        Runs a single episode and records the memories 
        """
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
        # obs = self.env.reset()

        obs, self.env = safe_reset(self.env)
        # obs, self.env = hard_reset(self.env)
        done = False

        # This is not really used in training
        # More just for us to estimate the success of an episode
        total_reward = 0

        while not done:
            # Preprocess image
            agent_obs = self.agent._env_obs_to_agent(obs)

            # Run the full model to get both heads and the
            # new hidden state
            # pi_distribution, v_prediction, state = self.agent.policy.get_output_for_observation(
            #     agent_obs,
            #     state,
            #     dummy_first
            # )
            agent_obs = tree_map(lambda x: x.unsqueeze(1), agent_obs)
            first = dummy_first.unsqueeze(1)

            # Need to run this with no_grad or else the gradient descent will crash during training lol
            with th.no_grad():
                (pi_h, v_h), state = self.agent.policy.net(
                    agent_obs, state, context={"first": first})

                pi_distribution = self.agent.policy.pi_head(pi_h)
                v_prediction = self.agent.policy.value_head(v_h)

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
            total_reward += reward

            memory = Memory(agent_obs, state, pi_h, v_h, action, action_log_prob,
                            reward, done, v_prediction)

            self.memories.append(memory)
            # Finally, render the environment to the screen
            # Comment this out if you are boring
            self.env.render()

        # Reset the reward calculator once we are done with the episode
        self.rc.clear()

        end = datetime.now()
        print(
            f"🏁 Episode finished (duration - {end - start} | Σreward - {total_reward})")
        # print(episode)

    def learn(self):
        # TODO: calcualte generalized advantage estimate
        # IDK if that is just for PPG or what, but it looked scary

        data = MemoryDataset(self.memories)
        dl = DataLoader(data, batch_size=self.minibatch_size, shuffle=True)

        if self.plot:
            pi_loss_history = []
            v_loss_history = []

        for _ in tqdm(range(self.epochs), desc="epochs"):

            # Shuffle the memories
            # Note: These are batches! Not individual samples
            for obs, states, pi_h, v_h, actions, old_action_log_probs, rewards, dones, v_old in dl:
                # Run the model on the batch using the latent state output
                pi_distribution = self.agent.policy.pi_head(pi_h)
                v_prediction = self.agent.policy.value_head(v_h)

                # Get log probs
                action_log_probs = self.agent.policy.get_logprob_of_action(
                    pi_distribution, actions)

                entropy = self.agent.policy.pi_head.entropy(pi_distribution)

                # Calculate clipped surrogate objective
                ratios = (action_log_probs - old_action_log_probs).exp()
                advantages = normalize(rewards - v_prediction.detach())
                surr1 = ratios * advantages
                surr2 = ratios.clamp(
                    1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = - th.min(surr1, surr2) - self.beta_s * entropy

                # print(type(policy_loss))
                # print(policy_loss.shape)

                # Calculate clipped value loss
                value_clipped = v_old + \
                    (v_prediction - v_old).clamp(-self.value_clip, self.value_clip)

                value_loss_1 = (value_clipped.squeeze() - rewards) ** 2
                value_loss_2 = (v_prediction.squeeze() - rewards) ** 2

                value_loss = th.mean(th.max(value_loss_1, value_loss_2))
                # Calculate clipped value loss
                # value_loss = clipped_value_loss(
                #     v_prediction, rewards, old_values, self.value_clip)

                if self.plot:
                    pi_loss_history.append(policy_loss.mean().item())
                    v_loss_history.append(value_loss.item())

                # Update the policy network
                self.optim_pi.zero_grad()
                policy_loss.mean().backward()
                self.optim_pi.step()

                # Update the value network
                self.optim_v.zero_grad()
                value_loss.backward()
                self.optim_v.step()

            # Update plot at the end of every epoch
            if self.plot:
                self.pi_loss_plot.set_ydata(pi_loss_history)
                self.pi_loss_plot.set_xdata(list(range(len(pi_loss_history))))

                self.v_loss_plot.set_ydata(v_loss_history)
                self.v_loss_plot.set_xdata(list(range(len(v_loss_history))))

                self.ax.relim()        # Recalculate limits
                self.ax.autoscale_view(True, True, True)
                self.fig.canvas.draw()
                # plt.pause(0.01)
                self.fig.canvas.flush_events()
            # update_network(value_loss, self.optim_v)

    def run_train_loop(self):
        """
        Runs the basic PPO training loop
        """
        if self.plot:
            # Create a plot to show the progress of the training
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 4))

            self.ax.set_autoscale_on(True)  # enable autoscale
            self.ax.autoscale_view(True, True, True)

            self.pi_loss_plot, = self.ax.plot(
                [], [], label="Policy Loss", color="blue")

            self.v_loss_plot, = self.ax.plot(
                [], [], label="Value Loss", color="orange")

        for i in range(self.ppo_iterations):
            for eps in range(self.episodes):
                print(
                    f"🚩 Starting {self.env_name} episode {eps + 1}/{self.episodes}")
                self.run_episode()
            self.learn()
            self.memories.clear()


if __name__ == "__main__":
    rc = RewardsCalculator(
        damage_dealt=1
    )
    ppo = ProximalPolicyOptimizer(
        "MineRLPunchCow-v0",
        "models/foundation-model-2x.model",
        "weights/foundation-model-2x.weights",

        rc=rc,
        ppo_iterations=100,
        episodes=3,
        epochs=12,
        minibatch_size=48,
        lr=1e-3,


        plot=True
    )

    ppo.run_train_loop()
