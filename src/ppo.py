
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
# device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
device = th.device("mps")  # apple silicon


class ProximalPolicyOptimizer:
    def __init__(
            self,
            env_name: str,
            model_path: str,
            weights_path: str,

            # A custom reward function to be used
            rc: RewardsCalculator,

            # Hyperparameters
            ppo_iterations: int,
            episodes: int,
            epochs: int,
            minibatch_size: int,
            lr: float,
            betas: tuple,
            beta_s: float,
            eps_clip: float,
            value_clip: float,
            value_loss_weight: float,
            gamma: float,
            lam: float,

            mem_buffer_size: int,
            # tau: float = 0.95,

            # Optional: plot stuff
            plot: bool,



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
        self.value_loss_weight = value_loss_weight
        self.gamma = gamma
        self.lam = lam
        # self.tau = tau

        self.plot = plot

        self.mem_buffer_size = mem_buffer_size

        if self.plot:
            self.pi_loss_history = []
            self.v_loss_history = []
            self.total_loss_history = []

            self.entropy_history = []
            self.expl_var_history = []

            self.surr1_history = []
            self.surr2_history = []

            self.reward_history = []

        # Load the agent parameters from the weight files
        agent_parameters = pickle.load(open(model_path, "rb"))
        policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

        self.agent = MineRLAgent(self.env, policy_kwargs=policy_kwargs,
                                 pi_head_kwargs=pi_head_kwargs)
        self.agent.load_weights(weights_path)

        # This can be adjusted later to only train certain heads
        # Unifying all parameters under one optimizer gives us much more flexibility
        trainable_parameters = self.agent.policy.parameters()

        self.optim = th.optim.Adam(trainable_parameters, lr=lr, betas=betas)

        # self.optim_pi = th.optim.Adam(
        #     self.agent.policy.pi_head.parameters(), lr=lr, betas=betas)

        # self.optim_v = th.optim.Adam(
        #     self.agent.policy.value_head.parameters(), lr=lr, betas=betas)

        # Internal buffer of the most recent episode memories
        # This will be a relatively large chunk of data
        # Potential memory issues / optimizations around here...
        self.memories: List[Memory] = []

    def run_episode(self, hard_reset: bool = False):
        """
        Runs a single episode and records the memories 
        """
        start = datetime.now()

        # Temporary buffer to put the memories in before extending self.memories
        episode_memories: List[Memory] = []

        # Initialize the hidden state vector
        # Note that batch size is 1 here because we are only running the agent
        # on a single episode
        state = self.agent.policy.initial_state(1)

        # I think these are just internal masks that should just be set to false
        dummy_first = th.from_numpy(np.array((False,))).to(device)
        dummy_first = dummy_first.unsqueeze(1)

        # Start the episode with gym
        # obs = self.env.reset()
        if hard_reset:
            self.env.close()
            self.env = gym.make(self.env_name)
            obs = self.env.reset()
        else:
            obs = self.env.reset()
        done = False

        # This is not really used in training
        # More just for us to estimate the success of an episode
        total_reward = 0

        while not done:
            # Preprocess image
            agent_obs = self.agent._env_obs_to_agent(obs)

            # Basically just adds a dimension to the tensor
            agent_obs = tree_map(lambda x: x.unsqueeze(1), agent_obs)

            # Need to run this with no_grad or else the gradient descent will crash during training lol
            with th.no_grad():
                (pi_h, v_h), state = self.agent.policy.net(
                    agent_obs, state, context={"first": dummy_first})

                pi_distribution = self.agent.policy.pi_head(pi_h)
                v_prediction = self.agent.policy.value_head(v_h)

            # Get action sampled from policy distribution
            # If deterministic==True, this just uses argmax
            action = self.agent.policy.pi_head.sample(
                pi_distribution, deterministic=False)

            # print(action)

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
                            reward, 0, done, v_prediction)

            episode_memories.append(memory)
            # Finally, render the environment to the screen
            # Comment this out if you are boring
            self.env.render()

        # Update all memories so we know the total reward of its episode
        # Intuition: memories from episodes where 0 reward was achieved are less valuable
        for mem in episode_memories:
            mem.total_reward = total_reward

        # Reset the reward calculator once we are done with the episode
        self.rc.clear()

        # Update internal memory buffer
        self.memories.extend(episode_memories)

        if self.plot:
            # Update the reward plot
            self.reward_history.append(total_reward)
            self.reward_plot.set_ydata(self.reward_history)
            self.reward_plot.set_xdata(range(len(self.reward_history)))

            self.ax[1, 2].relim()
            self.ax[1, 2].autoscale_view(True, True, True)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        end = datetime.now()
        print(
            f"✅ Episode finished (duration - {end - start} | memories - {len(episode_memories)} | total reward - {total_reward})")

    def learn(self):

        # Create dataloader from the memory buffer
        data = MemoryDataset(self.memories)
        dl = DataLoader(data, batch_size=self.minibatch_size, shuffle=True)

        # dummy_first = th.from_numpy(
        #     np.tile(np.array((False,)), self.minibatch_size)).to(device)

        # dummy_first = th.from_numpy(np.array((False,))).to(device)
        # dummy_first = dummy_first.unsqueeze(1)
        # # print(dummy_first.shape)

        # Shorthand
        policy = self.agent.policy

        for _ in tqdm(range(self.epochs), desc="epochs"):

            # Note: These are batches, not individual samples
            for agent_obs, state, recorded_pi_h, recorded_v_h, actions, old_action_log_probs, rewards, total_rewards, dones, v_old in dl:
                batch_size = len(dones)
                # print(agent_obs["img"].shape)
                v_old = v_old.to(device)

                # DATALOADER WHY DO YOU LOVE ADDING DIMENSIONS?!!
                for i in range(len(state)):
                    state[i][0] = state[i][0].squeeze(1)
                    state[i][1][0] = state[i][1][0].squeeze(1)
                    state[i][1][1] = state[i][1][1].squeeze(1)
                agent_obs = tree_map(lambda x: x.squeeze(1), agent_obs)

                dummy_first = th.from_numpy(np.full(
                    (batch_size, 1), False)).to(device)

                # print(dummy_first.shape)

                (pi_h, v_h), state_out = policy.net(
                    agent_obs, state, context={"first": dummy_first})

                pi_distribution = self.agent.policy.pi_head(pi_h)
                v_prediction = self.agent.policy.value_head(v_h).to(device)

                masks = list(map(lambda d: 1-float(d), dones))

                returns = []
                gae = 0

                for i in reversed(range(len(rewards))):
                    # hacky but necessary since we don't have "next_state"
                    v_next = v_old[i + 1] if i != len(rewards) - 1 else 0

                    delta = rewards[i] + self.gamma * \
                        v_next * masks[i] - v_old[i]
                    gae = delta + self.gamma * self.lam * masks[i] * gae
                    returns.insert(0, gae + v_old[i])

                # print(rewards)
                # Overwrite the rewards now
                rewards = th.tensor(returns).float().to(device)

                # Calculate the explained variance, to see how accurate the GAE really is...
                # print(rewards.shape)
                # print(v_prediction.shape)
                explained_variance = 1 - \
                    (rewards-v_prediction).var() / rewards.var()

                # print(rewards)
                # Get log probs
                action_log_probs = self.agent.policy.get_logprob_of_action(
                    pi_distribution, actions)

                entropy = self.agent.policy.pi_head.entropy(
                    pi_distribution).to(device)

                # Calculate clipped surrogate objective
                ratios = (action_log_probs -
                          old_action_log_probs).exp().to(device)
                advantages = normalize(
                    rewards - v_prediction.detach().to(device))
                surr1 = ratios * advantages
                surr2 = ratios.clamp(
                    1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = - th.min(surr1, surr2) - self.beta_s * entropy

                # Calculate clipped value loss
                value_clipped = (v_old +
                                 (v_prediction - v_old).clamp(-self.value_clip, self.value_clip)).to(device)

                value_loss_1 = (value_clipped.squeeze() - rewards) ** 2
                value_loss_2 = (v_prediction.squeeze() - rewards) ** 2

                value_loss = th.mean(th.max(value_loss_1, value_loss_2))

                loss = policy_loss.mean() + self.value_loss_weight * value_loss

                if self.plot:
                    self.pi_loss_history.append(policy_loss.mean().item())
                    self.v_loss_history.append(value_loss.item())
                    self.total_loss_history.append(loss.item())

                    self.expl_var_history.append(explained_variance.item())

                    self.entropy_history.append(entropy.mean().item())

                loss.backward()

                self.optim.step()
                self.optim.zero_grad()

            # Update plot at the end of every epoch
            if self.plot:
                # Update policy loss plot
                self.pi_loss_plot.set_ydata(self.pi_loss_history)
                self.pi_loss_plot.set_xdata(
                    range(len(self.pi_loss_history)))
                self.ax[0, 0].relim()
                self.ax[0, 0].autoscale_view(True, True, True)

                # Update value loss plot
                self.v_loss_plot.set_ydata(self.v_loss_history)
                self.v_loss_plot.set_xdata(
                    range(len(self.v_loss_history)))
                self.ax[0, 1].relim()
                self.ax[0, 1].autoscale_view(True, True, True)

                # Update total loss plot
                self.total_loss_plot.set_ydata(self.total_loss_history)
                self.total_loss_plot.set_xdata(
                    range(len(self.total_loss_history)))
                self.ax[0, 2].relim()
                self.ax[0, 2].autoscale_view(True, True, True)

                # Update the entropy plot
                self.entropy_plot.set_ydata(self.entropy_history)
                self.entropy_plot.set_xdata(range(len(self.entropy_history)))
                self.ax[1, 0].relim()
                self.ax[1, 0].autoscale_view(True, True, True)

                # Update the explained variance plot
                self.expl_var_plot.set_ydata(self.expl_var_history)
                self.expl_var_plot.set_xdata(
                    range(len(self.expl_var_history)))

                self.ax[1, 1].relim()
                self.ax[1, 1].autoscale_view(True, True, True)

                # Actually draw everything
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            # update_network(value_loss, self.optim_v)

    def run_train_loop(self):
        """
        Runs the basic PPO training loop
        """
        if self.plot:
            # Create a plot to show the progress of the training
            plt.ion()
            self.fig, self.ax = plt.subplots(2, 3, figsize=(10, 8))

            # Set up policy loss plot
            self.ax[0, 0].set_autoscale_on(True)
            self.ax[0, 0].autoscale_view(True, True, True)

            self.ax[0, 0].set_title("Policy Loss")

            self.pi_loss_plot, = self.ax[0, 0].plot(
                [], [], color="blue")

            # Setup value loss plot
            self.ax[0, 1].set_autoscale_on(True)
            self.ax[0, 1].autoscale_view(True, True, True)

            self.ax[0, 1].set_title("Value Loss")

            self.v_loss_plot, = self.ax[0, 1].plot(
                [], [], color="orange")

            # Set up total loss plot
            self.ax[0, 2].set_autoscale_on(True)
            self.ax[0, 2].autoscale_view(True, True, True)

            self.ax[0, 2].set_title("Total Loss")

            self.total_loss_plot, = self.ax[0, 2].plot(
                [], [], color="purple"
            )

            # Setup entropy plot
            self.ax[1, 0].set_autoscale_on(True)
            self.ax[1, 0].autoscale_view(True, True, True)
            self.ax[1, 0].set_title("Entropy")

            self.entropy_plot, = self.ax[1, 0].plot([], [], color="green")

            # Setup explained variance plot
            self.ax[1, 1].set_autoscale_on(True)
            self.ax[1, 1].autoscale_view(True, True, True)
            self.ax[1, 1].set_title("Explained Variance")

            self.expl_var_plot, = self.ax[1, 1].plot([], [], color="grey")

            # Setup reward plot
            self.ax[1, 2].set_autoscale_on(True)
            self.ax[1, 2].autoscale_view(True, True, True)
            self.ax[1, 2].set_title("Reward per Episode")

            self.reward_plot,  = self.ax[1, 2].plot([], [], color="red")

        for i in range(self.ppo_iterations):
            for eps in range(self.episodes):
                print(
                    f"🎬 Starting {self.env_name} episode {eps + 1}/{self.episodes}")
                self.run_episode(hard_reset=eps == 0)

            # Trim the size of memory buffer:
            if len(self.memories) > self.mem_buffer_size:
                self.memories.sort(key=lambda mem: mem.total_reward)
                self.memories = self.memories[-self.mem_buffer_size:]
                print(
                    f"⚠️ Trimmed memory buffer to length {self.mem_buffer_size} (worst - {self.memories[0].total_reward} | best - {self.memories[-1].total_reward})")

            self.learn()
            # self.memories.clear()


if __name__ == "__main__":
    rc = RewardsCalculator(
        damage_dealt=1,
        mob_kills=100
    )
    ppo = ProximalPolicyOptimizer(
        "MineRLPunchCowEz-v0",
        "models/foundation-model-1x.model",
        "weights/foundation-model-1x.weights",

        rc=rc,
        ppo_iterations=100,
        episodes=5,
        epochs=4,
        minibatch_size=20,
        lr=0.00001,
        betas=(0.9, 0.999),
        beta_s=0.999,
        eps_clip=0.1,
        value_clip=0.1,
        value_loss_weight=0.1,
        gamma=0.99,
        lam=0.95,
        mem_buffer_size=2000,
        plot=True,
    )

    ppo.run_train_loop()
