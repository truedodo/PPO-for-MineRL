
import pickle
import sys
from typing import List
import gym
import torch as th
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import copy

from datetime import datetime
from tqdm import tqdm
from rewards import RewardsCalculator
from memory import Memory, MemoryDataset
from util import to_torch_tensor, normalize, safe_reset, hard_reset
from vectorized_minerl import *

sys.path.insert(0, "vpt")  # nopep8

from agent import MineRLAgent  # nopep8
from lib.tree_util import tree_map  # nopep8

# For debugging purposes
# th.autograd.set_detect_anomaly(True)
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
# device = th.device("mps")  # apple silicon

TRAIN_WHOLE_MODEL = False


class ProximalPolicyOptimizer:
    def __init__(
            self,
            env_name: str,
            model_path: str,
            weights_path: str,

            # A custom reward function to be used
            rc: RewardsCalculator,

            # Hyperparameters
            num_rollouts: int, 
            num_steps: int, # the number of steps per rollout, T 
            epochs: int,
            minibatch_size: int,
            lr: float,
            weight_decay: float,
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
            num_envs: int,



    ):
        self.env_name = env_name
        self.num_envs = num_envs
        self.envs = init_vec_envs(self.env_name, self.num_envs)

        # Load the reward calculator
        self.rc = rc

        # Load hyperparameters unchanged
        self.num_rollouts = num_rollouts
        self.num_steps = num_steps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.lr = lr
        self.weight_decay = weight_decay
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

        self.agent = MineRLAgent(self.envs, policy_kwargs=policy_kwargs,
                                 pi_head_kwargs=pi_head_kwargs)
        self.agent.load_weights(weights_path)

        policy = self.agent.policy

        if TRAIN_WHOLE_MODEL:
            self.trainable_parameters = list(policy.parameters())
        else:
            self.trainable_parameters = list(
                policy.pi_head.parameters()) + list(policy.value_head.parameters())

        self.optim = th.optim.Adam(
            self.trainable_parameters, lr=lr, betas=betas, weight_decay=weight_decay)

        # self.optim_pi = th.optim.Adam(
        #     self.agent.policy.pi_head.parameters(), lr=lr, betas=betas)

        # self.optim_v = th.optim.Adam(
        #     self.agent.policy.value_head.parameters(), lr=lr, betas=betas)

        # Internal buffer of the most recent episode memories
        # This will be a relatively large chunk of data
        # Potential memory issues / optimizations around here...
        self.memories: List[Memory] = []

    def rollout(self, env, next_obs=None, next_done=False, next_hidden_state=None, hard_reset: bool = False):
        """
        Runs a rollout on the vectorized environments for `num_steps` timesteps and records the memories

        Returns `next_obs` and `next_done` for starting the next section of rollouts
        """
        start = datetime.now()

        # Temporary buffer to put the memories in before extending self.memories
        rollout_memories: List[Memory] = []

        # Only run this in the beginning
        if next_obs is None:
            # Initialize the hidden state vector
            next_hidden_state = self.agent.policy.initial_state(1)

            ## MineRL specific technical setup

            # Need to do a little bit of augmentation so the dataloader accepts the initial hidden state
            # This shouldn't affect anything; initial_state just uses None instead of empty tensors
            for i in range(len(next_hidden_state)):
                next_hidden_state[i] = list(next_hidden_state[i])
                next_hidden_state[i][0] = th.from_numpy(np.full(
                    (1, 1, 128), False)).to(device)


            if hard_reset:
                env.close()
                env = gym.make(self.env_name)
                next_obs = env.reset()
            else:
                next_obs = env.reset()
            
            next_done = False



        # This is a dummy tensor of shape (batchsize, 1) which was used as a mask internally
        dummy_first = th.from_numpy(np.array((False,))).to(device)
        dummy_first = dummy_first.unsqueeze(1)

        # This is not really used in training
        # More just for us to estimate the success of an episode
        episode_reward = 0

        for _ in range(self.num_steps):
            obs = next_obs
            done = next_done
            hidden_state = next_hidden_state

            # We have to do some resetting...
            if done:
                next_obs = env.reset()
                hidden_state = self.agent.policy.initial_state(1)
                for i in range(len(hidden_state)):
                    hidden_state[i] = list(hidden_state[i])
                    hidden_state[i][0] = th.from_numpy(np.full(
                        (1, 1, 128), False)).to(device)


            # Preprocess image
            agent_obs = self.agent._env_obs_to_agent(obs)

            # Basically just adds a dimension to both camera and button tensors
            agent_obs = tree_map(lambda x: x.unsqueeze(1), agent_obs)

            # print("Initial Hidden State:")
            # for i in range(len(hidden_state)):
            #     # print(hidden_state[i][0].shape)
            #     print(hidden_state[i][1][0].shape)
            #     print(hidden_state[i][1][1].shape)
            # Need to run this with no_grad or else the gradient descent will crash during training lol
            with th.no_grad():
                (pi_h, v_h), next_hidden_state = self.agent.policy.net(
                    agent_obs, hidden_state, context={"first": dummy_first})

                pi_distribution = self.agent.policy.pi_head(pi_h)
                v_prediction = self.agent.policy.value_head(v_h)

            action = self.agent.policy.pi_head.sample(
                pi_distribution, deterministic=False)

            # Get log probability of taking this action given pi
            action_log_prob = self.agent.policy.get_logprob_of_action(
                pi_distribution, action)

            # Process this so the env can accept it
            minerl_action = self.agent._agent_action_to_env(action)

            # Take action step in the environment
            next_obs, reward, next_done, info = env.step(minerl_action)

            # Immediately disregard the reward function from the environment
            reward = self.rc.get_rewards(obs, True)
            episode_reward += reward

            # Important! When we store a memory, we want the hidden state at the time of the observation as input! Not the step after
            memory = Memory(agent_obs, hidden_state, pi_h, v_h, action, action_log_prob,
                            reward, 0, done, v_prediction)

            rollout_memories.append(memory)

            # Finally, render the environment to the screen
            # Comment this out if you are boring
            env.render()



        # Reset the reward calculator once we are done with the episode
        self.rc.clear()

        # Calculate the generalized advantage estimate
        # This used to be done during each minibatch; however, the GAE should be more accurate
        # if we calculate it over the entire episode... I think?

        # It is not "more accurate," it MUST be calculated with rollouts after it, so in 
        # randomized minibatch would just be nonsense...
        
        # TODO need to use next_obs and next_done for this
        gae = 0
        returns = []

        v_preds = list(map(lambda mem: mem.value, rollout_memories))
        rewards = list(map(lambda mem: mem.reward, rollout_memories))

        masks = list(map(lambda mem: 1 - float(mem.done), rollout_memories))
        for i in reversed(range(len(rollout_memories))):

            # hacky but necessary since we don't have "next_state"
            v_next = v_preds[i + 1] if i != len(rollout_memories) - 1 else 0 # TODO insert next_obs here?

            delta = rewards[i] + self.gamma * \
                v_next * masks[i] - v_preds[i]
            gae = delta + self.gamma * self.lam * masks[i] * gae
            returns.insert(0, gae + v_preds[i])


        # Make changes to the memories for this episode before adding them to main buffer
        for i in range(len(rollout_memories)):
            # Replace raw reward with the GAE
            rollout_memories[i].reward = returns[i]

            # Remember the total reward for this episode
            rollout_memories[i].total_reward = episode_reward

        # Update internal memory buffer
        self.memories.extend(rollout_memories)

        if self.plot:
            # Update the reward plot
            self.reward_history.append(episode_reward)
            self.reward_plot.set_ydata(self.reward_history)
            self.reward_plot.set_xdata(range(len(self.reward_history)))

            self.ax[1, 2].relim()
            self.ax[1, 2].autoscale_view(True, True, True)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        end = datetime.now()
        print(
            f"✅ Rollout finished (duration - {end - start} | memories - {len(rollout_memories)} | total reward - {episode_reward})")
        
        return next_obs, next_done, next_hidden_state

    def learn(self):

        # Create dataloader from the memory buffer
        data = MemoryDataset(self.memories)

        # TODO should not be shuffling, rather rolling out again
        dl = DataLoader(data, batch_size=self.minibatch_size, shuffle=True)

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

                if TRAIN_WHOLE_MODEL:
                    (pi_h, v_h), state_out = policy.net(
                        agent_obs, state, context={"first": dummy_first})

                else:
                    # Use the hidden state calculated at the time since the model shouldn't change
                    pi_h, v_h = recorded_pi_h, recorded_v_h

                pi_distribution = self.agent.policy.pi_head(pi_h)
                v_prediction = self.agent.policy.value_head(v_h).to(device)

                masks = list(map(lambda d: 1-float(d), dones))

                # Overwrite the rewards now
                returns = rewards
                # returns = th.tensor(rewards).float().to(device)

                # Calculate the explained variance, to see how accurate the GAE really is...
                # print(rewards.shape)
                # print(v_prediction.shape)
                explained_variance = 1 - \
                    th.sub(returns, v_prediction).var() / returns.var()

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
                    returns - v_prediction.detach().to(device))
                surr1 = ratios * advantages
                surr2 = ratios.clamp(
                    1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = - th.min(surr1, surr2) - self.beta_s * entropy

                # Calculate clipped value loss
                # TODO we do not need to clip this - might even be worse than not
                value_clipped = (v_old +
                                 (v_prediction - v_old).clamp(-self.value_clip, self.value_clip)).to(device)

                # TODO what is this?
                value_loss_1 = (value_clipped.squeeze() - returns) ** 2
                value_loss_2 = (v_prediction.squeeze() - returns) ** 2

                value_loss = th.mean(th.max(value_loss_1, value_loss_2))

                loss = policy_loss.mean() + self.value_loss_weight * value_loss

                if self.plot:
                    self.pi_loss_history.append(policy_loss.mean().item())
                    self.v_loss_history.append(value_loss.item())
                    self.total_loss_history.append(loss.item())

                    self.expl_var_history.append(explained_variance.item())

                    self.entropy_history.append(entropy.mean().item())

                loss.backward()

                # th.nn.utils.clip_grad_norm_(
                #     self.trainable_parameters, MAX_GRAD_NORM)
                self.optim.step()

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

    def init_plot(self):
        # Create a plot to show the progress of the training
        plt.ion()
        self.fig, self.ax = plt.subplots(2, 3, figsize=(12, 8))

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

    def run_train_loop(self):
        """
        Runs the basic PPO training loop
        """
        if self.plot:
            self.init_plot()

        obss = [None]*self.num_envs
        dones = [False]*self.num_envs
        states = [None]*self.num_envs

        for i in range(self.num_rollouts):

            print(f"🎬 Starting {self.env_name} rollout {i + 1}/{self.num_rollouts}")

            obss_buffer = []
            dones_buffer = []
            states_buffer = []
            for env, next_obs, next_done, next_hidden_state in zip(self.envs, obss, dones, states):
                next_obs, next_done, next_hidden_state = self.rollout(env, next_obs, next_done, next_hidden_state)
                obss_buffer.append(next_obs)
                dones_buffer.append(next_done)
                states_buffer.append(next_hidden_state)
            
            obss = obss_buffer
            dones = dones_buffer
            states = states_buffer

            self.learn()

            # clear memories after every rollout
            self.memories.clear()


if __name__ == "__main__":
    rc = RewardsCalculator(
        damage_dealt=1,
        mob_kills=10000
    )
    rc.set_time_punishment(-10)
    ppo = ProximalPolicyOptimizer(
        "MineRLPunchCowEz-v0",
        "models/foundation-model-1x.model",
        "weights/foundation-model-1x.weights",
        num_envs=3,
        rc=rc,
        num_rollouts=100,
        num_steps=100,
        epochs=4,
        minibatch_size=20,
        lr=0.0001,
        weight_decay=0.00001,
        betas=(0.9, 0.999),
        beta_s=0.999,
        eps_clip=0.2,
        value_clip=0.1,
        value_loss_weight=0.15,
        gamma=0.99,
        lam=0.95,
        mem_buffer_size=10000,
        plot=True,
    )

    ppo.run_train_loop()
