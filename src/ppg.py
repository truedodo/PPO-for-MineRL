import pickle
import sys
import time
from typing import List
import gym
import torch as th
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy

from torch.utils.tensorboard import SummaryWriter


from datetime import datetime
from tqdm import tqdm
from memory import Memory, MemoryDataset, AuxMemory
from util import detach_hidden_states, fix_initial_hidden_states, squeeze_hidden_states, to_torch_tensor, normalize, safe_reset, hard_reset, calculate_gae
from vectorized_minerl import *

sys.path.insert(0, "vpt")  # nopep8

from agent import MineRLAgent  # nopep8
from lib.tree_util import tree_map  # nopep8

# For debugging purposes
th.autograd.set_detect_anomaly(True)
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
# device = th.device("mps")  # apple silicon

# TRAIN_WHOLE_MODEL = True
MAX_GRAD_NORM = 5.


class PhasicPolicyGradient:
    def __init__(
            self,
            env_name: str,
            model: str,
            weights: str,
            out_weights: str,
            save_every: int,


            # The number of PPG iterations to be run
            num_iterations: int,

            # The number of wake cycles (rollout/policy train) per iteration
            # This corresponds to the hyperparam N_pi in the PPG paper
            num_wake_cycles: int,

            # The number of steps per rollout
            T: int,

            # The length of subtrajectories used in training
            # Should satisfy these constraints:
            # 0 < l <= T
            # T % l == 0    this might actually not be necessary at all; todo: reevaluate
            l: int,


            # Number of epochs for the wake/policy phase
            # Note: this should always be 1!
            wake_epochs: int,

            # Number of epochs for the sleep/auxiliary phase
            sleep_epochs: int,

            # minibatch_size: int,
            lr: float,
            weight_decay: float,
            betas: tuple,
            beta_s: float,
            eps_clip: float,
            value_clip: float,
            # value_loss_weight: float,
            gamma: float,
            lam: float,
            beta_klp: float,


            # mem_buffer_size: int,
            beta_clone: float,

            # Optional: plot stuff
            plot: bool,
            num_envs: int

    ):
        model_path = f"models/{model}.model"
        weights_path = f"weights/{weights}.weights"
        self.out_weights_path = f"weights/{out_weights}.weights"
        self.training_name = f"ppo-{env_name}-{out_weights}-{int(time.time())}"

        self.env_name = env_name
        self.num_envs = num_envs
        self.envs = init_vec_envs(self.env_name, self.num_envs)

        self.save_every = save_every

        # Load hyperparameters unchanged
        self.num_iterations = num_iterations
        self.num_wake_cycles = num_wake_cycles

        self.T = T
        self.l = l
        self.wake_epochs = wake_epochs
        self.sleep_epochs = sleep_epochs
        # self.minibatch_size = minibatch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.beta_s = beta_s
        self.eps_clip = eps_clip
        self.value_clip = value_clip
        # self.value_loss_weight = value_loss_weight
        self.gamma = gamma
        self.lam = lam
        self.beta_klp = beta_klp
        self.beta_clone = beta_clone

        self.plot = plot

        # self.mem_buffer_size = mem_buffer_size

        if self.plot:
            # These statistics are calculated live during the episode running (rollout)
            self.live_reward_history = []
            self.live_value_history = []
            self.live_gae_history = []

        # Load the agent parameters from the weight files
        agent_parameters = pickle.load(open(model_path, "rb"))
        policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

        # Make our agents

        self.agent = MineRLAgent(self.envs, policy_kwargs=policy_kwargs,
                                 pi_head_kwargs=pi_head_kwargs)
        self.agent.load_weights(weights_path)

        # Set up the optimizer and learning rate scheduler for the agent
        agent_params = self.agent.policy.parameters()

        self.agent_optim = th.optim.Adam(
            agent_params, lr=lr, betas=betas, weight_decay=weight_decay)

        self.agent_scheduler = th.optim.lr_scheduler.LambdaLR(
            self.agent_optim, lambda x: 1 - x / num_iterations)

        # Create a SEPARATE VPT agent just for the critic
        self.critic = MineRLAgent(self.envs, policy_kwargs=policy_kwargs,
                                  pi_head_kwargs=pi_head_kwargs)
        self.critic.load_weights(weights_path)

        # Separate optimizer for the critic
        critic_params = self.critic.policy.parameters()

        self.critic_optim = th.optim.Adam(
            critic_params, lr=lr, betas=betas, weight_decay=weight_decay)

        self.critic_scheduler = th.optim.lr_scheduler.LambdaLR(
            self.critic_optim, lambda x: 1 - x / num_iterations)

        # Internal buffer of the most recent episode memories
        # This will be a relatively large chunk of data
        # Potential memory issues / optimizations around here...
        self.memories: List[List[Memory]] = []
        self.aux_memories: List[List[AuxMemory]] = []

        # Initialize the ORIGINAL MODEL for a KL divergence term during the Policy Phase
        # We will use KL divergence between our policy predictions and the original policy
        # This is to ensure that we don't deviate too far from the original policy
        self.orig_agent = MineRLAgent(self.envs, policy_kwargs=policy_kwargs,
                                      pi_head_kwargs=pi_head_kwargs)
        self.orig_agent.load_weights(weights_path)

        # Initialize the hidden states of this agent
        self.orig_hidden_states = {}
        for i in range(self.num_envs):
            self.orig_hidden_states[i] = self.orig_agent.policy.initial_state(
                1)

        # Setup tensorboard logging
        self.tb_writer = SummaryWriter()

        # Used for indexing tensorboard plots
        self.num_wake_updates = 0
        self.num_sleep_updates = 0
        # self.num_rollouts_so_far = 0  # name conflict with num_rollouts
        self.num_episodes_finished = 0

    def policy(self):
        '''
        Returns the policy network head, aux value head, and base
        '''

        return self.agent.policy.pi_head, self.agent.policy.value_head, self.agent.policy.net

    def value(self):
        '''
        Return the current value network  head and base
        '''
        return self.critic.policy.value_head, self.critic.policy.net

    def pi_and_v(self, agent_obs, policy_hidden_state, value_hidden_state, dummy_first, use_aux=False):
        """
        Returns the correct policy and value outputs
        """
        # Shorthand for networks
        policy, aux, policy_base = self.policy()
        value, value_base = self.value()

        (pi_h, aux_head), p_state_out = policy_base(
            agent_obs, policy_hidden_state, context={"first": dummy_first})
        (_, v_h), v_state_out = value_base(
            agent_obs, value_hidden_state, context={"first": dummy_first})

        if not use_aux:
            return policy(pi_h), value(v_h), p_state_out, v_state_out
        return policy(pi_h), value(v_h), aux(aux_head), p_state_out, v_state_out

    def init_plots(self):
        plt.ion()
        self.live_fig, self.live_ax = plt.subplots(1, 1, figsize=(6, 4))

        # Setup live plots
        self.live_ax.set_autoscale_on(True)
        self.live_ax.autoscale_view(True, True, True)
        self.live_ax.set_title("Episode Progress")
        self.live_ax.set_xlabel("steps")

        self.live_reward_plot, = self.live_ax.plot(
            [], [], color="red", label="Reward")
        self.live_value_plot, = self.live_ax.plot(
            [], [], color="blue", label="Value")
        self.live_gae_plot, = self.live_ax.plot(
            [], [], color="green", label="GAE")

        self.live_ax.legend(loc="upper right")

    def rollout(self, env, next_obs=None, next_done=False, next_policy_hidden_state=None, next_critic_hidden_state=None, hard_reset: bool = False):
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
            next_policy_hidden_state = self.agent.policy.initial_state(1)
            next_critic_hidden_state = self.critic.policy.initial_state(1)

            fix_initial_hidden_states(next_policy_hidden_state)
            fix_initial_hidden_states(next_critic_hidden_state)

            if hard_reset:
                env.close()
                env = gym.make(self.env_name)
                next_obs = env.reset()
            else:
                next_obs = env.reset()

            # Need this after every call of env.reset()
            # Ideally we would use a gym wrapper
            # But this works for now and is a quick patch
            env._cum_reward = 0

            next_done = False

        # This is a dummy tensor of shape (batchsize, 1) which was used as a mask internally
        dummy_first = th.from_numpy(np.array((False,))).to(device)
        dummy_first = dummy_first.unsqueeze(1)

        # Keep track of this because why not
        rollout_reward = 0

        if self.plot:
            self.live_reward_history.clear()
            self.live_value_history.clear()
            self.live_gae_history.clear()

        for _ in range(self.T):
            obs = next_obs
            done = next_done
            policy_hidden_state = next_policy_hidden_state
            critic_hidden_state = next_critic_hidden_state

            # We have to do some resetting...
            if done:

                next_obs = env.reset()
                env._cum_reward = 0
                policy_hidden_state = self.agent.policy.initial_state(1)
                critic_hidden_state = self.critic.policy.initial_state(1)

                fix_initial_hidden_states(policy_hidden_state)
                fix_initial_hidden_states(critic_hidden_state)

            # Preprocess image
            agent_obs = self.agent._env_obs_to_agent(obs)

            # Basically just adds a dimension to both camera and button tensors
            agent_obs = tree_map(lambda x: x.unsqueeze(1), agent_obs)

            with th.no_grad():
                pi_distribution, v_prediction, next_policy_hidden_state, next_critic_hidden_state \
                    = self.pi_and_v(agent_obs, policy_hidden_state, critic_hidden_state, dummy_first)

            action = self.agent.policy.pi_head.sample(
                pi_distribution, deterministic=False)

            # Get log probability of taking this action given pi
            action_log_prob = self.agent.policy.get_logprob_of_action(
                pi_distribution, action)

            # Process this so the env can accept it
            minerl_action = self.agent._agent_action_to_env(action)

            # Take action step in the environment
            next_obs, reward, next_done, info = env.step(minerl_action)
            env._cum_reward += reward  # Keep track of the episodic reawrd
            rollout_reward += reward

            if next_done:
                self.num_episodes_finished += 1
                self.tb_writer.add_scalar(
                    "Episodic Reward", env._cum_reward, self.num_episodes_finished)

            # Important! When we store a memory, we want the hidden state at the time of the observation as input! Not the step after
            # This is because we need to fully recreate the input when training the LSTM part of the network
            # memory = Memory(agent_obs, 0, 0, 0, action, action_log_prob,
            #                 reward, 0, next_done, v_prediction)
            memory = Memory(
                agent_obs=agent_obs,
                policy_hidden_state=policy_hidden_state,
                critic_hidden_state=critic_hidden_state,
                pi_h=0,
                v_h=0,
                action=action,
                action_log_prob=action_log_prob,
                reward=reward,
                total_reward=env._cum_reward,
                done=next_done,
                value=v_prediction
            )

            rollout_memories.append(memory)

            # Finally, render the environment to the screen
            # Comment this out if you are boring
            env.render()

            if self.plot:
                with torch.no_grad():
                    # Calculate the GAE up to this point
                    v_preds = list(
                        map(lambda mem: mem.value, rollout_memories))
                    rewards = list(
                        map(lambda mem: mem.reward, rollout_memories))
                    masks = list(
                        map(lambda mem: 1 - float(mem.done), rollout_memories))

                    agent_obs = self.agent._env_obs_to_agent(obs)
                    agent_obs = tree_map(lambda x: x.unsqueeze(1), agent_obs)
                    pi_distribution, v_prediction, next_policy_hidden_state, next_critic_hidden_state \
                        = self.pi_and_v(agent_obs, policy_hidden_state, critic_hidden_state, dummy_first)
                    returns = calculate_gae(
                        rewards, v_preds, masks, self.gamma, self.lam, v_prediction)

                    # Update data
                    self.live_reward_history.append(reward)
                    self.live_value_history.append(v_prediction.item())
                    self.live_gae_history = returns

                    # Update the plots
                    self.live_reward_plot.set_ydata(self.live_reward_history)
                    self.live_reward_plot.set_xdata(
                        range(len(self.live_reward_history)))

                    self.live_value_plot.set_ydata(self.live_value_history)
                    self.live_value_plot.set_xdata(
                        range(len(self.live_value_history)))

                    self.live_gae_plot.set_ydata(self.live_gae_history)
                    self.live_gae_plot.set_xdata(
                        range(len(self.live_gae_history)))

                    self.live_ax.relim()
                    self.live_ax.autoscale_view(True, True, True)

                    # Actually draw everything
                    self.live_fig.canvas.draw()
                    self.live_fig.canvas.flush_events()

        # Calculate the generalized advantage estimate
        v_preds = list(map(lambda mem: mem.value, rollout_memories))
        rewards = list(map(lambda mem: mem.reward, rollout_memories))
        masks = list(map(lambda mem: 1 - float(mem.done), rollout_memories))

        with th.no_grad():
            agent_obs = self.agent._env_obs_to_agent(obs)
            agent_obs = tree_map(lambda x: x.unsqueeze(1), agent_obs)
            pi_distribution, v_prediction, next_policy_hidden_state, next_critic_hidden_state \
                = self.pi_and_v(agent_obs, policy_hidden_state, critic_hidden_state, dummy_first)
            returns = calculate_gae(
                rewards, v_preds, masks, self.gamma, self.lam, v_prediction)

        # Make changes to the memories for this episode before adding them to main buffer
        for i in range(len(rollout_memories)):
            # Replace raw reward with the GAE
            rollout_memories[i].reward = returns[i]

            # Remember the total reward for this episode
            # TODO this is broken for PPG!
            rollout_memories[i].total_reward = rollout_reward

        # Update internal memory buffer
        # Now do this in run_train_loop for clarity
        self.memories.append(rollout_memories)

        end = datetime.now()
        print(
            f"✅ Rollout finished (duration: {end - start} | memories: {len(rollout_memories)} | rollout reward: {rollout_reward})")

        return next_obs, next_done, next_policy_hidden_state, next_critic_hidden_state

    def learn_policy_phase(self):

        # Check that we have consistent trajectory lengths
        for mem in self.memories:
            assert len(
                mem) == self.T, f"Got {len(mem)} memories but expected {self.T}"

        memories_arr = np.array(self.memories)

        # Note: In PPO this can be variable, but technically it should always be 1
        # In PPG, we always set it to 1 anyway
        # (Going over the data >1 times causes the policy to overfit)
        for _ in range(self.wake_epochs):

            # During training, the model reconstructs a hidden state for subtrajectories of length l
            # We have num_envs * T memories
            # We divide these into num_envs * (T // l) subtrajectories
            # Each time we run the model, we are stepping through each subtrajectory in parallel
            # (technically we aren't exactly running it in parallel, but the calculation is independent)
            # This allows us to run the model on num_envs * (T // l) inputs at once, rather than one at a time
            # As a result, the model is run and updated l times during training, hence this for loop
            for i in tqdm(range(self.l), desc="🧠 Policy Updates"):

                # Need to collect all the current memories in one list
                # This contains the i-th memory from each subtrajectory
                current_memories: List[Memory] = []
                # print(memories_arr.shape)

                for n in range(self.num_envs):
                    # Get the indices of the current memory in each subtrajectory
                    # For i == 0, l == 3, this is [0, 3, 6, 9, ... T]
                    idxs = np.arange(i, self.T, self.l)
                    mems = memories_arr[n][idxs]
                    # print(idxs)

                    current_memories.extend(mems)

                assert len(current_memories) == self.num_envs * (self.T //
                                                                 self.l), f"Got {len(data)} data points but expected {self.num_envs * (self.T // self.l)}"

                # Create dataloader so we can concatenate our subtrajectories into one input for the model
                data = MemoryDataset(current_memories)

                dl = DataLoader(
                    data, batch_size=len(data), shuffle=False)

                # We only get one batch containing step i from all subtrajectories
                agent_obss, policy_hidden_states, critic_hidden_states, _, _, actions, old_action_log_probs, rewards, _, dones, v_old = next(
                    iter(dl))

                if i == 0:
                    # In the 0-th step, we start with the hidden states from rollout
                    # So we just do the old thing of fixing dimensions from the dataloader
                    squeeze_hidden_states(policy_hidden_states)
                    squeeze_hidden_states(critic_hidden_states)

                else:
                    # Otherwise, we want to ignore the hidden states from rollout
                    # Instead, we use the calculated state from the previous step
                    # Apparently, reconstructing the hidden states during training is important
                    # See bible: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
                    policy_hidden_states = next_policy_hidden_states
                    critic_hidden_states = next_critic_hidden_states

                    detach_hidden_states(policy_hidden_states)
                    detach_hidden_states(critic_hidden_states)

                # Fix the dimensions that get skewed by the dataloader
                agent_obss = tree_map(lambda x: x.squeeze(1), agent_obss)

                # print(agent_obss)
                # print(agent_obss["img"].requires_grad)

                # Create dummy firsts tensors
                dummy_firsts = th.from_numpy(np.full(
                    (len(current_memories), 1), False)).to(device)

                # print(policy_hidden_states)
                # Run the agent to get the policy distribution
                (pi_h, _), next_policy_hidden_states = self.agent.policy.net(
                    agent_obss, policy_hidden_states, context={"first": dummy_firsts})

                pi_distribution = self.agent.policy.pi_head(pi_h)

                # See what the original model would predict for input
                # Note: We should really be running the entire original VPT instead of just the pi_head
                # Doing just the pi_head is cheaper, but not as useful
                with th.no_grad():
                    orig_pi_distribution = self.orig_agent.policy.pi_head(pi_h)

                # Calculate KL divergence
                kl_div = self.agent.policy.pi_head.kl_divergence(
                    pi_distribution, orig_pi_distribution)

                # Run the disjoint value network
                (_, v_h), next_critic_hidden_states = self.critic.policy.net(
                    agent_obss, critic_hidden_states, context={"first": dummy_firsts})

                v_prediction = self.critic.policy.value_head(v_h).to(device)

                # Calculate the log probs of the actual actions wrt the new policy distribution
                action_log_probs = self.agent.policy.get_logprob_of_action(
                    pi_distribution, actions)

                # The returns are stored in the `reward` field in memory, for some reason
                returns = normalize(rewards).to(device)
                # returns = rewards.to(device)

                # Calculate the explained variance, to see how accurate the GAE really is...
                explained_variance = 1 - \
                    th.sub(returns, v_prediction).var() / returns.var()

                # Calculate entropy
                entropy = self.agent.policy.pi_head.entropy(
                    pi_distribution).to(device)

                # Calculate clipped surrogate objective loss
                ratios = (action_log_probs -
                          old_action_log_probs).exp().to(device)
                advantages = (returns.detach() -
                              v_prediction.detach()).to(device)
                surr1 = ratios * advantages
                surr2 = ratios.clamp(
                    1 - self.eps_clip, 1 + self.eps_clip) * advantages

                policy_loss = - \
                    th.min(surr1, surr2) - self.beta_s * \
                    entropy - self.beta_klp * kl_div

                # Calculate clipped value loss
                value_clipped = (v_old +
                                 (v_prediction - v_old).clamp(-self.value_clip, self.value_clip)).to(device)
                value_loss_1 = (value_clipped.squeeze() - returns) ** 2

                # Calculate unclipped value loss
                value_loss_2 = (v_prediction.squeeze() - returns) ** 2

                # Our actual value loss
                # Previously, we removed the clipped value loss function
                # Although, I think intuitively it helps you stay closer to your prev value predictions
                # And since our value predictions are so bad, it might help
                # value_loss = th.mean(th.max(value_loss_1, value_loss_2))

                value_loss = value_loss_2.mean()
                # Backprop for policy

                th.nn.utils.clip_grad_norm_(
                    self.agent.policy.parameters(), MAX_GRAD_NORM)

                self.agent_optim.zero_grad()
                policy_loss.mean().backward()
                self.agent_optim.step()

                # Backprop for critic

                th.nn.utils.clip_grad_norm_(
                    self.critic.policy.parameters(), MAX_GRAD_NORM)

                self.critic_optim.zero_grad()
                value_loss.mean().backward()
                self.critic_optim.step()

                # Update tensorboard with metrics
                self.tb_writer.add_scalar(
                    "Wake Loss/Policy", policy_loss.mean().item(), self.num_wake_updates)
                self.tb_writer.add_scalar(
                    "Wake Loss/Value", value_loss.mean().item(), self.num_wake_updates)

                self.tb_writer.add_scalar(
                    "Wake Stats/Entropy", entropy.mean().item(), self.num_wake_updates)
                self.tb_writer.add_scalar(
                    "Wake Stats/KL Divergence from ORIGINAL", kl_div.mean().item(), self.num_wake_updates)
                self.tb_writer.add_scalar(
                    "Wake Stats/Explained Variance", explained_variance.item(), self.num_wake_updates)

                self.num_wake_updates += 1
        # self.agent_scheduler.step()
        # self.critic_scheduler.step()

    def calculate_policy_priors(self):
        '''
        This calculates the policy predictions for the memories using the current model
        This should be run after the last wake phase, before the aux phase
        During the aux phase, this is used to calculate the KL divergence between the current policy and the frozen policy from the last wake phase
        '''
        # Start by setting up the same batching system as used in wake/sleep learning

        # The auxiliary memory buffer contains the memories across multiple wake cycles
        # This will reshape the memories into the same structure as expected during wake
        # The difference is that each trajectory is of length T*num_wake_cycles instead of T
        memories_arr = np.hstack(
            np.split(np.array(self.aux_memories), self.num_wake_cycles))
        # print(memories_arr.shape)

        assert memories_arr.shape[0] == self.num_envs
        assert memories_arr.shape[1] == self.T * self.num_wake_cycles

        # We will collect all the policy predictions in a nice list here
        # This should be of size l
        pi_dists = []
        for i in tqdm(range(self.l), desc="⏸️  Policy Priors"):

            # Need to collect all the current memories in one list
            # This contains the i-th memory from each subtrajectory
            current_memories: List[Memory] = []
            # print(memories_arr.shape)
            for n in range(self.num_envs):
                # Get the indices of the current memory in each subtrajectory
                # For i == 0, l == 3, this is [0, 3, 6, 9, ... T]
                idxs = np.arange(i, self.T*self.num_wake_cycles, self.l)
                mems = memories_arr[n][idxs]
                # print(idxs)

                current_memories.extend(mems)

            data = MemoryDataset(current_memories)
            dl = DataLoader(
                data, batch_size=len(data), shuffle=False)

            # We only get one batch containing step i from all subtrajectories
            agent_obss, policy_hidden_states, critic_hidden_states, _, _, actions, old_action_log_probs, rewards, _, dones, values = next(
                iter(dl))

            if i == 0:
                # In the 0-th step, we start with the hidden states from rollout
                # So we just do the old thing of fixing dimensions from the dataloader
                squeeze_hidden_states(policy_hidden_states)
                # squeeze_hidden_states(critic_hidden_states)
            else:

                policy_hidden_states = next_policy_hidden_states
                # critic_hidden_states = next_critic_hidden_states

                detach_hidden_states(policy_hidden_states)
                # detach_hidden_states(critic_hidden_states)

            agent_obss = tree_map(lambda x: x.squeeze(1), agent_obss)
            dummy_firsts = th.from_numpy(np.full(
                (len(current_memories), 1), False)).to(device)

            # Run the policy model
            with th.no_grad():
                (pi_h, _), next_policy_hidden_states = self.agent.policy.net(
                    agent_obss, policy_hidden_states, context={"first": dummy_firsts})

                pi_distribution = self.agent.policy.pi_head(pi_h)

            pi_dists.append(pi_distribution)

        assert len(
            pi_dists) == self.l, f"Expected {self.l} policy priors, got {len(pi_dists)}"
        return pi_dists

    def learn_aux_phase(self, pi_priors):
        '''
        Run the auxiliary training phase for the value and aux value functions
        '''
        # Do the same reshaping trick as in calculate_policy_priors
        # self.aux_memories is set up slightly wrong for convenience!
        memories_arr = np.hstack(
            np.split(np.array(self.aux_memories), self.num_wake_cycles))

        for _ in tqdm(range(self.sleep_epochs), desc="😴 Auxiliary Epochs"):
            # Do the same batching procedure as in the wake phase
            for i in tqdm(range(self.l), desc="🧠 Auxiliary Updates", leave=False):
                current_memories: List[Memory] = []
                for n in range(self.num_envs):
                    # Get the indices of the current memory in each subtrajectory
                    # For i == 0, l == 3, this is [0, 3, 6, 9, ... T]
                    idxs = np.arange(i, self.T*self.num_wake_cycles, self.l)
                    mems = memories_arr[n][idxs]
                    # print(idxs)

                    current_memories.extend(mems)

                data = MemoryDataset(current_memories)
                dl = DataLoader(
                    data, batch_size=len(data), shuffle=False)

                # We only get one batch containing step i from all subtrajectories
                agent_obss, policy_hidden_states, critic_hidden_states, _, _, actions, old_action_log_probs, rewards, _, dones, values = next(
                    iter(dl))

                if i == 0:
                    # In the 0-th step, we start with the hidden states from rollout
                    # So we just do the old thing of fixing dimensions from the dataloader
                    squeeze_hidden_states(policy_hidden_states)
                    squeeze_hidden_states(critic_hidden_states)

                else:
                    # Otherwise, we want to ignore the hidden states from rollout
                    # Instead, we use the calculated state from the previous step
                    # Apparently, reconstructing the hidden states during training is important
                    # See bible: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
                    policy_hidden_states = next_policy_hidden_states
                    critic_hidden_states = next_critic_hidden_states

                    detach_hidden_states(policy_hidden_states)
                    detach_hidden_states(critic_hidden_states)

                # Fix the dimensions that get skewed by the dataloader
                agent_obss = tree_map(lambda x: x.squeeze(1), agent_obss)

                dummy_firsts = th.from_numpy(np.full(
                    (len(current_memories), 1), False)).to(device)

                # Run the agent to get the policy distribution and aux value prediction
                (pi_h, v_aux_h), next_policy_hidden_states = self.agent.policy.net(
                    agent_obss, policy_hidden_states, context={"first": dummy_firsts})

                pi_distribution = self.agent.policy.pi_head(pi_h)
                v_aux_prediction = self.agent.policy.value_head(
                    v_aux_h).to(device)

                # Normalize returns again
                returns = normalize(rewards).to(device)
                # returns = rewards.to(device)

                # Run the critic to get the main value prediction
                (_, v_h), next_critic_hidden_states = self.critic.policy.net(
                    agent_obss, critic_hidden_states, context={"first": dummy_firsts})

                v_prediction = self.critic.policy.value_head(v_h).to(device)

                aux_loss = .5 * (v_aux_prediction - returns.detach()) ** 2

                kl_term = self.agent.policy.pi_head.kl_divergence(
                    pi_distribution, pi_priors[i])

                joint_loss = aux_loss + self.beta_clone * kl_term

                # Calculate unclipped value loss
                value_loss = 0.5 * (v_prediction - returns.detach()) ** 2

                th.nn.utils.clip_grad_norm_(
                    self.agent.policy.parameters(), MAX_GRAD_NORM)

                # Optimize Ljoint wrt policy weights
                self.agent_optim.zero_grad()
                joint_loss.mean().backward()
                self.agent_optim.step()

                th.nn.utils.clip_grad_norm_(
                    self.critic.policy.parameters(), MAX_GRAD_NORM)

                # Optimize Lvalue wrt value weights
                self.critic_optim.zero_grad()
                value_loss.mean().backward()
                self.critic_optim.step()

                self.tb_writer.add_scalar(
                    "Sleep Loss/Value (Joint)", joint_loss.mean().item(), self.num_sleep_updates)

                self.tb_writer.add_scalar(
                    "Sleep Loss/Value (Critic)", value_loss.mean().item(), self.num_sleep_updates)

                self.tb_writer.add_scalar(
                    "Sleep Stats/KL Term", kl_term.mean().item(), self.num_sleep_updates)

                self.num_sleep_updates += 1

    def run_train_loop(self):
        """
        Runs the basic PPG training loop
        """
        if self.plot:
            self.init_plots()

        obss = [None]*self.num_envs
        dones = [False]*self.num_envs
        policy_states = [None]*self.num_envs
        critic_states = [None]*self.num_envs

        for i in range(self.num_iterations):

            if i % self.save_every == 0:
                state_dict = self.agent.policy.state_dict()
                th.save(state_dict, f'{self.out_weights_path}_{i}')
                print(
                    f"💾 Saved checkpoint weights to {self.out_weights_path}_{i}")

            print(
                f"🎬 Starting {self.env_name} rollout {i + 1}/{self.num_iterations}")

            # Do a server restart every 10 rollouts
            # Note: This is not one-to-one with the episodes
            # At T = 100, this is every 5 episodes
            should_hard_reset = i % 10 == 0 and i != 0

            obss_buffer = []
            dones_buffer = []
            policy_states_buffer = []
            critic_states_buffer = []

            # Run rollout until a reward is achieved in at least one environment
            actually_got_a_fucking_reward = False

            while not actually_got_a_fucking_reward:
                for env, next_obs, next_done, next_policy_hidden_state, next_critic_hidden_state \
                        in zip(self.envs, obss, dones, policy_states, critic_states):
                    next_obs, next_done, next_policy_hidden_state, next_critic_hidden_state = self.rollout(
                        env, next_obs, next_done, next_policy_hidden_state, next_critic_hidden_state, hard_reset=should_hard_reset)
                    obss_buffer.append(next_obs)
                    dones_buffer.append(next_done)
                    policy_states_buffer.append(next_policy_hidden_state)
                    critic_states_buffer.append(next_critic_hidden_state)

                # Check the rewards
                for traj in self.memories:
                    assert len(traj) == self.T
                    # Assuming our default reward is -1 per step
                    # Should generalize this; not gonna
                    if traj[0].total_reward != -self.T:
                        actually_got_a_fucking_reward = True

                if not actually_got_a_fucking_reward:
                    print(
                        "🔁 No rewards collected during this rollout! Clearing memories and starting new rollout")
                    self.memories.clear()

            # Need to give initial states from this rollout to re-rollout in learning with LSTM model
            self.learn_policy_phase()

            # Load this wake phase's memories into the aux memory buffer
            self.aux_memories.extend(self.memories)

            # NOW, we clear the memories so that we don't retrain on old data in the next wake phase
            self.memories.clear()

            # Every N_pi wake phases, we do an auxiliary phase
            if (i+1) % self.num_wake_cycles == 0:
                # Get pi predictions using current weights
                pi_priors = self.calculate_policy_priors()

                self.learn_aux_phase(pi_priors)

                self.aux_memories.clear()

            # Update from buffers AFTER learning...
            obss = obss_buffer
            dones = dones_buffer
            policy_states = policy_states_buffer
            critic_states = critic_states_buffer

            # Do learning rate annealing here so that it's more consistent i guess...
            self.agent_scheduler.step()
            self.critic_scheduler.step()


if __name__ == "__main__":

    ppg = PhasicPolicyGradient(
        env_name="MineRLPunchCowEz-v0",
        model="foundation-model-1x",
        weights="foundation-model-1x",
        out_weights="ppg-defeater-of-cows-1x",
        save_every=5,
        num_envs=4,
        num_iterations=500,
        num_wake_cycles=2,
        T=50,
        l=10,
        wake_epochs=1,
        sleep_epochs=4,
        lr=1e-5,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
        beta_s=0,  # no entropy in fine tuning!
        eps_clip=0.2,
        value_clip=0.2,
        # value_loss_weight=0.2,
        gamma=0.99,
        lam=0.95,
        beta_klp=1,

        beta_clone=1,
        plot=True,
    )

    ppg.run_train_loop()
